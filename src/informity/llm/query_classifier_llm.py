# ==============================================================================
# Informity AI — LLM-Based Query Classifier
# Uses the main LLM (via llm_engine) for query intent classification and
# metadata extraction.
# ==============================================================================

import json
import re
import time

import structlog

from informity.exceptions import LLMError
from informity.file_patterns import get_all_supported_extensions
from informity.llm.intent_normalization import normalize_intent_policy_fields
from informity.llm.intent_profiles import (
    IntentProfileId,
    rank_profile_candidates,
)
from informity.llm.metadata_filters import extract_metadata_filters
from informity.llm.model_adapter import get_profile
from informity.llm.nlp_heuristics import (
    extract_field_hint,
    has_aggregation_semantics,
    has_extraction_task,
    has_period_comparison_semantics,
    parse_query,
)
from informity.llm.nlp_heuristics import (
    is_filename_summary_query as nlp_is_filename_summary_query,
)
from informity.llm.nlp_heuristics import (
    is_inventory_plus_content_coverage as nlp_is_inventory_plus_content_coverage,
)
from informity.llm.query_classifier import QueryClassification
from informity.llm.query_patterns import (
    build_file_list_pattern,
)

log = structlog.get_logger(__name__)
_CLASSIFIER_RUNTIME_EXCEPTIONS = (RuntimeError, ValueError, TypeError, OSError)

_FILE_LIST_PATTERN = build_file_list_pattern()
_SUPPORTED_EXTENSIONS_CASEFOLD = tuple(ext.casefold() for ext in get_all_supported_extensions())


def _extract_phase2_constraints(query: str) -> dict[str, object]:
    """
    Extract deterministic Phase 2 constraints from query text.

    Returns structured fields used for trace visibility and downstream routing:
    - field_hint: normalized structured field hint (for example, 'box_1')
    - aggregation_semantics / period_comparison_semantics / extraction_task: semantic flags
    - source_terms: filename/source contains tokens extracted via metadata filters (regex fallback)
    - file_type: fallback extension filter when present in metadata filters

    Note: group_by and section are now extracted directly from LLM JSON output.
    source_terms here serves as a regex fallback if the LLM output is empty.
    """
    doc = parse_query(query)
    field_hint = extract_field_hint(doc)

    source_terms: list[str] = []
    file_type: str | None = None
    for metadata_filter in extract_metadata_filters(query):
        if metadata_filter.field == 'filename' and metadata_filter.operator == 'contains_any':
            if isinstance(metadata_filter.value, list):
                source_terms = [str(v) for v in metadata_filter.value if str(v).strip()]
        elif (
            metadata_filter.field == 'extension'
            and metadata_filter.operator == 'eq'
            and isinstance(metadata_filter.value, str)
        ):
            file_type = metadata_filter.value

    return {
        'field_hint': field_hint,
        'aggregation_semantics': has_aggregation_semantics(doc),
        'period_comparison_semantics': has_period_comparison_semantics(doc),
        'extraction_task': has_extraction_task(doc),
        'source_terms': source_terms,
        'file_type': file_type,
    }


def _is_exact_filename_constraint(value: str) -> bool:
    candidate = value.strip().strip('"\'')
    if not candidate:
        return False
    if any(separator in candidate for separator in ('/', '\\')):
        return True
    lowered = candidate.casefold()
    if lowered.endswith(' form') or lowered.endswith(' forms'):
        return False
    return any(lowered.endswith(extension) for extension in _SUPPORTED_EXTENSIONS_CASEFOLD)


def _is_inventory_plus_content_coverage_query(query: str) -> bool:
    return nlp_is_inventory_plus_content_coverage(parse_query(query))


def _is_filename_summary_query(query: str) -> bool:
    return nlp_is_filename_summary_query(parse_query(query))


def _resolve_route_candidate(
    *,
    llm_route_candidate: str | None,
    intent: str,
    response_shape: str,
    subtype: str | None,
    group_by: str | None,
) -> tuple[IntentProfileId, bool, list[str]]:
    """
    Resolve the routing profile and provenance for the given classification signals.

    Returns:
        (route_candidate, deterministic_override, reason_codes)
        deterministic_override: True when a hard aggregate rule fired (policy_aggregate_route_enforced),
            meaning the LLM route was not consulted or was overridden.
    """
    reason_codes: list[str] = []
    force_aggregate_route = intent == 'coverage' and subtype == 'aggregate_by_period'
    default_route_candidate: IntentProfileId
    if intent == 'metadata':
        default_route_candidate = 'metadata_inventory'
    elif intent == 'simple':
        default_route_candidate = 'clarification_or_disambiguation'
    elif force_aggregate_route:
        default_route_candidate = 'comparative_analysis'
    elif response_shape == 'structured_extract' or subtype == 'extract_structured_values':
        default_route_candidate = 'structured_field_extraction'
    elif intent == 'coverage' and group_by in {'year', 'category', 'file'}:
        default_route_candidate = 'comparative_analysis'
    elif intent == 'coverage':
        default_route_candidate = 'cross_document_synthesis'
    else:
        default_route_candidate = 'targeted_fact_lookup'
    valid_candidates: set[str] = {
        'metadata_inventory',
        'targeted_fact_lookup',
        'structured_field_extraction',
        'cross_document_synthesis',
        'comparative_analysis',
        'audit_or_compliance_brief',
        'continuation_or_refinement',
        'clarification_or_disambiguation',
    }
    allowed_by_intent: dict[str, set[str]] = {
        'metadata': {'metadata_inventory'},
        'simple': {'clarification_or_disambiguation'},
        'focused': {'targeted_fact_lookup', 'structured_field_extraction', 'continuation_or_refinement'},
        'coverage': {'cross_document_synthesis', 'comparative_analysis', 'audit_or_compliance_brief'},
    }
    route_candidate: IntentProfileId = default_route_candidate
    deterministic_override: bool
    if force_aggregate_route:
        # Hard aggregate rule: aggregate_by_period always maps to comparative_analysis.
        route_candidate = 'comparative_analysis'
        reason_codes.append('policy_aggregate_route_enforced')
        deterministic_override = True
    elif (
        isinstance(llm_route_candidate, str)
        and llm_route_candidate in valid_candidates
        and llm_route_candidate in allowed_by_intent.get(intent, set())
        # When the response shape is structured_extract, the default route is
        # structured_field_extraction. Don't let an LLM-selected coverage-only route
        # override this: structured_field_extraction better handles explicit table/extract
        # output contracts, and coverage routes don't allow structured_extract shapes.
        and not (response_shape == 'structured_extract' and llm_route_candidate in allowed_by_intent.get('coverage', set()))
    ):
        route_candidate = llm_route_candidate  # LLM-selected profile, validated against enum set
        reason_codes.append('llm_profile_selected')
        deterministic_override = False
    else:
        reason_codes.append('policy_default_selected')
        deterministic_override = False
    return route_candidate, deterministic_override, reason_codes


# ==============================================================================
# Category Inference Helper (uses centralized mapping)
# ==============================================================================

def _infer_category_from_file_type(file_type: str | None) -> str | None:
    """
    Infer category from file_type using centralized mapping.

    Uses category_patterns.get_category_for_extension() as single source of truth.

    Args:
        file_type: File extension (e.g., ".pdf", ".xlsx") or None

    Returns:
        Category string ("document", "plaintext", "data", "web") or None
    """
    if not file_type:
        return None

    from informity.category_patterns import get_category_for_extension
    from informity.db.models import FileCategory

    category = get_category_for_extension(file_type)

    # Convert FileCategory enum to string, exclude OTHER
    if category == FileCategory.OTHER:
        return None

    return category.value


# ==============================================================================
# Classification System Prompt and Few-Shot Priming
# ==============================================================================

# Few-shot examples prepended to every classification call.
# Qwen3 with GBNF JSON grammar can produce empty {} on the very first call in a
# fresh process; these examples establish reliable JSON output.
# User content is stored WITHOUT the no_think suffix — it is appended at call
# time based on profile.no_think_token so the KV cache prefix always matches.
_CLASSIFICATION_FEW_SHOT_PAIRS: tuple[tuple[str, str], ...] = (
    (
        'list all PDFs',
        '{"intent": "metadata", "response_shape": "narrative_synthesis",'
        ' "route_candidate": "metadata_inventory", "year": null,'
        ' "category": "document", "file_type": ".pdf", "filename": null,'
        ' "block_type": null, "section": null, "group_by": null, "source_terms": [],'
        ' "is_continuation": false, "is_scope_reset": false}',
    ),
    (
        'what information is in 2022 Acme Invoice.pdf?',
        '{"intent": "focused", "response_shape": "narrative_synthesis",'
        ' "route_candidate": "targeted_fact_lookup", "year": 2022,'
        ' "category": null, "file_type": ".pdf",'
        ' "filename": "2022 Acme Invoice.pdf",'
        ' "block_type": null, "section": null, "group_by": null, "source_terms": [],'
        ' "is_continuation": false, "is_scope_reset": false}',
    ),
    (
        'how many documents are indexed?',
        '{"intent": "metadata", "response_shape": "narrative_synthesis",'
        ' "route_candidate": "metadata_inventory", "year": null,'
        ' "category": null, "file_type": null, "filename": null,'
        ' "block_type": null, "section": null, "group_by": null, "source_terms": [],'
        ' "is_continuation": false, "is_scope_reset": false}',
    ),
    (
        'continue',
        '{"intent": "simple", "response_shape": "narrative_synthesis",'
        ' "route_candidate": "continuation_or_refinement", "year": null,'
        ' "category": null, "file_type": null, "filename": null,'
        ' "block_type": null, "section": null, "group_by": null, "source_terms": [],'
        ' "is_continuation": true, "is_scope_reset": false}',
    ),
    (
        'start over, I want to ask about something else entirely',
        '{"intent": "simple", "response_shape": "narrative_synthesis",'
        ' "route_candidate": "clarification_or_disambiguation", "year": null,'
        ' "category": null, "file_type": null, "filename": null,'
        ' "block_type": null, "section": null, "group_by": null, "source_terms": [],'
        ' "is_continuation": false, "is_scope_reset": true}',
    ),
    (
        'what kind of documents do you have indexed?',
        '{"intent": "simple", "response_shape": "narrative_synthesis",'
        ' "route_candidate": "clarification_or_disambiguation", "year": null,'
        ' "category": null, "file_type": null, "filename": null,'
        ' "block_type": null, "section": null, "group_by": null, "source_terms": [],'
        ' "is_continuation": false, "is_scope_reset": false}',
    ),
    (
        'tell me about the documents from 2021',
        '{"intent": "coverage", "response_shape": "narrative_synthesis",'
        ' "route_candidate": "cross_document_synthesis", "year": 2021,'
        ' "category": null, "file_type": null, "filename": null,'
        ' "block_type": null, "section": null, "group_by": null, "source_terms": [],'
        ' "is_continuation": false, "is_scope_reset": false}',
    ),
    (
        'find any document mentioning FATCA or foreign account reporting',
        '{"intent": "focused", "response_shape": "narrative_synthesis",'
        ' "route_candidate": "targeted_fact_lookup", "year": null,'
        ' "category": null, "file_type": null, "filename": null,'
        ' "block_type": null, "section": null, "group_by": null, "source_terms": [],'
        ' "is_continuation": false, "is_scope_reset": false}',
    ),
    (
        'what does the employment contract say about termination?',
        '{"intent": "focused", "response_shape": "narrative_synthesis",'
        ' "route_candidate": "targeted_fact_lookup", "year": null,'
        ' "category": null, "file_type": null, "filename": null,'
        ' "block_type": null, "section": null, "group_by": null, "source_terms": ["employment contract"],'
        ' "is_continuation": false, "is_scope_reset": false}',
    ),
    (
        'compare revenue by year across all annual reports',
        '{"intent": "coverage", "response_shape": "narrative_synthesis",'
        ' "route_candidate": "comparative_analysis", "year": null,'
        ' "category": null, "file_type": null, "filename": null,'
        ' "block_type": null, "section": null, "group_by": "year", "source_terms": [],'
        ' "is_continuation": false, "is_scope_reset": false}',
    ),
    (
        'which indexed documents mention project deadlines? list all files and summarize what each one says',
        '{"intent": "coverage", "response_shape": "narrative_synthesis",'
        ' "route_candidate": "cross_document_synthesis", "year": null,'
        ' "category": null, "file_type": null, "filename": null,'
        ' "block_type": null, "section": null, "group_by": null, "source_terms": [],'
        ' "is_continuation": false, "is_scope_reset": false}',
    ),
    (
        'what are the most common themes mentioned across all indexed documents?',
        '{"intent": "coverage", "response_shape": "narrative_synthesis",'
        ' "route_candidate": "cross_document_synthesis", "year": null,'
        ' "category": null, "file_type": null, "filename": null,'
        ' "block_type": null, "section": null, "group_by": null, "source_terms": [],'
        ' "is_continuation": false, "is_scope_reset": false}',
    ),
    (
        'create a report with sections ## Overview, ## Supporting Documents, ## Key Data, ## Action Items. Under ## Key Data include a markdown table.',
        '{"intent": "coverage", "response_shape": "narrative_synthesis",'
        ' "route_candidate": "cross_document_synthesis", "year": null,'
        ' "category": null, "file_type": null, "filename": null,'
        ' "block_type": null, "section": null, "group_by": null, "source_terms": [],'
        ' "is_continuation": false, "is_scope_reset": false}',
    ),
)


def build_classification_messages(query: str, no_think_suffix: str) -> list[dict[str, str]]:
    """Build the full classification message list with profile-appropriate no_think suffix."""
    messages: list[dict[str, str]] = [{'role': 'system', 'content': CLASSIFICATION_SYSTEM_PROMPT}]
    for user_text, assistant_text in _CLASSIFICATION_FEW_SHOT_PAIRS:
        messages.append({'role': 'user', 'content': user_text + no_think_suffix})
        messages.append({'role': 'assistant', 'content': assistant_text})
    messages.append({'role': 'user', 'content': query + no_think_suffix})
    return messages

CLASSIFICATION_SYSTEM_PROMPT = """OUTPUT FORMAT: Return ONLY valid JSON. No markdown, no explanations, no code blocks.

You are a query intent classifier. Output this exact JSON structure:
{
  "intent": "<metadata|focused|coverage|simple>",
  "response_shape": "<structured_extract|narrative_synthesis>",
  "route_candidate": "<metadata_inventory|targeted_fact_lookup|structured_field_extraction|cross_document_synthesis|comparative_analysis|audit_or_compliance_brief|continuation_or_refinement|clarification_or_disambiguation>",
  "year": <null|integer>,
  "category": <null|"document"|"plaintext"|"data"|"web"|"code">,
  "file_type": <null|".pdf"|".txt"|".docx"|".xlsx"|...>,
  "filename": <null|string>,
  "block_type": <null|"table"|"form"|"narrative">,
  "section": <null|string>,
  "group_by": <null|"year"|"category"|"file">,
  "source_terms": <[]|[string...]>,
  "is_continuation": <true|false>,
  "is_scope_reset": <true|false>
}

FIELD RULES:
- year: Explicit years only (2022, 2023). If asking ABOUT years, use null
- category: Only for metadata queries. Always null for focused/coverage
- file_type: File extensions mentioned. Null if none
- filename: Set when query has explicit filename constraint (exact file name or filename contains pattern)
- block_type: For focused/coverage only when user explicitly asks for table/form/narrative content
- section: For focused/coverage section targeting (e.g., "introduction", "conclusion"). Null if none
- group_by: Set to "year", "category", or "file" when query explicitly groups or compares by that dimension (e.g. "by year", "per year", "by category", "group by file"). Null otherwise
- source_terms: Array of informal document reference terms (e.g. ["employment contract", "Q3 report"]). Only include named document references that narrow which files to search. Use [] for concept keywords (FATCA, inflation) or when no specific document is referenced
- is_continuation: true ONLY for brief bare continuation phrases ("continue", "go on", "more", "next section", "the rest", "keep going"). False for queries with any specific content — "more about the cash flow section" is a NEW focused query (false), not a continuation
- is_scope_reset: true ONLY when user explicitly abandons the current topic ("start over", "ignore everything above", "never mind, different question"). "start over on X" is a NEW focused query on X (false), not a scope reset
- response_shape:
  - structured_extract: user asks for a SINGLE table or field-by-field extraction as the ENTIRE response (no surrounding sections)
  - narrative_synthesis: user asks for summaries, briefs, comparisons, evidence maps, risks, recommendations — and ANY multi-section response with ## headings (even if one section contains an embedded table)

INTENT TYPES:
- metadata: Structural questions about the file index: file counts, file listings, which years/types/categories are present. The answer comes from file metadata only, not from document content.
- focused: Read or extract content from a specific document or a small set of documents identified by name or concept.
- coverage: Synthesize, compare, or aggregate content across many documents — or find which documents contain a concept.
- simple: Greetings, capability questions ("what can you do", "what kind of documents do you have", "what information is available"), or general conversation unrelated to reading document content.

CRITICAL ROUTING RULE:
- If the answer is a list or count of files themselves = metadata. File type, category, and year are filters on the file list, not content operations.
- If the answer requires reading what is inside documents = focused (one/few docs) or coverage (many docs).
- "List all X files" / "How many X files" / "Show files from year Y" = metadata (file inventory).
- "What is in X.pdf" / "Summarize X.pdf" / "What does X say about Y" = focused or coverage (content).
- "What kind of documents do you have" / "What can you help with" / "What information is available" = simple (capability question, not a file listing or content retrieval).
- "Tell me about documents from 2021" / "What do the 2022 records show" = coverage (requires reading content, not just listing files).
- "Find any document mentioning X" / "Which document contains X" = focused (stop at first/best match; targeted concept lookup).
- "Which documents contain X, list all of them" / "Which indexed documents have Y, list files and details" = coverage (corpus-wide survey requiring content reading from every document).
- "X mentioned across all indexed documents" / "across all documents" / "across all indexed" = coverage (aggregation across entire corpus).
- "Produce a response with sections ## A, ## B, ## C" (multi-section output with required headings) = coverage + narrative_synthesis even if one section contains a table. Only use structured_extract when the ENTIRE response is a single table or field list with no surrounding sections.
- category must be null for focused/coverage; use category only for metadata queries.

EXAMPLES (output JSON only):

Query: "how many PDFs do I have from 2022?"
Output: {"intent": "metadata", "response_shape": "narrative_synthesis", "route_candidate": "metadata_inventory", "year": 2022, "category": "document", "file_type": ".pdf", "filename": null, "block_type": null, "section": null, "group_by": null, "source_terms": [], "is_continuation": false, "is_scope_reset": false}

Query: "how many years of documents do I have?"
Output: {"intent": "metadata", "response_shape": "narrative_synthesis", "route_candidate": "metadata_inventory", "year": null, "category": null, "file_type": null, "filename": null, "block_type": null, "section": null, "group_by": null, "source_terms": [], "is_continuation": false, "is_scope_reset": false}

Query: "list all files from 2023"
Output: {"intent": "metadata", "response_shape": "narrative_synthesis", "route_candidate": "metadata_inventory", "year": 2023, "category": null, "file_type": null, "filename": null, "block_type": null, "section": null, "group_by": null, "source_terms": [], "is_continuation": false, "is_scope_reset": false}

Query: "continue"
Output: {"intent": "simple", "response_shape": "narrative_synthesis", "route_candidate": "continuation_or_refinement", "year": null, "category": null, "file_type": null, "filename": null, "block_type": null, "section": null, "group_by": null, "source_terms": [], "is_continuation": true, "is_scope_reset": false}

Query: "more detail on the cash flow section"
Output: {"intent": "focused", "response_shape": "narrative_synthesis", "route_candidate": "targeted_fact_lookup", "year": null, "category": null, "file_type": null, "filename": null, "block_type": null, "section": "cash flow", "group_by": null, "source_terms": [], "is_continuation": false, "is_scope_reset": false}

Query: "start over, I want to ask something completely different"
Output: {"intent": "simple", "response_shape": "narrative_synthesis", "route_candidate": "clarification_or_disambiguation", "year": null, "category": null, "file_type": null, "filename": null, "block_type": null, "section": null, "group_by": null, "source_terms": [], "is_continuation": false, "is_scope_reset": true}

Query: "what does the employment contract say about vacation days?"
Output: {"intent": "focused", "response_shape": "narrative_synthesis", "route_candidate": "targeted_fact_lookup", "year": null, "category": null, "file_type": null, "filename": null, "block_type": null, "section": null, "group_by": null, "source_terms": ["employment contract"], "is_continuation": false, "is_scope_reset": false}

Query: "what information is in 2022 Acme Invoice.pdf?"
Output: {"intent": "focused", "response_shape": "narrative_synthesis", "route_candidate": "targeted_fact_lookup", "year": 2022, "category": null, "file_type": ".pdf", "filename": "2022 Acme Invoice.pdf", "block_type": null, "section": null, "group_by": null, "source_terms": [], "is_continuation": false, "is_scope_reset": false}

Query: "summarize the content of 2018 Company Report.pdf"
Output: {"intent": "focused", "response_shape": "narrative_synthesis", "route_candidate": "targeted_fact_lookup", "year": 2018, "category": null, "file_type": ".pdf", "filename": "2018 Company Report.pdf", "block_type": null, "section": null, "group_by": null, "source_terms": [], "is_continuation": false, "is_scope_reset": false}

Query: "extract the summary table values from the annual report"
Output: {"intent": "focused", "response_shape": "structured_extract", "route_candidate": "structured_field_extraction", "year": null, "category": null, "file_type": null, "filename": null, "block_type": "table", "section": null, "group_by": null, "source_terms": ["annual report"], "is_continuation": false, "is_scope_reset": false}

Query: "summarize the key findings across all annual reports"
Output: {"intent": "coverage", "response_shape": "narrative_synthesis", "route_candidate": "cross_document_synthesis", "year": null, "category": null, "file_type": null, "filename": null, "block_type": null, "section": null, "group_by": null, "source_terms": [], "is_continuation": false, "is_scope_reset": false}

Query: "compare revenue by year across all annual reports"
Output: {"intent": "coverage", "response_shape": "narrative_synthesis", "route_candidate": "comparative_analysis", "year": null, "category": null, "file_type": null, "filename": null, "block_type": null, "section": null, "group_by": "year", "source_terms": [], "is_continuation": false, "is_scope_reset": false}

Query: "Build a compliance reconciliation brief across 3 years with required sections and evidence-backed deltas."
Output: {"intent": "coverage", "response_shape": "narrative_synthesis", "route_candidate": "audit_or_compliance_brief", "year": null, "category": null, "file_type": null, "filename": null, "block_type": null, "section": null, "group_by": "year", "source_terms": [], "is_continuation": false, "is_scope_reset": false}

Query: "hello, what can you do?"
Output: {"intent": "simple", "response_shape": "narrative_synthesis", "route_candidate": "clarification_or_disambiguation", "year": null, "category": null, "file_type": null, "filename": null, "block_type": null, "section": null, "group_by": null, "source_terms": [], "is_continuation": false, "is_scope_reset": false}

Query: "what kind of documents do you have indexed?"
Output: {"intent": "simple", "response_shape": "narrative_synthesis", "route_candidate": "clarification_or_disambiguation", "year": null, "category": null, "file_type": null, "filename": null, "block_type": null, "section": null, "group_by": null, "source_terms": [], "is_continuation": false, "is_scope_reset": false}

Query: "tell me about the documents from 2021"
Output: {"intent": "coverage", "response_shape": "narrative_synthesis", "route_candidate": "cross_document_synthesis", "year": 2021, "category": null, "file_type": null, "filename": null, "block_type": null, "section": null, "group_by": null, "source_terms": [], "is_continuation": false, "is_scope_reset": false}

Query: "find any document mentioning FATCA or foreign account reporting"
Output: {"intent": "focused", "response_shape": "narrative_synthesis", "route_candidate": "targeted_fact_lookup", "year": null, "category": null, "file_type": null, "filename": null, "block_type": null, "section": null, "group_by": null, "source_terms": [], "is_continuation": false, "is_scope_reset": false}

Query: "how much is the largest amount mentioned in any document?"
Output: {"intent": "focused", "response_shape": "narrative_synthesis", "route_candidate": "targeted_fact_lookup", "year": null, "category": null, "file_type": null, "filename": null, "block_type": null, "section": null, "group_by": null, "source_terms": [], "is_continuation": false, "is_scope_reset": false}

Query: "which indexed documents mention project deadlines? list all files and summarize what each one says"
Output: {"intent": "coverage", "response_shape": "narrative_synthesis", "route_candidate": "cross_document_synthesis", "year": null, "category": null, "file_type": null, "filename": null, "block_type": null, "section": null, "group_by": null, "source_terms": [], "is_continuation": false, "is_scope_reset": false}

Query: "what are the most common themes mentioned across all indexed documents?"
Output: {"intent": "coverage", "response_shape": "narrative_synthesis", "route_candidate": "cross_document_synthesis", "year": null, "category": null, "file_type": null, "filename": null, "block_type": null, "section": null, "group_by": null, "source_terms": [], "is_continuation": false, "is_scope_reset": false}

Query: "create a report with sections ## Overview, ## Supporting Documents, ## Key Data, ## Action Items. Under ## Key Data include a markdown table."
Output: {"intent": "coverage", "response_shape": "narrative_synthesis", "route_candidate": "cross_document_synthesis", "year": null, "category": null, "file_type": null, "filename": null, "block_type": null, "section": null, "group_by": null, "source_terms": [], "is_continuation": false, "is_scope_reset": false}

EDGE CASES:
- Ambiguous (single-doc vs multi-doc unclear, no scope signal) = prefer "focused"
- Queries with "across all documents", "across all indexed", "all indexed documents", or asking to survey/list from the entire corpus = "coverage" regardless of other signals
- Multiple intents = choose the intent that determines retrieval strategy
- Greeting + non-document question = "simple"
- Greeting + document question = classify by document task (metadata/focused/coverage), not "simple"
- "What kind of documents" / "what do you have" / "what can you help with" / "what information is available" = "simple" even if they mention "documents" — these are capability questions, not file listings or content retrievals

CRITICAL: Output MUST start with { and end with }. No code blocks, no explanations."""


# ==============================================================================
# LLM Query Classifier
# ==============================================================================


def classify_query_llm(query: str) -> QueryClassification:
    """
    Classify query using the main LLM engine with structured JSON output.

    Args:
        query: User query string

    Returns:
        QueryClassification with intent and extracted filters

    Raises:
        LLMError: If inference fails
    """
    from informity.llm.engine import llm_engine

    classify_start = time.perf_counter()
    profile = get_profile()
    # Build messages with profile-appropriate no_think suffix on every user turn.
    # The suffix must be identical in both few-shot and real query for KV cache reuse.
    no_think_suffix = f'\n{profile.no_think_token}' if profile.no_think_token else ''
    messages = build_classification_messages(query, no_think_suffix)
    stops = profile.get_stop_sequences(reasoning_enabled=False)
    # JSON mode (GBNF grammar) constrains output to valid JSON; /no_think suppresses thinking tokens

    try:
        response = llm_engine.chat_complete(
            messages=messages,
            max_tokens=400,   # larger to fit group_by, source_terms array, is_continuation/is_scope_reset, and long filenames
            temperature=0.0,  # deterministic
            stop=stops,
            response_format={'type': 'json_object'},  # Enforces valid JSON via GBNF grammar
        )
    except _CLASSIFIER_RUNTIME_EXCEPTIONS as exc:
        raise LLMError(f'Classification inference failed: {exc}') from exc

    # Extract content from response
    if not response or 'choices' not in response or not response['choices']:
        log.error(
            'classifier_empty_response',
            response_present=bool(response),
            choices_present=bool(response and 'choices' in response),
        )
        raise LLMError('Empty response from classification model')

    choice = response['choices'][0]
    if 'message' not in choice or 'content' not in choice['message']:
        log.error(
            'classifier_invalid_response_structure',
            choice_keys=sorted(choice.keys()) if isinstance(choice, dict) else [],
        )
        raise LLMError('Invalid response structure from classification model')

    content = choice['message']['content']
    if content is None:
        content = ''
    content = content.strip()

    if not content:
        log.error(
            'classifier_empty_content',
            query_length=len(query),
            payload_redacted=True,
            finish_reason=choice.get('finish_reason'),
        )
        raise LLMError('Empty content in classification response - model may have stopped early')

    # Parse JSON (JSON mode ensures valid JSON structure, so parse should always succeed)
    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        log.error(
            'classifier_json_parse_failed',
            content_length=len(content),
            query_length=len(query),
            payload_redacted=True,
            error=str(exc),
            finish_reason=choice.get('finish_reason'),
        )
        raise LLMError(f'Failed to parse classification JSON (unexpected with JSON mode): {exc}. Content: {content[:200]}') from exc

    # Validate and extract fields
    intent = data.get('intent', 'focused')
    if intent not in ('metadata', 'focused', 'coverage', 'simple'):
        log.warning(
            'invalid_intent_from_classifier',
            intent=intent,
            query_length=len(query),
        )
        intent = 'focused'  # Safe fallback

    response_shape = str(data.get('response_shape', 'narrative_synthesis')).strip().lower()
    if response_shape not in {'structured_extract', 'narrative_synthesis'}:
        response_shape = 'narrative_synthesis'

    llm_intent = intent

    year = data.get('year')
    if year is not None:
        try:
            year = int(year)
            if not (1900 <= year <= 2099):
                year = None
        except (ValueError, TypeError):
            year = None

    # Guardrail: multi-year queries should not collapse into a single-year classification filter.
    explicit_years = {int(m.group(0)) for m in re.finditer(r'\b(?:19|20)\d{2}\b', query)}
    if len(explicit_years) > 1:
        year = None

    # Category extraction: only for metadata queries
    category = None
    if intent == 'metadata':
        category = data.get('category')
        if category not in ('document', 'plaintext', 'data', 'web', 'code'):
            category = None

    file_type = data.get('file_type')
    if file_type and not file_type.startswith('.'):
        file_type = f'.{file_type}'

    phase2_constraints = _extract_phase2_constraints(query)
    explicit_file_type = (
        str(phase2_constraints['file_type'])
        if isinstance(phase2_constraints.get('file_type'), str)
        else None
    )
    if intent in ('focused', 'coverage') and file_type is not None and explicit_file_type is None:
        log.debug(
            'classifier_file_type_filter_dropped_no_explicit_extension',
            file_type=file_type,
            intent=intent,
        )
        file_type = None

    if file_type is None and isinstance(phase2_constraints.get('file_type'), str):
        file_type = str(phase2_constraints['file_type'])

    if intent == 'metadata' and category is None and file_type:
        inferred_category = _infer_category_from_file_type(file_type)
        if inferred_category:
            category = inferred_category
            log.debug(
                'category_inferred_from_file_type',
                file_type=file_type,
                inferred_category=category,
            )

    filename_raw = data.get('filename')
    filename = str(filename_raw).strip() or None if isinstance(filename_raw, str) else None
    if intent in ('focused', 'coverage') and filename is not None and not _is_exact_filename_constraint(filename):
        log.debug(
            'classifier_filename_filter_dropped_non_exact',
            filename=filename,
            intent=intent,
        )
        filename = None
    block_type = data.get('block_type')
    if block_type not in ('table', 'form', 'narrative'):
        block_type = None

    # Section: use LLM output directly (LLM has full query context).
    section_raw = data.get('section')
    section = str(section_raw).strip() if isinstance(section_raw, str) and str(section_raw).strip() else None

    # group_by: use LLM output directly.
    llm_group_by = data.get('group_by')
    group_by = llm_group_by if isinstance(llm_group_by, str) and llm_group_by in {'year', 'category', 'file'} else None

    field_hint_value = phase2_constraints.get('field_hint')
    field_hint = str(field_hint_value) if isinstance(field_hint_value, str) else None
    aggregation_semantics = bool(phase2_constraints.get('aggregation_semantics'))
    period_comparison_semantics = bool(phase2_constraints.get('period_comparison_semantics'))
    extraction_task = bool(phase2_constraints.get('extraction_task'))
    has_multi_year_scope = len(explicit_years) > 1
    subtype: str | None = None
    if intent == 'metadata':
        subtype = 'file_inventory'
    elif intent in ('focused', 'coverage'):
        if (
            (aggregation_semantics or period_comparison_semantics)
            and (
                group_by == 'year'
                or has_multi_year_scope
                or extraction_task
                or period_comparison_semantics
            )
        ):
            subtype = 'aggregate_by_period'
        elif response_shape == 'structured_extract' or field_hint is not None:
            subtype = 'extract_structured_values'

    (
        intent,
        subtype,
        response_shape,
        normalization_reason_codes,
    ) = normalize_intent_policy_fields(
        query=query,
        intent=intent,
        subtype=subtype,
        response_shape=response_shape,
        group_by=group_by,
        filename_filter=filename,
        has_multi_year_scope=has_multi_year_scope,
    )
    if intent == 'focused' and _is_inventory_plus_content_coverage_query(query):
        intent = 'coverage'
        response_shape = 'narrative_synthesis'
        subtype = None
        if 'policy_inventory_plus_content_to_coverage' not in normalization_reason_codes:
            normalization_reason_codes.append('policy_inventory_plus_content_to_coverage')

    # source_terms: use LLM output first; fall back to regex-based metadata_filters extraction.
    llm_source_terms = data.get('source_terms')
    if isinstance(llm_source_terms, list) and llm_source_terms:
        source_terms = [str(v) for v in llm_source_terms if str(v).strip()]
    else:
        fallback_source_terms = phase2_constraints.get('source_terms')
        source_terms = [str(v) for v in fallback_source_terms] if isinstance(fallback_source_terms, list) else []

    # is_continuation / is_scope_reset: use LLM output directly.
    is_continuation = bool(data.get('is_continuation'))
    is_scope_reset = bool(data.get('is_scope_reset'))

    is_metadata_query = (intent == 'metadata')
    is_file_list_query = bool(intent == 'metadata' and _FILE_LIST_PATTERN.search(query))

    effective_block_type_filter = block_type if intent in ('focused', 'coverage') else None
    effective_section_filter = section if intent in ('focused', 'coverage') else None
    llm_route_candidate_raw = data.get('route_candidate')
    llm_route_candidate = str(llm_route_candidate_raw).strip() if isinstance(llm_route_candidate_raw, str) else None
    route_candidate, deterministic_override, reason_codes = _resolve_route_candidate(
        llm_route_candidate=llm_route_candidate,
        intent=intent,
        response_shape=response_shape,
        subtype=subtype,
        group_by=group_by,
    )
    # llm_confidence: reserved for future LLM self-reported confidence; 0.0 until LLM emits a "confidence" field.
    llm_confidence = 0.0
    # Derive numeric confidence from provenance for downstream confidence_band consumers.
    # deterministic_override (hard aggregate rule) → 'high' band.
    # llm_profile_selected → 'high' band.
    # policy_default_selected → 'medium' band.
    if deterministic_override:
        confidence = 0.9
    elif 'llm_profile_selected' in reason_codes:
        confidence = 0.85
    else:
        confidence = 0.75
    if (
        intent == 'focused'
        and filename is not None
        and _is_filename_summary_query(query)
        and route_candidate == 'structured_field_extraction'
    ):
        route_candidate = 'targeted_fact_lookup'
        response_shape = 'narrative_synthesis'
        if 'policy_filename_summary_to_targeted_fact' not in reason_codes:
            reason_codes.append('policy_filename_summary_to_targeted_fact')
    for reason_code in normalization_reason_codes:
        if reason_code not in reason_codes:
            reason_codes.append(reason_code)
    alternatives = rank_profile_candidates(
        route_candidate=route_candidate,
        confidence=confidence,
    )[1:]
    missing_slots: list[str] = []
    if route_candidate == 'continuation_or_refinement' and not effective_section_filter:
        missing_slots.append('section_hint')

    classification = QueryClassification(
        intent=intent,
        response_shape=response_shape,
        route_candidate=route_candidate,
        confidence=confidence,
        alternatives=alternatives,
        reason_codes=reason_codes,
        missing_slots=missing_slots,
        subtype=subtype,
        has_multi_year_scope=has_multi_year_scope,
        group_by=group_by,
        field_hint=field_hint,
        source_terms=source_terms,
        year_filter=year,
        category_filter=category,
        file_type_filter=file_type,
        filename_filter=filename,
        block_type_filter=effective_block_type_filter,
        section_filter=effective_section_filter,
        is_metadata_query=is_metadata_query,
        is_file_list_query=is_file_list_query,
        is_continuation=is_continuation,
        is_scope_reset=is_scope_reset,
        deterministic_override=deterministic_override,
        llm_confidence=llm_confidence,
    )

    log.info(
        'query_classification',
        query_length=len(query),
        payload_redacted=True,
        duration_ms=round((time.perf_counter() - classify_start) * 1000, 2),
        classification={
            'intent': classification.intent,
            'llm_intent': llm_intent,
            'response_shape': classification.response_shape,
            'route_candidate': classification.route_candidate,
            'confidence': classification.confidence,
            'alternatives': classification.alternatives,
            'reason_codes': classification.reason_codes,
            'missing_slots': classification.missing_slots,
            'subtype': classification.subtype,
            'has_multi_year_scope': classification.has_multi_year_scope,
            'group_by': classification.group_by,
            'field_hint': classification.field_hint,
            'source_terms': classification.source_terms,
            'year_filter': classification.year_filter,
            'category_filter': classification.category_filter,
            'file_type_filter': classification.file_type_filter,
            'filename_filter': classification.filename_filter,
            'block_type_filter': classification.block_type_filter,
            'section_filter': classification.section_filter,
            'is_continuation': classification.is_continuation,
            'is_scope_reset': classification.is_scope_reset,
            'deterministic_override': classification.deterministic_override,
            'llm_confidence': classification.llm_confidence,
        },
    )

    return classification
