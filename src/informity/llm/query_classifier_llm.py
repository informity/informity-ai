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
    extract_group_by,
    extract_mention_target,
    extract_section_hint,
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
# Confidence scores for route candidate resolution.
# Ordering: aggregate policy (deterministic multi-signal) > LLM-selected (single-signal) > policy default (fallback).
# Values are intentionally non-equal so tie-breaking is deterministic.
_ROUTE_CONFIDENCE_AGGREGATE_POLICY = 0.88  # multi-signal deterministic policy agreement
_ROUTE_CONFIDENCE_LLM_SELECTED     = 0.86  # LLM output alone, no additional policy signals
_ROUTE_CONFIDENCE_POLICY_DEFAULT   = 0.76  # fallback when no strong signal present


def _derive_policy_default_confidence(
    *,
    intent: str,
    response_shape: str,
    subtype: str | None,
    group_by: str | None,
) -> float:
    signal_count = 0
    if response_shape == 'structured_extract':
        signal_count += 1
    if subtype in {'aggregate_by_period', 'extract_structured_values'}:
        signal_count += 1
    if intent == 'coverage' and group_by in {'year', 'category', 'file'}:
        signal_count += 1
    if intent in {'metadata', 'simple'}:
        signal_count += 1
    confidence = _ROUTE_CONFIDENCE_POLICY_DEFAULT + (0.04 * min(signal_count, 2))
    return min(confidence, _ROUTE_CONFIDENCE_AGGREGATE_POLICY)


def _extract_phase2_constraints(query: str) -> dict[str, object]:
    """
    Extract deterministic Phase 2 constraints from query text.

    Returns structured fields used for trace visibility and downstream routing:
    - group_by: 'year'|'category'|'file'|None
    - field_hint: normalized structured field hint (for example, 'box_1')
    - section_hint: short section hint
    - source_terms: filename/source contains tokens extracted via metadata filters
    - file_type: fallback extension filter when present in metadata filters
    """
    doc = parse_query(query)
    group_by = extract_group_by(doc)
    field_hint = extract_field_hint(doc)
    section_hint = extract_section_hint(doc)

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

    if not source_terms:
        mention_term = extract_mention_target(doc)
        if mention_term:
            source_terms = [mention_term]

    return {
        'group_by': group_by,
        'field_hint': field_hint,
        'section_hint': section_hint,
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
) -> tuple[IntentProfileId, float, list[str]]:
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
    if force_aggregate_route:
        route_candidate = 'comparative_analysis'
        reason_codes.append('policy_aggregate_route_enforced')
        confidence = _ROUTE_CONFIDENCE_AGGREGATE_POLICY
    elif (
        isinstance(llm_route_candidate, str)
        and llm_route_candidate in valid_candidates
        and llm_route_candidate in allowed_by_intent.get(intent, set())
    ):
        route_candidate = llm_route_candidate  # LLM-selected profile, validated against enum set
        reason_codes.append('llm_profile_selected')
        confidence = _ROUTE_CONFIDENCE_LLM_SELECTED
    else:
        reason_codes.append('policy_default_selected')
        confidence = _derive_policy_default_confidence(
            intent=intent,
            response_shape=response_shape,
            subtype=subtype,
            group_by=group_by,
        )
    return route_candidate, confidence, reason_codes


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
# Classification System Prompt
# ==============================================================================

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
  "section": <null|string>
}

FIELD RULES:
- year: Explicit years only (2022, 2023). If asking ABOUT years, use null
- category: Only for metadata queries. Always null for focused/coverage
- file_type: File extensions mentioned. Null if none
- filename: Set when query has explicit filename constraint (exact file name or filename contains pattern)
- block_type: For focused/coverage only when user explicitly asks for table/form/narrative content
- section: For focused/coverage section targeting (e.g., "introduction", "conclusion"). Null if none
- response_shape:
  - structured_extract: user asks for direct field/row extraction or numeric extraction tables from source text
  - narrative_synthesis: user asks for summaries, briefs, comparisons, evidence maps, risks, recommendations

INTENT TYPES:
- metadata: File inventory operations only (count/list/enumerate file metadata)
- focused: Find specific information in documents
- coverage: Synthesize across multiple documents
- simple: Greetings, clarifications, system questions

CRITICAL ROUTING RULE:
- If the operation is performed ON document content values (extract/calculate/sum/total/compare/aggregate),
  this is NOT a metadata inventory request. Use focused (single-document/small set) or coverage
  (cross-document synthesis), even if the user says "list all".
- If the operation counts or lists files themselves (inventory of files/years/types), use metadata.
- category must be null for focused/coverage; use category only for metadata queries.

EXAMPLES (output JSON only):

Query: "how many PDFs do I have from 2022?"
Output: {"intent": "metadata", "response_shape": "narrative_synthesis", "route_candidate": "metadata_inventory", "year": 2022, "category": "document", "file_type": ".pdf", "filename": null, "block_type": null, "section": null}

Query: "what does the employment contract say about vacation days?"
Output: {"intent": "focused", "response_shape": "narrative_synthesis", "route_candidate": "targeted_fact_lookup", "year": null, "category": null, "file_type": null, "filename": null, "block_type": null, "section": null}

Query: "summarize the key findings across all annual reports"
Output: {"intent": "coverage", "response_shape": "narrative_synthesis", "route_candidate": "cross_document_synthesis", "year": null, "category": null, "file_type": null, "filename": null, "block_type": null, "section": null}

Query: "hello, what can you do?"
Output: {"intent": "simple", "response_shape": "narrative_synthesis", "route_candidate": "clarification_or_disambiguation", "year": null, "category": null, "file_type": null, "filename": null, "block_type": null, "section": null}

Query: "list all files from 2023"
Output: {"intent": "metadata", "response_shape": "narrative_synthesis", "route_candidate": "metadata_inventory", "year": 2023, "category": null, "file_type": null, "filename": null, "block_type": null, "section": null}

Query: "how many years of invoices do I have?"
Output: {"intent": "metadata", "response_shape": "narrative_synthesis", "route_candidate": "metadata_inventory", "year": null, "category": null, "file_type": null, "filename": null, "block_type": null, "section": null}

Query: "show me files named report.pdf"
Output: {"intent": "metadata", "response_shape": "narrative_synthesis", "route_candidate": "metadata_inventory", "year": null, "category": null, "file_type": null, "filename": "report.pdf", "block_type": null, "section": null}

Query: "extract the summary table values from the annual report"
Output: {"intent": "focused", "response_shape": "structured_extract", "route_candidate": "structured_field_extraction", "year": null, "category": null, "file_type": null, "filename": null, "block_type": "table", "section": null}

Query: "summarize the annual report findings"
Output: {"intent": "focused", "response_shape": "narrative_synthesis", "route_candidate": "targeted_fact_lookup", "year": null, "category": null, "file_type": null, "filename": null, "block_type": null, "section": null}

Query: "Build a compliance reconciliation brief across 3 years with required sections and evidence-backed deltas."
Output: {"intent": "coverage", "response_shape": "narrative_synthesis", "route_candidate": "audit_or_compliance_brief", "year": null, "category": null, "file_type": null, "filename": null, "block_type": null, "section": null}

EDGE CASES:
- Ambiguous = prefer "focused"
- Multiple intents = choose the intent that determines retrieval strategy
- Greeting + non-document question = "simple"
- Greeting + document question = classify by document task (metadata/focused/coverage), not "simple"

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
    messages = profile.prepare_messages(
        [
            {'role': 'system', 'content': CLASSIFICATION_SYSTEM_PROMPT},
            {'role': 'user', 'content': query},
        ],
        query_type='simple',  # reasoning=False → /no_think injected for think-capable models
    )
    stops = profile.get_stop_sequences(reasoning_enabled=False)
    # JSON mode (GBNF grammar) constrains output to valid JSON; reasoning is disabled via /no_think

    try:
        response = llm_engine.model.create_chat_completion(
            messages=messages,
            max_tokens=120,   # classification is always short
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
    # Section constraints must be explicitly grounded in user phrasing
    phase2_section_hint = phase2_constraints.get('section_hint')
    section = str(phase2_section_hint) if isinstance(phase2_section_hint, str) else None

    group_by_value = phase2_constraints.get('group_by')
    if isinstance(group_by_value, str) and group_by_value in {'year', 'category', 'file'}:
        group_by = group_by_value
    else:
        group_by = None

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

    source_terms_value = phase2_constraints.get('source_terms')
    source_terms = [str(v) for v in source_terms_value] if isinstance(source_terms_value, list) else []

    is_metadata_query = (intent == 'metadata')
    is_file_list_query = bool(intent == 'metadata' and _FILE_LIST_PATTERN.search(query))

    effective_block_type_filter = block_type if intent in ('focused', 'coverage') else None
    effective_section_filter = section if intent in ('focused', 'coverage') else None
    llm_route_candidate_raw = data.get('route_candidate')
    llm_route_candidate = str(llm_route_candidate_raw).strip() if isinstance(llm_route_candidate_raw, str) else None
    route_candidate, confidence, reason_codes = _resolve_route_candidate(
        llm_route_candidate=llm_route_candidate,
        intent=intent,
        response_shape=response_shape,
        subtype=subtype,
        group_by=group_by,
    )
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
        },
    )

    return classification
