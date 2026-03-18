# ==============================================================================
# Informity AI — Query Planner
# Produces a QueryPlan from a user query using the active model.
# The plan covers retrieval sub-steps (Phase 5) and an answer section outline
# (Phases 2–3) for structured multi-section generation.
# ==============================================================================

import json
import time
from dataclasses import dataclass, field
from typing import Literal

import structlog

from informity.config import settings
from informity.llm.model_adapter import get_profile

log = structlog.get_logger(__name__)

_PLANNER_RUNTIME_EXCEPTIONS = (RuntimeError, ValueError, TypeError, OSError)

# Routes where planning adds structural value: multi-step retrieval + section outline.
# Used to gate build_plan() in routes_chat.py and multi-step retrieval in handlers/rag.py.
PLANNING_ELIGIBLE_ROUTES: frozenset[str] = frozenset({
    'cross_document_synthesis', 'comparative_analysis', 'audit_or_compliance_brief',
})

# ==============================================================================
# Data Model
# ==============================================================================


@dataclass
class RetrievalFilters:
    """Structured metadata filters for a plan retrieval step.

    Maps 1:1 to retrieve_chunks keyword arguments. Only structured metadata
    fields (year, category, extension) are supported — no content-pattern filtering.
    """

    year_filter: int | None = None
    category_filter: str | None = None
    extension_filter: str | None = None
    filename_filter: str | None = None
    source_terms_filter: list[str] | None = None
    block_type_filter: str | None = None
    section_filter: str | None = None


@dataclass
class PlanStep:
    """A single retrieval sub-step within a query plan."""

    step_id: int
    description: str            # "Get revenue figures from 2024 reports"
    sub_query: str              # Reformulated query for this step
    filters: RetrievalFilters   # Year / category / file constraints
    retrieval_mode: Literal['focused', 'coverage']
    expected_output: str        # "list of revenue numbers by file"


@dataclass
class AnswerSection:
    """A single output section in the planned answer outline."""

    heading: str                             # "## Scenario 1: Conservative"
    scope: str                               # 1–2 sentence description of what this section covers
    estimated_complexity: Literal['simple', 'detailed']


@dataclass
class QueryPlan:
    """Full plan for query decomposition and answer generation."""

    steps: list[PlanStep]
    answer_sections: list[AnswerSection]
    aggregation_mode: Literal['merge', 'compare', 'synthesize']
    output_shape: Literal['structured_extract', 'narrative_synthesis', 'metadata_table', 'hybrid']


# ==============================================================================
# Helper Functions
# ==============================================================================

_VALID_OUTPUT_SHAPES: frozenset[str] = frozenset({
    'structured_extract', 'narrative_synthesis', 'metadata_table', 'hybrid',
})
_VALID_AGGREGATION_MODES: frozenset[str] = frozenset({'merge', 'compare', 'synthesize'})
_VALID_RETRIEVAL_MODES: frozenset[str] = frozenset({'focused', 'coverage'})
_VALID_COMPLEXITY: frozenset[str] = frozenset({'simple', 'detailed'})
_VALID_CATEGORIES: frozenset[str] = frozenset({'document', 'plaintext', 'data', 'web', 'code'})
_VALID_BLOCK_TYPES: frozenset[str] = frozenset({'table', 'form', 'narrative'})


def _normalize_output_shape(
    raw: str,
) -> Literal['structured_extract', 'narrative_synthesis', 'metadata_table', 'hybrid']:
    """Normalize an unknown output_shape string, defaulting to narrative_synthesis."""
    normalized = str(raw).strip().lower()
    if normalized in _VALID_OUTPUT_SHAPES:
        return normalized  # type: ignore[return-value]
    return 'narrative_synthesis'


def _filters_to_kwargs(filters: RetrievalFilters) -> dict[str, object]:
    """Convert a RetrievalFilters instance to retrieve_chunks keyword arguments.

    This is the single adapter between the plan data model and the retrieval API.
    Call sites must use this function — never reference retrieve_chunks parameter
    names directly in plan-execution code.
    """
    kwargs: dict[str, object] = {}
    if filters.year_filter is not None:
        kwargs['year_filter'] = filters.year_filter
    if filters.category_filter is not None:
        kwargs['category_filter'] = filters.category_filter
    if filters.extension_filter is not None:
        kwargs['extension_filter'] = filters.extension_filter
    if filters.filename_filter is not None:
        kwargs['filename_filter'] = filters.filename_filter
    if filters.source_terms_filter is not None:
        kwargs['source_terms_filter'] = filters.source_terms_filter
    if filters.block_type_filter is not None:
        kwargs['block_type_filter'] = filters.block_type_filter
    if filters.section_filter is not None:
        kwargs['section_filter'] = filters.section_filter
    return kwargs


def build_corpus_summary(
    years: list[int],
    categories: list[str],
    file_count: int,
) -> str:
    """Build a short corpus summary string for the planner prompt.

    Args:
        years: Distinct indexed years (from get_distinct_years).
        categories: Distinct indexed categories (from get_distinct_categories).
        file_count: Total indexed file count.

    Returns:
        A compact summary string suitable for inclusion in the planner user message.
    """
    parts: list[str] = [f'{file_count} files indexed']
    if years:
        year_range = (
            str(years[0])
            if len(years) == 1
            else f'{min(years)}–{max(years)} ({len(years)} years)'
        )
        parts.append(f'years: {year_range}')
    if categories:
        parts.append(f'categories: {", ".join(sorted(categories))}')
    return '; '.join(parts)


# ==============================================================================
# Planner Prompt
# ==============================================================================

_PLANNER_SYSTEM_PROMPT = """OUTPUT FORMAT: Return ONLY valid JSON. No markdown, no explanations, no code blocks.

You are a query planner for a local document retrieval system. Given a user query and corpus metadata, produce a structured plan.

Output this exact JSON structure:
{
  "steps": [
    {
      "step_id": 1,
      "description": "<short description of what to retrieve>",
      "sub_query": "<reformulated query for this step>",
      "filters": {
        "year_filter": <null|integer>,
        "category_filter": <null|"document"|"plaintext"|"data"|"web"|"code">,
        "extension_filter": <null|string>,
        "filename_filter": <null|string>,
        "source_terms_filter": <null|[string,...]>,
        "block_type_filter": <null|"table"|"form"|"narrative">,
        "section_filter": <null|string>
      },
      "retrieval_mode": "<focused|coverage>",
      "expected_output": "<what this step should return>"
    }
  ],
  "answer_sections": [
    {
      "heading": "<## Section Title>",
      "scope": "<1-2 sentences describing what this section covers>",
      "estimated_complexity": "<simple|detailed>"
    }
  ],
  "aggregation_mode": "<merge|compare|synthesize>",
  "output_shape": "<structured_extract|narrative_synthesis|metadata_table|hybrid>"
}

FIELD RULES:
- steps: One step per distinct retrieval dimension (different files, years, topics). Maximum 5 steps.
- answer_sections: One entry per top-level section in the expected answer. Maximum 8 sections.
- aggregation_mode:
  - merge: Combine findings from multiple steps into a unified answer
  - compare: Show differences between steps side-by-side
  - synthesize: Draw higher-level conclusions across all steps
- output_shape:
  - structured_extract: User asks for direct field or numeric extraction (tables, structured data)
  - narrative_synthesis: User asks for summaries, analysis, comparisons, recommendations
  - metadata_table: User asks for a file or document inventory table
  - hybrid: Mixed structured and narrative content

ANSWER SECTIONS RULES:
- Each heading must start with ## (markdown heading level 2)
- The scope must be 1-2 sentences describing specifically what content this section covers
- estimated_complexity: "detailed" for scenarios requiring multi-paragraph explanation, "simple" for short factual answers
- If the query asks for N scenarios, plans, or cases — produce exactly N answer_sections entries
- If the query is a single focused question with one answer, produce 1 answer_sections entry with an appropriate heading

STEPS RULES:
- sub_query must be a complete, self-contained question for retrieval (not a fragment)
- filters: use only structured metadata fields from the corpus; set to null if not applicable
- retrieval_mode: "focused" for targeted single-document queries, "coverage" for multi-document synthesis

CRITICAL: Output MUST start with { and end with }. No code blocks, no explanations."""


# ==============================================================================
# Planner LLM Call
# ==============================================================================


def build_plan(query: str, corpus_summary: str) -> QueryPlan | None:
    """Build a QueryPlan for the given query using the active model.

    Args:
        query: The user query to plan for.
        corpus_summary: A short text description of corpus metadata
                        (years available, categories, file count) — use
                        build_corpus_summary() to construct this.

    Returns:
        A validated QueryPlan, or None if planning fails for any reason.
        The caller must fall through to the single-pass path on None.

    Never raises — all failures return None.
    """
    from informity.llm.engine import llm_engine

    plan_start = time.perf_counter()
    profile = get_profile()
    no_think_suffix = f'\n{profile.no_think_token}' if profile.no_think_token else ''
    stops = profile.get_stop_sequences(reasoning_enabled=False)

    user_content = (
        f'User query: {query}\n\nCorpus metadata: {corpus_summary}'
        + no_think_suffix
    )
    messages = [
        {'role': 'system', 'content': _PLANNER_SYSTEM_PROMPT},
        {'role': 'user', 'content': user_content},
    ]

    try:
        response = llm_engine.model.create_chat_completion(
            messages=messages,
            max_tokens=settings.planner_max_tokens,
            temperature=0.0,
            stop=stops,
            response_format={'type': 'json_object'},
        )
    except _PLANNER_RUNTIME_EXCEPTIONS as exc:
        log.warning('planner_inference_failed', error=str(exc), query_length=len(query))
        return None

    if not response or 'choices' not in response or not response['choices']:
        log.warning('planner_empty_response', query_length=len(query))
        return None

    choice = response['choices'][0]
    if 'message' not in choice or 'content' not in choice['message']:
        log.warning('planner_invalid_response_structure', query_length=len(query))
        return None

    content = choice['message']['content']
    if content is None:
        content = ''
    content = content.strip()

    if not content:
        log.warning(
            'planner_empty_content',
            query_length=len(query),
            finish_reason=choice.get('finish_reason'),
        )
        return None

    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        log.warning(
            'planner_json_parse_failed',
            error=str(exc),
            content_length=len(content),
            query_length=len(query),
        )
        return None

    if not isinstance(data, dict):
        log.warning('planner_response_not_dict', query_length=len(query))
        return None

    latency_ms = round((time.perf_counter() - plan_start) * 1000, 2)
    plan = _validate_and_build_plan(data, query)

    if plan is None:
        log.warning('planner_validation_failed', query_length=len(query), latency_ms=latency_ms)
        return None

    log.info(
        'query_plan_built',
        query_length=len(query),
        latency_ms=latency_ms,
        steps_count=len(plan.steps),
        answer_sections_count=len(plan.answer_sections),
        aggregation_mode=plan.aggregation_mode,
        output_shape=plan.output_shape,
    )
    return plan


# ==============================================================================
# Plan Validation
# ==============================================================================


def _validate_and_build_plan(data: dict, query: str) -> QueryPlan | None:
    """Parse and validate plan JSON into a QueryPlan. Returns None on any validation failure."""
    raw_steps = data.get('steps')
    if not isinstance(raw_steps, list):
        raw_steps = []
    steps: list[PlanStep] = []
    for raw_step in raw_steps[:settings.planner_max_steps]:
        step = _parse_plan_step(raw_step)
        if step is not None:
            steps.append(step)

    raw_sections = data.get('answer_sections')
    if not isinstance(raw_sections, list):
        raw_sections = []
    answer_sections: list[AnswerSection] = []
    for raw_section in raw_sections[:settings.planner_max_sections]:
        section = _parse_answer_section(raw_section)
        if section is not None:
            answer_sections.append(section)

    raw_agg = str(data.get('aggregation_mode', '')).strip().lower()
    aggregation_mode: Literal['merge', 'compare', 'synthesize'] = (
        raw_agg  # type: ignore[assignment]
        if raw_agg in _VALID_AGGREGATION_MODES
        else 'synthesize'
    )

    output_shape = _normalize_output_shape(str(data.get('output_shape', '')))

    if not answer_sections:
        log.debug('planner_no_valid_sections', query_length=len(query))
        return None

    return QueryPlan(
        steps=steps,
        answer_sections=answer_sections,
        aggregation_mode=aggregation_mode,
        output_shape=output_shape,
    )


def _parse_plan_step(raw: object) -> PlanStep | None:
    """Parse a single plan step from raw JSON. Returns None if invalid."""
    if not isinstance(raw, dict):
        return None
    sub_query = str(raw.get('sub_query', '')).strip()
    if not sub_query:
        return None
    description = str(raw.get('description', '')).strip()
    expected_output = str(raw.get('expected_output', '')).strip()
    step_id_raw = raw.get('step_id')
    try:
        step_id = int(step_id_raw) if step_id_raw is not None else 0
    except (ValueError, TypeError):
        step_id = 0
    raw_mode = str(raw.get('retrieval_mode', '')).strip().lower()
    retrieval_mode: Literal['focused', 'coverage'] = (
        raw_mode  # type: ignore[assignment]
        if raw_mode in _VALID_RETRIEVAL_MODES
        else 'coverage'
    )
    filters = _parse_retrieval_filters(raw.get('filters'))
    return PlanStep(
        step_id=step_id,
        description=description,
        sub_query=sub_query,
        filters=filters,
        retrieval_mode=retrieval_mode,
        expected_output=expected_output,
    )


def _parse_answer_section(raw: object) -> AnswerSection | None:
    """Parse a single answer section from raw JSON. Returns None if invalid."""
    if not isinstance(raw, dict):
        return None
    heading = str(raw.get('heading', '')).strip()
    scope = str(raw.get('scope', '')).strip()
    if not heading or not scope:
        return None
    raw_complexity = str(raw.get('estimated_complexity', '')).strip().lower()
    estimated_complexity: Literal['simple', 'detailed'] = (
        raw_complexity  # type: ignore[assignment]
        if raw_complexity in _VALID_COMPLEXITY
        else 'detailed'
    )
    return AnswerSection(
        heading=heading,
        scope=scope,
        estimated_complexity=estimated_complexity,
    )


def _parse_retrieval_filters(raw: object) -> RetrievalFilters:
    """Parse retrieval filters from raw JSON, defaulting all fields to None on any issue."""
    if not isinstance(raw, dict):
        return RetrievalFilters()

    year_filter: int | None = None
    year_raw = raw.get('year_filter')
    if year_raw is not None:
        try:
            year_val = int(year_raw)
            if 1900 <= year_val <= 2099:
                year_filter = year_val
        except (ValueError, TypeError):
            pass

    category_filter: str | None = None
    cat_raw = raw.get('category_filter')
    if isinstance(cat_raw, str) and cat_raw.strip().lower() in _VALID_CATEGORIES:
        category_filter = cat_raw.strip().lower()

    extension_filter: str | None = None
    ext_raw = raw.get('extension_filter')
    if isinstance(ext_raw, str) and ext_raw.strip():
        extension_filter = ext_raw.strip()

    filename_filter: str | None = None
    fn_raw = raw.get('filename_filter')
    if isinstance(fn_raw, str) and fn_raw.strip():
        filename_filter = fn_raw.strip()

    source_terms_filter: list[str] | None = None
    st_raw = raw.get('source_terms_filter')
    if isinstance(st_raw, list):
        terms = [str(t).strip() for t in st_raw if str(t).strip()]
        if terms:
            source_terms_filter = terms

    block_type_filter: str | None = None
    bt_raw = raw.get('block_type_filter')
    if isinstance(bt_raw, str) and bt_raw.strip().lower() in _VALID_BLOCK_TYPES:
        block_type_filter = bt_raw.strip().lower()

    section_filter: str | None = None
    sf_raw = raw.get('section_filter')
    if isinstance(sf_raw, str) and sf_raw.strip():
        section_filter = sf_raw.strip()

    return RetrievalFilters(
        year_filter=year_filter,
        category_filter=category_filter,
        extension_filter=extension_filter,
        filename_filter=filename_filter,
        source_terms_filter=source_terms_filter,
        block_type_filter=block_type_filter,
        section_filter=section_filter,
    )
