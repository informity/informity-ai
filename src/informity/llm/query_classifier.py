# ==============================================================================
# Informity AI — Query Classifier (v2)
# Deterministic slot extraction and intent routing.
# ==============================================================================

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import structlog

from informity.config import settings
from informity.llm.intent_router import get_intent_router
from informity.llm.query_patterns import (
    build_aggregate_listing_scope_pattern,
    build_analysis_action_pattern,
    build_continuation_pattern,
    build_count_pattern,
    build_coverage_pattern,
    build_enumeration_pattern,
    build_evidence_value_extraction_pattern,
    build_extreme_value_lookup_pattern,
    build_file_list_pattern,
    build_inventory_capability_pattern,
    build_meta_query_pattern,
    build_structured_output_schema_pattern,
)
from informity.llm.types import (
    BlockType,
    ConfidenceBand,
    GroupBy,
    IntentLabel,
    IntentProfileId,
    OutputShape,
    QuerySubtype,
)

if TYPE_CHECKING:
    from promptcue import PromptCueQueryObject

log = structlog.get_logger(__name__)

# Module-level constants mirror Settings thresholds for local classifier consumers.
CONFIDENCE_HIGH_THRESHOLD: float = float(settings.classification_confidence_high_threshold)
CONFIDENCE_MEDIUM_THRESHOLD: float = float(settings.classification_confidence_medium_threshold)

_FILE_LIST_PATTERN = build_file_list_pattern()
_CONTINUATION_PATTERN = build_continuation_pattern()
_COUNT_PATTERN = build_count_pattern()
_ENUMERATION_PATTERN = build_enumeration_pattern()
_COVERAGE_PATTERN = build_coverage_pattern()
_EVIDENCE_VALUE_EXTRACTION_PATTERN = build_evidence_value_extraction_pattern()
_STRUCTURED_OUTPUT_SCHEMA_PATTERN = build_structured_output_schema_pattern()
_ANALYSIS_ACTION_PATTERN = build_analysis_action_pattern()
_INVENTORY_CAPABILITY_PATTERN = build_inventory_capability_pattern()
_META_QUERY_PATTERN = build_meta_query_pattern()
_CONTENT_ANALYSIS_PATTERN = re.compile(
    r'\b('
    r'summarize|summary|compare|contrast|contradictions?|conflicts?|overview|'
    r'main subject|describe|analy[sz]e|findings?|mentioned|tell me about|key fields?|'
    r'what does'
    r')\b',
    re.IGNORECASE,
)
_PLURAL_CORPUS_SCOPE_PATTERN = re.compile(r'\b(documents|files|records)\b', re.IGNORECASE)
_SINGLE_TARGET_PATTERN = re.compile(r'\b(any|one|single|this|that)\s+(document|file|record)\b', re.IGNORECASE)
_YEAR_AGGREGATE_CUE_PATTERN = re.compile(
    r'\b('
    r'by year|year[-\s]*by[-\s]*year|year[-\s]*over[-\s]*year|cross[-\s]*year|'
    r'findings by year|evidence map by year|coverage matrix|largest increase|largest decrease|'
    r'deltas?|per indexed year|years covered'
    r')\b',
    re.IGNORECASE,
)
_BROAD_SCOPE_EXTRA_PATTERN = re.compile(
    r'\b(across|all|cross[\s-]*document|year[\s-]*by[\s-]*year)\b',
    re.IGNORECASE,
)
_MULTI_DOC_LISTING_PATTERN = re.compile(
    r'\b(which|list|show)\b.*\b(files?|documents?)\b',
    re.IGNORECASE,
)
_GLOBAL_ENTITY_LISTING_PATTERN = re.compile(
    r'\b('
    r'names?\s+of\s+people|people\s+names?|people\s+mentioned|'
    r'important\s+dates?|key\s+dates?|'
    r'numeric\s+amounts?|key\s+amounts?|financial\s+figures?|financial\s+amounts?'
    r')\b',
    re.IGNORECASE,
)
_GENERIC_CAPABILITY_PATTERN = re.compile(
    r'\b(can\s+you\s+help|help\s+me\s+understand|what\s+information\s+is\s+available)\b',
    re.IGNORECASE,
)
_EXTREME_VALUE_LOOKUP_PATTERN = build_extreme_value_lookup_pattern()
_AGGREGATE_LISTING_SCOPE_PATTERN = build_aggregate_listing_scope_pattern()
_ANCHOR_DOCUMENT_TERM_PATTERN = re.compile(
    r'\b(?:19|20)\d{2}\s+[a-z0-9][a-z0-9\s-]{1,64}\b(?:receipt|statement|report|return|form|record|invoice|summary)\b',
    re.IGNORECASE,
)
_QUOTED_PHRASE_PATTERN = re.compile(r'["\']([^"\']{3,80})["\']')


def _has_structured_schema_request(text: str) -> bool:
    return bool(_STRUCTURED_OUTPUT_SCHEMA_PATTERN.search(text))


def _has_analysis_action_request(text: str) -> bool:
    return bool(_ANALYSIS_ACTION_PATTERN.search(text))


def _is_inventory_metadata_request(text: str) -> bool:
    return bool(
        _COUNT_PATTERN.search(text)
        or _ENUMERATION_PATTERN.search(text)
        or _FILE_LIST_PATTERN.search(text)
        or _INVENTORY_CAPABILITY_PATTERN.search(text)
    )


def _looks_broad_scope(text: str) -> bool:
    if _COVERAGE_PATTERN.search(text):
        return True
    return bool(_BROAD_SCOPE_EXTRA_PATTERN.search(text))


def _has_evidence_value_extraction_request(text: str) -> bool:
    return bool(_EVIDENCE_VALUE_EXTRACTION_PATTERN.search(text))


def _has_corpus_document_scope_request(text: str) -> bool:
    return bool(re.search(r'\b(indexed\s+)?(files?|documents?|records?)\b', text))


def _looks_multi_document_listing_request(text: str) -> bool:
    return bool(_MULTI_DOC_LISTING_PATTERN.search(text))


def _has_global_entity_listing_request(text: str) -> bool:
    return bool(_GLOBAL_ENTITY_LISTING_PATTERN.search(text))


def _has_content_analysis_request(text: str) -> bool:
    return bool(_CONTENT_ANALYSIS_PATTERN.search(text))


def _looks_plural_corpus_scope_request(text: str) -> bool:
    return bool(_PLURAL_CORPUS_SCOPE_PATTERN.search(text))


def _looks_single_target_request(text: str) -> bool:
    return bool(_SINGLE_TARGET_PATTERN.search(text))


def _has_multi_year_scope_signal(text: str) -> bool:
    years = re.findall(r'\b(?:19|20)\d{2}\b', text)
    if len(set(years)) >= 2:
        return True
    return bool(_YEAR_AGGREGATE_CUE_PATTERN.search(text))


def _is_general_capability_query(text: str) -> bool:
    return bool(_META_QUERY_PATTERN.search(text) or _GENERIC_CAPABILITY_PATTERN.search(text))


def _has_extreme_value_lookup_request(text: str) -> bool:
    return bool(_EXTREME_VALUE_LOOKUP_PATTERN.search(text))


def _has_aggregate_listing_scope_request(text: str) -> bool:
    return bool(_AGGREGATE_LISTING_SCOPE_PATTERN.search(text))


def _extract_source_terms(*, text: str, filename_filter: str | None) -> list[str]:
    terms: list[str] = []
    seen: set[str] = set()

    def _add(value: str) -> None:
        normalized = re.sub(r'\s+', ' ', str(value or '').strip())
        if len(normalized) < 3:
            return
        key = normalized.casefold()
        if key in seen:
            return
        seen.add(key)
        terms.append(normalized)

    if filename_filter:
        _add(filename_filter)
        stem = re.sub(r'\.[a-z0-9]{2,6}$', '', filename_filter, flags=re.IGNORECASE)
        if stem and stem != filename_filter:
            _add(stem)

    for match in _QUOTED_PHRASE_PATTERN.findall(text):
        _add(match)
        if len(terms) >= 6:
            return terms

    for match in _ANCHOR_DOCUMENT_TERM_PATTERN.findall(text):
        _add(match)
        if len(terms) >= 6:
            return terms

    return terms


@dataclass
class QueryClassification:
    """
    Query classification result with intent and filters.

    Attributes:
        intent: Query intent ('metadata', 'focused', 'coverage', 'simple')
        subtype: Internal subtype for routing/diagnostics (not user-facing)
        group_by: Optional grouping dimension extracted from query ('year'|'category'|'file')
        field_hint: Optional field extraction hint (for example, 'box_1')
        source_terms: Source constraint terms extracted from query (for example, filename contains terms)
        year_filter: Extracted year filter (int | None)
        category_filter: Extracted category filter (str | None)
        file_type_filter: Extracted file type filter (str | None)
        filename_filter: Extracted filename filter (str | None)
        block_type_filter: Extracted structural block filter ('table'|'form'|'narrative')
        section_filter: Extracted section hint (str | None)
        is_metadata_query: Whether this is a metadata query (count/enumeration)
        is_file_list_query: Whether this is a file listing query
    """
    intent: IntentLabel
    response_shape: OutputShape = OutputShape.NARRATIVE_SYNTHESIS
    route_candidate: IntentProfileId = IntentProfileId.TARGETED_FACT_LOOKUP
    confidence: float = 0.5
    alternatives: list[tuple[IntentLabel, float]] = field(default_factory=list)
    reason_codes: list[str] = field(default_factory=list)
    missing_slots: list[str] = field(default_factory=list)
    subtype: QuerySubtype | None = None
    has_multi_year_scope: bool = False
    group_by: GroupBy | None = None
    field_hint: str | None = None
    source_terms: list[str] = field(default_factory=list)
    year_filter: int | None = None
    category_filter: str | None = None
    file_type_filter: str | None = None
    filename_filter: str | None = None
    block_type_filter: BlockType | None = None
    section_filter: str | None = None
    is_metadata_query: bool = False
    is_file_list_query: bool = False
    is_continuation: bool = False
    is_scope_reset: bool = False
    # action_hints: forwarded from PromptCueQueryObject when the promptcue adapter is active.
    # Empty dict when a non-promptcue router is used (e.g. test fakes).
    action_hints: dict[str, bool] = field(default_factory=dict)
    # Provenance flags — describe how route_candidate was selected.
    # deterministic_override: True when a hard aggregate rule fired (e.g. policy_aggregate_route_enforced).
    # llm_confidence: raw confidence reported by the LLM (0.0 when LLM did not emit a confidence field).
    deterministic_override: bool = False
    llm_confidence: float = 0.0

    @property
    def confidence_band(self) -> ConfidenceBand:
        if self.confidence >= settings.classification_confidence_high_threshold:
            return ConfidenceBand.HIGH
        if self.confidence >= settings.classification_confidence_medium_threshold:
            return ConfidenceBand.MEDIUM
        return ConfidenceBand.LOW


def classify_query(query: str) -> QueryClassification:
    """Classify query via pluggable intent router + deterministic slot extraction."""
    text = str(query or '').strip()
    lowered = text.casefold()
    year_filter: int | None = None

    year_candidates = re.findall(r'\b(?:19|20)\d{2}\b', lowered)
    if len(year_candidates) == 1:
        _yr = int(year_candidates[0])
        if 1900 <= _yr <= 2099:
            year_filter = _yr

    file_type_filter: str | None = None
    ext_match = re.search(r'\.(pdf|txt|md|csv|json|docx?|xlsx?)\b', lowered)
    if ext_match:
        file_type_filter = f".{ext_match.group(1)}"
    elif 'pdf' in lowered:
        file_type_filter = '.pdf'

    filename_filter: str | None = None
    filename_match = re.search(r'([a-z0-9][a-z0-9._-]*\.(?:pdf|txt|md|csv|json|docx?|xlsx?))', lowered)
    if filename_match:
        filename_filter = filename_match.group(1)

    is_continuation = bool(_CONTINUATION_PATTERN.search(lowered))

    # --- Intent routing ---------------------------------------------------
    # When the active router is PromptCueIntentAdapter, call classify() to get
    # both the IntentPrediction and the full PromptCueQueryObject in one pass.
    # When any other router is active (e.g. test fakes), fall back to
    # classify_intent() and derive the additional signals from regex.
    from informity.llm.promptcue_adapter import PromptCueIntentAdapter

    router = get_intent_router()
    pcue: PromptCueQueryObject | None = None

    if isinstance(router, PromptCueIntentAdapter):
        try:
            prediction, pcue = router.classify(text)
        except Exception:  # noqa: BLE001
            log.warning('promptcue_adapter_classify_failed', query=text[:120])
            prediction = router.classify_intent(text)
    else:
        prediction = router.classify_intent(text)

    intent      = prediction.intent
    reason_codes = list(prediction.reason_codes)

    # --- Signals: prefer PromptCue when available, fall back to regex -----
    # Corpus-specific signals (always from regex — PromptCue has no knowledge
    # of indexed files, evidence values, or inventory structure).
    is_inventory_metadata     = _is_inventory_metadata_request(lowered)
    has_evidence_value_request = _has_evidence_value_extraction_request(lowered)
    has_corpus_scope          = _has_corpus_document_scope_request(lowered)
    has_multi_doc_listing     = _looks_multi_document_listing_request(lowered)
    is_general_capability     = _is_general_capability_query(lowered)
    source_terms              = _extract_source_terms(text=text, filename_filter=filename_filter)

    if pcue is not None:
        # PromptCue path — richer signals from 12-type classification + scope.
        semantic_hints = getattr(pcue, 'semantic_hints', None)
        is_continuation      = pcue.is_continuation
        # Blend PromptCue scope with deterministic lexical scope cues so
        # broad/focused routing remains stable when PromptCue scope confidence
        # is imperfect for corpus-specific phrasing.
        broad_scope          = (str(pcue.scope) == 'broad') or _looks_broad_scope(lowered)
        plural_scope         = broad_scope or _looks_plural_corpus_scope_request(lowered)
        single_target_scope  = (str(pcue.scope) == 'focused') or _looks_single_target_request(lowered)
        has_structured_schema = (
            bool(pcue.routing_hints.get('needs_structure'))
            or bool(getattr(semantic_hints, 'requests_structure', False))
            or _has_structured_schema_request(lowered)
        )
        has_analysis_action  = (
            bool(pcue.action_hints.get('should_compare'))
            or bool(getattr(semantic_hints, 'requests_comparison', False))
            or _has_analysis_action_request(lowered)
        )
        has_multi_year_scope = (
            bool(pcue.routing_hints.get('has_temporal_scope'))
            or bool(getattr(semantic_hints, 'requires_multi_period_analysis', False))
            or _has_multi_year_scope_signal(lowered)
        )
        has_content_analysis = pcue.primary_query_type in {
            'analysis', 'comparison', 'summarization', 'coverage',
        }
    else:
        # Regex fallback path — used when a non-PromptCue router is active.
        broad_scope          = _looks_broad_scope(lowered)
        plural_scope         = _looks_plural_corpus_scope_request(lowered)
        single_target_scope  = _looks_single_target_request(lowered)
        has_structured_schema = _has_structured_schema_request(lowered)
        has_analysis_action  = _has_analysis_action_request(lowered)
        has_multi_year_scope = _has_multi_year_scope_signal(lowered)
        has_content_analysis = _has_content_analysis_request(lowered)
    has_extreme_value_lookup = _has_extreme_value_lookup_request(lowered)
    has_aggregate_listing_scope = _has_aggregate_listing_scope_request(lowered)
    has_global_entity_listing = _has_global_entity_listing_request(lowered)

    # --- Corpus metadata promotion ----------------------------------------
    # Inventory queries (count, enumeration, file listing, capability) are
    # always 'metadata' intent regardless of the router prediction.  PromptCue
    # has no knowledge of the indexed corpus, so it cannot classify these.
    if is_inventory_metadata and intent != IntentLabel.METADATA:
        intent = IntentLabel.METADATA
        reason_codes.append('deterministic_inventory_metadata_promoted')
    elif (
        is_general_capability
        and not is_inventory_metadata
        and not has_structured_schema
        and not has_analysis_action
        and not has_evidence_value_request
        and not has_corpus_scope
    ):
        intent = IntentLabel.SIMPLE
        reason_codes.append('deterministic_general_capability_to_simple')

    deterministic_override = False
    response_shape: OutputShape = OutputShape.NARRATIVE_SYNTHESIS
    subtype: QuerySubtype | None = None
    group_by: GroupBy | None = GroupBy.YEAR if has_multi_year_scope else None

    def apply_override(
        *,
        reason_code: str,
        new_intent: IntentLabel | None = None,
        new_route: IntentProfileId | None = None,
        new_shape: OutputShape | None = None,
        new_subtype: QuerySubtype | None = None,
    ) -> None:
        nonlocal intent, route_candidate, response_shape, subtype, deterministic_override
        deterministic_override = True
        if new_intent is not None:
            intent = new_intent
        if new_route is not None:
            route_candidate = new_route
        if new_shape is not None:
            response_shape = new_shape
        if new_subtype is not None:
            subtype = new_subtype
        reason_codes.append(reason_code)

    if intent == IntentLabel.METADATA:
        route_candidate = IntentProfileId.METADATA_INVENTORY
        if has_structured_schema:
            response_shape = OutputShape.METADATA_TABLE
        metadata_rules = [
            (
                has_structured_schema and not is_inventory_metadata and broad_scope,
                lambda: apply_override(
                    reason_code='deterministic_override_structured_schema_request',
                    new_intent=IntentLabel.COVERAGE,
                    new_route=IntentProfileId.COMPARATIVE_ANALYSIS,
                    new_shape=OutputShape.METADATA_TABLE,
                    new_subtype=QuerySubtype.EXTRACT_STRUCTURED_VALUES,
                ),
            ),
            (
                has_structured_schema and not is_inventory_metadata and not broad_scope,
                lambda: apply_override(
                    reason_code='deterministic_override_structured_schema_request',
                    new_intent=IntentLabel.FOCUSED,
                    new_route=IntentProfileId.STRUCTURED_FIELD_EXTRACTION,
                    new_shape=OutputShape.STRUCTURED_EXTRACT,
                    new_subtype=QuerySubtype.EXTRACT_STRUCTURED_VALUES,
                ),
            ),
            (
                has_analysis_action and not is_inventory_metadata and broad_scope,
                lambda: apply_override(
                    reason_code='deterministic_override_analysis_request',
                    new_intent=IntentLabel.COVERAGE,
                    new_route=IntentProfileId.CROSS_DOCUMENT_SYNTHESIS,
                    new_shape=OutputShape.NARRATIVE_SYNTHESIS,
                ),
            ),
            (
                has_analysis_action and not is_inventory_metadata and not broad_scope,
                lambda: apply_override(
                    reason_code='deterministic_override_analysis_request',
                    new_intent=IntentLabel.FOCUSED,
                    new_route=IntentProfileId.TARGETED_FACT_LOOKUP,
                    new_shape=OutputShape.NARRATIVE_SYNTHESIS,
                ),
            ),
            (
                has_evidence_value_request and (is_inventory_metadata or has_corpus_scope)
                and (broad_scope or has_multi_doc_listing),
                lambda: apply_override(
                    reason_code='deterministic_override_inventory_with_evidence_request',
                    new_intent=IntentLabel.COVERAGE,
                    new_route=IntentProfileId.CROSS_DOCUMENT_SYNTHESIS,
                    new_shape=OutputShape.NARRATIVE_SYNTHESIS,
                ),
            ),
            (
                has_evidence_value_request and (is_inventory_metadata or has_corpus_scope)
                and not (broad_scope or has_multi_doc_listing),
                lambda: apply_override(
                    reason_code='deterministic_override_inventory_with_evidence_request',
                    new_intent=IntentLabel.FOCUSED,
                    new_route=IntentProfileId.TARGETED_FACT_LOOKUP,
                    new_shape=OutputShape.NARRATIVE_SYNTHESIS,
                ),
            ),
            (
                has_corpus_scope and has_global_entity_listing and filename_filter is None and not has_extreme_value_lookup,
                lambda: apply_override(
                    reason_code='deterministic_override_global_entity_listing_to_coverage',
                    new_intent=IntentLabel.COVERAGE,
                    new_route=IntentProfileId.CROSS_DOCUMENT_SYNTHESIS,
                    new_shape=OutputShape.NARRATIVE_SYNTHESIS,
                ),
            ),
        ]
        for condition, action in metadata_rules:
            if condition:
                action()
                break
    elif intent == IntentLabel.SIMPLE:
        route_candidate = IntentProfileId.CLARIFICATION_OR_DISAMBIGUATION
    elif intent == IntentLabel.COVERAGE:
        route_candidate = IntentProfileId.CROSS_DOCUMENT_SYNTHESIS
        coverage_rules = [
            (
                has_extreme_value_lookup and not has_structured_schema and not has_multi_year_scope,
                lambda: apply_override(
                    reason_code='deterministic_override_extreme_value_lookup_to_focused',
                    new_intent=IntentLabel.FOCUSED,
                    new_route=IntentProfileId.TARGETED_FACT_LOOKUP,
                    new_shape=OutputShape.NARRATIVE_SYNTHESIS,
                ),
            ),
            (
                has_structured_schema,
                lambda: apply_override(
                    reason_code='deterministic_override_structured_schema_for_coverage',
                    new_route=IntentProfileId.COMPARATIVE_ANALYSIS,
                    new_shape=OutputShape.METADATA_TABLE,
                    new_subtype=QuerySubtype.EXTRACT_STRUCTURED_VALUES,
                ),
            ),
            (
                not broad_scope
                and not has_corpus_scope
                and not has_multi_year_scope
                and (filename_filter is not None or bool(source_terms)),
                lambda: apply_override(
                    reason_code='deterministic_override_narrow_scope_to_focused',
                    new_intent=IntentLabel.FOCUSED,
                    new_route=IntentProfileId.TARGETED_FACT_LOOKUP,
                ),
            ),
        ]
        for condition, action in coverage_rules:
            if condition:
                action()
                break
    else:
        route_candidate = IntentProfileId.TARGETED_FACT_LOOKUP
        focused_rules = [
            (
                has_corpus_scope
                and (broad_scope or has_multi_doc_listing)
                and not single_target_scope
                and filename_filter is None
                and not has_extreme_value_lookup,
                lambda: apply_override(
                    reason_code='deterministic_override_corpus_scope_to_coverage',
                    new_intent=IntentLabel.COVERAGE,
                    new_route=IntentProfileId.CROSS_DOCUMENT_SYNTHESIS,
                ),
            ),
            (
                has_aggregate_listing_scope
                and not single_target_scope
                and filename_filter is None
                and not has_extreme_value_lookup,
                lambda: apply_override(
                    reason_code='deterministic_override_aggregate_listing_to_coverage',
                    new_intent=IntentLabel.COVERAGE,
                    new_route=IntentProfileId.CROSS_DOCUMENT_SYNTHESIS,
                ),
            ),
            (
                has_corpus_scope
                and has_global_entity_listing
                and not single_target_scope
                and filename_filter is None
                and not has_extreme_value_lookup,
                lambda: apply_override(
                    reason_code='deterministic_override_global_entity_listing_to_coverage',
                    new_intent=IntentLabel.COVERAGE,
                    new_route=IntentProfileId.CROSS_DOCUMENT_SYNTHESIS,
                ),
            ),
            (
                has_multi_year_scope and has_content_analysis and filename_filter is None,
                lambda: apply_override(
                    reason_code='deterministic_override_multi_year_analysis_request',
                    new_intent=IntentLabel.COVERAGE,
                    new_route=IntentProfileId.CROSS_DOCUMENT_SYNTHESIS,
                ),
            ),
            (
                plural_scope and not single_target_scope and has_content_analysis and filename_filter is None,
                lambda: apply_override(
                    reason_code='deterministic_override_plural_scope_analysis_request',
                    new_intent=IntentLabel.COVERAGE,
                    new_route=IntentProfileId.CROSS_DOCUMENT_SYNTHESIS,
                ),
            ),
            (
                has_structured_schema and (broad_scope or has_corpus_scope),
                lambda: apply_override(
                    reason_code='deterministic_override_structured_schema_for_coverage_scope',
                    new_intent=IntentLabel.COVERAGE,
                    new_route=IntentProfileId.COMPARATIVE_ANALYSIS,
                    new_shape=OutputShape.METADATA_TABLE,
                    new_subtype=QuerySubtype.EXTRACT_STRUCTURED_VALUES,
                ),
            ),
            (
                has_structured_schema and not (broad_scope or has_corpus_scope),
                lambda: apply_override(
                    reason_code='deterministic_override_structured_schema_for_focused',
                    new_route=IntentProfileId.STRUCTURED_FIELD_EXTRACTION,
                    new_shape=OutputShape.STRUCTURED_EXTRACT,
                    new_subtype=QuerySubtype.EXTRACT_STRUCTURED_VALUES,
                ),
            ),
        ]
        for condition, action in focused_rules:
            if condition:
                action()
                break

    if (
        intent == IntentLabel.METADATA
        and has_content_analysis
        and not is_inventory_metadata
    ):
        if broad_scope or (has_corpus_scope and plural_scope):
            apply_override(
                reason_code='deterministic_override_metadata_content_request',
                new_intent=IntentLabel.COVERAGE,
                new_route=IntentProfileId.CROSS_DOCUMENT_SYNTHESIS,
            )
        else:
            apply_override(
                reason_code='deterministic_override_metadata_content_request',
                new_intent=IntentLabel.FOCUSED,
                new_route=IntentProfileId.TARGETED_FACT_LOOKUP,
            )

    if (
        intent == IntentLabel.COVERAGE
        and filename_filter is None
        and (has_multi_year_scope or group_by == GroupBy.YEAR)
    ):
        if subtype != QuerySubtype.AGGREGATE_BY_PERIOD:
            subtype = QuerySubtype.AGGREGATE_BY_PERIOD
            reason_codes.append('deterministic_override_coverage_year_aggregate_subtype')
        if response_shape != OutputShape.NARRATIVE_SYNTHESIS:
            response_shape = OutputShape.NARRATIVE_SYNTHESIS
            reason_codes.append('deterministic_override_year_aggregate_narrative_shape')

    return QueryClassification(
        intent=intent,
        response_shape=response_shape,
        route_candidate=route_candidate,
        confidence=prediction.confidence,
        alternatives=prediction.alternatives,
        reason_codes=reason_codes,
        subtype=subtype,
        has_multi_year_scope=has_multi_year_scope,
        group_by=group_by,
        source_terms=source_terms,
        year_filter=year_filter,
        file_type_filter=file_type_filter,
        filename_filter=filename_filter,
        is_metadata_query=(intent == IntentLabel.METADATA),
        is_file_list_query=(intent == IntentLabel.METADATA and bool(_FILE_LIST_PATTERN.search(lowered))),
        is_continuation=is_continuation,
        action_hints=(pcue.action_hints if pcue is not None else {}),
        deterministic_override=deterministic_override,
        llm_confidence=0.0,
    )


__all__ = [
    'CONFIDENCE_HIGH_THRESHOLD',
    'CONFIDENCE_MEDIUM_THRESHOLD',
    'classify_query',
    'QueryClassification',
]
