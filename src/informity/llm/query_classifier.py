# ==============================================================================
# Informity AI — Query Classifier (v2)
# Deterministic slot extraction and intent routing.
# ==============================================================================
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import structlog

from informity.config import settings
from informity.llm.contract_prompt_parser import EXPLICIT_YEAR_PATTERN
from informity.llm.intent_router import get_intent_router
from informity.llm.query_patterns import (
    build_aggregate_listing_scope_pattern,
    build_analysis_action_pattern,
    build_anchor_document_term_pattern,
    build_comparative_pattern,
    build_continuation_pattern,
    build_corpus_document_scope_pattern,
    build_count_pattern,
    build_coverage_pattern,
    build_enumeration_pattern,
    build_evidence_value_extraction_pattern,
    build_fact_lookup_pattern,
    build_file_list_pattern,
    build_filename_exclusion_pattern,
    build_generic_capability_pattern,
    build_global_entity_listing_pattern,
    build_inventory_capability_pattern,
    build_meta_query_pattern,
    build_multi_document_listing_pattern,
    build_negation_pattern,
    build_output_format_bullets_pattern,
    build_output_format_csv_pattern,
    build_output_format_list_pattern,
    build_output_format_narrative_pattern,
    build_output_format_table_pattern,
    build_plural_corpus_scope_pattern,
    build_quoted_phrase_pattern,
    build_single_target_pattern,
    build_structured_output_schema_pattern,
    build_year_aggregate_cue_pattern,
)
from informity.llm.term_dictionary import expand_query_for_routing
from informity.llm.types import (
    BlockType,
    ConfidenceBand,
    GroupBy,
    IntentLabel,
    IntentProfileId,
    OutputFormat,
    OutputShape,
    QuerySubtype,
)

if TYPE_CHECKING:
    from promptcue import PromptCueQueryObject
else:
    PromptCueQueryObject = object

log = structlog.get_logger(__name__)
_PROMPTCUE_CLASSIFY_EXCEPTIONS = (RuntimeError, ValueError, TypeError, AttributeError)

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
_PLURAL_CORPUS_SCOPE_PATTERN = build_plural_corpus_scope_pattern()
_SINGLE_TARGET_PATTERN = build_single_target_pattern()
_YEAR_AGGREGATE_CUE_PATTERN = build_year_aggregate_cue_pattern()
_MULTI_DOC_LISTING_PATTERN = build_multi_document_listing_pattern()
_GLOBAL_ENTITY_LISTING_PATTERN = build_global_entity_listing_pattern()
_GENERIC_CAPABILITY_PATTERN = build_generic_capability_pattern()
_FACT_LOOKUP_PATTERN = build_fact_lookup_pattern()
_AGGREGATE_LISTING_SCOPE_PATTERN = build_aggregate_listing_scope_pattern()
_ANCHOR_DOCUMENT_TERM_PATTERN = build_anchor_document_term_pattern()
_QUOTED_PHRASE_PATTERN = build_quoted_phrase_pattern()
_CORPUS_DOCUMENT_SCOPE_PATTERN = build_corpus_document_scope_pattern()
_COMPARATIVE_PATTERN = build_comparative_pattern()
_OUTPUT_FORMAT_TABLE_PATTERN = build_output_format_table_pattern()
_OUTPUT_FORMAT_BULLETS_PATTERN = build_output_format_bullets_pattern()
_OUTPUT_FORMAT_CSV_PATTERN = build_output_format_csv_pattern()
_OUTPUT_FORMAT_LIST_PATTERN = build_output_format_list_pattern()
_OUTPUT_FORMAT_NARRATIVE_PATTERN = build_output_format_narrative_pattern()
_NEGATION_PATTERN = build_negation_pattern()
_FILENAME_EXCLUSION_PATTERN = build_filename_exclusion_pattern()
_COMPARATIVE_FILE_CONTENT_CUE_PATTERN = re.compile(r'\b(mention|mentions|contain|contains|include|includes)\b')


def _has_structured_schema_request(text: str) -> bool:
    return bool(_STRUCTURED_OUTPUT_SCHEMA_PATTERN.search(text))


def _has_analysis_action_request(text: str) -> bool:
    return bool(_ANALYSIS_ACTION_PATTERN.search(text))


def _is_inventory_metadata_request(text: str) -> bool:
    how_much_inventory_cue = bool(
        re.search(
            r'\bhow\s+much\s+(?:data|information|files?|documents?|records?)\s+'
            r'(?:do\s+i\s+have|are\s+indexed|is\s+indexed|in\s+the\s+index|in\s+my\s+index|total)\b',
            text,
            re.IGNORECASE,
        )
    )
    metadata_aggregation_cue = bool(
        re.search(
            r'\b(time\s+span|date\s+span|span\s+of|date\s+range|range\s+of\s+dates?|from\s+when|to\s+when)\b',
            text,
            re.IGNORECASE,
        )
        and re.search(r'\b(files?|documents?|records?|data|index(?:ed)?)\b', text, re.IGNORECASE)
    )
    return bool(
        _COUNT_PATTERN.search(text)
        or _ENUMERATION_PATTERN.search(text)
        or _FILE_LIST_PATTERN.search(text)
        or _INVENTORY_CAPABILITY_PATTERN.search(text)
        or how_much_inventory_cue
        or metadata_aggregation_cue
    )


def _has_evidence_value_extraction_request(text: str) -> bool:
    return bool(_EVIDENCE_VALUE_EXTRACTION_PATTERN.search(text))


def _has_corpus_document_scope_request(text: str) -> bool:
    return bool(_CORPUS_DOCUMENT_SCOPE_PATTERN.search(text))


def _looks_multi_document_listing_request(text: str) -> bool:
    return bool(_MULTI_DOC_LISTING_PATTERN.search(text))


def _has_global_entity_listing_request(text: str) -> bool:
    return bool(_GLOBAL_ENTITY_LISTING_PATTERN.search(text))


def _looks_plural_corpus_scope_request(text: str) -> bool:
    return bool(_PLURAL_CORPUS_SCOPE_PATTERN.search(text))


def _looks_single_target_request(text: str) -> bool:
    return bool(_SINGLE_TARGET_PATTERN.search(text))


def _has_multi_year_scope_signal(text: str) -> bool:
    years = [match.group(0) for match in EXPLICIT_YEAR_PATTERN.finditer(text)]
    if len(set(years)) >= 2:
        return True
    return bool(_YEAR_AGGREGATE_CUE_PATTERN.search(text))


def _is_general_capability_query(text: str) -> bool:
    return bool(_META_QUERY_PATTERN.search(text) or _GENERIC_CAPABILITY_PATTERN.search(text))


def _looks_fact_lookup_query(text: str) -> bool:
    return bool(_FACT_LOOKUP_PATTERN.search(text))


def _has_aggregate_listing_scope_request(text: str) -> bool:
    return bool(_AGGREGATE_LISTING_SCOPE_PATTERN.search(text))


def _resolve_comparative_group_by(text: str) -> GroupBy | None:
    match = re.search(r'\bwhich\s+(file|document|year|category)\b', text)
    if not match:
        return None
    token = str(match.group(1) or '')
    if token in {'year'}:
        return GroupBy.YEAR
    if token in {'category'}:
        return GroupBy.CATEGORY
    if token in {'file', 'document'}:
        return GroupBy.FILE
    return None


def _detect_output_format(text: str) -> OutputFormat | None:
    if _OUTPUT_FORMAT_CSV_PATTERN.search(text):
        return OutputFormat.CSV
    if _OUTPUT_FORMAT_TABLE_PATTERN.search(text):
        return OutputFormat.TABLE
    if _OUTPUT_FORMAT_BULLETS_PATTERN.search(text):
        return OutputFormat.BULLETS
    if _OUTPUT_FORMAT_LIST_PATTERN.search(text):
        return OutputFormat.LIST
    if _OUTPUT_FORMAT_NARRATIVE_PATTERN.search(text):
        return OutputFormat.NARRATIVE
    return None


def _extract_filename_exclusions(text: str) -> list[str]:
    exclusions: list[str] = []
    seen: set[str] = set()
    for match in _FILENAME_EXCLUSION_PATTERN.finditer(text):
        value = str(match.group(1) or '').strip()
        if not value:
            continue
        key = value.casefold()
        if key in seen:
            continue
        seen.add(key)
        exclusions.append(value)
    return exclusions


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


@dataclass(frozen=True)
class _ClassifierSignals:
    is_inventory_metadata: bool
    has_evidence_value_request: bool
    has_corpus_scope: bool
    has_multi_doc_listing: bool
    is_general_capability: bool
    single_target_scope: bool
    has_structured_schema: bool
    has_analysis_action: bool
    has_multi_year_scope: bool
    has_aggregate_listing_scope: bool
    has_global_entity_listing: bool
    looks_fact_lookup: bool


def _resolve_classifier_signals(
    *,
    lowered: str,
    pcue: PromptCueQueryObject | None,
) -> _ClassifierSignals:
    is_inventory_metadata = _is_inventory_metadata_request(lowered)
    has_evidence_value_request = _has_evidence_value_extraction_request(lowered)
    has_corpus_scope = _has_corpus_document_scope_request(lowered)
    has_multi_doc_listing = _looks_multi_document_listing_request(lowered)
    is_general_capability = _is_general_capability_query(lowered)
    looks_fact_lookup = _looks_fact_lookup_query(lowered)
    has_aggregate_listing_scope = _has_aggregate_listing_scope_request(lowered)
    has_global_entity_listing = _has_global_entity_listing_request(lowered)

    if pcue is not None:
        semantic_hints = getattr(pcue, 'semantic_hints', None)
        single_target_scope = (str(pcue.scope) == 'focused') or _looks_single_target_request(lowered)
        has_structured_schema = (
            bool(pcue.routing_hints.get('needs_structure'))
            or bool(getattr(semantic_hints, 'requests_structure', False))
            or _has_structured_schema_request(lowered)
        )
        has_analysis_action = (
            bool(pcue.action_hints.get('should_compare'))
            or bool(getattr(semantic_hints, 'requests_comparison', False))
            or _has_analysis_action_request(lowered)
        )
        has_multi_year_scope = (
            bool(pcue.routing_hints.get('has_temporal_scope'))
            or bool(getattr(semantic_hints, 'requires_multi_period_analysis', False))
            or _has_multi_year_scope_signal(lowered)
        )
    else:
        single_target_scope = _looks_single_target_request(lowered)
        has_structured_schema = _has_structured_schema_request(lowered)
        has_analysis_action = _has_analysis_action_request(lowered)
        has_multi_year_scope = _has_multi_year_scope_signal(lowered)

    if (
        is_inventory_metadata
        and looks_fact_lookup
        and not has_corpus_scope
        and not has_structured_schema
        and not has_analysis_action
        and not has_evidence_value_request
        and not is_general_capability
    ):
        is_inventory_metadata = False

    return _ClassifierSignals(
        is_inventory_metadata=is_inventory_metadata,
        has_evidence_value_request=has_evidence_value_request,
        has_corpus_scope=has_corpus_scope,
        has_multi_doc_listing=has_multi_doc_listing,
        is_general_capability=is_general_capability,
        single_target_scope=single_target_scope,
        has_structured_schema=has_structured_schema,
        has_analysis_action=has_analysis_action,
        has_multi_year_scope=has_multi_year_scope,
        has_aggregate_listing_scope=has_aggregate_listing_scope,
        has_global_entity_listing=has_global_entity_listing,
        looks_fact_lookup=looks_fact_lookup,
    )


def _resolve_base_route(
    *,
    intent: IntentLabel,
    has_structured_schema: bool,
) -> tuple[IntentProfileId, OutputShape]:
    response_shape = OutputShape.NARRATIVE_SYNTHESIS
    if intent == IntentLabel.METADATA:
        route_candidate = IntentProfileId.METADATA_INVENTORY
        if has_structured_schema:
            response_shape = OutputShape.METADATA_TABLE
    elif intent == IntentLabel.SIMPLE:
        route_candidate = IntentProfileId.CLARIFICATION_OR_DISAMBIGUATION
    elif intent == IntentLabel.COVERAGE:
        route_candidate = IntentProfileId.CROSS_DOCUMENT_SYNTHESIS
    else:
        route_candidate = IntentProfileId.TARGETED_FACT_LOOKUP
    return route_candidate, response_shape


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
        output_format: Preferred output format extracted from query.
        secondary_intent: Optional secondary intent for compound queries.
        filename_exclude: Filename exclusions extracted from query.
        is_negation_query: Whether query contains explicit negation/exclusion semantics.
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
    output_format: OutputFormat | None = None
    secondary_intent: IntentLabel | None = None
    filename_exclude: list[str] = field(default_factory=list)
    is_negation_query: bool = False
    is_metadata_query: bool = False
    is_file_list_query: bool = False
    is_continuation: bool = False
    is_scope_reset: bool = False
    # needs_current_info: forwarded from PromptCue routing_hints when available.
    # Used by assistant-mode orchestration (for example, web-search freshness routing).
    needs_current_info: bool = False
    # mentions_time: forwarded from PromptCue semantic_hints when available.
    # Captures explicit temporal wording (for example, today/now/tomorrow/latest).
    mentions_time: bool = False
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

    year_candidates = [match.group(0) for match in EXPLICIT_YEAR_PATTERN.finditer(lowered)]
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
    output_format = _detect_output_format(lowered)
    filename_exclude = _extract_filename_exclusions(lowered)
    is_negation_query = bool(_NEGATION_PATTERN.search(lowered))
    has_comparative_request = bool(_COMPARATIVE_PATTERN.search(lowered))
    comparative_group_by = _resolve_comparative_group_by(lowered)
    secondary_intent: IntentLabel | None = None

    is_continuation = bool(_CONTINUATION_PATTERN.search(lowered))

    # --- Intent routing ---------------------------------------------------
    # When the active router is PromptCueIntentAdapter, call classify() to get
    # both the IntentPrediction and the full PromptCueQueryObject in one pass.
    # When any other router is active (e.g. test fakes), fall back to
    # classify_intent() and derive the additional signals from regex.
    from informity.llm.promptcue_adapter import PromptCueIntentAdapter

    router = get_intent_router()
    pcue: PromptCueQueryObject | None = None

    routing_expansion = expand_query_for_routing(text)
    router_query_text = routing_expansion.expanded_query or text

    if isinstance(router, PromptCueIntentAdapter):
        try:
            prediction, pcue = router.classify(router_query_text)
        except _PROMPTCUE_CLASSIFY_EXCEPTIONS:
            log.warning('promptcue_adapter_classify_failed', query=router_query_text[:120])
            prediction = router.classify_intent(router_query_text)
    else:
        prediction = router.classify_intent(router_query_text)

    intent      = prediction.intent
    reason_codes = list(prediction.reason_codes)
    if routing_expansion.canonical_terms:
        reason_codes.append('term_dictionary_routing_expansion_applied')

    # --- Signals: normalized once for deterministic policy application -------
    source_terms = _extract_source_terms(text=text, filename_filter=filename_filter)
    signals = _resolve_classifier_signals(
        lowered=lowered,
        pcue=pcue,
    )
    if pcue is not None:
        is_continuation = pcue.is_continuation

    # --- Corpus metadata promotion ----------------------------------------
    # Inventory queries (count, enumeration, file listing, capability) are
    # always 'metadata' intent regardless of the router prediction.  PromptCue
    # has no knowledge of the indexed corpus, so it cannot classify these.
    inventory_metadata_signal = signals.is_inventory_metadata
    if has_comparative_request and comparative_group_by == GroupBy.FILE:
        inventory_metadata_signal = False

    if inventory_metadata_signal and intent != IntentLabel.METADATA:
        intent = IntentLabel.METADATA
        reason_codes.append('deterministic_inventory_metadata_promoted')
    elif (
        signals.is_general_capability
        and not signals.is_inventory_metadata
        and not signals.has_structured_schema
        and not signals.has_analysis_action
        and not signals.has_evidence_value_request
        and not signals.has_corpus_scope
    ):
        intent = IntentLabel.SIMPLE
        reason_codes.append('deterministic_general_capability_to_simple')

    deterministic_override = False
    route_candidate, response_shape = _resolve_base_route(
        intent=intent,
        has_structured_schema=signals.has_structured_schema,
    )
    subtype: QuerySubtype | None = None
    if has_comparative_request:
        subtype = QuerySubtype.COMPARATIVE
        if comparative_group_by in {GroupBy.YEAR, GroupBy.CATEGORY} and intent != IntentLabel.METADATA:
            intent = IntentLabel.METADATA
            route_candidate, response_shape = _resolve_base_route(
                intent=intent,
                has_structured_schema=signals.has_structured_schema,
            )
            reason_codes.append('deterministic_comparative_metadata_group_detected')
        elif intent != IntentLabel.METADATA:
            route_candidate = IntentProfileId.COMPARATIVE_ANALYSIS
        reason_codes.append('deterministic_comparative_subtype_detected')

    if re.search(r'\b(and|also|as\s+well\s+as)\b', lowered):
        if _COUNT_PATTERN.search(lowered) and re.search(r'\blist\s+(?:them|those|files?|documents?)\b', lowered):
            secondary_intent = IntentLabel.METADATA
            reason_codes.append('deterministic_compound_count_list_detected')
        elif _FILE_LIST_PATTERN.search(lowered) and _COMPARATIVE_FILE_CONTENT_CUE_PATTERN.search(lowered):
            secondary_intent = IntentLabel.FOCUSED
            reason_codes.append('deterministic_compound_list_content_detected')
    group_by: GroupBy | None = GroupBy.YEAR if signals.has_multi_year_scope else None
    if has_comparative_request and comparative_group_by is not None:
        group_by = comparative_group_by

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

    if has_comparative_request and comparative_group_by == GroupBy.FILE:
        apply_override(
            reason_code='deterministic_comparative_file_scope_to_focused',
            new_intent=IntentLabel.FOCUSED,
            new_route=IntentProfileId.COMPARATIVE_ANALYSIS,
            new_shape=OutputShape.NARRATIVE_SYNTHESIS,
        )

    # Minimal deterministic guardrail: single-target requests should remain
    # focused even when base classification predicts broad synthesis.
    if (
        intent == IntentLabel.COVERAGE
        and signals.single_target_scope
        and not _looks_plural_corpus_scope_request(lowered)
        and not signals.has_multi_doc_listing
        and not signals.has_multi_year_scope
        and not signals.has_aggregate_listing_scope
        and not signals.has_global_entity_listing
    ):
        apply_override(
            reason_code='deterministic_override_single_target_to_focused',
            new_intent=IntentLabel.FOCUSED,
            new_route=IntentProfileId.TARGETED_FACT_LOOKUP,
            new_shape=OutputShape.NARRATIVE_SYNTHESIS,
        )

    # Minimal deterministic guardrail: plural corpus synthesis/listing prompts
    # should route to coverage even when base classification predicts focused.
    if (
        intent == IntentLabel.FOCUSED
        and filename_filter is None
        and not (has_comparative_request and comparative_group_by == GroupBy.FILE)
        and signals.has_corpus_scope
        and (
            signals.has_multi_doc_listing
            or signals.has_aggregate_listing_scope
            or signals.has_global_entity_listing
            or (signals.has_structured_schema and _looks_plural_corpus_scope_request(lowered))
            or (
                signals.has_analysis_action
                and _looks_plural_corpus_scope_request(lowered)
                and not signals.single_target_scope
            )
        )
    ):
        apply_override(
            reason_code='deterministic_override_plural_corpus_to_coverage',
            new_intent=IntentLabel.COVERAGE,
            new_route=IntentProfileId.CROSS_DOCUMENT_SYNTHESIS,
            new_shape=OutputShape.NARRATIVE_SYNTHESIS,
        )

    if (
        intent == IntentLabel.COVERAGE
        and filename_filter is None
        and (signals.has_multi_year_scope or group_by == GroupBy.YEAR)
    ):
        if subtype != QuerySubtype.AGGREGATE_BY_PERIOD:
            subtype = QuerySubtype.AGGREGATE_BY_PERIOD
            reason_codes.append('deterministic_override_coverage_year_aggregate_subtype')
        if response_shape != OutputShape.NARRATIVE_SYNTHESIS and not signals.has_evidence_value_request:
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
        has_multi_year_scope=signals.has_multi_year_scope,
        group_by=group_by,
        source_terms=source_terms,
        year_filter=year_filter,
        file_type_filter=file_type_filter,
        filename_filter=filename_filter,
        output_format=output_format,
        secondary_intent=secondary_intent,
        filename_exclude=filename_exclude,
        is_negation_query=is_negation_query,
        is_metadata_query=(intent == IntentLabel.METADATA),
        is_file_list_query=(intent == IntentLabel.METADATA and bool(_FILE_LIST_PATTERN.search(lowered))),
        is_continuation=is_continuation,
        needs_current_info=bool(pcue and pcue.routing_hints.get('needs_current_info')),
        mentions_time=bool(getattr(getattr(pcue, 'semantic_hints', None), 'mentions_time', False)),
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
