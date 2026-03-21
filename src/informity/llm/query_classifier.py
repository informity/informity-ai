# ==============================================================================
# Informity AI — Query Classifier (v2)
# Deterministic slot extraction and intent routing.
# ==============================================================================

from dataclasses import dataclass, field
from typing import Literal

import structlog

from informity.config import settings
from informity.llm.intent_router import get_intent_router
from informity.llm.query_patterns import (
    build_analysis_action_pattern,
    build_continuation_pattern,
    build_count_pattern,
    build_coverage_pattern,
    build_enumeration_pattern,
    build_evidence_value_extraction_pattern,
    build_file_list_pattern,
    build_inventory_capability_pattern,
    build_structured_output_schema_pattern,
)

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
    import re
    return bool(re.search(r'\b(across|all|cross[\s-]*document|year[\s-]*by[\s-]*year)\b', text))


def _has_evidence_value_extraction_request(text: str) -> bool:
    return bool(_EVIDENCE_VALUE_EXTRACTION_PATTERN.search(text))


def _has_corpus_document_scope_request(text: str) -> bool:
    import re
    return bool(re.search(r'\b(indexed\s+)?(files?|documents?)\b', text))


def _looks_multi_document_listing_request(text: str) -> bool:
    import re
    return bool(re.search(r'\b(which|list|show)\b.*\b(files?|documents?)\b', text))


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
    intent: Literal['metadata', 'focused', 'coverage', 'simple']
    response_shape: Literal['structured_extract', 'narrative_synthesis', 'metadata_table', 'hybrid'] = 'narrative_synthesis'
    route_candidate: Literal[
        'metadata_inventory',
        'targeted_fact_lookup',
        'structured_field_extraction',
        'cross_document_synthesis',
        'comparative_analysis',
        'audit_or_compliance_brief',
        'continuation_or_refinement',
        'clarification_or_disambiguation',
    ] = 'targeted_fact_lookup'
    confidence: float = 0.5
    alternatives: list[tuple[str, float]] = field(default_factory=list)
    reason_codes: list[str] = field(default_factory=list)
    missing_slots: list[str] = field(default_factory=list)
    subtype: Literal['extract_structured_values', 'aggregate_by_period', 'file_inventory'] | None = None
    has_multi_year_scope: bool = False
    group_by: Literal['year', 'category', 'file'] | None = None
    field_hint: str | None = None
    source_terms: list[str] = field(default_factory=list)
    year_filter: int | None = None
    category_filter: str | None = None
    file_type_filter: str | None = None
    filename_filter: str | None = None
    block_type_filter: Literal['table', 'form', 'narrative'] | None = None
    section_filter: str | None = None
    is_metadata_query: bool = False
    is_file_list_query: bool = False
    is_continuation: bool = False
    is_scope_reset: bool = False
    # Provenance flags — describe how route_candidate was selected.
    # deterministic_override: True when a hard aggregate rule fired (e.g. policy_aggregate_route_enforced).
    # llm_confidence: raw confidence reported by the LLM (0.0 when LLM did not emit a confidence field).
    deterministic_override: bool = False
    llm_confidence: float = 0.0

    @property
    def confidence_band(self) -> Literal['high', 'medium', 'low']:
        if self.confidence >= settings.classification_confidence_high_threshold:
            return 'high'
        if self.confidence >= settings.classification_confidence_medium_threshold:
            return 'medium'
        return 'low'


def classify_query(query: str) -> QueryClassification:
    """Classify query via pluggable intent router + deterministic slot extraction."""
    text = str(query or '').strip()
    lowered = text.casefold()
    import re

    year_filter: int | None = None
    year_match = None

    year_candidates = re.findall(r'\b(?:19|20)\d{2}\b', lowered)
    if len(year_candidates) == 1:
        year_match = int(year_candidates[0])
        if 1900 <= year_match <= 2099:
            year_filter = year_match

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

    prediction = get_intent_router().classify_intent(text)
    intent = prediction.intent
    reason_codes = list(prediction.reason_codes)
    deterministic_override = False
    response_shape: Literal['structured_extract', 'narrative_synthesis', 'metadata_table', 'hybrid'] = 'narrative_synthesis'
    subtype: Literal['extract_structured_values', 'aggregate_by_period', 'file_inventory'] | None = None

    if intent == 'metadata':
        has_structured_schema = _has_structured_schema_request(lowered)
        has_analysis_action = _has_analysis_action_request(lowered)
        is_inventory_metadata = _is_inventory_metadata_request(lowered)
        if has_structured_schema and not is_inventory_metadata:
            deterministic_override = True
            if _looks_broad_scope(lowered):
                intent = 'coverage'
                route_candidate = 'comparative_analysis'
                response_shape = 'metadata_table'
            else:
                intent = 'focused'
                route_candidate = 'structured_field_extraction'
                response_shape = 'structured_extract'
            subtype = 'extract_structured_values'
            reason_codes.append('deterministic_override_structured_schema_request')
        elif has_analysis_action and not is_inventory_metadata:
            deterministic_override = True
            if _looks_broad_scope(lowered):
                intent = 'coverage'
                route_candidate = 'cross_document_synthesis'
            else:
                intent = 'focused'
                route_candidate = 'targeted_fact_lookup'
            response_shape = 'narrative_synthesis'
            reason_codes.append('deterministic_override_analysis_request')
        elif (
            _has_evidence_value_extraction_request(lowered)
            and (is_inventory_metadata or _has_corpus_document_scope_request(lowered))
        ):
            deterministic_override = True
            if _looks_broad_scope(lowered) or _looks_multi_document_listing_request(lowered):
                intent = 'coverage'
                route_candidate = 'cross_document_synthesis'
            else:
                intent = 'focused'
                route_candidate = 'targeted_fact_lookup'
            response_shape = 'narrative_synthesis'
            reason_codes.append('deterministic_override_inventory_with_evidence_request')
        else:
            route_candidate = 'metadata_inventory'
            if has_structured_schema:
                response_shape = 'metadata_table'
    elif intent == 'simple':
        route_candidate = 'clarification_or_disambiguation'
    elif intent == 'coverage':
        route_candidate = 'cross_document_synthesis'
        if _has_structured_schema_request(lowered):
            deterministic_override = True
            route_candidate = 'comparative_analysis'
            response_shape = 'metadata_table'
            subtype = 'extract_structured_values'
            reason_codes.append('deterministic_override_structured_schema_for_coverage')
    else:
        route_candidate = 'targeted_fact_lookup'
        if _has_structured_schema_request(lowered):
            deterministic_override = True
            route_candidate = 'structured_field_extraction'
            response_shape = 'structured_extract'
            subtype = 'extract_structured_values'
            reason_codes.append('deterministic_override_structured_schema_for_focused')

    return QueryClassification(
        intent=intent,
        response_shape=response_shape,
        route_candidate=route_candidate,
        confidence=prediction.confidence,
        alternatives=prediction.alternatives,
        reason_codes=reason_codes,
        subtype=subtype,
        year_filter=year_filter,
        file_type_filter=file_type_filter,
        filename_filter=filename_filter,
        is_metadata_query=(intent == 'metadata'),
        is_file_list_query=(intent == 'metadata' and bool(_FILE_LIST_PATTERN.search(lowered))),
        is_continuation=is_continuation,
        deterministic_override=deterministic_override,
        llm_confidence=0.0,
    )


__all__ = [
    'CONFIDENCE_HIGH_THRESHOLD',
    'CONFIDENCE_MEDIUM_THRESHOLD',
    'classify_query',
    'QueryClassification',
]
