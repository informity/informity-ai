# ==============================================================================
# Informity AI — Intent/Subtype Normalization Policy
# Shared deterministic normalization for classifier + runtime paths.
# ==============================================================================

from __future__ import annotations

import re

from informity.llm.types import GroupBy, OutputShape, QuerySubtype, QueryType

_NARRATIVE_BRIEF_PATTERN = re.compile(r'\b(report|brief|evidence map|risk[s]? and gaps|action checklist)\b', re.IGNORECASE)
_CROSS_SET_COMPARISON_PATTERN = re.compile(
    r'\b(compare|comparison)\b[\s\S]{0,80}\b(to|between|versus|vs\.?)\b',
    re.IGNORECASE,
)
_CROSS_SET_SCOPE_PATTERN = re.compile(r'\b(files?|documents?|categories?)\b', re.IGNORECASE)
_BROAD_SCOPE_SIGNAL_PATTERN = re.compile(
    r'\b(all|every|each|across all|across the|throughout)\b',
    re.IGNORECASE,
)
_PLURAL_DOCUMENT_PATTERN = re.compile(
    r'\b(documents|files|reports|sources)\b',
    re.IGNORECASE,
)
# Signals that a query is operating on document *content*, not just listing files.
# Used to guard policy_plural_multi_scope_to_coverage from firing on inventory queries
# like "list all document files" which have "all" + "files" but no content operation.
_CONTENT_OPERATION_PATTERN = re.compile(
    r'\b(what (is|are|does|do|did)|summarize|summary|contains?|content|extract|analyze|analysis|'
    r'findings?|compare|contrast|about|say|says|mention|describe|tell me|show me what)\b',
    re.IGNORECASE,
)


def normalize_intent_policy_fields(
    *,
    query: str,
    intent: QueryType,
    subtype: QuerySubtype | None,
    response_shape: OutputShape,
    group_by: GroupBy | None,
    filename_filter: str | None,
    has_multi_year_scope: bool,
) -> tuple[QueryType, QuerySubtype | None, OutputShape, list[str]]:
    reason_codes: list[str] = []
    normalized_intent = intent
    normalized_subtype = subtype
    normalized_response_shape = response_shape

    if (
        normalized_intent == QueryType.FOCUSED
        and normalized_subtype == QuerySubtype.AGGREGATE_BY_PERIOD
        and (group_by == GroupBy.YEAR or has_multi_year_scope)
        and filename_filter is None
    ):
        normalized_intent = QueryType.COVERAGE
        reason_codes.append('policy_cross_year_focus_to_coverage')

    if (
        normalized_intent == QueryType.FOCUSED
        and _BROAD_SCOPE_SIGNAL_PATTERN.search(query)
        and _PLURAL_DOCUMENT_PATTERN.search(query)
        and _CONTENT_OPERATION_PATTERN.search(query)
        and filename_filter is None
    ):
        normalized_intent = QueryType.COVERAGE
        reason_codes.append('policy_plural_multi_scope_to_coverage')

    if (
        normalized_intent in {QueryType.FOCUSED, QueryType.METADATA}
        and _CROSS_SET_COMPARISON_PATTERN.search(query)
        and _CROSS_SET_SCOPE_PATTERN.search(query)
        and filename_filter is None
    ):
        normalized_intent = QueryType.COVERAGE
        if normalized_subtype == QuerySubtype.FILE_INVENTORY:
            normalized_subtype = None
        reason_codes.append('policy_cross_set_comparison_to_coverage')

    if normalized_intent in {QueryType.FOCUSED, QueryType.COVERAGE} and _NARRATIVE_BRIEF_PATTERN.search(query):
        normalized_response_shape = OutputShape.NARRATIVE_SYNTHESIS
        reason_codes.append('policy_narrative_brief_shape')
        if has_multi_year_scope or group_by == GroupBy.YEAR:
            normalized_subtype = QuerySubtype.AGGREGATE_BY_PERIOD
            reason_codes.append('policy_narrative_brief_aggregate_subtype')

    return normalized_intent, normalized_subtype, normalized_response_shape, reason_codes
