# ==============================================================================
# Informity AI — Intent/Subtype Normalization Policy
# Shared deterministic normalization for classifier + runtime paths.
# ==============================================================================

from __future__ import annotations

import re

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


def normalize_intent_policy_fields(
    *,
    query: str,
    intent: str,
    subtype: str | None,
    response_shape: str,
    group_by: str | None,
    filename_filter: str | None,
    has_multi_year_scope: bool,
) -> tuple[str, str | None, str, list[str]]:
    reason_codes: list[str] = []
    normalized_intent = intent
    normalized_subtype = subtype
    normalized_response_shape = response_shape

    if (
        normalized_intent == 'focused'
        and normalized_subtype == 'aggregate_by_period'
        and (group_by == 'year' or has_multi_year_scope)
        and filename_filter is None
    ):
        normalized_intent = 'coverage'
        reason_codes.append('policy_cross_year_focus_to_coverage')

    if (
        normalized_intent == 'focused'
        and _BROAD_SCOPE_SIGNAL_PATTERN.search(query)
        and _PLURAL_DOCUMENT_PATTERN.search(query)
        and filename_filter is None
    ):
        normalized_intent = 'coverage'
        reason_codes.append('policy_plural_multi_scope_to_coverage')

    if (
        normalized_intent in {'focused', 'metadata'}
        and _CROSS_SET_COMPARISON_PATTERN.search(query)
        and _CROSS_SET_SCOPE_PATTERN.search(query)
        and filename_filter is None
    ):
        normalized_intent = 'coverage'
        if normalized_subtype == 'file_inventory':
            normalized_subtype = None
        reason_codes.append('policy_cross_set_comparison_to_coverage')

    if normalized_intent in {'focused', 'coverage'} and _NARRATIVE_BRIEF_PATTERN.search(query):
        normalized_response_shape = 'narrative_synthesis'
        reason_codes.append('policy_narrative_brief_shape')
        if has_multi_year_scope or group_by == 'year':
            normalized_subtype = 'aggregate_by_period'
            reason_codes.append('policy_narrative_brief_aggregate_subtype')

    return normalized_intent, normalized_subtype, normalized_response_shape, reason_codes
