from informity.llm.intent_normalization import (
    normalize_intent_policy_fields,
    normalize_query_classification,
)
from informity.llm.query_classifier import QueryClassification


def test_normalize_intent_policy_promotes_cross_year_aggregate_to_coverage() -> None:
    intent, subtype, response_shape, reasons = normalize_intent_policy_fields(
        query='Build a year-by-year summary across 2022 and 2023.',
        intent='focused',
        subtype='aggregate_by_period',
        response_shape='structured_extract',
        group_by='year',
        filename_filter=None,
        has_multi_year_scope=True,
    )

    assert intent == 'coverage'
    assert subtype == 'aggregate_by_period'
    assert response_shape == 'structured_extract'
    assert 'policy_cross_year_focus_to_coverage' in reasons


def test_normalize_intent_policy_promotes_cross_set_comparison_to_coverage() -> None:
    intent, subtype, _shape, reasons = normalize_intent_policy_fields(
        query='Compare files to documents and summarize differences.',
        intent='metadata',
        subtype='file_inventory',
        response_shape='metadata_table',
        group_by=None,
        filename_filter=None,
        has_multi_year_scope=False,
    )

    assert intent == 'coverage'
    assert subtype is None
    assert 'policy_cross_set_comparison_to_coverage' in reasons


def test_normalize_query_classification_applies_shared_runtime_policy() -> None:
    classification = QueryClassification(
        intent='focused',
        response_shape='structured_extract',
        route_candidate='structured_field_extraction',
        reason_codes=['llm_profile_selected'],
        subtype='aggregate_by_period',
        has_multi_year_scope=True,
        group_by='year',
        is_metadata_query=False,
    )

    normalized, reasons = normalize_query_classification(
        query='Create a report with risks and gaps across years.',
        classification=classification,
    )

    assert normalized.intent == 'coverage'
    assert normalized.response_shape == 'narrative_synthesis'
    assert normalized.subtype == 'aggregate_by_period'
    assert normalized.is_metadata_query is False
    assert 'policy_cross_year_focus_to_coverage' in reasons
    assert 'policy_narrative_brief_shape' in normalized.reason_codes
