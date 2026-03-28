from informity.llm.intent_normalization import normalize_intent_policy_fields
from informity.llm.types import GroupBy, OutputShape, QuerySubtype, QueryType


def test_normalize_intent_policy_promotes_cross_year_aggregate_to_coverage() -> None:
    intent, subtype, response_shape, reasons = normalize_intent_policy_fields(
        query='Build a year-by-year summary across 2022 and 2023.',
        intent=QueryType.FOCUSED,
        subtype=QuerySubtype.AGGREGATE_BY_PERIOD,
        response_shape=OutputShape.STRUCTURED_EXTRACT,
        group_by=GroupBy.YEAR,
        filename_filter=None,
        has_multi_year_scope=True,
    )

    assert intent == QueryType.COVERAGE
    assert subtype == QuerySubtype.AGGREGATE_BY_PERIOD
    assert response_shape == OutputShape.STRUCTURED_EXTRACT
    assert 'policy_cross_year_focus_to_coverage' in reasons


def test_normalize_intent_policy_promotes_cross_set_comparison_to_coverage() -> None:
    intent, subtype, _shape, reasons = normalize_intent_policy_fields(
        query='Compare files to documents and summarize differences.',
        intent=QueryType.METADATA,
        subtype=QuerySubtype.FILE_INVENTORY,
        response_shape=OutputShape.METADATA_TABLE,
        group_by=None,
        filename_filter=None,
        has_multi_year_scope=False,
    )

    assert intent == QueryType.COVERAGE
    assert subtype is None
    assert 'policy_cross_set_comparison_to_coverage' in reasons


def test_normalize_intent_policy_promotes_plural_multi_scope_to_coverage() -> None:
    intent, subtype, _shape, reasons = normalize_intent_policy_fields(
        query='What are the findings across all documents?',
        intent=QueryType.FOCUSED,
        subtype=None,
        response_shape=OutputShape.NARRATIVE_SYNTHESIS,
        group_by=None,
        filename_filter=None,
        has_multi_year_scope=False,
    )

    assert intent == QueryType.COVERAGE
    assert 'policy_plural_multi_scope_to_coverage' in reasons


def test_normalize_intent_policy_plural_multi_scope_skipped_with_filename_filter() -> None:
    intent, _subtype, _shape, reasons = normalize_intent_policy_fields(
        query='What are the findings across all documents?',
        intent=QueryType.FOCUSED,
        subtype=None,
        response_shape=OutputShape.NARRATIVE_SYNTHESIS,
        group_by=None,
        filename_filter='report_2024.pdf',
        has_multi_year_scope=False,
    )

    assert intent == QueryType.FOCUSED
    assert 'policy_plural_multi_scope_to_coverage' not in reasons


def test_normalize_intent_policy_combined_cross_year_and_narrative_brief() -> None:
    intent, subtype, response_shape, reasons = normalize_intent_policy_fields(
        query='Create a report with risks and gaps across years.',
        intent=QueryType.FOCUSED,
        subtype=QuerySubtype.AGGREGATE_BY_PERIOD,
        response_shape=OutputShape.STRUCTURED_EXTRACT,
        group_by=GroupBy.YEAR,
        filename_filter=None,
        has_multi_year_scope=True,
    )

    assert intent == QueryType.COVERAGE
    assert response_shape == OutputShape.NARRATIVE_SYNTHESIS
    assert subtype == QuerySubtype.AGGREGATE_BY_PERIOD
    assert 'policy_cross_year_focus_to_coverage' in reasons
    assert 'policy_narrative_brief_shape' in reasons
