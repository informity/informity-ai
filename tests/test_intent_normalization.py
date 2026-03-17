from informity.llm.intent_normalization import normalize_intent_policy_fields


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


def test_normalize_intent_policy_promotes_plural_multi_scope_to_coverage() -> None:
    intent, subtype, _shape, reasons = normalize_intent_policy_fields(
        query='What are the findings across all documents?',
        intent='focused',
        subtype=None,
        response_shape='narrative_synthesis',
        group_by=None,
        filename_filter=None,
        has_multi_year_scope=False,
    )

    assert intent == 'coverage'
    assert 'policy_plural_multi_scope_to_coverage' in reasons


def test_normalize_intent_policy_plural_multi_scope_skipped_with_filename_filter() -> None:
    intent, _subtype, _shape, reasons = normalize_intent_policy_fields(
        query='What are the findings across all documents?',
        intent='focused',
        subtype=None,
        response_shape='narrative_synthesis',
        group_by=None,
        filename_filter='report_2024.pdf',
        has_multi_year_scope=False,
    )

    assert intent == 'focused'
    assert 'policy_plural_multi_scope_to_coverage' not in reasons


def test_normalize_intent_policy_combined_cross_year_and_narrative_brief() -> None:
    intent, subtype, response_shape, reasons = normalize_intent_policy_fields(
        query='Create a report with risks and gaps across years.',
        intent='focused',
        subtype='aggregate_by_period',
        response_shape='structured_extract',
        group_by='year',
        filename_filter=None,
        has_multi_year_scope=True,
    )

    assert intent == 'coverage'
    assert response_shape == 'narrative_synthesis'
    assert subtype == 'aggregate_by_period'
    assert 'policy_cross_year_focus_to_coverage' in reasons
    assert 'policy_narrative_brief_shape' in reasons
