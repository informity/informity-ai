from informity.llm.query_classifier_llm import (
    _extract_phase2_constraints,
    _is_filename_summary_query,
    _is_inventory_plus_content_coverage_query,
    _resolve_route_candidate,
)


def test_route_candidate_prefers_structured_profile_for_aggregation_extract() -> None:
    route_candidate, confidence, reason_codes = _resolve_route_candidate(
        llm_route_candidate=None,
        intent='focused',
        response_shape='structured_extract',
        subtype='aggregate_by_period',
        group_by='year',
    )
    assert route_candidate == 'structured_field_extraction'
    assert confidence >= 0.8
    assert 'policy_default_selected' in reason_codes


def test_route_candidate_enforces_comparative_for_coverage_aggregate() -> None:
    route_candidate, confidence, reason_codes = _resolve_route_candidate(
        llm_route_candidate='structured_field_extraction',
        intent='coverage',
        response_shape='structured_extract',
        subtype='aggregate_by_period',
        group_by='year',
    )
    assert route_candidate == 'comparative_analysis'
    assert confidence >= 0.85
    assert 'policy_aggregate_route_enforced' in reason_codes


def test_phase2_extracts_grouping_field_and_source_terms() -> None:
    query = 'Extract Box 1 values by year from FormA forms and files where filenames contain "FormA" or "FormB".'
    constraints = _extract_phase2_constraints(query)
    assert constraints['group_by'] == 'year'
    assert constraints['field_hint'] == 'box_1'
    assert 'FormA' in constraints['source_terms']
    assert 'FormB' in constraints['source_terms']


def test_phase2_extracts_source_term_from_mention_clause() -> None:
    query = 'Which files mention FormA, and what do those files say about withholding or wages?'
    constraints = _extract_phase2_constraints(query)
    assert constraints['source_terms'] == ['FormA']


def test_phase2_extracts_section_hint() -> None:
    query = 'Compare values per file from section liabilities across the indexed documents.'
    constraints = _extract_phase2_constraints(query)
    assert constraints['group_by'] == 'file'
    assert isinstance(constraints['section_hint'], str)
    assert constraints['section_hint'].startswith('liabilities')


def test_phase2_ignores_numeric_section_instruction() -> None:
    query = 'In section 3, use nested bullets with exactly 3 levels and then summarize.'
    constraints = _extract_phase2_constraints(query)
    assert constraints['section_hint'] is None


def test_phase2_detects_year_by_year_aggregation_semantics() -> None:
    query = 'Create a year-by-year evidence map across 2022-2024 with totals.'
    constraints = _extract_phase2_constraints(query)
    assert constraints['aggregation_semantics'] is True


def test_phase2_detects_year_over_year_period_comparison_semantics() -> None:
    query = 'Produce a year-over-year comparison with largest increase and deltas.'
    constraints = _extract_phase2_constraints(query)
    assert constraints['period_comparison_semantics'] is True


def test_route_candidate_prefers_audit_brief_for_multi_section_coverage_report() -> None:
    route_candidate, confidence, reason_codes = _resolve_route_candidate(
        llm_route_candidate='audit_or_compliance_brief',
        intent='coverage',
        response_shape='narrative_synthesis',
        subtype='extract_structured_values',
        group_by='year',
    )
    assert route_candidate == 'audit_or_compliance_brief'
    assert confidence >= 0.8
    assert 'llm_profile_selected' in reason_codes


def test_inventory_plus_content_query_pattern_detects_coverage_shape() -> None:
    assert _is_inventory_plus_content_coverage_query(
        'Which files mention FormA, and what do those files say about withholding or wages?'
    ) is True
    assert _is_inventory_plus_content_coverage_query(
        'Which files mention FormA?'
    ) is False


def test_route_candidate_rejects_focused_profile_for_coverage_intent() -> None:
    route_candidate, confidence, reason_codes = _resolve_route_candidate(
        llm_route_candidate='targeted_fact_lookup',
        intent='coverage',
        response_shape='narrative_synthesis',
        subtype=None,
        group_by=None,
    )
    assert route_candidate == 'cross_document_synthesis'
    assert confidence >= 0.75
    assert 'policy_default_selected' in reason_codes


def test_filename_summary_query_pattern_detects_summary_phrasing() -> None:
    assert _is_filename_summary_query('Summarize the content of portfolio_notes.md') is True
    assert _is_filename_summary_query('What does annual_statement.pdf contain?') is True
    assert _is_filename_summary_query('Extract box 1 from FormA by year') is False
