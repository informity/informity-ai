from informity.llm.query_classifier import QueryClassification
from informity.llm.rag_runtime.structured_numeric import (
    _derive_format_requirements,
    _extract_candidate_values,
    _extract_requested_table_columns,
    _parse_numeric_token,
    _render_structured_rows_answer,
    _render_year_aggregate_answer,
    _should_run_structured_extraction,
    _validate_structured_rows,
)


def test_parse_numeric_token_ignores_plain_year() -> None:
    assert _parse_numeric_token('2023') is None


def test_parse_numeric_token_parses_currency() -> None:
    parsed = _parse_numeric_token('$12,345.67')
    assert parsed is not None
    value, has_currency_shape = parsed
    assert value == 12345.67
    assert has_currency_shape is True


def test_extract_candidate_values_prefers_near_field_hint() -> None:
    text = (
        'Box 1 wages: $4,500.00 for this tax year. '
        'Unrelated amount appears later as $15.00.'
    )
    candidates = _extract_candidate_values(
        chunk_text=text,
        field_hint='box_1',
        base_score=0.0,
    )
    assert candidates
    assert float(candidates[0]['value']) == 4500.00


def test_render_structured_rows_answer_renders_markdown_table() -> None:
    answer = _render_structured_rows_answer(
        rows=[
            {
                'field_label': 'box_1_wages',
                'raw_value': '$1,200.00',
                'evidence_span': 'Box 1 wages are listed as $1,200.00 on the payroll form.',
            },
            {
                'field_label': 'box_2_tax_withheld',
                'raw_value': '$240.00',
                'evidence_span': 'Box 2 tax withheld is shown as $240.00.',
            },
        ],
    )
    assert 'Deterministic Structured Extraction' in answer
    assert '| Field | Value | Source Snippet |' in answer
    assert 'box_1_wages' in answer
    assert '$240.00' in answer


def test_render_structured_rows_answer_uses_requested_column_headers() -> None:
    answer = _render_structured_rows_answer(
        rows=[
            {
                'field_label': 'mortgage_interest',
                'raw_value': '$5,095.64',
                'evidence_span': 'Mortgage interest paid: $5,095.64.',
            },
        ],
        table_columns=['Line Item', 'Amount', 'Source Snippet'],
    )
    assert '| Line Item | Amount | Source Snippet |' in answer
    assert '| mortgage_interest | $5,095.64 |' in answer


def test_extract_requested_table_columns_from_question() -> None:
    question = (
        'Output only a markdown table with columns: '
        'Line Item, Amount, Source Snippet.'
    )
    columns = _extract_requested_table_columns(question)
    assert columns == ['Line Item', 'Amount', 'Source Snippet']


def test_should_not_run_for_narrative_forensic_report_prompt() -> None:
    classification = QueryClassification(
        intent='coverage',
        subtype='aggregate_by_period',
        group_by='year',
        field_hint=None,
    )
    question = (
        'Build a forensic reconciliation report across 2022-2024 with sections '
        'for Scope, Method, Findings by Year, Cross-Year Deltas, and Confidence Notes. '
        'Include extracted amounts and contradictions.'
    )
    assert _should_run_structured_extraction(
        question=question,
        classification=classification,
        response_shape='narrative_synthesis',
    ) is False


def test_should_not_run_for_summary_prompt_with_global_contract() -> None:
    classification = QueryClassification(
        intent='focused',
        subtype='aggregate_by_period',
        group_by=None,
        field_hint=None,
    )
    question = (
        'Summarize the content of planning_scenarios.md in <= 180 words. '
        'Include exactly 3 bullets: objective, key tradeoff, decision implication.'
    )
    assert _should_run_structured_extraction(
        question=question,
        classification=classification,
        response_shape='structured_extract',
    ) is False


def test_should_not_run_when_missing_evidence_callout_is_required() -> None:
    classification = QueryClassification(
        intent='focused',
        subtype='extract_structured_values',
        group_by=None,
        field_hint=None,
    )
    question = (
        'Build an evidence map by year (2022-2024) showing document groups present vs missing. '
        'Call out missing evidence for each year/group.'
    )
    assert _should_run_structured_extraction(
        question=question,
        classification=classification,
        response_shape='structured_extract',
    ) is False


def test_render_year_aggregate_answer_marks_missing_years() -> None:
    answer = _render_year_aggregate_answer(
        rows=[
            {
                'file_id': 1,
                'raw_value': '$1,000.00',
                'value': 1000.0,
                'evidence_span': 'Box 1 wages are listed as $1,000.00.',
            },
        ],
        metadata_by_file_id={
            1: {'year': 2022},
            2: {'year': 2023},
            3: {'year': 2024},
        },
        required_years=[2022, 2023, 2024],
    )
    assert '### Deterministic Numeric Extraction' in answer
    assert '| 2022 |' in answer
    assert '| 2023 | Missing evidence | N/A |' in answer
    assert '| 2024 | Missing evidence | N/A |' in answer
    assert 'Grand total' in answer


def test_derive_format_requirements_extracts_headings_and_depth() -> None:
    question = (
        'Create a brief with these sections in order: 1) Executive Summary, 2) Findings, '
        '3) Action Checklist. Use nested bullets with exactly 3 levels and call out missing evidence.'
    )
    requirements = _derive_format_requirements(question)
    assert any('in the requested order' in requirement for requirement in requirements)
    assert any('include heading: Executive Summary' in requirement for requirement in requirements)
    assert any('exactly 3 levels' in requirement for requirement in requirements)
    assert any('missing evidence' in requirement for requirement in requirements)


def test_derive_format_requirements_extracts_year_subsection_contract() -> None:
    question = (
        'Build a forensic reconciliation report across all indexed records with exact headings: '
        '## Scope, ## Method, ## Findings by Year. '
        'Under ## Findings by Year include one subsection per indexed year.'
    )
    requirements = _derive_format_requirements(question)
    assert any('one subsection per year' in requirement for requirement in requirements)
    assert any('at least 2 distinct year subsections' in requirement for requirement in requirements)


def test_validate_structured_rows_filters_ssn_like_values() -> None:
    rows = [
        {
            'file_id': 1,
            'field_label': 'tax id',
            'raw_value': '000-00-0000',
            'evidence_span': 'Synthetic SSN-like value 000-00-0000 appears on this page.',
            'confidence': 0.91,
        },
        {
            'file_id': 1,
            'field_label': 'mortgage interest',
            'raw_value': '$5,095.64',
            'evidence_span': 'Mortgage interest paid: $5,095.64.',
            'confidence': 0.88,
        },
    ]
    validated = _validate_structured_rows(rows)
    assert len(validated) == 1
    assert validated[0]['raw_value'] == '$5,095.64'
