from __future__ import annotations

from informity.llm import contract_prompt_parser as parser


def test_extract_required_headings_supports_markdown_and_numbered_sections() -> None:
    question = (
        "Output must contain:\n## Scope\n## Method\n"
        "Also use sections in order: 1) Findings by Year, 2) Next Steps."
    )
    headings = parser.extract_required_headings(question)
    assert headings == ['Scope', 'Method', 'Findings by Year', 'Next Steps']


def test_has_year_subsection_cue_detects_findings_by_year() -> None:
    assert parser.has_year_subsection_cue('Provide Findings by Year across all indexed records.') is True


def test_extract_required_years_returns_sorted_unique_years() -> None:
    years = parser.extract_required_years('Compare 2024 vs 2022 and include 2024 again.')
    assert years == [2022, 2024]


def test_extract_required_headings_inline_exact_order_list_stops_before_under_clauses() -> None:
    question = (
        "Using all indexed records, produce one response with headings in exact order: "
        "## Executive Summary, ## Evidence Map by Year, ## Financial Deltas, "
        "## Contradictions and Gaps, ## Verification Actions. "
        "Under ## Evidence Map by Year include one subsection per indexed year."
    )
    headings = parser.extract_required_headings(question)
    assert headings == [
        'Executive Summary',
        'Evidence Map by Year',
        'Financial Deltas',
        'Contradictions and Gaps',
        'Verification Actions',
    ]


def test_extract_required_headings_inline_with_exact_headings_phrase() -> None:
    question = (
        'Build a forensic report with exact headings: '
        '## Scope, ## Method, ## Findings by Year, ## Cross-Year Deltas, '
        '## Confidence Notes, ## Next Verification Steps. '
        'Under ## Findings by Year include one subsection per indexed year.'
    )
    headings = parser.extract_required_headings(question)
    assert headings == [
        'Scope',
        'Method',
        'Findings by Year',
        'Cross-Year Deltas',
        'Confidence Notes',
        'Next Verification Steps',
    ]


def test_extract_required_headings_supports_exactly_n_sections_phrase() -> None:
    question = (
        'Summarize cross-year changes in extracted numeric values. '
        'Output exactly 3 sections: ## Biggest Increase, ## Biggest Decrease, ## Ambiguous Delta.'
    )
    headings = parser.extract_required_headings(question)
    assert headings == ['Biggest Increase', 'Biggest Decrease', 'Ambiguous Delta']


def test_extract_required_labels_supports_columns_and_format_cues() -> None:
    question = (
        'Output bullets in format: Field | Value | Source Snippet. '
        'Under details include one markdown table with columns: Year, Amount, Evidence Note.'
    )
    labels = parser.extract_required_labels(question)
    assert 'Field' in labels
    assert 'Value' in labels
    assert 'Source Snippet' in labels
    assert 'Year' in labels


def test_extract_required_labels_supports_include_list_cue() -> None:
    question = (
        'Compare records from different years and explain where totals or balances disagree. '
        'Include conflict statement, involved documents, conflicting values, and likely reason grounded in evidence.'
    )
    labels = parser.extract_required_labels(question)
    assert 'conflict statement' in [label.casefold() for label in labels]
    assert 'involved documents' in [label.casefold() for label in labels]
    assert 'conflicting values' in [label.casefold() for label in labels]


def test_has_year_subsection_cue_does_not_trigger_on_generic_by_year_phrase() -> None:
    assert parser.has_year_subsection_cue('Provide a concise synthesis of trends by year.') is False
