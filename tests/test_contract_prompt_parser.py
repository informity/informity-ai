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
