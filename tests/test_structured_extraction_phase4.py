"""Test module for tests test structured extraction phase4."""

# pylint: disable=line-too-long

from informity.llm.rag_runtime.structured_numeric import (
    _derive_format_requirements,
    _extract_candidate_values,
    _extract_requested_table_columns,
    _parse_numeric_token,
)


def test_parse_numeric_token_ignores_plain_year() -> None:
    """Test parse numeric token ignores plain year."""
    assert _parse_numeric_token("2023") is None


def test_parse_numeric_token_parses_currency() -> None:
    """Test parse numeric token parses currency."""
    parsed = _parse_numeric_token("$12,345.67")
    assert parsed is not None
    value, has_currency_shape = parsed
    assert value == 12345.67
    assert has_currency_shape is True


def test_extract_candidate_values_prefers_near_field_hint() -> None:
    """Test extract candidate values prefers near field hint."""
    text = "Box 1 wages: $4,500.00 for this tax year. Unrelated amount appears later as $15.00."
    candidates = _extract_candidate_values(
        chunk_text=text,
        field_hint="box_1",
        base_score=0.0,
    )
    assert candidates
    assert float(candidates[0]["value"]) == 4500.00


def test_extract_requested_table_columns_from_question() -> None:
    """Test extract requested table columns from question."""
    question = "Output only a markdown table with columns: Line Item, Amount, Source Snippet."
    columns = _extract_requested_table_columns(question)
    assert columns == ["Line Item", "Amount", "Source Snippet"]


def test_derive_format_requirements_extracts_headings_and_depth() -> None:
    """Test derive format requirements extracts headings and depth."""
    question = (
        "Create a brief with these sections in order: 1) Executive Summary, 2) Findings, "
        "3) Action Checklist. Use nested bullets with exactly 3 levels and call out missing evidence."
    )
    requirements = _derive_format_requirements(question)
    assert any("in the requested order" in requirement for requirement in requirements)
    assert any("include heading: Executive Summary" in requirement for requirement in requirements)
    assert any("exactly 3 levels" in requirement for requirement in requirements)
    assert any("missing evidence" in requirement for requirement in requirements)


def test_derive_format_requirements_extracts_year_subsection_contract() -> None:
    """Test derive format requirements extracts year subsection contract."""
    question = (
        "Build a forensic reconciliation report across all indexed records with exact headings: "
        "## Scope, ## Method, ## Findings by Year. "
        "Under ## Findings by Year include one subsection per indexed year."
    )
    requirements = _derive_format_requirements(question)
    assert any("one subsection per year" in requirement for requirement in requirements)
    assert any(
        "at least 2 distinct year subsections" in requirement for requirement in requirements
    )
