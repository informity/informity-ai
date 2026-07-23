"""Test module for tests test query classifier."""

from __future__ import annotations

from informity.llm.five_q_classifier import _SYSTEM_PROMPT
from informity.llm.query_classifier import QueryClassification, classify_query


def test_metadata_inventory_query_routes_to_metadata() -> None:
    """Test metadata inventory query routes to metadata."""
    result = classify_query("How many PDF files from 2023 are indexed?")
    assert isinstance(result, QueryClassification)
    assert result.intent == "metadata"
    assert result.route_candidate == "metadata_inventory"
    assert result.year_filter == 2023
    assert result.file_type_filter is None
    assert result.is_metadata_query is True


def test_chat_summary_query_routes_to_simple_chat_history() -> None:
    """Test chat summary query routes to simple chat history."""
    result = classify_query("What have we been chatting about?")
    assert result.intent == "simple"
    assert result.route_candidate == "continuation_or_refinement"
    assert result.needs_chat_history is True


def test_app_help_query_routes_to_simple_disambiguation() -> None:
    """Test app help query routes to simple disambiguation."""
    result = classify_query("What does this app do?")
    assert result.intent == "simple"
    assert result.route_candidate == "clarification_or_disambiguation"


def test_mortgage_lookup_routes_to_focused_fact_lookup() -> None:
    """Test mortgage lookup routes to focused fact lookup."""
    result = classify_query("What is the interest rate on my mortgage?")
    assert result.intent == "focused"
    assert result.route_candidate == "targeted_fact_lookup"


def test_comparison_query_routes_to_coverage() -> None:
    """Test comparison query routes to coverage."""
    result = classify_query("Compare the 2023 and 2025 closing documents.")
    assert result.intent == "coverage"
    assert result.route_candidate == "comparative_analysis"
    assert result.subtype == "aggregate_by_period"


def test_filename_filter_is_extracted() -> None:
    """Test filename filter is extracted."""
    result = classify_query("Summarize content in sample-lender-statement.pdf")
    assert result.filename_filter == "sample-lender-statement.pdf"


def test_continuation_signal_is_detected() -> None:
    """Test continuation signal is detected."""
    result = classify_query("Show me the rest")
    assert result.is_continuation is True


def test_agent_mode_subquery_instruction_is_conservative() -> None:
    """Test agent mode subquery instruction is conservative."""
    lowered = " ".join(_SYSTEM_PROMPT.casefold().split())
    assert "include subqueries only if the query clearly benefits from" in lowered
    assert "single coherent retrieval pass would likely miss important evidence" in lowered
    assert "keep subqueries empty for ordinary single-topic questions" in lowered
    assert "inventory" in lowered
    assert "style wording" in lowered
    assert "when agent_mode is true and operation=compare" in lowered
    assert "do not collapse the work into one broad subquery" in lowered
