"""Test module for tests test query classifier."""

from __future__ import annotations

from informity.llm.five_q_classifier import _AGENT_SYSTEM_PROMPT, _SYSTEM_PROMPT
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


def test_rag_prompt_remains_on_the_baseline_classifier_rules() -> None:
    """Test RAG prompt remains on the baseline classifier rules."""
    lowered = " ".join(_SYSTEM_PROMPT.casefold().split())
    assert "subqueries" not in lowered
    assert "agent_mode" not in lowered
    assert "list the mortgage-related files in bullet points." in lowered
    assert "show me all files related to topic a." not in lowered


def test_agent_mode_prompt_adds_subquery_guidance() -> None:
    """Test agent mode prompt adds subquery guidance."""
    lowered = " ".join(_AGENT_SYSTEM_PROMPT.casefold().split())
    assert "subqueries" in lowered
    assert "agent_mode only changes how subqueries are produced" in lowered
    assert "for compare questions, create separate subqueries" in lowered
    assert "keep subqueries empty for simple lookups, narrow targeted questions, or pure inventory requests." in lowered
    assert _AGENT_SYSTEM_PROMPT != _SYSTEM_PROMPT
