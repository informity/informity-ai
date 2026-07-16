"""Test module for tests test rag patterns."""

from informity.db.models import ChatMessage
from informity.llm.query_classifier import QueryClassification
from informity.llm.rag_patterns import (
    evaluate_substantive_evidence,
    extract_explicit_title_reference,
    has_explicit_title_reference,
    has_topic_overlap_with_previous_user,
    has_topic_shift_cue,
    is_plot_or_chapter_request,
    is_summary_style_request,
    should_prefer_title_alignment,
)


def test_is_summary_style_request_allows_coverage_and_focused_intents() -> None:
    """Test is summary style request allows coverage and focused intents."""
    coverage = QueryClassification(intent="coverage")
    focused = QueryClassification(intent="focused")
    metadata = QueryClassification(intent="metadata")
    assert is_summary_style_request("Summarize this document", coverage) is True
    assert is_summary_style_request("Summarize this document", focused) is True
    assert is_summary_style_request("Summarize this document", metadata) is False


def test_is_summary_style_request_detects_document_about_prompt() -> None:
    """Test is summary style request detects document about prompt."""
    focused = QueryClassification(intent="focused")
    assert is_summary_style_request("What is this document about?", focused) is True


def test_is_plot_or_chapter_request_detects_plot_and_chapter() -> None:
    """Test is plot or chapter request detects plot and chapter."""
    assert is_plot_or_chapter_request("What is the plot?") is True
    assert is_plot_or_chapter_request("Summarize chapter 1") is True
    assert is_plot_or_chapter_request("Give me key points") is False


def test_evaluate_substantive_evidence_profiles_structural_only() -> None:
    """Test evaluate substantive evidence profiles structural only."""
    profile = evaluate_substantive_evidence(
        [
            {"block_type": "table"},
            {"block_type": "form"},
        ]
    )
    assert profile["chunk_count"] == 2
    assert profile["structural_count"] == 2
    assert profile["substantive_count"] == 0
    assert profile["substantive_ratio"] == 0.0


def test_has_topic_overlap_with_previous_user_detects_shared_terms() -> None:
    """Test has topic overlap with previous user detects shared terms."""
    history = [
        ChatMessage(chat_id="chat", role="user", content="List characters in The Three Musketeers"),
        ChatMessage(chat_id="chat", role="assistant", content="Done"),
    ]
    assert (
        has_topic_overlap_with_previous_user(
            question="give character description for each character",
            history=history,
        )
        is True
    )


def test_should_prefer_title_alignment_for_compare_prompt() -> None:
    """Test should prefer title alignment for compare prompt."""
    classification = QueryClassification(intent="focused")
    assert (
        should_prefer_title_alignment(
            question="Compare D'Artagnan in The Three Musketeers and Twenty Years After",
            classification=classification,
        )
        is True
    )


def test_should_not_prefer_title_alignment_for_generic_prompt() -> None:
    """Test should not prefer title alignment for generic prompt."""
    classification = QueryClassification(intent="focused")
    assert (
        should_prefer_title_alignment(
            question="What is the weather today?",
            classification=classification,
        )
        is False
    )


def test_should_prefer_title_alignment_when_source_terms_include_title_phrase() -> None:
    """Test should prefer title alignment when source terms include title phrase."""
    classification = QueryClassification(intent="focused", source_terms=["the three musketeers"])
    assert (
        should_prefer_title_alignment(
            question="List all characters in this document",
            classification=classification,
        )
        is True
    )


def test_has_explicit_title_reference_detects_prepositional_title_phrase() -> None:
    """Test has explicit title reference detects prepositional title phrase."""
    assert (
        has_explicit_title_reference("What is the general plot of The Three Musketeers book?")
        is True
    )


def test_has_explicit_title_reference_detects_title_before_document_noun() -> None:
    """Test has explicit title reference detects title before document noun."""
    assert has_explicit_title_reference("What is The Three Musketeers book about?") is True


def test_has_explicit_title_reference_detects_quoted_title_phrase() -> None:
    """Test has explicit title reference detects quoted title phrase."""
    assert has_explicit_title_reference('Summarize "The Three Musketeers" with key themes.') is True


def test_has_explicit_title_reference_ignores_generic_question() -> None:
    """Test has explicit title reference ignores generic question."""
    assert has_explicit_title_reference("What is this file about?") is False


def test_extract_explicit_title_reference_returns_normalized_title() -> None:
    """Test extract explicit title reference returns normalized title."""
    title = extract_explicit_title_reference("What is The Three Musketeers book about?")
    assert title == "The Three Musketeers"


def test_has_topic_shift_cue_excludes_on_another_note_phrase() -> None:
    """Test has topic shift cue excludes on another note phrase."""
    assert has_topic_shift_cue("On another note, summarize this file.") is False
