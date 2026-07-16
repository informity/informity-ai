"""Test module for tests test vectors fts query."""

from __future__ import annotations

from informity.db.vectors import _sanitize_fts5_query


def test_sanitize_fts5_query_removes_match_operators_and_punctuation() -> None:
    """Test sanitize fts5 query removes match operators and punctuation."""
    query = 'What are the achievements of Glenn Perez? AND "Form-1099"'
    sanitized = _sanitize_fts5_query(query)
    assert sanitized == "What are the achievements of Glenn Perez Form 1099"


def test_sanitize_fts5_query_returns_empty_for_non_word_input() -> None:
    """Test sanitize fts5 query returns empty for non word input."""
    assert _sanitize_fts5_query("??? --- (( ))") == ""
