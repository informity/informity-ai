"""Test module for tests test chat mode policy."""

from __future__ import annotations

from informity.llm.chat_mode import is_assistant_mode, resolve_chat_mode


def test_resolve_chat_mode_defaults_to_researcher_for_invalid() -> None:
    """Test resolve chat mode defaults to researcher for invalid."""
    assert resolve_chat_mode(None) == "researcher"
    assert resolve_chat_mode("") == "researcher"
    assert resolve_chat_mode("invalid") == "researcher"


def test_is_assistant_mode_resolves_case_insensitively() -> None:
    """Test is assistant mode resolves case insensitively."""
    assert is_assistant_mode("Assistant") is True
    assert is_assistant_mode("researcher") is False
