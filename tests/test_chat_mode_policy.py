from __future__ import annotations

from informity.llm.chat_mode import is_assistant_mode, resolve_chat_mode


def test_resolve_chat_mode_defaults_to_researcher_for_invalid() -> None:
    assert resolve_chat_mode(None) == 'researcher'
    assert resolve_chat_mode('') == 'researcher'
    assert resolve_chat_mode('invalid') == 'researcher'


def test_is_assistant_mode_resolves_case_insensitively() -> None:
    assert is_assistant_mode('Assistant') is True
    assert is_assistant_mode('researcher') is False
