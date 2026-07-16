"""Test module for tests test chat out of scope."""

from __future__ import annotations

from informity.api.chat_out_of_scope import resolve_out_of_scope_next_action
from informity.llm.types import NextAction


def test_resolve_out_of_scope_next_action_switches_for_researcher_signal() -> None:
    """Test resolve out of scope next action switches for researcher signal."""
    action, reason = resolve_out_of_scope_next_action(
        chat_mode="researcher",
        researcher_out_of_scope=True,
        next_action=NextAction.NONE,
        next_action_reason=None,
        answer_signals_out_of_scope=False,
    )
    assert action == NextAction.ASSISTANT_SWITCH
    assert reason == "out_of_scope"


def test_resolve_out_of_scope_next_action_preserves_existing_action() -> None:
    """Test resolve out of scope next action preserves existing action."""
    action, reason = resolve_out_of_scope_next_action(
        chat_mode="researcher",
        researcher_out_of_scope=True,
        next_action=NextAction.REGENERATE,
        next_action_reason="stopped",
        answer_signals_out_of_scope=True,
    )
    assert action == NextAction.REGENERATE
    assert reason == "stopped"
