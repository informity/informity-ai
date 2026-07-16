"""Test module for tests test chat completion policy."""

from __future__ import annotations

from informity.api.chat_completion_policy import resolve_completion_and_action
from informity.llm.types import CompletionMode, NextAction


def test_resolve_completion_and_action_defaults_to_complete_no_action() -> None:
    """Test resolve completion and action defaults to complete no action."""
    mode, has_remaining_scope, next_action, reason = resolve_completion_and_action(
        completion_mode_override=None,
        timeout_occurred=False,
        timeout_reason=None,
        has_remaining_scope=False,
        stopped_by_user=False,
        continuation_resolution_reason=None,
        chat_mode="researcher",
        researcher_out_of_scope=False,
        answer_signals_out_of_scope=False,
    )
    assert mode == CompletionMode.COMPLETE
    assert has_remaining_scope is False
    assert next_action == NextAction.NONE
    assert reason is None


def test_resolve_completion_and_action_sets_assistant_switch_for_out_of_scope() -> None:
    """Test resolve completion and action sets assistant switch for out of scope."""
    mode, has_remaining_scope, next_action, reason = resolve_completion_and_action(
        completion_mode_override=None,
        timeout_occurred=False,
        timeout_reason=None,
        has_remaining_scope=False,
        stopped_by_user=False,
        continuation_resolution_reason=None,
        chat_mode="researcher",
        researcher_out_of_scope=True,
        answer_signals_out_of_scope=False,
    )
    assert mode == CompletionMode.COMPLETE
    assert has_remaining_scope is False
    assert next_action == NextAction.ASSISTANT_SWITCH
    assert reason == "out_of_scope"
