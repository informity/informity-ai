from informity.api.chat_continuation import (
    enforce_completion_action_consistency,
    is_continuation_request,
)
from informity.llm.types import CompletionMode, NextAction


def test_assistant_switch_forces_terminal_completion_state() -> None:
    completion_mode, has_remaining_scope = enforce_completion_action_consistency(
        completion_mode=CompletionMode.PARTIAL,
        has_remaining_scope=True,
        next_action=NextAction.ASSISTANT_SWITCH,
        next_action_reason='out_of_corpus',
    )
    assert completion_mode == CompletionMode.COMPLETE
    assert has_remaining_scope is False


def test_is_continuation_request_supports_conversational_followups() -> None:
    assert is_continuation_request('what else') is True
    assert is_continuation_request('tell me more') is True
    assert is_continuation_request('show me the rest') is True
    assert is_continuation_request('anything else?') is True


def test_is_continuation_request_supports_filter_update_phrase() -> None:
    assert is_continuation_request('same question but for 2023') is True
