from informity.api.chat_continuation import enforce_completion_action_consistency
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

