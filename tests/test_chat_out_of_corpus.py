from __future__ import annotations

from informity.api.chat_out_of_corpus import resolve_out_of_corpus_next_action
from informity.llm.types import NextAction


def test_resolve_out_of_corpus_next_action_switches_for_researcher_signal() -> None:
    action, reason = resolve_out_of_corpus_next_action(
        chat_mode='researcher',
        researcher_out_of_corpus=True,
        next_action=NextAction.NONE,
        next_action_reason=None,
        answer_signals_out_of_corpus=False,
    )
    assert action == NextAction.ASSISTANT_SWITCH
    assert reason == 'out_of_corpus'


def test_resolve_out_of_corpus_next_action_preserves_existing_action() -> None:
    action, reason = resolve_out_of_corpus_next_action(
        chat_mode='researcher',
        researcher_out_of_corpus=True,
        next_action=NextAction.REGENERATE,
        next_action_reason='stopped',
        answer_signals_out_of_corpus=True,
    )
    assert action == NextAction.REGENERATE
    assert reason == 'stopped'
