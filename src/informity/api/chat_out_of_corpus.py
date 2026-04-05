# ==============================================================================
# Informity AI — Out-of-Corpus Action Policy
# ==============================================================================

from __future__ import annotations

from informity.llm.types import NextAction


def resolve_out_of_corpus_next_action(
    *,
    chat_mode: str,
    researcher_out_of_corpus: bool,
    next_action: NextAction,
    next_action_reason: str | None,
    answer_signals_out_of_corpus: bool,
) -> tuple[NextAction, str | None]:
    """
    Resolve researcher-mode out-of-corpus policy to assistant-switch in one place.
    """
    if chat_mode != 'researcher':
        return next_action, next_action_reason
    if next_action != NextAction.NONE:
        return next_action, next_action_reason
    if researcher_out_of_corpus or answer_signals_out_of_corpus:
        return NextAction.ASSISTANT_SWITCH, 'out_of_corpus'
    return next_action, next_action_reason
