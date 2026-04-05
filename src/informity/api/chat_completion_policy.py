# ==============================================================================
# Informity AI — Chat Completion Policy
# Centralized final resolution of completion mode/scope/action.
# ==============================================================================

from __future__ import annotations

from informity.api.chat_continuation import (
    enforce_completion_action_consistency,
    resolve_completion_state,
    resolve_next_action,
)
from informity.api.chat_out_of_corpus import resolve_out_of_corpus_next_action
from informity.llm.types import (
    CompletionMode,
    ContinuationResolutionReason,
    NextAction,
    StructuralGapReason,
    TimeoutReason,
)


def resolve_completion_and_action(
    *,
    completion_mode_override: CompletionMode | str | None,
    timeout_occurred: bool,
    timeout_reason: TimeoutReason | str | None,
    has_remaining_scope: bool,
    stopped_by_user: bool,
    continuation_resolution_reason: ContinuationResolutionReason | StructuralGapReason | TimeoutReason | str | None,
    chat_mode: str,
    researcher_out_of_corpus: bool,
    answer_signals_out_of_corpus: bool,
) -> tuple[CompletionMode, bool, NextAction, str | None]:
    completion_mode, resolved_has_remaining_scope = resolve_completion_state(
        completion_mode_override=completion_mode_override,
        timeout_occurred=timeout_occurred,
        timeout_reason=timeout_reason,
        has_remaining_scope=has_remaining_scope,
    )
    next_action, next_action_reason = resolve_next_action(
        stopped_by_user=stopped_by_user,
        timeout_occurred=timeout_occurred,
        has_remaining_scope=resolved_has_remaining_scope,
        continuation_resolution_reason=continuation_resolution_reason,
    )
    next_action, next_action_reason = resolve_out_of_corpus_next_action(
        chat_mode=chat_mode,
        researcher_out_of_corpus=researcher_out_of_corpus,
        next_action=next_action,
        next_action_reason=next_action_reason,
        answer_signals_out_of_corpus=answer_signals_out_of_corpus,
    )
    completion_mode, resolved_has_remaining_scope = enforce_completion_action_consistency(
        completion_mode=completion_mode,
        has_remaining_scope=resolved_has_remaining_scope,
        next_action=next_action,
        next_action_reason=next_action_reason,
    )
    return completion_mode, resolved_has_remaining_scope, next_action, next_action_reason
