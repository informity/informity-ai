from __future__ import annotations

from collections.abc import Callable

from informity.db.models import ChatMessage


def build_history_aware_retrieval_query(
    *,
    question: str,
    history: list[ChatMessage] | None,
    rag_query_rewrite_enabled: bool,
    allow_intent: bool,
    is_scope_reset: bool,
    has_topic_shift_cue: bool,
    has_referential_followup: bool,
    has_topical_overlap_fn: Callable[[str, list[ChatMessage] | None], bool],
    normalize_text_fn: Callable[[str], str],
    history_limit: int,
    max_chars_per_turn: int,
    max_query_chars: int,
    preferred_previous_user: str | None = None,
) -> tuple[str | None, bool]:
    normalized_question = normalize_text_fn(question)
    if not normalized_question:
        return None, False
    if not rag_query_rewrite_enabled:
        return None, False
    if not history:
        return None, False
    if has_topic_shift_cue:
        return None, False
    if is_scope_reset:
        return None, False
    if not allow_intent:
        return None, False
    has_topical_overlap = has_topical_overlap_fn(normalized_question, history)
    if not has_referential_followup and not has_topical_overlap:
        return None, False
    if history_limit == 0:
        return None, False

    previous_user = normalize_text_fn(preferred_previous_user or '')
    if not previous_user:
        for message in reversed(history[-history_limit:]):
            content = normalize_text_fn(message.content or '')
            if not content:
                continue
            if not previous_user and message.role == 'user':
                previous_user = content
            if previous_user:
                break
    if not previous_user:
        return None, False

    rewritten_query = (
        f"{normalized_question}\n\nFollow-up context:\n"
        f"- Previous user question: {previous_user[:max_chars_per_turn]}"
    )
    return rewritten_query[:max_query_chars], True
