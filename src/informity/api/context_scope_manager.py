# ==============================================================================
# Informity AI — Context Scope Manager
# Centralized context-generation key resolution for chat history isolation.
# ==============================================================================

from __future__ import annotations

from informity.db.models import ChatMessage
from informity.llm.rag_patterns import (
    has_explicit_title_reference,
    has_referential_followup_language,
    has_topic_overlap_with_previous_user,
    has_topic_shift_cue,
)

INDEXED_CORPUS_SCOPE_KIND = 'indexed_corpus'
INDEXED_CORPUS_GENERATION_PREFIX = f'{INDEXED_CORPUS_SCOPE_KIND}|g:'
_TOPIC_SHIFT_THRESHOLD = 0.45


def _extract_indexed_generation(scope_key: str | None) -> int | None:
    normalized = str(scope_key or '').strip()
    if not normalized:
        return None
    if normalized == INDEXED_CORPUS_SCOPE_KIND:
        return 0
    if not normalized.startswith(INDEXED_CORPUS_GENERATION_PREFIX):
        return None
    suffix = normalized[len(INDEXED_CORPUS_GENERATION_PREFIX):].strip()
    try:
        generation = int(suffix)
    except (TypeError, ValueError):
        return None
    if generation < 0:
        return 0
    return generation


def normalize_indexed_corpus_scope_key(scope_key: str | None) -> str:
    generation = _extract_indexed_generation(scope_key)
    if generation is None:
        return str(scope_key or '').strip()
    return f'{INDEXED_CORPUS_GENERATION_PREFIX}{generation}'


def indexed_corpus_scope_key_for_generation(generation: int) -> str:
    safe_generation = max(0, int(generation))
    return f'{INDEXED_CORPUS_GENERATION_PREFIX}{safe_generation}'


def _max_indexed_generation(history: list[ChatMessage], *, chat_mode: str) -> int:
    max_generation = 0
    for message in history:
        message_chat_mode = str(message.chat_mode or '').strip()
        if message_chat_mode and message_chat_mode != chat_mode:
            continue
        message_scope_kind = str(message.retrieval_scope_kind or '').strip()
        if message_scope_kind and message_scope_kind != INDEXED_CORPUS_SCOPE_KIND:
            continue
        generation = _extract_indexed_generation(message.retrieval_scope_key)
        if generation is None:
            # Legacy indexed-corpus rows (including pre-scope) are generation 0.
            generation = 0
        if generation > max_generation:
            max_generation = generation
    return max_generation


def _evaluate_topic_shift_signal(
    *,
    message_text: str,
    history: list[ChatMessage],
) -> tuple[bool, float, list[str]]:
    normalized = str(message_text or '').strip()
    if not normalized:
        return False, 0.0, ['empty_message']
    if not history:
        return False, 0.0, ['no_history']

    score = 0.0
    reasons: list[str] = []

    explicit_shift_cue = has_topic_shift_cue(normalized)
    has_referential_language = has_referential_followup_language(normalized)
    if explicit_shift_cue and not has_referential_language:
        return True, 1.0, ['explicit_shift_cue_override']

    if explicit_shift_cue:
        score += 0.45
        reasons.append('explicit_shift_cue')
    if has_referential_language:
        score -= 0.35
        reasons.append('referential_followup_language')

    overlap = has_topic_overlap_with_previous_user(question=normalized, history=history)
    if overlap:
        score -= 0.25
        reasons.append('topic_overlap_with_previous_user')
    else:
        score += 0.25
        reasons.append('no_topic_overlap_with_previous_user')

    if has_explicit_title_reference(normalized) and not overlap:
        score += 0.2
        reasons.append('explicit_title_reference_with_no_overlap')

    bounded_score = max(0.0, min(1.0, score))
    return bounded_score >= _TOPIC_SHIFT_THRESHOLD, bounded_score, reasons


def resolve_retrieval_context_scope_key(
    *,
    chat_mode: str,
    retrieval_scope_kind: str,
    retrieval_scope_key: str,
    message_text: str,
    history: list[ChatMessage],
) -> tuple[str, dict[str, object]]:
    normalized_scope_kind = str(retrieval_scope_kind or '').strip()
    normalized_scope_key = str(retrieval_scope_key or '').strip()
    if chat_mode != 'researcher' or normalized_scope_kind != INDEXED_CORPUS_SCOPE_KIND:
        return normalized_scope_key, {
            'topic_shift_reset': False,
            'scope_transition_reset': False,
            'generation': None,
        }

    latest_researcher_message: ChatMessage | None = None
    for message in reversed(history):
        message_chat_mode = str(message.chat_mode or '').strip()
        if message_chat_mode and message_chat_mode != chat_mode:
            continue
        latest_researcher_message = message
        break

    scope_transition_reset = False
    latest_generation: int | None = None
    if latest_researcher_message is not None:
        latest_scope_kind = str(latest_researcher_message.retrieval_scope_kind or '').strip()
        latest_scope_key = str(latest_researcher_message.retrieval_scope_key or '').strip()
        latest_generation = _extract_indexed_generation(latest_scope_key)
        if latest_scope_kind and latest_scope_kind != INDEXED_CORPUS_SCOPE_KIND:
            scope_transition_reset = True

    topic_shift_reset, topic_shift_score, topic_shift_reasons = _evaluate_topic_shift_signal(
        message_text=message_text,
        history=history,
    )
    max_generation = _max_indexed_generation(history, chat_mode=chat_mode)

    if topic_shift_reset or scope_transition_reset:
        resolved_generation = max_generation + 1
    elif latest_generation is not None:
        resolved_generation = latest_generation
    else:
        resolved_generation = max_generation

    resolved_key = indexed_corpus_scope_key_for_generation(resolved_generation)
    return resolved_key, {
        'topic_shift_reset': topic_shift_reset,
        'topic_shift_score': round(topic_shift_score, 4),
        'topic_shift_reasons': topic_shift_reasons,
        'scope_transition_reset': scope_transition_reset,
        'generation': resolved_generation,
    }
