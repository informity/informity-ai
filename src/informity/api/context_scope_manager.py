# ==============================================================================
# Informity AI — Context Scope Manager
# Centralized context-generation key resolution for chat history isolation.
# ==============================================================================

from __future__ import annotations

from informity.db.models import ChatMessage
from informity.llm.rag_patterns import has_topic_shift_cue

INDEXED_CORPUS_SCOPE_KIND = 'indexed_corpus'
INDEXED_CORPUS_GENERATION_PREFIX = f'{INDEXED_CORPUS_SCOPE_KIND}|g:'


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

    topic_shift_reset = has_topic_shift_cue(message_text)
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
        'scope_transition_reset': scope_transition_reset,
        'generation': resolved_generation,
    }
