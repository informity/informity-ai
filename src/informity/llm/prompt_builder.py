# ==============================================================================
# Informity AI — Prompt Builder (v2)
# Static system prompt + context formatting + token-budget-aware history trim
# ==============================================================================

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

from informity.config import settings
from informity.db.models import ChatMessage
from informity.llm.chat_mode import normalize_chat_mode
from informity.llm.model_adapter import get_effective_context_length
from informity.llm.specializations import compose_prompt
from informity.llm.tokenization import count_tokens

if TYPE_CHECKING:
    from informity.llm.model_adapter import ModelProfile

log = structlog.get_logger(__name__)

# Prompt budgeting constants:
# - reserve tokens for generation so prompt assembly cannot consume the full context
# - keep a tokenizer mismatch buffer because cl100k is an estimate for runtime tokenizers
# - include fixed per-message protocol overhead in budgeting calculations
_GENERATION_RESERVE_TOKENS = 2000
_TOKENIZER_MISMATCH_BUFFER_RATIO = 0.12
_MESSAGE_OVERHEAD_TOKENS = 6


def _coerce_source_rank(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        rank = int(value)
    except (TypeError, ValueError):
        return None
    return rank if rank > 0 else None


def _estimate_message_tokens(*, role: str, content: str) -> int:
    return _MESSAGE_OVERHEAD_TOKENS + count_tokens(role) + count_tokens(content)


def _reorder_context_chunks_for_attention(context_chunks: list[dict]) -> list[dict]:
    if len(context_chunks) <= 2:
        return context_chunks

    ranked_chunks = [
        (
            _coerce_source_rank(chunk.get('source_rank')) or index,
            index,
            chunk,
        )
        for index, chunk in enumerate(context_chunks, start=1)
    ]
    ranked_chunks.sort(key=lambda item: (item[0], item[1]))
    ordered_chunks = [chunk for _, _, chunk in ranked_chunks]
    if len(ordered_chunks) <= 2:
        return ordered_chunks

    highest_rank_chunk = ordered_chunks[0]
    second_highest_rank_chunk = ordered_chunks[1]
    middle_chunks = ordered_chunks[2:]
    return [highest_rank_chunk, *middle_chunks, second_highest_rank_chunk]


def resolve_history_limit(chat_mode: str | None) -> int:
    mode = normalize_chat_mode(chat_mode)
    if mode == 'assistant':
        return max(0, int(settings.chat_history_messages_assistant))
    if mode == 'researcher':
        return max(0, int(settings.chat_history_messages_researcher))
    # Fallback for unresolved modes.
    return max(0, int(settings.chat_history_messages))


def _trim_history_by_token_budget(
    *,
    history: list[ChatMessage],
    system_content: str,
    question: str,
    model_profile: ModelProfile | None,
    chat_mode: str | None,
) -> list[ChatMessage]:
    history_limit = resolve_history_limit(chat_mode)
    if history_limit == 0:
        return []
    capped_history = history[-history_limit:]
    if not capped_history or model_profile is None:
        return capped_history

    context_length = get_effective_context_length(model_profile)
    rag_context_ratio = float(getattr(model_profile, 'rag_context_ratio', 0.75) or 0.75)
    rag_context_ratio = min(max(rag_context_ratio, 0.0), 0.95)

    base_tokens = (
        _estimate_message_tokens(role='system', content=system_content)
        + _estimate_message_tokens(role='user', content=question)
    )
    prompt_budget = max(0, context_length - _GENERATION_RESERVE_TOKENS)
    history_budget_by_window = max(0, prompt_budget - base_tokens)
    history_budget_by_ratio = max(0, int(context_length * (1.0 - rag_context_ratio)))
    raw_history_budget = min(history_budget_by_window, history_budget_by_ratio)
    effective_history_budget = max(0, int(raw_history_budget * (1.0 - _TOKENIZER_MISMATCH_BUFFER_RATIO)))

    selected_reversed: list[ChatMessage] = []
    used_history_tokens = 0
    for message in reversed(capped_history):
        message_tokens = _estimate_message_tokens(role=message.role, content=message.content or '')
        if selected_reversed and used_history_tokens + message_tokens > effective_history_budget:
            break
        if not selected_reversed and message_tokens > effective_history_budget:
            # Keep the most recent turn as a floor; engine-level truncation remains
            # the final backstop if this still overflows.
            selected_reversed.append(message)
            used_history_tokens += message_tokens
            break
        selected_reversed.append(message)
        used_history_tokens += message_tokens

    selected = list(reversed(selected_reversed))
    trimmed_count = len(capped_history) - len(selected)
    if trimmed_count > 0:
        log.warning(
            'history_trimmed_by_token_budget',
            trimmed_count=trimmed_count,
            kept_count=len(selected),
            history_limit=history_limit,
            context_length=context_length,
            rag_context_ratio=rag_context_ratio,
            base_tokens=base_tokens,
            history_budget_tokens=effective_history_budget,
            estimated_history_tokens=used_history_tokens,
        )
    return selected


def build_messages(
    question: str,
    context_chunks: list[dict],
    history: list[ChatMessage] | None = None,
    output_constraints: dict[str, int] | None = None,
    format_requirements: list[str] | None = None,
    model_profile: ModelProfile | None = None,
    system_prompt: str | None = None,
    chat_mode: str | None = None,
    specialization_id: str | None = None,
) -> list[dict[str, str]]:
    # Build messages for LLM. Context chunks formatted with [Source: N] labels
    # for LLM understanding (document boundaries, structure, provenance).
    # Labels are informational only — not for citation in answers.
    # Format context
    ordered_context_chunks = _reorder_context_chunks_for_attention(context_chunks)
    context_parts = []
    for i, chunk in enumerate(ordered_context_chunks, start=1):
        source_rank = _coerce_source_rank(chunk.get('source_rank')) or i
        source_label = f"[Source: {source_rank}] {chunk.get('filename', 'unknown')}"
        if isinstance(chunk.get('year'), int):
            source_label += f", Year: {chunk['year']}"
        category = str(chunk.get('category', '') or '').strip()
        if category:
            source_label += f", Category: {category}"
        start_page = chunk.get('start_page')
        end_page = chunk.get('end_page')
        if start_page and end_page and start_page != end_page:
            source_label += f", Pages {start_page}-{end_page}"
        elif chunk.get('page_number'):
            source_label += f", Page {chunk['page_number']}"
        if chunk.get('section_path'):
            source_label += f", Section: {chunk['section_path']}"
        if chunk.get('block_type'):
            source_label += f", Block: {chunk['block_type']}"
        context_parts.append(f"{source_label}\n{chunk.get('chunk_text', '')}")

    context_text = "\n\n".join(context_parts)

    contract_lines: list[str] = []
    if isinstance(output_constraints, dict):
        max_words = output_constraints.get('max_words')
        if isinstance(max_words, int) and max_words > 0:
            contract_lines.append(f'- Maximum words: {max_words}')
        exact_bullets = output_constraints.get('exact_top_level_bullets')
        if isinstance(exact_bullets, int) and exact_bullets > 0:
            contract_lines.append(f'- Exactly {exact_bullets} top-level bullets when bullets are requested')

    if isinstance(format_requirements, list):
        for requirement in format_requirements:
            text = str(requirement or '').strip()
            if text:
                contract_lines.append(f'- {text}')

    contract_block = ''
    if contract_lines:
        contract_block = '\n\nOutput Contract:\n' + '\n'.join(contract_lines[:24])

    # Build system message
    active_system_prompt = (
        compose_prompt(mode_id='researcher_rag', chat_mode=chat_mode, specialization_id=specialization_id)
        if system_prompt is None
        else str(system_prompt)
    )
    system_content = f"{active_system_prompt}{contract_block}\n\nContext:\n{context_text}"

    # Build messages list
    messages = [{'role': 'system', 'content': system_content}]

    # Add history (count ceiling + token-budget-aware trim when model profile is available)
    if history:
        selected_history = _trim_history_by_token_budget(
            history=history,
            system_content=system_content,
            question=question,
            model_profile=model_profile,
            chat_mode=chat_mode,
        )
        for msg in selected_history:
            history_content = msg.content or ''
            messages.append({'role': msg.role, 'content': history_content})

    # Add current question
    messages.append({'role': 'user', 'content': question})

    return messages
