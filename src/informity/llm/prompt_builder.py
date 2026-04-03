# ==============================================================================
# Informity AI — Prompt Builder (v2)
# Static system prompt + context formatting + token-budget-aware history trim
# ==============================================================================

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

import structlog
import tiktoken

from informity.config import settings
from informity.db.models import ChatMessage

if TYPE_CHECKING:
    from informity.llm.model_adapter import ModelProfile

log = structlog.get_logger(__name__)

_SYSTEM_PROMPT = """You are a research assistant answering questions from a private document corpus.

Rules:
1. Answer using ONLY the provided documents. Never infer, speculate, or use outside knowledge.
2. If the documents do not contain enough information to answer, say so directly.
3. Follow the user's requested output format exactly when specified (for example: "output only a markdown table", exact column names, exact section headings, exact bullet format).
4. Preserve required labels/terms from the user request verbatim when they are part of the requested output schema (for example: source, snippet, objective, tradeoff, decision).
5. For delimiter schemas like "A | B | C", include an exact header/template line with those labels before listing values.
6. Use markdown: headers for multi-topic answers, tables for comparisons, bullet lists for enumerations.
7. If values conflict across documents, report each value with its source document.
8. Start with the answer directly. Do not start with meta-commentary.
"""

_GENERATION_RESERVE_TOKENS = 2000
_TOKENIZER_MISMATCH_BUFFER_RATIO = 0.12
_MESSAGE_OVERHEAD_TOKENS = 6


@lru_cache(maxsize=1)
def _encoding() -> tiktoken.Encoding:
    return tiktoken.get_encoding('cl100k_base')


def _count_tokens(text: str) -> int:
    value = str(text or '')
    if not value:
        return 0
    try:
        return len(_encoding().encode(value))
    except Exception:  # noqa: BLE001 - fallback must remain non-fatal
        return max(1, len(value) // 4)


def _estimate_message_tokens(*, role: str, content: str) -> int:
    return _MESSAGE_OVERHEAD_TOKENS + _count_tokens(role) + _count_tokens(content)


def resolve_history_limit(chat_mode: str | None) -> int:
    mode = str(chat_mode or '').strip().lower()
    if mode == 'assistant':
        return max(0, int(settings.chat_history_messages_assistant))
    if mode == 'researcher':
        return max(0, int(settings.chat_history_messages_researcher))
    # Keep unresolved/legacy modes backward compatible.
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

    context_length = int(getattr(model_profile, 'context_length', settings.llm_context_length) or settings.llm_context_length)
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
) -> list[dict[str, str]]:
    # Build messages for LLM. Context chunks formatted with [Source: N] labels
    # for LLM understanding (document boundaries, structure, provenance).
    # Labels are informational only — not for citation in answers.
    # Format context
    context_parts = []
    for i, chunk in enumerate(context_chunks, start=1):
        source_label = f"[Source: {i}] {chunk.get('filename', 'unknown')}"
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
    active_system_prompt = _SYSTEM_PROMPT if system_prompt is None else str(system_prompt)
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
