# ==============================================================================
# Informity AI — Prompt Builder (v2)
# Static system prompt + context formatting + token-budget-aware history trim
# ==============================================================================

"""Prompt assembly helpers for LLM chat and retrieval requests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import structlog

from informity.answer_sanitization import strip_source_artifacts, strip_think_blocks
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
_RUNNING_SUMMARY_MAX_TURNS = 6
_RUNNING_SUMMARY_MAX_CHARS_PER_TURN = 180


@dataclass(frozen=True)
class BuildMessagesRequest:
    """BuildMessagesRequest model."""

    question: str
    context_chunks: list[dict]
    history: list[ChatMessage] | None = None
    output_constraints: dict[str, int] | None = None
    format_requirements: list[str] | None = None
    model_profile: ModelProfile | None = None
    system_prompt: str | None = None
    chat_mode: str | None = None
    specialization_id: str | None = None
    agent_mode: bool = False
    agent_synthesis_focus: str | None = None


def _coerce_source_rank(value: object) -> int | None:
    """coerce source rank."""
    if isinstance(value, bool):
        return None
    try:
        rank = int(value)
    except (TypeError, ValueError):
        return None
    return rank if rank > 0 else None


def _estimate_message_tokens(*, role: str, content: str) -> int:
    """estimate message tokens."""
    return _MESSAGE_OVERHEAD_TOKENS + count_tokens(role) + count_tokens(content)


def _running_summary_enabled() -> bool:
    """Whether running summary injection is enabled."""
    return str(getattr(settings, "diagnostics_profile", "standard")) == "troubleshooting"


def _normalize_summary_text(content: str) -> str:
    """Normalize summary text."""
    cleaned = strip_source_artifacts(strip_think_blocks(content or ""))
    cleaned = " ".join(cleaned.split())
    return cleaned.strip()


def _truncate_summary_text(content: str, max_chars: int) -> str:
    """Truncate summary text."""
    cleaned = _normalize_summary_text(content)
    if len(cleaned) <= max_chars:
        return cleaned
    return f"{cleaned[: max(0, max_chars - 1)].rstrip()}…"


def _build_running_summary_message(omitted_history: list[ChatMessage]) -> ChatMessage | None:
    """Build a compact synthetic summary message for omitted history."""
    if not omitted_history:
        return None
    recent_omitted_history = omitted_history[-_RUNNING_SUMMARY_MAX_TURNS:]
    summary_lines = [
        "Conversation summary of earlier turns:",
    ]
    summarized_turns = 0
    for message in recent_omitted_history:
        summary_text = _truncate_summary_text(
            message.content or "", _RUNNING_SUMMARY_MAX_CHARS_PER_TURN
        )
        if not summary_text:
            continue
        role_label = "User" if message.role == "user" else "Assistant"
        summary_lines.append(f"- {role_label}: {summary_text}")
        summarized_turns += 1
    omitted_count = len(omitted_history) - summarized_turns
    if omitted_count > 0:
        summary_lines.append(f"- {omitted_count} earlier messages omitted.")
    summary_content = "\n".join(summary_lines).strip()
    if not summary_content:
        return None
    return ChatMessage(
        chat_id=omitted_history[0].chat_id,
        role="assistant",
        content=summary_content,
        is_internal=True,
    )


def _reorder_context_chunks_for_attention(context_chunks: list[dict]) -> list[dict]:
    """reorder context chunks for attention."""
    if len(context_chunks) <= 2:
        return context_chunks

    ranked_chunks = [
        (
            _coerce_source_rank(chunk.get("source_rank")) or index,
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
    """resolve history limit."""
    mode = normalize_chat_mode(chat_mode)
    if mode == "assistant":
        return max(0, int(settings.chat_history_messages_assistant))
    if mode == "researcher":
        return max(0, int(settings.chat_history_messages_researcher))
    # Fallback for unresolved modes.
    return max(0, int(settings.chat_history_messages))


def _compute_history_budget(
    *,
    history: list[ChatMessage],
    system_content: str,
    question: str,
    model_profile: ModelProfile,
    chat_mode: str | None,
) -> tuple[list[ChatMessage], int, int, float, int]:
    """compute history budget."""
    history_limit = resolve_history_limit(chat_mode)
    capped_history = history[-history_limit:]
    context_length = get_effective_context_length(model_profile)
    rag_context_ratio = float(getattr(model_profile, "rag_context_ratio", 0.75) or 0.75)
    rag_context_ratio = min(max(rag_context_ratio, 0.0), 0.95)
    base_tokens = _estimate_message_tokens(
        role="system", content=system_content
    ) + _estimate_message_tokens(role="user", content=question)
    prompt_budget = max(0, context_length - _GENERATION_RESERVE_TOKENS)
    history_budget_by_window = max(0, prompt_budget - base_tokens)
    history_budget_by_ratio = max(0, int(context_length * (1.0 - rag_context_ratio)))
    raw_history_budget = min(history_budget_by_window, history_budget_by_ratio)
    effective_history_budget = max(
        0, int(raw_history_budget * (1.0 - _TOKENIZER_MISMATCH_BUFFER_RATIO))
    )
    return capped_history, effective_history_budget, context_length, rag_context_ratio, base_tokens


def _select_history_messages(
    capped_history: list[ChatMessage], effective_history_budget: int
) -> tuple[list[ChatMessage], int]:
    """select history messages."""
    selected_reversed: list[ChatMessage] = []
    used_history_tokens = 0
    for message in reversed(capped_history):
        message_tokens = _estimate_message_tokens(role=message.role, content=message.content or "")
        if selected_reversed and used_history_tokens + message_tokens > effective_history_budget:
            break
        if not selected_reversed and message_tokens > effective_history_budget:
            selected_reversed.append(message)
            used_history_tokens += message_tokens
            break
        selected_reversed.append(message)
        used_history_tokens += message_tokens
    return selected_reversed, used_history_tokens


def _trim_history_by_token_budget(
    *,
    history: list[ChatMessage],
    system_content: str,
    question: str,
    model_profile: ModelProfile | None,
    chat_mode: str | None,
) -> list[ChatMessage]:
    """trim history by token budget."""
    history_limit = resolve_history_limit(chat_mode)
    if history_limit == 0:
        return []
    capped_history = history[-history_limit:]
    if not capped_history:
        return capped_history

    if model_profile is None:
        selected = capped_history
        omitted_count = len(history) - len(selected)
        if omitted_count > 0 and _running_summary_enabled():
            summary_message = _build_running_summary_message(history[:omitted_count])
            if summary_message is not None:
                selected = [summary_message, *selected]
        return selected

    (
        capped_history,
        effective_history_budget,
        context_length,
        rag_context_ratio,
        base_tokens,
    ) = _compute_history_budget(
        history=history,
        system_content=system_content,
        question=question,
        model_profile=model_profile,
        chat_mode=chat_mode,
    )
    selected_reversed, used_history_tokens = _select_history_messages(
        capped_history, effective_history_budget
    )
    selected = list(reversed(selected_reversed))
    omitted_count = len(history) - len(selected)
    if omitted_count > 0 and _running_summary_enabled():
        summary_message = _build_running_summary_message(history[:omitted_count])
        if summary_message is not None:
            selected = [summary_message, *selected]
    if omitted_count > 0:
        log.warning(
            "history_trimmed_by_token_budget",
            trimmed_count=omitted_count,
            kept_count=len(selected),
            history_limit=history_limit,
            context_length=context_length,
            rag_context_ratio=rag_context_ratio,
            base_tokens=base_tokens,
            history_budget_tokens=effective_history_budget,
            estimated_history_tokens=used_history_tokens,
        )
    return selected


def _build_context_text(context_chunks: list[dict]) -> str:
    """build context text."""
    ordered_context_chunks = _reorder_context_chunks_for_attention(context_chunks)
    context_parts: list[str] = []
    for i, chunk in enumerate(ordered_context_chunks, start=1):
        source_rank = _coerce_source_rank(chunk.get("source_rank")) or i
        source_label = f"[Source: {source_rank}] {chunk.get('filename', 'unknown')}"
        if isinstance(chunk.get("year"), int):
            source_label += f", Year: {chunk['year']}"
        category = str(chunk.get("category", "") or "").strip()
        if category:
            source_label += f", Category: {category}"
        start_page = chunk.get("start_page")
        end_page = chunk.get("end_page")
        if start_page and end_page and start_page != end_page:
            source_label += f", Pages {start_page}-{end_page}"
        elif chunk.get("page_number"):
            source_label += f", Page {chunk['page_number']}"
        if chunk.get("section_path"):
            source_label += f", Section: {chunk['section_path']}"
        if chunk.get("block_type"):
            source_label += f", Block: {chunk['block_type']}"
        context_parts.append(f"{source_label}\n{chunk.get('chunk_text', '')}")
    return "\n\n".join(context_parts)


def _build_output_contract_block(
    output_constraints: dict[str, int] | None,
    format_requirements: list[str] | None,
) -> str:
    """build output contract block."""
    contract_lines: list[str] = []
    if isinstance(output_constraints, dict):
        max_words = output_constraints.get("max_words")
        if isinstance(max_words, int) and max_words > 0:
            contract_lines.append(f"- Maximum words: {max_words}")
        exact_bullets = output_constraints.get("exact_top_level_bullets")
        if isinstance(exact_bullets, int) and exact_bullets > 0:
            contract_lines.append(
                f"- Exactly {exact_bullets} top-level bullets when bullets are requested"
            )

    if isinstance(format_requirements, list):
        for requirement in format_requirements:
            text = str(requirement or "").strip()
            if text:
                contract_lines.append(f"- {text}")

    if not contract_lines:
        return ""
    return "\n\nOutput Contract:\n" + "\n".join(contract_lines[:24])


def _build_messages_impl(request: BuildMessagesRequest) -> list[dict[str, str]]:
    # Build messages for LLM. Context chunks formatted with [Source: N] labels
    # for LLM understanding (document boundaries, structure, provenance).
    # Labels are informational only — not for citation in answers.
    # Format context
    """build messages."""
    context_text = _build_context_text(request.context_chunks)
    contract_block = _build_output_contract_block(
        request.output_constraints, request.format_requirements
    )

    # Build system message
    active_system_prompt = (
        compose_prompt(
            mode_id="researcher_agent_rag" if request.agent_mode else "researcher_rag",
            chat_mode=request.chat_mode,
            specialization_id=request.specialization_id,
        )
        if request.system_prompt is None
        else str(request.system_prompt)
    )
    synthesis_focus_block = ""
    if request.agent_synthesis_focus:
        synthesis_focus_block = (
            "\n\nAgent Synthesis Focus:\n"
            f"{request.agent_synthesis_focus.strip()}"
        )
    system_content = (
        f"{active_system_prompt}{contract_block}{synthesis_focus_block}\n\nContext:\n{context_text}"
    )

    # Build messages list
    messages = [{"role": "system", "content": system_content}]

    # Add history (count ceiling + token-budget-aware trim when model profile is available)
    if request.history:
        selected_history = _trim_history_by_token_budget(
            history=request.history,
            system_content=system_content,
            question=request.question,
            model_profile=request.model_profile,
            chat_mode=request.chat_mode,
        )
        for msg in selected_history:
            history_content = msg.content or ""
            messages.append({"role": msg.role, "content": history_content})

    # Add current question
    messages.append({"role": "user", "content": request.question})

    return messages


def build_messages(*args: object, **kwargs: object) -> list[dict[str, str]]:
    """Build messages using either the request object or the legacy call signature."""
    if len(args) == 1 and isinstance(args[0], BuildMessagesRequest) and not kwargs:
        request = args[0]
    else:
        if args and isinstance(args[0], str):
            question = args[0]
            context_chunks = args[1] if len(args) > 1 else kwargs.pop("context_chunks", None)
            history = args[2] if len(args) > 2 else kwargs.pop("history", None)
            if len(args) > 3:
                raise TypeError("build_messages() takes at most 3 positional arguments")
        else:
            question = kwargs.pop("question", None)
            context_chunks = kwargs.pop("context_chunks", None)
            history = kwargs.pop("history", None)

        output_constraints = kwargs.pop("output_constraints", None)
        format_requirements = kwargs.pop("format_requirements", None)
        model_profile = kwargs.pop("model_profile", None)
        system_prompt = kwargs.pop("system_prompt", None)
        chat_mode = kwargs.pop("chat_mode", None)
        specialization_id = kwargs.pop("specialization_id", None)
        agent_mode = kwargs.pop("agent_mode", False)
        agent_synthesis_focus = kwargs.pop("agent_synthesis_focus", None)

        if kwargs:
            unexpected = ", ".join(sorted(str(key) for key in kwargs))
            raise TypeError(f"build_messages() got unexpected keyword argument(s): {unexpected}")
        if question is None or context_chunks is None:
            missing = []
            if question is None:
                missing.append("question")
            if context_chunks is None:
                missing.append("context_chunks")
            raise TypeError(f"build_messages() missing required argument(s): {', '.join(missing)}")

        request = BuildMessagesRequest(
            question=str(question),
            context_chunks=(
                list(context_chunks) if not isinstance(context_chunks, list) else context_chunks
            ),
            history=history,  # legacy API accepts any sequence-like history list
            output_constraints=(
                output_constraints if isinstance(output_constraints, dict) else None
            ),
            format_requirements=(
                format_requirements if isinstance(format_requirements, list) else None
            ),
            model_profile=model_profile,
            system_prompt=str(system_prompt) if system_prompt is not None else None,
            chat_mode=str(chat_mode) if chat_mode is not None else None,
            specialization_id=str(specialization_id) if specialization_id is not None else None,
            agent_mode=bool(agent_mode),
            agent_synthesis_focus=(
                str(agent_synthesis_focus) if agent_synthesis_focus is not None else None
            ),
        )
    return _build_messages_impl(request)
