# ==============================================================================
# Informity AI — Prompt Builder (v2)
# 3-rule system prompt, structured context formatting with source labels
# ==============================================================================

from datetime import UTC, datetime

from informity.answer_sanitization import sanitize_display_answer
from informity.config import settings
from informity.db.models import ChatMessage

_SYSTEM_PROMPT = """You are a document Q&A assistant.

Rules:
1. Answer using ONLY the provided context excerpts.
2. Use markdown formatting when the answer has structure (multiple values, lists, or comparisons). For single-fact answers, use plain text.
3. If the context contains NO relevant information, say "The available documents do not contain enough information to answer this question."
4. Never infer missing facts. If evidence is insufficient for a specific required claim, output "Not found" for that claim.
5. If retrieved chunks contain conflicting values for the same field, report each value with its source document. Do not blend, average, or choose silently.
6. Start directly with answer content. Do not begin with meta-commentary phrases (for example "Based on", "According to", "The documents show", "The context indicates").
7. Do not cite [Source: N] labels in the answer; these labels are metadata only.
8. Do not output HTML tags (for example <br>). In markdown table cells, separate multiple items with "; ".
"""

_RESEARCH_MODE_PROMPT_ADDENDUM = """Research mode instructions:
- Prefer comprehensive, evidence-grounded coverage over brevity.
- Fully satisfy every required heading and subsection before concluding.
- When evidence exists, include concrete figures, dates, and identifiers.
- If evidence is missing, write "Missing evidence for <topic>".
- If no evidence exists for a required section, write "Not found" under that section.
- Do not generate verification actions that are not explicitly supported by context."""


def _current_utc_date_iso() -> str:
    # Canonical date anchor for temporal reasoning in prompts.
    return datetime.now(UTC).strftime('%Y-%m-%d')


def build_messages(
    question: str,
    context_chunks: list[dict],
    history: list[ChatMessage] | None = None,
    output_constraints: dict[str, int] | None = None,
    format_requirements: list[str] | None = None,
    response_mode: str = 'analysis',
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

    # Build system message
    system_content = (
        f"{_SYSTEM_PROMPT}\n\n"
        f"Current date: {_current_utc_date_iso()}\n\n"
        f"Context:\n{context_text}"
    )
    if str(response_mode or 'analysis').strip().lower() == 'research':
        system_content += f"\n\n{_RESEARCH_MODE_PROMPT_ADDENDUM}"
    if output_constraints:
        constraints: list[str] = []
        max_sections = output_constraints.get('max_sections')
        max_rows = output_constraints.get('max_rows')
        max_words = output_constraints.get('max_words')
        if isinstance(max_sections, int) and max_sections > 0:
            constraints.append(f'use at most {max_sections} sections')
        if isinstance(max_rows, int) and max_rows > 0:
            constraints.append(f'use at most {max_rows} table rows or list items')
        if isinstance(max_words, int) and max_words > 0:
            constraints.append(f'keep the answer under about {max_words} words')
        if constraints:
            system_content += '\n\nOutput budget constraints: ' + '; '.join(constraints) + '.'
    if format_requirements:
        normalized_requirements = [item.strip() for item in format_requirements if item and item.strip()]
        if normalized_requirements:
            system_content += '\n\nRequired output format:\n- ' + '\n- '.join(normalized_requirements)

    # Build messages list
    messages = [{'role': 'system', 'content': system_content}]

    # Add history (truncate if needed)
    if history:
        history_limit = settings.chat_history_messages
        for msg in history[-history_limit:]:  # Last N messages (configurable)
            history_content = msg.content or ''
            if msg.role == 'assistant':
                history_content = sanitize_display_answer(history_content)
                if not history_content:
                    continue
            messages.append({'role': msg.role, 'content': history_content})

    # Add current question
    messages.append({'role': 'user', 'content': question})

    return messages
