# ==============================================================================
# Informity AI — Prompt Builder (v2)
# Static system prompt + context formatting only
# ==============================================================================

from informity.config import settings
from informity.db.models import ChatMessage

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


def build_messages(
    question: str,
    context_chunks: list[dict],
    history: list[ChatMessage] | None = None,
    output_constraints: dict[str, int] | None = None,
    format_requirements: list[str] | None = None,
    response_mode: str = 'analysis',
) -> list[dict[str, str]]:
    _ = (output_constraints, format_requirements, response_mode)
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
    system_content = f"{_SYSTEM_PROMPT}\n\nContext:\n{context_text}"

    # Build messages list
    messages = [{'role': 'system', 'content': system_content}]

    # Add history (truncate if needed)
    if history:
        history_limit = settings.chat_history_messages
        for msg in history[-history_limit:]:  # Last N messages (configurable)
            history_content = msg.content or ''
            messages.append({'role': msg.role, 'content': history_content})

    # Add current question
    messages.append({'role': 'user', 'content': question})

    return messages
