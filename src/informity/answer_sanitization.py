# ==============================================================================
# Informity AI — Answer Sanitization
# Shared deterministic sanitization for display-channel answer payloads.
# ==============================================================================

import re

DISPLAY_FALLBACK_MESSAGE = (
    'I could not generate a final answer from the model output. Please try rephrasing your question.'
)


def strip_think_blocks(text: str) -> str:
    """
    Strip <think> reasoning blocks from text for display.
    Handles complete and orphaned tags.
    """
    # Intentionally handle the model variant `<<think>>...</think>>` in addition to
    # canonical `<think>...</think>` blocks.
    cleaned = re.sub(r'<<think>>.*?</think>>', '', text, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r'<think>.*?</think>', '', cleaned, flags=re.DOTALL | re.IGNORECASE)

    think_idx = cleaned.lower().find('<think>')
    if think_idx != -1 and cleaned.lower().find('</think>', think_idx) == -1:
        cleaned = cleaned[:think_idx]

    think_idx_double = cleaned.lower().find('<<think>>')
    if think_idx_double != -1 and cleaned.lower().find('</think>>', think_idx_double) == -1:
        cleaned = cleaned[:think_idx_double]

    return cleaned.strip()


def strip_source_artifacts(text: str) -> str:
    # Remove citation/source markers from display text.
    cleaned = re.sub(r'\[source:\s*\d+\]', '', text, flags=re.IGNORECASE)
    cleaned = re.sub(r'\(source\s*\d+\)', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\(\s*source\s*\d+(?:\s*,\s*source\s*\d+)*\s*\)', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\(\s*sources?\s*\d+(?:\s*,\s*\d+)*\s*\)', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'(?im)^\s*sources?\s*:\s*.*$', '', cleaned)
    cleaned = re.sub(r'(?im)^\s*source\s+\d+(?:\s*,\s*source\s+\d+)*\s*$', '', cleaned)
    return cleaned


def _normalize_inline_whitespace_preserve_indentation(text: str) -> str:
    normalized_lines: list[str] = []
    for line in text.splitlines():
        match = re.match(r'^([ \t]*)(.*)$', line)
        if match is None:
            normalized_lines.append(line)
            continue
        leading_ws = match.group(1)
        content = re.sub(r'[ \t]{2,}', ' ', match.group(2))
        normalized_lines.append(f'{leading_ws}{content}')
    return '\n'.join(normalized_lines)


def sanitize_display_answer(text: str) -> str:
    cleaned = strip_think_blocks(text)
    cleaned = strip_source_artifacts(cleaned)
    # Normalize line-break HTML artifacts commonly emitted inside markdown table cells.
    cleaned = re.sub(r'(?i)<br\s*/?>', '; ', cleaned)
    cleaned = _normalize_inline_whitespace_preserve_indentation(cleaned)
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    return cleaned.strip()


def build_display_answer(raw_answer: str, fallback_message: str = DISPLAY_FALLBACK_MESSAGE) -> tuple[str, bool]:
    """
    Build UI-safe answer while preserving canonical raw answer in storage.
    Returns (display_answer, reasoning_only_output).
    """
    cleaned_answer = sanitize_display_answer(raw_answer)
    reasoning_only_output = bool(raw_answer) and not cleaned_answer and (
        '<think>' in raw_answer.lower() or '<<think>>' in raw_answer.lower()
    )
    if reasoning_only_output:
        return fallback_message, True
    return cleaned_answer, False
