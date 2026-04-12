# ==============================================================================
# Informity AI — Answer Sanitization
# Shared deterministic sanitization for display-channel answer payloads.
# ==============================================================================

import re

from thinkstrip import strip_think

DISPLAY_FALLBACK_MESSAGE = (
    'I could not generate a final answer from the model output. Please try rephrasing your question.'
)
_TABLE_SEPARATOR_PATTERN = re.compile(r'^\s*\|(?:\s*:?-{3,}:?\s*\|)+\s*$')
_ANSWER_LABEL_PATTERN = re.compile(
    r'(?im)(^|\n{2,}[ \t]*)(?:\*\*)?[ \t]*answer[ \t]*(?:\*\*)?[ \t]*:[ \t]*'
)
_ANSWER_LABEL_BOLD_COLON_INSIDE_PATTERN = re.compile(
    r'(?im)(^|\n{2,}[ \t]*)\*\*[ \t]*answer[ \t]*:[ \t]*\*\*[ \t]*'
)
_OUT_OF_CORPUS_SENTENCE_PATTERN = re.compile(
    r'(?is)\bhowever,\s*this information is not (?:contained|present|available)\s+in\s+the\s+provided\s+documents\.?'
)
_OUT_OF_CORPUS_SIGNAL_PATTERN = re.compile(
    r'(?is)\b(?:documents?|context)\b.{0,120}\b(?:do\s+not|does\s+not|cannot|can\'t|not)\b.{0,120}\b(?:contain|include|cover|mention|provide)\b'
)
_IDENTITY_LEAK_PATTERNS = (
    re.compile(r'^\s*(?:my name is|i am|i\'m)\s+qwen\b[^.!?\n]*[.!?]?\s*', re.IGNORECASE),
    re.compile(
        r'^\s*(?:i am|i\'m)\s+(?:an?\s+)?large language model\b[^.!?\n]*(?:alibaba(?:\s+cloud)?)?[^.!?\n]*[.!?]?\s*',
        re.IGNORECASE,
    ),
    re.compile(
        r'^\s*(?:i was|i am|i\'m)\s+(?:created|developed|built)\s+by\s+alibaba(?:\s+cloud)?\b[^.!?\n]*[.!?]?\s*',
        re.IGNORECASE,
    ),
)
_IDENTITY_LEAK_HINT_PATTERN = re.compile(r'(?i)\bqwen\b|\balibaba(?:\s+cloud)?\b')
_IDENTITY_BRAND_LINE = 'I’m Informity AI, your local assistant.'
MAX_WORDS_PATTERN = re.compile(
    r'(?:<=?|at\s+most|max(?:imum)?|less than or equal to)\s*(\d+)\s*words?\b',
    re.IGNORECASE,
)
_WORD_PATTERN = re.compile(r'\S+')


def strip_think_blocks(text: str) -> str:
    """
    Strip <think> reasoning blocks from text for display.
    Handles complete and orphaned tags.
    """
    return strip_think(text).strip()


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


def _trim_truncated_trailing_markdown_table_row(text: str) -> str:
    lines = text.splitlines()
    if not lines:
        return text
    while lines:
        trailing_idx = len(lines) - 1
        while trailing_idx >= 0 and not lines[trailing_idx].strip():
            trailing_idx -= 1
        if trailing_idx < 0:
            return '\n'.join(lines)
        trailing_line = lines[trailing_idx].rstrip()
        if not trailing_line.lstrip().startswith('|') or trailing_line.endswith('|'):
            return '\n'.join(lines)
        has_table_separator = any(
            _TABLE_SEPARATOR_PATTERN.match(line.rstrip()) is not None
            for line in lines[:trailing_idx]
        )
        if not has_table_separator:
            return '\n'.join(lines)
        del lines[trailing_idx]
    return ''


def sanitize_display_answer(text: str) -> str:
    cleaned = strip_think_blocks(text)
    cleaned = strip_source_artifacts(cleaned)
    cleaned = _ANSWER_LABEL_BOLD_COLON_INSIDE_PATTERN.sub(lambda m: m.group(1), cleaned)
    cleaned = _ANSWER_LABEL_PATTERN.sub(lambda m: m.group(1), cleaned)
    if (
        len(_OUT_OF_CORPUS_SIGNAL_PATTERN.findall(cleaned)) >= 1
        and _OUT_OF_CORPUS_SENTENCE_PATTERN.search(cleaned) is not None
    ):
        cleaned = _OUT_OF_CORPUS_SENTENCE_PATTERN.sub('', cleaned)
    # Normalize line-break HTML artifacts commonly emitted inside markdown table cells.
    cleaned = re.sub(r'(?i)<br\s*/?>', '; ', cleaned)
    cleaned = _normalize_inline_whitespace_preserve_indentation(cleaned)
    cleaned = _trim_truncated_trailing_markdown_table_row(cleaned)
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    return cleaned.strip()


def normalize_assistant_identity_claim(text: str) -> str:
    """
    Deterministically replace leaked base-model self-identification at answer start.
    Only rewrites an opening identity sentence; never rewrites general content.
    """
    raw = str(text or '')
    if not raw:
        return raw
    opening_window = raw.lstrip()[:240]
    if _IDENTITY_LEAK_HINT_PATTERN.search(opening_window) is None:
        return raw

    for pattern in _IDENTITY_LEAK_PATTERNS:
        match = pattern.match(raw)
        if match is None:
            continue
        remainder = raw[match.end():].lstrip()
        return f'{_IDENTITY_BRAND_LINE} {remainder}'.strip() if remainder else _IDENTITY_BRAND_LINE
    return raw


def build_display_answer(raw_answer: str, fallback_message: str = DISPLAY_FALLBACK_MESSAGE) -> tuple[str, bool]:
    """
    Build UI-safe answer while preserving canonical raw answer in storage.
    Returns (display_answer, reasoning_only_output).
    """
    normalized_raw_answer = normalize_assistant_identity_claim(raw_answer)
    cleaned_answer = sanitize_display_answer(normalized_raw_answer)
    reasoning_only_output = bool(normalized_raw_answer) and not cleaned_answer and (
        '<think>' in normalized_raw_answer.lower() or '<<think>>' in normalized_raw_answer.lower()
    )
    if reasoning_only_output:
        return fallback_message, True
    return cleaned_answer, False


def extract_requested_max_words(text: str) -> int | None:
    match = MAX_WORDS_PATTERN.search(str(text or ''))
    if match is None:
        return None
    try:
        value = int(match.group(1))
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def count_words(text: str) -> int:
    return len(_WORD_PATTERN.findall(str(text or '')))


def truncate_to_word_limit(text: str, max_words: int) -> tuple[str, bool]:
    raw_text = str(text or '')
    if not raw_text or max_words <= 0:
        return raw_text, False

    matches = list(_WORD_PATTERN.finditer(raw_text))
    if len(matches) <= max_words:
        return raw_text, False

    cutoff = matches[max_words - 1].end()
    truncated = raw_text[:cutoff].rstrip()
    if not truncated:
        return truncated, True

    # Prefer a clean sentence boundary if it is very close to the hard cutoff.
    min_boundary = max(0, len(truncated) - 120)
    boundary = max(
        truncated.rfind('. ', min_boundary),
        truncated.rfind('! ', min_boundary),
        truncated.rfind('? ', min_boundary),
        truncated.rfind('.\n', min_boundary),
        truncated.rfind('!\n', min_boundary),
        truncated.rfind('?\n', min_boundary),
    )
    if boundary >= min_boundary:
        truncated = truncated[: boundary + 1].rstrip()

    return truncated, True
