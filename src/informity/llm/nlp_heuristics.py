# ==============================================================================
# Informity AI — NLP Heuristics (pure regex/string implementation)
# Lightweight query feature extraction using only re and string operations.
# Spacy dependency removed: all extraction is now regex-based.
# ==============================================================================

from __future__ import annotations

import re

_ALNUM_TOKEN_PATTERN = re.compile(r'^[A-Za-z0-9][A-Za-z0-9._-]{1,31}$')
_ORDINAL_ONLY_PATTERN = re.compile(r'^(?:\d+|[ivxlcdm]+|[a-z])$', re.IGNORECASE)
_STOP_SECTION_TOKENS = {'in', 'for', 'then', 'under', 'with', 'using'}
_PERIOD_COMPARISON_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r'\byear[-\s]*over[-\s]*year\b', re.IGNORECASE),
    re.compile(r'\byoy\b', re.IGNORECASE),
    re.compile(r'\byear[-\s]*to[-\s]*year\b', re.IGNORECASE),
    re.compile(r'\bcross[-\s]*year\b', re.IGNORECASE),
    re.compile(r'\b(?:largest|biggest)\s+(?:increase|decrease)\b', re.IGNORECASE),
    re.compile(r'\bdeltas?\b', re.IGNORECASE),
)

# Regex patterns replacing spacy Matcher patterns
_GROUP_BY_PATTERN = re.compile(
    r'\b(?:group(?:ed)?\s+by\s+(year|category|file)|(by|per)\s+(year|category|file))\b',
    re.IGNORECASE,
)
_FIELD_HINT_PATTERN = re.compile(r'\b(box|line|field)\s+(\d+[A-Za-z]?)\b', re.IGNORECASE)
_SECTION_ANCHOR_PATTERN = re.compile(r'\b(?:section|part|schedule)\s+', re.IGNORECASE)
_MENTION_ANCHOR_PATTERN = re.compile(
    r'\b(?:mention|mentions|mentioned|mentioning)\s+(\S+)', re.IGNORECASE
)

# Pre-compiled patterns for semantic checks
BY_PER_YEAR_PATTERN = re.compile(r'\b(?:by|per)\s+year\b', re.IGNORECASE)
_AGGREGATION_TERM_PATTERN = re.compile(
    r'\b(?:aggregate|aggregated|summary|summaries|total)\b', re.IGNORECASE
)
_EXTRACTION_TASK_PATTERN = re.compile(
    r'\b(?:create|produce|extract|calculate|sum|total|compare|compile|build)\b', re.IGNORECASE
)


class _ParsedQuery:
    """Lightweight parsed query container replacing spacy Doc."""

    __slots__ = ('text',)

    def __init__(self, text: str) -> None:
        self.text = text


def parse_query(query: str) -> _ParsedQuery:
    return _ParsedQuery(query or '')


def extract_group_by(doc: _ParsedQuery) -> str | None:
    for m in _GROUP_BY_PATTERN.finditer(doc.text):
        # Group 1: "group by X" or "grouped by X"
        if m.group(1):
            return m.group(1).lower()
        # Group 3: "by X" or "per X"
        if m.group(3):
            return m.group(3).lower()
    return None


def extract_field_hint(doc: _ParsedQuery) -> str | None:
    m = _FIELD_HINT_PATTERN.search(doc.text)
    if m:
        return f'{m.group(1).lower()}_{m.group(2).lower()}'
    return None


def extract_section_hint(doc: _ParsedQuery) -> str | None:
    for m in _SECTION_ANCHOR_PATTERN.finditer(doc.text):
        remainder = doc.text[m.end():]
        word_tokens: list[str] = []
        for wm in re.finditer(r'\S+', remainder):
            word = wm.group()
            stripped = word.rstrip('.,;:!?')
            if not stripped:
                break
            if stripped.lower() in _STOP_SECTION_TOKENS:
                break
            word_tokens.append(stripped)
            # Word ended with punctuation — include token then stop
            if word != stripped:
                break
            if len(word_tokens) >= 6:
                break
        candidate = ' '.join(word_tokens).strip(' .,:;')
        if not candidate:
            continue
        if _ORDINAL_ONLY_PATTERN.fullmatch(candidate):
            continue
        return candidate
    return None


def extract_mention_target(doc: _ParsedQuery) -> str | None:
    m = _MENTION_ANCHOR_PATTERN.search(doc.text)
    if m:
        candidate = m.group(1).strip().strip('.,:;!?')
        if _ALNUM_TOKEN_PATTERN.fullmatch(candidate):
            return candidate
    return None


def has_aggregation_semantics(doc: _ParsedQuery) -> bool:
    text = doc.text
    if 'year-by-year' in text.lower():
        return True
    if _AGGREGATION_TERM_PATTERN.search(text):
        return True
    return bool(BY_PER_YEAR_PATTERN.search(text))


def has_period_comparison_semantics(doc: _ParsedQuery) -> bool:
    lowered = doc.text.lower()
    return any(pattern.search(lowered) for pattern in _PERIOD_COMPARISON_PATTERNS)


def has_extraction_task(doc: _ParsedQuery) -> bool:
    return bool(_EXTRACTION_TASK_PATTERN.search(doc.text))


def is_inventory_plus_content_coverage(doc: _ParsedQuery) -> bool:
    text = doc.text.lower()
    has_file_term = bool(re.search(r'\bfiles?\b', text))
    has_mention = bool(re.search(r'\bmentions?\b|\bmentioned\b|\bmentioning\b', text))
    has_what_do = bool(re.search(r'\bwhat\s+do\b', text))
    has_say = bool(re.search(r'\bsay\b', text))
    has_connector = bool(re.search(r'\b(?:and|then)\b', text))
    return has_file_term and has_mention and has_what_do and has_say and has_connector


def is_filename_summary_query(doc: _ParsedQuery) -> bool:
    text = doc.text.lower()
    has_summary_verb = bool(re.search(r'\b(?:summarize|summarise|describe)\b', text))
    has_what_does = bool(re.search(r'\bwhat\s+does\b', text))
    has_content_term = bool(re.search(r'\b(?:content|contain|contains|summary)\b', text))
    return has_content_term and (has_summary_verb or has_what_does)
