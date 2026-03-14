from __future__ import annotations

import re

import spacy
from spacy.matcher import Matcher
from spacy.tokens import Doc

_ALNUM_TOKEN_PATTERN = re.compile(r'^[A-Za-z0-9][A-Za-z0-9._-]{1,31}$')
_ORDINAL_ONLY_PATTERN = re.compile(r'^(?:\d+|[ivxlcdm]+|[a-z])$', re.IGNORECASE)
_STOP_SECTION_TOKENS = {'in', 'for', 'then', 'under', 'with', 'using'}
_AGGREGATION_TERMS = {
    'aggregate',
    'aggregated',
    'summary',
    'summaries',
    'total',
}
_EXTRACTION_TASK_TERMS = {
    'create',
    'produce',
    'extract',
    'calculate',
    'sum',
    'total',
    'compare',
    'compile',
    'build',
}
_SUMMARY_VERBS = {'summarize', 'summarise', 'describe'}
_SUMMARY_CONTENT_TOKENS = {'content', 'contain', 'contains', 'summary'}
_PERIOD_COMPARISON_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r'\byear[-\s]*over[-\s]*year\b', re.IGNORECASE),
    re.compile(r'\byoy\b', re.IGNORECASE),
    re.compile(r'\byear[-\s]*to[-\s]*year\b', re.IGNORECASE),
    re.compile(r'\bcross[-\s]*year\b', re.IGNORECASE),
    re.compile(r'\b(?:largest|biggest)\s+(?:increase|decrease)\b', re.IGNORECASE),
    re.compile(r'\bdeltas?\b', re.IGNORECASE),
)


def _build_nlp() -> tuple[spacy.Language, Matcher]:
    nlp = spacy.blank('en')
    matcher = Matcher(nlp.vocab)
    matcher.add('GROUP_BY_YEAR', [[{'LOWER': {'IN': ['by', 'per']}}, {'LOWER': 'year'}]])
    matcher.add('GROUP_BY_CATEGORY', [[{'LOWER': {'IN': ['by', 'per']}}, {'LOWER': 'category'}]])
    matcher.add('GROUP_BY_FILE', [[{'LOWER': {'IN': ['by', 'per']}}, {'LOWER': 'file'}]])
    matcher.add(
        'GROUP_BY_GROUPED',
        [[{'LOWER': {'IN': ['group', 'grouped']}}, {'LOWER': 'by'}, {'LOWER': {'IN': ['year', 'category', 'file']}}]],
    )
    matcher.add('FIELD_HINT', [[{'LOWER': {'IN': ['box', 'line', 'field']}}, {'TEXT': {'REGEX': r'^\d+[A-Za-z]?$'}}]])
    matcher.add('SECTION_ANCHOR', [[{'LOWER': {'IN': ['section', 'part', 'schedule']}}]])
    matcher.add('MENTION_ANCHOR', [[{'LOWER': {'IN': ['mention', 'mentions', 'mentioned', 'mentioning']}}]])
    return nlp, matcher


_NLP, _MATCHER = _build_nlp()


def parse_query(query: str) -> Doc:
    return _NLP(query or '')


def extract_group_by(doc: Doc) -> str | None:
    for match_id, start, end in _MATCHER(doc):
        label = doc.vocab.strings[match_id]
        if label == 'GROUP_BY_YEAR':
            return 'year'
        if label == 'GROUP_BY_CATEGORY':
            return 'category'
        if label == 'GROUP_BY_FILE':
            return 'file'
        if label == 'GROUP_BY_GROUPED' and end - start >= 3:
            value = doc[start + 2].lower_
            if value in {'year', 'category', 'file'}:
                return value
    return None


def extract_field_hint(doc: Doc) -> str | None:
    for match_id, start, end in _MATCHER(doc):
        if doc.vocab.strings[match_id] != 'FIELD_HINT':
            continue
        if end - start < 2:
            continue
        return f"{doc[start].lower_}_{doc[start + 1].text.lower()}"
    return None


def extract_section_hint(doc: Doc) -> str | None:
    for match_id, _start, end in _MATCHER(doc):
        if doc.vocab.strings[match_id] != 'SECTION_ANCHOR':
            continue
        tokens: list[str] = []
        for token in doc[end:end + 6]:
            text = token.text.strip()
            if not text:
                continue
            if token.is_punct:
                break
            if token.lower_ in _STOP_SECTION_TOKENS:
                break
            tokens.append(text)
        candidate = ' '.join(tokens).strip(' .,:;')
        if not candidate:
            continue
        if _ORDINAL_ONLY_PATTERN.fullmatch(candidate):
            continue
        return candidate
    return None


def extract_mention_target(doc: Doc) -> str | None:
    for match_id, _start, end in _MATCHER(doc):
        if doc.vocab.strings[match_id] != 'MENTION_ANCHOR':
            continue
        if end >= len(doc):
            continue
        candidate = doc[end].text.strip().strip('.,:;!?')
        if _ALNUM_TOKEN_PATTERN.fullmatch(candidate):
            return candidate
    return None


def has_aggregation_semantics(doc: Doc) -> bool:
    lowered = doc.text.lower()
    if 'year-by-year' in lowered:
        return True
    if any(token.lower_ in _AGGREGATION_TERMS for token in doc):
        return True
    return any(
        doc[idx].lower_ in {'by', 'per'} and doc[idx + 1].lower_ == 'year'
        for idx in range(len(doc) - 1)
    )


def has_period_comparison_semantics(doc: Doc) -> bool:
    lowered = doc.text.lower()
    return any(pattern.search(lowered) for pattern in _PERIOD_COMPARISON_PATTERNS)


def has_extraction_task(doc: Doc) -> bool:
    return any(token.lower_ in _EXTRACTION_TASK_TERMS for token in doc)


def is_inventory_plus_content_coverage(doc: Doc) -> bool:
    has_file_term = any(token.lower_ in {'file', 'files'} for token in doc)
    has_mention = any(token.lower_.startswith('mention') for token in doc)
    has_what_do = any(
        doc[idx].lower_ == 'what' and doc[idx + 1].lower_ == 'do'
        for idx in range(len(doc) - 1)
    )
    has_say = any(token.lower_ == 'say' for token in doc)
    has_connector = any(token.lower_ in {'and', 'then'} for token in doc)
    return has_file_term and has_mention and has_what_do and has_say and has_connector


def is_filename_summary_query(doc: Doc) -> bool:
    has_summary_verb = any(token.lower_ in _SUMMARY_VERBS for token in doc)
    has_what_does = any(
        doc[idx].lower_ == 'what' and doc[idx + 1].lower_ == 'does'
        for idx in range(len(doc) - 1)
    )
    has_content_term = any(token.lower_ in _SUMMARY_CONTENT_TOKENS for token in doc)
    return has_content_term and (has_summary_verb or has_what_does)
