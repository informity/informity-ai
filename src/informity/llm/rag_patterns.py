"""
Informity AI — RAG Pattern Utilities

Deterministic pattern and helper utilities used by the RAG runtime.
"""

import re

from informity.db.models import ChatMessage
from informity.llm.query_classifier import QueryClassification
from informity.llm.query_patterns import build_referential_followup_pattern
from informity.llm.types import QueryType

SUMMARY_STYLE_REQUEST_PATTERN = re.compile(
    r'\b('
    r'summar(?:y|ize|ized)|overview|key\s+points?|main\s+points?|chapter|plot|story'
    r'|what\s+(?:is|does)\s+(?:this|the)\s+'
    r'(?:document|file|text|record|entry|item|source|material|attachment|note|paper)\s+'
    r'(?:about|cover)'
    r')\b',
    re.IGNORECASE,
)
PLOT_CHAPTER_REQUEST_PATTERN = re.compile(r'\b(plot|chapter)\b', re.IGNORECASE)
ANAPHORIC_SCOPE_PATTERN = re.compile(
    r'\b(this|that|it|'
    r'this\s+(?:document|file|text|record|entry|item|source|material|attachment|note|paper)|'
    r'that\s+(?:document|file|text|record|entry|item|source|material|attachment|note|paper)'
    r')\b',
    re.IGNORECASE,
)
EXPLICIT_SCOPE_RESET_PATTERN = re.compile(
    r'\b('
    r'all\s+(?:documents?|files?|records?)'
    r'|across\s+all'
    r'|another\s+(?:document|file|text|record|entry|item|source|material|attachment|note|paper)'
    r'|different\s+(?:document|file|text|record|entry|item|source|material|attachment|note|paper)'
    r'|other\s+(?:document|file|text|record|entry|item|source|material|attachment|note|paper)'
    r'|[a-z0-9][a-z0-9._-]*\.(?:pdf|txt|md|csv|json|docx?|xlsx?)'
    r')\b',
    re.IGNORECASE,
)
TITLE_ALIGNMENT_CUE_PATTERN = re.compile(
    r'\b(compare|between|versus|vs)\b'
    r'|'
    r'\b(?:in|from)\s+.{0,120}\b(document|file|text|record|entry|item|source|material|attachment|note|paper)\b',
    re.IGNORECASE,
)
TOPIC_SHIFT_CUE_PATTERN = re.compile(
    r'\b('
    r'new\s+topic'
    r'|change\s+(?:the\s+)?topic'
    r'|switch\s+(?:topics?|context)'
    r'|different\s+topic'
    r'|instead'
    r'|unrelated'
    r'|now\s+(?:about|switch(?:ing)?)'
    r')\b',
    re.IGNORECASE,
)
_TITLE_IN_PREPOSITION_PATTERN = re.compile(
    r'\b(?:of|in|about|from)\s+'
    r'((?:[A-Z][A-Za-z0-9\'_-]*)(?:\s+[A-Z][A-Za-z0-9\'_-]*){1,8})'
    r'(?:\s+(?:file|document|text|record|entry|item|source|material|attachment|note|paper))?\b'
)
_QUOTED_TITLE_PATTERN = re.compile(r'["“](.{3,120}?)[”"]')
STRUCTURAL_BLOCK_TYPES = {'table', 'form'}
SUMMARY_BLOCK_TYPE_EXCLUDE = ['table', 'form']
REFERENTIAL_FOLLOWUP_PATTERN = build_referential_followup_pattern()
EXTRACTION_CUE_PATTERN = re.compile(r'\bextract\b', re.IGNORECASE)
REWRITE_STOPWORDS = {
    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'do', 'for', 'from', 'give', 'has', 'have',
    'how', 'i', 'in', 'is', 'it', 'its', 'list', 'me', 'of', 'on', 'or', 'our', 'please', 'show',
    'that', 'the', 'their', 'them', 'there', 'these', 'they', 'this', 'to', 'us', 'what', 'when',
    'where', 'which', 'who', 'why', 'with', 'you', 'your',
}


def is_summary_style_request(
    question: str,
    classification: QueryClassification,
) -> bool:
    if classification.intent not in {QueryType.COVERAGE, QueryType.FOCUSED}:
        return False
    return bool(SUMMARY_STYLE_REQUEST_PATTERN.search(str(question or '')))


def is_plot_or_chapter_request(question: str) -> bool:
    return bool(PLOT_CHAPTER_REQUEST_PATTERN.search(str(question or '')))


def has_extraction_cue(question: str) -> bool:
    return bool(EXTRACTION_CUE_PATTERN.search(str(question or '')))


def normalize_query_text(text: str) -> str:
    return ' '.join(str(text or '').strip().split())


def has_referential_followup_language(question: str) -> bool:
    normalized = normalize_query_text(question).lower()
    if not normalized:
        return False
    return bool(REFERENTIAL_FOLLOWUP_PATTERN.search(normalized))


def has_topic_shift_cue(question: str) -> bool:
    normalized = normalize_query_text(question)
    if not normalized:
        return False
    return bool(TOPIC_SHIFT_CUE_PATTERN.search(normalized))


def has_explicit_title_reference(question: str) -> bool:
    text = str(question or '').strip()
    if not text:
        return False
    if _QUOTED_TITLE_PATTERN.search(text):
        return True
    return bool(_TITLE_IN_PREPOSITION_PATTERN.search(text))


def tokenize_query_terms(text: str) -> set[str]:
    lowered = normalize_query_text(text).lower()
    if not lowered:
        return set()
    raw_terms = set(re.findall(r"[a-z0-9][a-z0-9'_-]{2,}", lowered))
    terms: set[str] = set()
    for term in raw_terms:
        if term in REWRITE_STOPWORDS:
            continue
        terms.add(term)
        if term.endswith('s') and len(term) > 4:
            terms.add(term[:-1])
    return terms


def has_topic_overlap_with_previous_user(
    *,
    question: str,
    history: list[ChatMessage],
) -> bool:
    current_terms = tokenize_query_terms(question)
    if not current_terms:
        return False
    for message in reversed(history):
        if message.role != 'user':
            continue
        previous_terms = tokenize_query_terms(message.content or '')
        if not previous_terms:
            continue
        return bool(current_terms & previous_terms)
    return False


def should_prefer_title_alignment(
    *,
    question: str,
    classification: QueryClassification,
) -> bool:
    if classification.intent not in {QueryType.FOCUSED, QueryType.COVERAGE}:
        return False
    if has_explicit_title_reference(question):
        return True
    if TITLE_ALIGNMENT_CUE_PATTERN.search(str(question or '')):
        return True
    source_terms = [str(term or '').strip() for term in (classification.source_terms or [])]
    return any(len(term) >= 6 and ' ' in term for term in source_terms)


def evaluate_substantive_evidence(chunks: list[dict]) -> dict[str, float | int]:
    if not chunks:
        return {
            'chunk_count': 0,
            'narrative_count': 0,
            'structural_count': 0,
            'substantive_count': 0,
            'substantive_ratio': 0.0,
        }
    narrative_count = 0
    structural_count = 0
    substantive_count = 0
    for chunk in chunks:
        block_type = str(chunk.get('block_type') or '').strip().casefold()
        if block_type == 'narrative':
            narrative_count += 1
            substantive_count += 1
            continue
        if block_type in STRUCTURAL_BLOCK_TYPES:
            structural_count += 1
            continue
        # Unknown/missing block_type is treated as potentially substantive.
        substantive_count += 1
    chunk_count = len(chunks)
    substantive_ratio = substantive_count / max(1, chunk_count)
    return {
        'chunk_count': chunk_count,
        'narrative_count': narrative_count,
        'structural_count': structural_count,
        'substantive_count': substantive_count,
        'substantive_ratio': substantive_ratio,
    }


def should_block_summary_generation_for_structural_only_evidence(
    *,
    question: str,
    classification: QueryClassification,
    evidence_profile: dict[str, float | int],
) -> bool:
    if not is_summary_style_request(question, classification):
        return False
    # Plot/chapter mismatch is handled by reframing guidance later; do not
    # force a refusal gate for those prompts.
    if is_plot_or_chapter_request(question):
        return False
    chunk_count = int(evidence_profile.get('chunk_count') or 0)
    structural_count = int(evidence_profile.get('structural_count') or 0)
    substantive_count = int(evidence_profile.get('substantive_count') or 0)
    if chunk_count == 0:
        return False
    return structural_count > 0 and substantive_count == 0


def resolve_followup_scope_anchor_filename(
    *,
    question: str,
    history: list[ChatMessage] | None,
    classification: QueryClassification,
) -> str | None:
    if classification.filename_filter:
        return None
    if classification.intent != QueryType.FOCUSED:
        return None
    lowered_question = str(question or '').casefold()
    if not ANAPHORIC_SCOPE_PATTERN.search(lowered_question):
        return None
    if EXPLICIT_SCOPE_RESET_PATTERN.search(lowered_question):
        return None
    if not history:
        return None

    for message in reversed(history):
        if message.role != 'assistant':
            continue
        sources = list(message.sources or [])
        if not sources:
            continue
        filenames: list[str] = []
        for source in sources:
            filename = str((source or {}).get('filename') or '').strip()
            if filename:
                filenames.append(filename)
        unique_filenames = sorted(set(filenames))
        if len(unique_filenames) == 1:
            return unique_filenames[0]
    return None
