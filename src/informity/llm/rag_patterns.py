"""
Informity AI — RAG Pattern Utilities

Deterministic pattern and helper utilities used by the RAG runtime.
"""

import re

from informity.db.models import ChatMessage
from informity.llm.query_classifier import QueryClassification
from informity.llm.types import QueryType

SUMMARY_STYLE_REQUEST_PATTERN = re.compile(
    r'\b(summar(?:y|ize|ized)|overview|key\s+points?|main\s+points?|chapter|plot|story)\b',
    re.IGNORECASE,
)
PLOT_CHAPTER_REQUEST_PATTERN = re.compile(r'\b(plot|chapter)\b', re.IGNORECASE)
ANAPHORIC_SCOPE_PATTERN = re.compile(
    r'\b(this|that|it|this\s+(?:book|document|file)|that\s+(?:book|document|file))\b',
    re.IGNORECASE,
)
EXPLICIT_SCOPE_RESET_PATTERN = re.compile(
    r'\b('
    r'all\s+(?:documents?|files?|records?)'
    r'|across\s+all'
    r'|another\s+(?:document|file|book)'
    r'|different\s+(?:document|file|book)'
    r'|other\s+(?:document|file|book)'
    r'|[a-z0-9][a-z0-9._-]*\.(?:pdf|txt|md|csv|json|docx?|xlsx?)'
    r')\b',
    re.IGNORECASE,
)
STRUCTURAL_BLOCK_TYPES = {'table', 'form'}
SUMMARY_BLOCK_TYPE_EXCLUDE = ['table', 'form']


def is_summary_style_request(
    question: str,
    classification: QueryClassification,
) -> bool:
    if classification.intent not in {QueryType.COVERAGE, QueryType.FOCUSED}:
        return False
    return bool(SUMMARY_STYLE_REQUEST_PATTERN.search(str(question or '')))


def is_plot_or_chapter_request(question: str) -> bool:
    return bool(PLOT_CHAPTER_REQUEST_PATTERN.search(str(question or '')))


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
