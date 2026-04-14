# ==============================================================================
# Informity AI — Generation Closeout Runtime
# Post-stream metrics/trace/log/source assembly extracted from RAG handler.
# ==============================================================================

from __future__ import annotations

import re

from informity.api.schemas import ChatSourceReference

try:
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as _SKLEARN_ENGLISH_STOP_WORDS
except Exception:  # pragma: no cover - defensive import fallback
    _SKLEARN_ENGLISH_STOP_WORDS = frozenset()

_SOURCE_TOKEN_MIN_LENGTH = 3
_SOURCE_OVERLAP_MIN_TOKENS = 2
_SOURCE_FALLBACK_LIMIT = 5
_SOURCE_STOPWORDS = {str(token).casefold() for token in _SKLEARN_ENGLISH_STOP_WORDS}


def _tokenize_for_source_overlap(text: str) -> set[str]:
    tokens = {
        token.casefold()
        for token in re.findall(r'[A-Za-z0-9]+', text or '')
        if len(token) >= _SOURCE_TOKEN_MIN_LENGTH
    }
    return {token for token in tokens if token not in _SOURCE_STOPWORDS}


def _source_overlap_score(*, answer_tokens: set[str], chunk_text: str) -> int:
    if not answer_tokens:
        return 0
    chunk_tokens = _tokenize_for_source_overlap(chunk_text)
    if not chunk_tokens:
        return 0
    return len(answer_tokens.intersection(chunk_tokens))


def build_source_references(
    *,
    chunks: list[dict],
    answer_text: str,
    truncate_preview_fn: object,
    normalize_relevance_score_fn: object,
) -> list[ChatSourceReference]:
    answer_tokens = _tokenize_for_source_overlap(answer_text)
    candidate_chunks = chunks

    if answer_tokens:
        filtered_chunks: list[dict] = []
        for chunk in chunks:
            overlap_score = _source_overlap_score(
                answer_tokens=answer_tokens,
                chunk_text=str(chunk.get('chunk_text', '') or ''),
            )
            if overlap_score >= _SOURCE_OVERLAP_MIN_TOKENS:
                filtered_chunks.append(chunk)
        candidate_chunks = filtered_chunks or chunks[:_SOURCE_FALLBACK_LIMIT]

    return [
        ChatSourceReference(
            filename=chunk.get('filename', 'unknown'),
            path=chunk.get('file_path', ''),
            chunk_preview=truncate_preview_fn(str(chunk.get('chunk_text', '') or '')),
            relevance_score=normalize_relevance_score_fn(chunk.get('score', 0.0)),
        )
        for chunk in candidate_chunks
    ]


def record_sources_trace(
    *,
    trace: object | None,
    sources: list[ChatSourceReference],
) -> None:
    if trace is None:
        return
    trace.record('sources', {
        'count': len(sources),
        'sources': [source.model_dump(mode='json') for source in sources],
    })
