"""Generation-closeout helpers for chat source assembly and tracing."""

from __future__ import annotations

import re

import structlog

from informity.api.schemas import ChatSourceReference
from informity.llm.rag_runtime.citation_verification import (
    assess_answer_support,
    filter_verified_sources,
)

try:
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as _SKLEARN_ENGLISH_STOP_WORDS
except ImportError:  # pragma: no cover - defensive import fallback
    _SKLEARN_ENGLISH_STOP_WORDS = frozenset()

log = structlog.get_logger(__name__)
_SOURCE_TOKEN_MIN_LENGTH = 3
_SOURCE_OVERLAP_MIN_TOKENS = 2
_SOURCE_FALLBACK_LIMIT = 5
_SOURCE_STOPWORDS = {str(token).casefold() for token in _SKLEARN_ENGLISH_STOP_WORDS}


def _tokenize_for_source_overlap(text: str) -> set[str]:
    """Tokenize text for overlap matching against candidate sources."""
    tokens = {
        token.casefold()
        for token in re.findall(r"[A-Za-z0-9]+", text or "")
        if len(token) >= _SOURCE_TOKEN_MIN_LENGTH
    }
    return {token for token in tokens if token not in _SOURCE_STOPWORDS}


def _source_overlap_score(*, answer_tokens: set[str], chunk_text: str) -> int:
    """Score how much a source chunk overlaps the generated answer."""
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
    """Build source references for the final chat response."""
    answer_tokens = _tokenize_for_source_overlap(answer_text)
    candidate_chunks = chunks

    if answer_tokens:
        filtered_chunks: list[dict] = []
        for chunk in chunks:
            overlap_score = _source_overlap_score(
                answer_tokens=answer_tokens,
                chunk_text=str(chunk.get("chunk_text", "") or ""),
            )
            if overlap_score >= _SOURCE_OVERLAP_MIN_TOKENS:
                filtered_chunks.append(chunk)
        candidate_chunks = filtered_chunks or chunks[:_SOURCE_FALLBACK_LIMIT]

    sources = [
        ChatSourceReference(
            filename=chunk.get("filename", "unknown"),
            path=chunk.get("file_path", ""),
            chunk_preview=truncate_preview_fn(str(chunk.get("chunk_text", "") or "")),
            relevance_score=normalize_relevance_score_fn(chunk.get("score", 0.0)),
            file_id=int(chunk["file_id"]) if chunk.get("file_id") is not None else None,
        )
        for chunk in candidate_chunks
    ]
    support_result = assess_answer_support(
        answer_text=answer_text,
        source_texts=[source.chunk_preview or "" for source in sources],
    )
    if support_result.should_fail_closed:
        log.warning(
            "generation_closeout_fail_closed",
            evaluated_claim_count=support_result.evaluated_claim_count,
            supported_claim_count=support_result.supported_claim_count,
            unsupported_claim_count=support_result.unsupported_claim_count,
            evidence_coverage_rate=support_result.evidence_coverage_rate,
            verified_source_count=support_result.verified_source_count,
        )
        return []

    verified_sources = filter_verified_sources(sources, answer_text=answer_text)
    # If the support assessor says the answer is thin, fail closed by hiding
    # sources entirely instead of surfacing unverified references.
    return verified_sources or sources


def record_sources_trace(
    *,
    trace: object | None,
    sources: list[ChatSourceReference],
) -> None:
    """Record source references in the active trace object."""
    if trace is None:
        return
    trace.record(
        "sources",
        {
            "count": len(sources),
            "sources": [source.model_dump(mode="json") for source in sources],
        },
    )
