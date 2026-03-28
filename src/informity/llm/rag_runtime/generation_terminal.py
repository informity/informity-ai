# ==============================================================================
# Informity AI — Generation Terminal Outcomes
# Shared helpers for generation-skipped metrics and deterministic fallback sources.
# ==============================================================================

from __future__ import annotations

from informity.api.schemas import ChatSourceReference
from informity.llm.types import CompletionMode, RetrievalMode


def build_generation_skipped_metrics_payload(
    *,
    query_type: RetrievalMode,
    timeout_seconds: int,
    retrieval_elapsed_ms: float,
    preflight_projected_seconds: float,
    preflight_ratio: float,
    applied_degradations: list[dict[str, object]],
    fallback_events: list[dict[str, object]],
    has_remaining_scope: bool,
    suggested_completion_mode: CompletionMode = CompletionMode.COMPLETE,
    post_retrieval_projected_seconds: float | None = None,
    post_retrieval_ratio: float | None = None,
    validation_gates: dict[str, bool] | None = None,
    retrieval_relevance_score: float | None = None,
    pre_closeout_quality_check: dict[str, object] | None = None,
    extra_fields: dict[str, object] | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        'query_type': query_type,
        'timeout_seconds': timeout_seconds,
        'retrieval_duration_ms': round(retrieval_elapsed_ms, 1),
        'prompt_duration_ms': 0.0,
        'first_token_latency_ms': None,
        'stream_duration_ms': 0.0,
        'timeout_reason': None,
        'soft_budget_checkpoints_hit': [],
        'suggested_completion_mode': suggested_completion_mode,
        'budget_preflight_projected_seconds': round(preflight_projected_seconds, 1),
        'budget_preflight_ratio': round(preflight_ratio, 3),
        'applied_degradations': applied_degradations,
        'fallback_events': fallback_events,
        'has_remaining_scope': has_remaining_scope,
        'generation_skipped': True,
    }
    if post_retrieval_projected_seconds is not None:
        payload['budget_post_retrieval_projected_seconds'] = round(post_retrieval_projected_seconds, 1)
    if post_retrieval_ratio is not None:
        payload['budget_post_retrieval_ratio'] = round(post_retrieval_ratio, 3)
    if validation_gates is not None:
        payload['validation_gates'] = validation_gates
    if retrieval_relevance_score is not None:
        payload['retrieval_relevance_score'] = round(retrieval_relevance_score, 3)
    if pre_closeout_quality_check is not None:
        payload['pre_closeout_quality_check'] = pre_closeout_quality_check
    if extra_fields:
        payload.update(extra_fields)
    return payload


def build_limited_fallback_sources(
    *,
    chunks: list[dict],
    limit: int,
    truncate_preview_fn: object,
    normalize_relevance_score_fn: object,
) -> list[ChatSourceReference]:
    return [
        ChatSourceReference(
            filename=chunk.get('filename', 'unknown'),
            path=chunk.get('file_path', ''),
            chunk_preview=truncate_preview_fn(str(chunk.get('chunk_text', '') or '')),
            relevance_score=normalize_relevance_score_fn(chunk.get('score', 0.0)),
        )
        for chunk in chunks[:limit]
    ]
