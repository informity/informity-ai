# ==============================================================================
# Informity AI — Generation Closeout Runtime
# Post-stream metrics/trace/log/source assembly extracted from RAG handler.
# ==============================================================================

from __future__ import annotations

from informity.api.schemas import ChatSourceReference


def build_generation_metrics_payload(
    *,
    query_type: str,
    response_mode_used: str,
    mode_adjustments_applied: list[str],
    timeout_seconds: int,
    retrieval_elapsed_ms: float,
    prompt_elapsed_ms: float,
    first_token_ms: float | None,
    llm_elapsed_ms: float,
    timeout_reason: str | None,
    checkpoints_hit: list[int],
    completion_mode: str,
    preflight_projected_seconds: float,
    preflight_ratio: float,
    post_retrieval_projected_seconds: float,
    post_retrieval_ratio: float,
    fit_to_budget_rollout_stage: str,
    fit_to_budget_enabled: bool,
    fit_to_budget_sample_count: int,
    fit_to_budget_timeout_rate: float,
    fit_to_budget_first_token_p95_ms: float | None,
    fit_to_budget_completion_p95_seconds: float | None,
    applied_degradations: list[dict[str, object]],
    fallback_events: list[dict[str, object]],
    has_remaining_scope: bool,
    stream_recovery_reason: str | None,
    output_contract_check: dict[str, object],
) -> dict[str, object]:
    return {
        'query_type': query_type,
        'response_mode_used': response_mode_used,
        'mode_adjustments_applied': mode_adjustments_applied,
        'timeout_seconds': timeout_seconds,
        'retrieval_duration_ms': round(retrieval_elapsed_ms, 1),
        'prompt_duration_ms': round(prompt_elapsed_ms, 1),
        'first_token_latency_ms': round(first_token_ms, 1) if first_token_ms is not None else None,
        'stream_duration_ms': round(llm_elapsed_ms, 1),
        'timeout_reason': timeout_reason,
        'soft_budget_checkpoints_hit': checkpoints_hit,
        'suggested_completion_mode': completion_mode,
        'budget_preflight_projected_seconds': round(preflight_projected_seconds, 1),
        'budget_preflight_ratio': round(preflight_ratio, 3),
        'budget_post_retrieval_projected_seconds': round(post_retrieval_projected_seconds, 1),
        'budget_post_retrieval_ratio': round(post_retrieval_ratio, 3),
        'fit_to_budget_rollout_stage': fit_to_budget_rollout_stage,
        'fit_to_budget_enabled': fit_to_budget_enabled,
        'fit_to_budget_sample_count': fit_to_budget_sample_count,
        'fit_to_budget_timeout_rate': fit_to_budget_timeout_rate,
        'fit_to_budget_first_token_p95_ms': fit_to_budget_first_token_p95_ms,
        'fit_to_budget_completion_p95_seconds': fit_to_budget_completion_p95_seconds,
        'applied_degradations': applied_degradations,
        'fallback_events': fallback_events,
        'has_remaining_scope': has_remaining_scope,
        'stream_recovery_reason': stream_recovery_reason,
        'output_contract_check': output_contract_check,
        'generation_skipped': False,
    }


def record_generation_trace(
    *,
    trace: object | None,
    token_count: int,
    max_tokens: int,
    first_token_ms: float | None,
    llm_elapsed_ms: float,
    profile_name: str,
    stream_recovery_reason: str | None,
    output_contract_check: dict[str, object],
) -> None:
    if trace is None:
        return
    trace.record('llm', {
        'token_count': token_count,
        'max_tokens': max_tokens,
        'first_token_ms': round(first_token_ms, 1) if first_token_ms is not None else None,
        'total_elapsed_ms': round(llm_elapsed_ms, 1),
        'model_profile': profile_name,
        'stream_recovery_reason': stream_recovery_reason,
        'output_contract_check': output_contract_check,
    })


def log_generation_completion(
    *,
    log: object,
    query_type: str,
    question_length: int,
    context_chunks: int,
    history_messages: int,
    max_tokens: int,
    timeout_seconds: int,
    prompt_elapsed_ms: float,
    first_token_ms: float | None,
    llm_elapsed_ms: float,
    token_count: int,
    preflight_ratio: float,
    post_retrieval_ratio: float,
    response_mode_used: str,
    mode_adjustments_applied: list[str],
    applied_degradations: list[dict[str, object]],
    stream_recovery_reason: str | None,
) -> None:
    log.info(
        'rag_pipeline_completed',
        query_type=query_type,
        query_length=question_length,
        context_chunks=context_chunks,
        history_messages=history_messages,
        max_tokens=max_tokens,
        timeout_seconds=timeout_seconds,
        prompt_duration_ms=round(prompt_elapsed_ms, 1),
        llm_first_token_ms=round(first_token_ms, 1) if first_token_ms is not None else None,
        llm_duration_ms=round(llm_elapsed_ms, 1),
        tokens_generated=token_count,
        budget_preflight_ratio=round(preflight_ratio, 3),
        budget_post_retrieval_ratio=round(post_retrieval_ratio, 3),
        response_mode_used=response_mode_used,
        mode_adjustments_applied=mode_adjustments_applied,
        applied_degradations=applied_degradations,
        stream_recovery_reason=stream_recovery_reason,
    )


def build_source_references(
    *,
    chunks: list[dict],
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
        for chunk in chunks
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
