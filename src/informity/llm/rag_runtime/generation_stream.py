# ==============================================================================
# Informity AI — Generation Stream Runtime
# Streaming execution + completion summary extraction for RAG handler.
# ==============================================================================

from __future__ import annotations

import time
from collections.abc import AsyncGenerator, Callable
from dataclasses import dataclass

import structlog

from informity.llm.rag_runtime import generation_runtime as _generation_runtime
from informity.llm.streaming import stream_llm
from informity.llm.types import CompletionMode, QueryType, StreamSignalTag, TimeoutReason

log = structlog.get_logger(__name__)

STREAM_SUMMARY_EVENT = StreamSignalTag.STREAM_SUMMARY


@dataclass
class StreamExecutionSummary:
    token_count: int
    first_token_ms: float | None
    total_elapsed_ms: float
    timeout_reason: TimeoutReason | str | None
    stream_recovery_reason: str | None
    soft_budget_checkpoints_hit: list[int]
    completion_mode: CompletionMode
    has_remaining_scope: bool
    final_answer: str = ''
    # Per-stage latency breakdown (set by rag.py after streaming completes).
    # All values are wall-clock milliseconds measured with perf_counter.
    embed_ms: float | None = None           # Query embedding time
    vector_search_ms: float | None = None   # Vector ANN search time
    rerank_ms: float | None = None          # Cross-encoder reranker time
    prompt_build_ms: float | None = None    # Context assembly + message build time
    ttft_ms: float | None = None            # Time to first generated token


def _is_section_boundary(token: str) -> bool:
    return (
        '\n\n' in token
        or token.rstrip().endswith('.')
        or token.rstrip().endswith('!')
        or token.rstrip().endswith('?')
        or token.rstrip().endswith(':')
    )


async def stream_generation_with_budget(
    *,
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float,
    top_p: float,
    timeout_seconds: int,
    stop_sequences: list[str],
    fit_to_budget_enabled: bool,
    stream_soft_limit_ratio: float,
    soft_closeout_allowed: bool,
    checkpoint_query_type: QueryType | None,
    dedupe_insufficient_context_after_stream: bool,
    insufficient_context_response: str,
    applied_degradations: list[dict[str, object]],
    output_contract_plan: object | None,
    collapse_duplicate_message_fn: Callable[[str], tuple[str, bool]],
    stream_llm_fn: Callable[..., AsyncGenerator[str | tuple[str, object]]] = stream_llm,
) -> AsyncGenerator[str | tuple[str, object]]:
    checkpoint_targets = [0.6, 0.8]
    checkpoints_emitted: set[float] = set()
    timeout_reason: TimeoutReason | str | None = None
    stream_recovery_reason: str | None = None
    stream_soft_limit_ms = timeout_seconds * stream_soft_limit_ratio * 1000
    should_close_after_boundary = False

    llm_start = time.perf_counter()
    token_count = 0
    first_token_ms: float | None = None
    answer_parts: list[str] = []
    _ = output_contract_plan  # Contract enforcement occurs in closeout validator.
    async for item in stream_llm_fn(
        messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        timeout_seconds=timeout_seconds,
        stop_sequences=stop_sequences,
    ):
        if isinstance(item, tuple) and len(item) == 2 and item[0] == StreamSignalTag.TIMEOUT:
            timeout_payload = item[1] if isinstance(item[1], dict) else {}
            raw_timeout_reason = str(timeout_payload.get('reason') or TimeoutReason.UNKNOWN_TIMEOUT.value).strip().lower()
            try:
                timeout_reason = TimeoutReason(raw_timeout_reason)
            except ValueError:
                timeout_reason = raw_timeout_reason
            yield (StreamSignalTag.TIMEOUT, timeout_payload)
            continue

        if not isinstance(item, str):
            continue

        if dedupe_insufficient_context_after_stream and answer_parts:
            projected_answer = ''.join(answer_parts) + item
            if projected_answer.count(insufficient_context_response) > 1:
                stream_recovery_reason = 'duplicate_insufficient_context_guard'
                applied_degradations.append({
                    'step': 'duplicate_insufficient_context_guard',
                    'reason': 'duplicate_insufficient_context_phrase_detected',
                })
                break

        stream_elapsed_ms = (time.perf_counter() - llm_start) * 1000
        for checkpoint_ratio in checkpoint_targets:
            checkpoint_ms = timeout_seconds * checkpoint_ratio * 1000
            if checkpoint_ratio in checkpoints_emitted:
                continue
            if stream_elapsed_ms >= checkpoint_ms:
                checkpoints_emitted.add(checkpoint_ratio)
                checkpoint_payload: dict[str, object] = {
                    'ratio': checkpoint_ratio,
                    'elapsed_seconds': round(stream_elapsed_ms / 1000, 1),
                    'timeout_seconds': timeout_seconds,
                }
                if checkpoint_query_type:
                    checkpoint_payload['query_type'] = checkpoint_query_type
                yield (StreamSignalTag.BUDGET_CHECKPOINT, checkpoint_payload)

        if first_token_ms is None:
            first_token_ms = stream_elapsed_ms

        if fit_to_budget_enabled and soft_closeout_allowed and stream_elapsed_ms >= stream_soft_limit_ms:
            should_close_after_boundary = True

        token_count += 1
        answer_parts.append(item)
        yield item

        if should_close_after_boundary and _is_section_boundary(item):
            stream_recovery_reason = 'soft_limit_section_closeout'
            applied_degradations.append({
                'step': 'mid_stream_recovery_soft_limit',
                'elapsed_seconds': round(stream_elapsed_ms / 1000, 1),
                'soft_limit_seconds': round(stream_soft_limit_ms / 1000, 1),
                'reason': 'soft_budget_limit_crossed',
            })
            break

    llm_elapsed_ms = (time.perf_counter() - llm_start) * 1000
    if dedupe_insufficient_context_after_stream and answer_parts:
        deduped_answer, dedup_applied = collapse_duplicate_message_fn(''.join(answer_parts))
        if dedup_applied:
            answer_parts = [deduped_answer]
            applied_degradations.append({
                'step': 'post_stream_duplicate_insufficient_context_dedup',
                'reason': 'duplicate_insufficient_context_phrase_collapsed',
            })
    completion_mode = CompletionMode.PARTIAL if timeout_reason else CompletionMode.COMPLETE
    if stream_recovery_reason is not None:
        completion_mode = CompletionMode.SCOPED_COMPLETE
    has_remaining_scope = _generation_runtime._has_remaining_scope(
        timeout_reason=timeout_reason,
        stream_recovery_reason=stream_recovery_reason,
        generation_skipped=False,
        applied_degradations=applied_degradations,
    )

    yield (STREAM_SUMMARY_EVENT, StreamExecutionSummary(
        token_count=token_count,
        first_token_ms=first_token_ms,
        total_elapsed_ms=llm_elapsed_ms,
        timeout_reason=timeout_reason,
        stream_recovery_reason=stream_recovery_reason,
        soft_budget_checkpoints_hit=sorted(int(ratio * 100) for ratio in checkpoints_emitted),
        completion_mode=completion_mode,
        has_remaining_scope=has_remaining_scope,
        final_answer=''.join(answer_parts),
    ))
