# ==============================================================================
# Informity AI — Generation Stream Runtime
# Streaming execution + completion summary extraction for RAG handler.
# ==============================================================================

"""Module for llm rag runtime generation stream."""

from __future__ import annotations

import time
from collections.abc import AsyncGenerator, Callable
from dataclasses import dataclass

import structlog

from informity.llm.rag_runtime import generation_runtime as _generation_runtime
from informity.llm.streaming import stream_llm
from informity.llm.timeout_policy import normalize_timeout_reason
from informity.llm.types import CompletionMode, StreamSignalTag, TimeoutReason

log = structlog.get_logger(__name__)

STREAM_SUMMARY_EVENT = StreamSignalTag.STREAM_SUMMARY


@dataclass
class StreamExecutionSummary:
    """Class docstring."""
    token_count: int
    first_token_ms: float | None
    total_elapsed_ms: float
    submit_ms: float | None
    queue_wait_ms: float | None
    submit_at_s: float | None
    first_token_at_s: float | None
    timeout_reason: TimeoutReason | str | None
    stream_recovery_reason: str | None
    soft_budget_checkpoints_hit: list[int]
    completion_mode: CompletionMode
    has_remaining_scope: bool
    final_answer: str = ""
    runtime_metrics: dict[str, object] | None = None
    # Per-stage latency breakdown (set by rag.py after streaming completes).
    # All values are wall-clock milliseconds measured with perf_counter.
    embed_ms: float | None = None  # Query embedding time
    vector_search_ms: float | None = None  # Vector ANN search time
    rerank_ms: float | None = None  # Cross-encoder reranker time
    prompt_build_ms: float | None = None  # Context assembly + message build time
    ttft_ms: float | None = None  # Time to first generated token


async def stream_generation_with_budget(
    *,
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float,
    top_p: float,
    timeout_seconds: int,
    stop_sequences: list[str],
    dedupe_insufficient_context_after_stream: bool,
    insufficient_context_response: str,
    applied_degradations: list[dict[str, object]],
    output_contract_plan: object | None,
    collapse_duplicate_message_fn: Callable[[str], tuple[str, bool]],
    chat_template_kwargs_override: dict[str, object] | None = None,
    timing_context: dict[str, object] | None = None,
    stream_llm_fn: Callable[..., AsyncGenerator[str | tuple[str, object]]] = stream_llm,
) -> AsyncGenerator[str | tuple[str, object]]:
    """Stream generation with budget."""
    timeout_reason: TimeoutReason | str | None = None
    stream_recovery_reason: str | None = None
    engine_submit_ms: float | None = None
    engine_queue_wait_ms: float | None = None

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
        chat_template_kwargs_override=chat_template_kwargs_override,
        timing_context=timing_context,
    ):
        if isinstance(item, tuple) and len(item) == 2 and item[0] == StreamSignalTag.TIMEOUT:
            timeout_payload = item[1] if isinstance(item[1], dict) else {}
            timeout_reason = normalize_timeout_reason(timeout_payload.get("reason"))
            yield (StreamSignalTag.TIMEOUT, timeout_payload)
            continue
        if isinstance(item, tuple) and len(item) == 2 and item[0] == STREAM_SUMMARY_EVENT:
            summary_payload = item[1] if isinstance(item[1], dict) else {}
            submit_ms = summary_payload.get("submit_ms")
            queue_wait_ms = summary_payload.get("queue_wait_ms")
            if isinstance(submit_ms, (int, float)):
                engine_submit_ms = float(submit_ms)
            if isinstance(queue_wait_ms, (int, float)):
                engine_queue_wait_ms = float(queue_wait_ms)
            continue

        if not isinstance(item, str):
            continue

        if dedupe_insufficient_context_after_stream and answer_parts:
            projected_answer = "".join(answer_parts) + item
            if projected_answer.count(insufficient_context_response) > 1:
                stream_recovery_reason = "duplicate_insufficient_context_guard"
                applied_degradations.append(
                    {
                        "step": "duplicate_insufficient_context_guard",
                        "reason": "duplicate_insufficient_context_phrase_detected",
                    }
                )
                break

        if first_token_ms is None:
            first_token_ms = (time.perf_counter() - llm_start) * 1000
            if timing_context is not None:
                timing_context.setdefault("runtime_first_token_at_s", time.time())

        token_count += 1
        answer_parts.append(item)
        yield item

    llm_elapsed_ms = (time.perf_counter() - llm_start) * 1000
    if dedupe_insufficient_context_after_stream and answer_parts:
        deduped_answer, dedup_applied = collapse_duplicate_message_fn("".join(answer_parts))
        if dedup_applied:
            answer_parts = [deduped_answer]
            applied_degradations.append(
                {
                    "step": "post_stream_duplicate_insufficient_context_dedup",
                    "reason": "duplicate_insufficient_context_phrase_collapsed",
                }
            )
    completion_mode = CompletionMode.PARTIAL if timeout_reason else CompletionMode.COMPLETE
    if stream_recovery_reason is not None:
        completion_mode = CompletionMode.SCOPED_COMPLETE
    has_remaining_scope = _generation_runtime._has_remaining_scope(
        timeout_reason=timeout_reason,
        stream_recovery_reason=stream_recovery_reason,
        generation_skipped=False,
        applied_degradations=applied_degradations,
    )

    yield (
        STREAM_SUMMARY_EVENT,
        StreamExecutionSummary(
            token_count=token_count,
            first_token_ms=first_token_ms,
            total_elapsed_ms=llm_elapsed_ms,
            submit_ms=engine_submit_ms,
            queue_wait_ms=engine_queue_wait_ms,
            submit_at_s=timing_context.get("payload_sent_to_model_runtime_at_s")
            if timing_context is not None
            else None,
            first_token_at_s=timing_context.get("runtime_first_token_at_s")
            if timing_context is not None
            else None,
            timeout_reason=timeout_reason,
            stream_recovery_reason=stream_recovery_reason,
            soft_budget_checkpoints_hit=[],
            completion_mode=completion_mode,
            has_remaining_scope=has_remaining_scope,
            final_answer="".join(answer_parts),
            ttft_ms=first_token_ms,
            runtime_metrics=(
                timing_context.get("runtime_metrics") if timing_context is not None else None
            ),
        ),
    )
