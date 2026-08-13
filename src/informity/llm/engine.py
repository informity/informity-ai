# ==============================================================================
# Informity AI — LLM Engine
# Lazy-loads a GGUF model via xllamacpp with Metal acceleration.
# Provides async streaming generation. Downloads the model from Hugging Face
# Hub if not present locally.
#
# Runtime: xllamacpp (CommonParams + Server, no-server in-process path).
# Token counting: tiktoken cl100k_base approximation (±15%); 100-token safety
# margin in _truncate_messages_to_fit absorbs the variance.
# Chat template: read from GGUF metadata via gguf.GGUFReader at load time.
# EOS suppression (min_tokens): not available in xllamacpp — pipeline-level
# word-count gate (generation_runtime.py) is the enforcement mechanism.
# ==============================================================================

"""LLM engine and model-loading utilities for local GGUF inference."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import socket
import threading
import time
import urllib.error
import urllib.request
from collections.abc import AsyncGenerator, Callable
from contextlib import suppress
from contextvars import ContextVar, Token, copy_context
from pathlib import Path

import structlog
from thinkstrip import ThinkStrip, strip_think_prefill

from informity.config import DEFAULT_OLLAMA_BASE_URL, settings
from informity.diagnostics.resource_snapshot import capture_resource_snapshot
from informity.config import STARTUP_RAM_HEADROOM_RATIO
from informity.exceptions import LLMError
from informity.llm.model_adapter import (
    get_effective_context_length,
    get_model_alias_filenames,
    get_profile,
    get_profile_for_filename,
)
from informity.llm.model_bootstrap import GGUFModelSpec, download_gguf_model
from informity.llm.timeout_policy import normalize_timeout_reason
from informity.llm.tokenization import count_tokens
from informity.llm.types import StreamSignalTag, TimeoutReason

log = structlog.get_logger(__name__)
_PROMPT_RENDER_EXCEPTIONS = (ValueError, TypeError, AttributeError, RuntimeError)
_RUNTIME_CALL_PROBE_CONTEXT: ContextVar[dict[str, object] | None] = ContextVar(
    "informity_runtime_call_probe_context",
    default=None,
)

# ==============================================================================
# Constants
# ==============================================================================

# Sentinel put on the queue when the stream worker is done
_STREAM_END: object = object()
_FIRST_TOKEN_WATCHDOG_MIN_SECONDS = max(
    45.0,
    float(getattr(settings, "diagnostics_alert_max_first_token_seconds", 90.0) or 90.0),
)
_FIRST_TOKEN_WATCHDOG_MAX_SECONDS = 180.0
_FIRST_TOKEN_WATCHDOG_RATIO = 0.30
_SLOW_PROFILE_TPS_THRESHOLD = 6.0
_SLOW_PROFILE_WATCHDOG_RATIO = 0.50
_SLOW_PROFILE_WATCHDOG_MAX_SECONDS = 600.0


def _merge_chat_template_kwargs(
    base_kwargs: dict[str, object] | None,
    override_kwargs: dict[str, object] | None,
) -> dict[str, object]:
    """Merge template kwargs with request-scoped overrides."""
    merged: dict[str, object] = {}
    if base_kwargs:
        merged.update(base_kwargs)
    if override_kwargs:
        merged.update(override_kwargs)
    return merged


def _pick_free_loopback_port() -> int:
    """Pick an ephemeral loopback port for the embedded xllamacpp server."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _check_generation_model_memory_headroom(*, model_path: Path, stage: str) -> None:
    """Fail fast when the generation model would exceed the configured RAM headroom."""
    # Local GGUF only: OllamaProvider is a separate synthesis backend and should
    # not use this in-process memory gate.
    if model_path.name != settings.llm_model_filename:
        return
    if not model_path.is_file():
        return

    snapshot = capture_resource_snapshot()
    available_mb = snapshot.get("system_memory_available_mb")
    if not isinstance(available_mb, (int, float)):
        return

    model_size_gb = model_path.stat().st_size / (1024**3)
    available_ram_gb = float(available_mb) / 1024.0
    allowed = model_size_gb <= available_ram_gb * STARTUP_RAM_HEADROOM_RATIO
    log.info(
        "llm_generation_memory_gate",
        stage=stage,
        model=model_path.name,
        model_size_gb=round(model_size_gb, 1),
        available_ram_gb=round(available_ram_gb, 1),
        headroom_ratio=STARTUP_RAM_HEADROOM_RATIO,
        allowed=allowed,
    )
    if not allowed:
        raise LLMError("insufficient memory available for this model")


def _profile_chat_template_kwargs(profile: object) -> dict[str, object]:
    """Return profile chat-template kwargs, defaulting to an empty mapping."""
    template_kwargs = getattr(profile, "chat_template_kwargs", {})
    if isinstance(template_kwargs, dict):
        return template_kwargs
    return {}


def set_runtime_call_probe_context(context: dict[str, object] | None) -> Token[dict[str, object] | None]:
    """Set the current request-scoped runtime call probe context."""
    return _RUNTIME_CALL_PROBE_CONTEXT.set(context)


def reset_runtime_call_probe_context(token: Token[dict[str, object] | None]) -> None:
    """Reset the current request-scoped runtime call probe context."""
    _RUNTIME_CALL_PROBE_CONTEXT.reset(token)


def _get_runtime_call_probe_context() -> dict[str, object] | None:
    """Return the active runtime call probe context, if any."""
    context = _RUNTIME_CALL_PROBE_CONTEXT.get()
    if not isinstance(context, dict):
        return None
    return context


def _record_runtime_call_probe(
    *,
    call_kind: str,
    max_tokens: int | None,
    enable_thinking: bool | None,
    duration_ms: float,
    prompt_token_count: int | None,
    completion_token_count: int | None,
    runtime_metrics: dict[str, object] | None,
    status: str,
    error: str | None = None,
) -> None:
    """Record and log a runtime-call probe for the active request context."""
    context = _get_runtime_call_probe_context()
    if context is None:
        return
    records = context.setdefault("runtime_call_records", [])
    if not isinstance(records, list):
        return
    record: dict[str, object] = {
        "sequence": len(records) + 1,
        "call_kind": call_kind,
        "status": status,
        "max_tokens": max_tokens,
        "enable_thinking": enable_thinking,
        "prompt_tokens": prompt_token_count,
        "completion_tokens": completion_token_count,
        "duration_ms": round(duration_ms, 1),
    }
    if error:
        record["error"] = error
    if runtime_metrics is not None:
        prompt_token_source = runtime_metrics.get("prompt_token_count_source")
        cached_token_count = runtime_metrics.get("cached_token_count")
        prefix_cache_hit = runtime_metrics.get("prefix_cache_hit")
        prompt_eval_tps = runtime_metrics.get("prompt_eval_tokens_per_second")
        generation_tps = runtime_metrics.get("generation_tokens_per_second")
        prompt_eval_duration_ms = runtime_metrics.get("prompt_eval_duration_ms")
        generation_duration_ms = runtime_metrics.get("generation_duration_ms")
        if prompt_token_source is not None:
            record["prompt_tokens_source"] = prompt_token_source
        if cached_token_count is not None:
            record["cached_tokens"] = cached_token_count
        if prefix_cache_hit is not None:
            record["prefix_cache_hit"] = prefix_cache_hit
        if prompt_eval_tps is not None:
            record["prompt_eval_tokens_per_second"] = prompt_eval_tps
        if generation_tps is not None:
            record["generation_tokens_per_second"] = generation_tps
        if prompt_eval_duration_ms is not None:
            record["prompt_eval_duration_ms"] = prompt_eval_duration_ms
        if generation_duration_ms is not None:
            record["generation_duration_ms"] = generation_duration_ms

    records.append(record)
    context["runtime_call_count"] = len(records)
    log.info("researcher_runtime_call", **record)


def _record_generation_memory_snapshot(
    *,
    point: str,
    timing_context: dict[str, object] | None,
) -> None:
    """Capture a compact generation-time memory snapshot."""
    snapshot = capture_resource_snapshot()
    previous_snapshot: dict[str, object] | None = None
    if isinstance(timing_context, dict):
        snapshots = timing_context.setdefault("generation_memory_snapshots", [])
        if isinstance(snapshots, list):
            if snapshots and isinstance(snapshots[-1], dict):
                previous_snapshot = snapshots[-1]
            snapshot["point"] = point
            snapshot["generation_snapshot_index"] = len(snapshots) + 1
            if previous_snapshot is not None:
                for key, delta_key in (
                    ("system_pageins", "pageins_delta"),
                    ("system_pageouts", "pageouts_delta"),
                ):
                    before_value = previous_snapshot.get(key)
                    after_value = snapshot.get(key)
                    if isinstance(before_value, (int, float)) and isinstance(
                        after_value, (int, float)
                    ):
                        snapshot[delta_key] = round(float(after_value) - float(before_value), 2)
            snapshots.append(snapshot)
    log_snapshot = dict(snapshot)
    log_snapshot.pop("point", None)
    log.info("llm_generation_memory_snapshot", **log_snapshot)


# ==============================================================================
# Finish reason normalisation
# ==============================================================================


def _normalize_finish_reason(reason: str | None) -> str | None:
    """normalize finish reason."""
    if not reason:
        return None
    reason_lower = str(reason).lower().strip()
    if reason_lower in ("stop", "eos", "end_of_sequence"):
        return "stop"
    if reason_lower in ("length", "max_tokens", "max_tokens_reached"):
        return "length"
    if reason_lower == "cancelled":
        return "cancelled"
    log.debug("llm_unknown_finish_reason", reason=reason)
    return reason_lower


def _raise_llm_streaming_failed(exc: Exception) -> None:
    """raise llm streaming failed."""
    raise LLMError(f"LLM streaming failed: {exc}") from exc


def _resolve_first_token_deadline_seconds(*, wall_clock: float, profile_tps: float) -> float:
    """Compute first-token watchdog deadline with a slow-profile override."""
    base_deadline = max(
        _FIRST_TOKEN_WATCHDOG_MIN_SECONDS,
        min(_FIRST_TOKEN_WATCHDOG_MAX_SECONDS, wall_clock * _FIRST_TOKEN_WATCHDOG_RATIO),
    )
    if profile_tps <= _SLOW_PROFILE_TPS_THRESHOLD:
        slow_deadline = max(
            _FIRST_TOKEN_WATCHDOG_MIN_SECONDS,
            min(_SLOW_PROFILE_WATCHDOG_MAX_SECONDS, wall_clock * _SLOW_PROFILE_WATCHDOG_RATIO),
        )
        return max(base_deadline, slow_deadline)
    return base_deadline


def _extract_runtime_metrics_from_chunks(
    chunks: list[dict[str, object]],
    *,
    prompt_token_estimate: int | None = None,
    total_elapsed_ms: float | None = None,
    token_count: int | None = None,
) -> dict[str, object]:
    """Extract model runtime metrics from collected response chunks."""
    def _to_int(value: object) -> int | None:
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, int):
            return value if value >= 0 else 0
        if isinstance(value, float):
            value = int(value)
            return value if value >= 0 else 0
        if isinstance(value, str):
            with suppress(ValueError):
                return max(0, int(float(value.strip())))
        return None

    def _to_float(value: object) -> float | None:
        if isinstance(value, bool):
            return float(value)
        if isinstance(value, (int, float)):
            value = float(value)
            return value if value >= 0.0 else 0.0
        if isinstance(value, str):
            with suppress(ValueError):
                value = float(value.strip())
                return value if value >= 0.0 else 0.0
        return None

    def _duration_to_seconds(key: str, value: object) -> float | None:
        numeric = _to_float(value)
        if numeric is None:
            return None
        lowered = key.casefold()
        if lowered.endswith("_ms") or "_ms_" in lowered or "milliseconds" in lowered:
            return numeric / 1000.0
        if lowered.endswith("_us") or "microseconds" in lowered:
            return numeric / 1_000_000.0
        if lowered.endswith("_s") or "seconds" in lowered or "duration_s" in lowered:
            return numeric
        return numeric / 1_000_000_000.0

    dict_chunks = [chunk for chunk in chunks if isinstance(chunk, dict)]
    if not dict_chunks:
        return {}
    final_chunk = dict_chunks[-1]
    usage = final_chunk.get("usage")
    usage_dict = usage if isinstance(usage, dict) else {}

    prompt_token_count = None
    prompt_token_source = None
    for candidate_key in (
        "prompt_tokens",
        "prompt_eval_count",
        "prompt_token_count",
        "prompt_eval_tokens",
    ):
        prompt_token_count = _to_int(usage_dict.get(candidate_key))
        if prompt_token_count is not None:
            prompt_token_source = "usage"
            break
        prompt_token_count = _to_int(final_chunk.get(candidate_key))
        if prompt_token_count is not None:
            prompt_token_source = "response"
            break
    if prompt_token_count is None and prompt_token_estimate is not None:
        prompt_token_count = int(prompt_token_estimate)
        prompt_token_source = "estimate"

    completion_token_count = None
    completion_token_source = None
    for candidate_key in (
        "completion_tokens",
        "completion_token_count",
        "eval_count",
        "generated_tokens",
        "generation_tokens",
        "output_tokens",
    ):
        completion_token_count = _to_int(usage_dict.get(candidate_key))
        if completion_token_count is not None:
            completion_token_source = "usage"
            break
        completion_token_count = _to_int(final_chunk.get(candidate_key))
        if completion_token_count is not None:
            completion_token_source = "response"
            break
    if completion_token_count is None and token_count is not None:
        completion_token_count = int(token_count)
        completion_token_source = "stream_count"

    cached_token_count = None
    for candidate_key in (
        "cached_tokens",
        "cache_hit_tokens",
        "prompt_cached_tokens",
        "kv_cache_tokens",
        "cached_token_count",
    ):
        cached_token_count = _to_int(usage_dict.get(candidate_key))
        if cached_token_count is not None:
            break
        cached_token_count = _to_int(final_chunk.get(candidate_key))
        if cached_token_count is not None:
            break

    prefix_cache_hit = None
    for candidate_key in (
        "prefix_cache_hit",
        "cache_hit",
        "prompt_cache_hit",
        "kv_cache_hit",
    ):
        value = usage_dict.get(candidate_key)
        if isinstance(value, bool):
            prefix_cache_hit = value
            break
        value = final_chunk.get(candidate_key)
        if isinstance(value, bool):
            prefix_cache_hit = value
            break
    if prefix_cache_hit is None and cached_token_count is not None:
        prefix_cache_hit = cached_token_count > 0

    prompt_eval_duration_seconds = None
    generation_duration_seconds = None
    prompt_eval_tps = None
    generation_tps = None
    prompt_eval_duration_keys = (
        "prompt_eval_duration",
        "prompt_eval_duration_ms",
        "prompt_eval_duration_s",
    )
    generation_duration_keys = ("eval_duration", "generation_duration", "completion_duration")

    for candidate_key in prompt_eval_duration_keys:
        prompt_eval_duration_seconds = _duration_to_seconds(candidate_key, usage_dict.get(candidate_key))
        if prompt_eval_duration_seconds is not None:
            break
        prompt_eval_duration_seconds = _duration_to_seconds(candidate_key, final_chunk.get(candidate_key))
        if prompt_eval_duration_seconds is not None:
            break

    for candidate_key in generation_duration_keys:
        generation_duration_seconds = _duration_to_seconds(candidate_key, usage_dict.get(candidate_key))
        if generation_duration_seconds is not None:
            break
        generation_duration_seconds = _duration_to_seconds(candidate_key, final_chunk.get(candidate_key))
        if generation_duration_seconds is not None:
            break

    for candidate_key in ("prompt_eval_tps", "prompt_eval_tokens_per_second"):
        prompt_eval_tps = _to_float(usage_dict.get(candidate_key))
        if prompt_eval_tps is not None:
            break
        prompt_eval_tps = _to_float(final_chunk.get(candidate_key))
        if prompt_eval_tps is not None:
            break

    for candidate_key in ("eval_tps", "generation_tokens_per_second", "completion_tokens_per_second"):
        generation_tps = _to_float(usage_dict.get(candidate_key))
        if generation_tps is not None:
            break
        generation_tps = _to_float(final_chunk.get(candidate_key))
        if generation_tps is not None:
            break

    if prompt_eval_tps is None and prompt_token_count is not None and prompt_eval_duration_seconds:
        prompt_eval_tps = prompt_token_count / max(prompt_eval_duration_seconds, 1e-9)
    if generation_tps is None and token_count is not None and total_elapsed_ms is not None:
        generation_tps = token_count / max(total_elapsed_ms / 1000.0, 1e-9)

    runtime_metrics: dict[str, object] = {
        "prompt_token_count": prompt_token_count,
        "prompt_token_count_source": prompt_token_source,
        "completion_token_count": completion_token_count,
        "completion_token_count_source": completion_token_source,
        "cached_token_count": cached_token_count,
        "prefix_cache_hit": prefix_cache_hit,
        "prompt_eval_tokens_per_second": round(prompt_eval_tps, 3)
        if prompt_eval_tps is not None
        else None,
        "generation_tokens_per_second": round(generation_tps, 3) if generation_tps is not None else None,
    }
    if prompt_eval_duration_seconds is not None:
        runtime_metrics["prompt_eval_duration_ms"] = round(prompt_eval_duration_seconds * 1000.0, 3)
    if generation_duration_seconds is not None:
        runtime_metrics["generation_duration_ms"] = round(generation_duration_seconds * 1000.0, 3)
    if total_elapsed_ms is not None:
        runtime_metrics["total_elapsed_ms"] = round(total_elapsed_ms, 3)
    return runtime_metrics


async def _stream_from_queue(
    queue: asyncio.Queue[str | object],
    _cancel_event: threading.Event,
    *,
    start: float,
    wall_clock: float,
    first_token_deadline_seconds: float,
    stripper: ThinkStrip,
    timing_context: dict[str, object] | None = None,
) -> AsyncGenerator[tuple[str, object]]:
    """stream from queue."""
    first_token_seen = False
    think_to_visible_snapshot_taken = False
    while True:
        elapsed = time.perf_counter() - start
        if elapsed >= wall_clock:
            yield (
                "timeout",
                {
                    "reason": TimeoutReason.UNKNOWN_TIMEOUT.value,
                    "elapsed_seconds": round(elapsed, 1),
                    "timeout_seconds": wall_clock,
                },
            )
            return
        if not first_token_seen and elapsed >= first_token_deadline_seconds:
            yield (
                "timeout",
                {
                    "reason": TimeoutReason.FIRST_TOKEN_WATCHDOG_TIMEOUT.value,
                    "elapsed_seconds": round(elapsed, 1),
                    "timeout_seconds": wall_clock,
                },
            )
            return

        remaining_timeout = wall_clock - elapsed
        queue_poll_timeout = min(remaining_timeout, 2.0)
        try:
            item = await asyncio.wait_for(queue.get(), timeout=queue_poll_timeout)
        except TimeoutError:
            continue

        if item is _STREAM_END:
            emit_text = stripper.flush()
            if emit_text:
                first_token_seen = True
                yield ("text", emit_text)
            return
        if isinstance(item, tuple) and len(item) == 2 and item[0] == StreamSignalTag.FINISH_REASON:
            yield ("finish_reason", item[1])
            continue

        raw_token = str(item)
        if (
            timing_context is not None
            and raw_token
            and "runtime_first_raw_token_at_s" not in timing_context
        ):
            timing_context["runtime_first_raw_token_at_s"] = time.time()
        in_think_before = stripper.in_think_block
        emit_text = stripper.feed(raw_token)
        if not emit_text:
            continue

        if (
            timing_context is not None
            and not think_to_visible_snapshot_taken
            and in_think_before
            and not stripper.in_think_block
        ):
            think_to_visible_snapshot_taken = True
            _record_generation_memory_snapshot(
                point="think_to_visible",
                timing_context=timing_context,
            )

        first_token_seen = True
        yield ("text", emit_text)


# ==============================================================================
# HuggingFace cache cleanup
# ==============================================================================


def remove_models_dir_cache() -> None:
    # Remove any nested .cache directories left by huggingface_hub after download.
    """remove models dir cache."""
    for models_dir in [settings.models_dir]:
        if models_dir is None:
            continue
        cache_dir = models_dir / ".cache"
        if cache_dir.is_dir():
            try:
                shutil.rmtree(cache_dir)
                log.info("models_dir_cache_removed", path=str(cache_dir))
            except OSError as exc:
                log.warning("models_dir_cache_remove_failed", path=str(cache_dir), error=str(exc))


# ==============================================================================
# GGUF metadata
# ==============================================================================


def _read_gguf_chat_template(model_path: Path) -> str:
    # Read the chat template from GGUF metadata using gguf.GGUFReader.
    # Returns empty string if not found or on any error.
    """read gguf chat template."""
    try:
        from gguf import GGUFReader  # type: ignore[import-untyped]

        reader = GGUFReader(str(model_path), mode="r")
        field = reader.fields.get("tokenizer.chat_template")
        if field is not None and field.parts:
            return bytes(field.parts[-1]).decode("utf-8")
    except (ImportError, OSError, RuntimeError, ValueError, TypeError) as exc:
        log.debug("gguf_template_read_failed", error=str(exc))
    return ""


# ==============================================================================
# Prompt rendering
# ==============================================================================


def _messages_to_prompt(chat_template: str, messages: list[dict[str, str]]) -> str:
    # Convert chat messages to a prompt string.
    # Uses the GGUF's embedded Jinja2 chat template when available (preferred),
    # falls back to ChatML when the template is empty or fails to render.
    """messages to prompt."""
    if chat_template:
        try:
            prompt = _render_gguf_template(chat_template, messages)
            if prompt and prompt.strip():
                # Strip trailing <think> prefill that some GGUF builds add.
                # The model will generate <think> itself; having it in the prompt
                # breaks our streaming <think> block detection.
                prompt = strip_think_prefill(prompt)
                log.debug(
                    "gguf_template_used",
                    prompt_len=len(prompt),
                    prompt_tail=repr(prompt[-80:]),
                )
                return prompt
            log.warning(
                "gguf_template_empty",
                msg="GGUF template rendered to empty string; using ChatML fallback",
            )
        except _PROMPT_RENDER_EXCEPTIONS as exc:
            log.warning("gguf_template_render_failed", error=str(exc))
    else:
        log.debug("gguf_template_not_found", msg="No chat template in GGUF metadata")

    prompt = _fallback_chatml_prompt(messages)
    log.debug("chatml_fallback_used", prompt_len=len(prompt), prompt_tail=repr(prompt[-80:]))
    return prompt


def _render_gguf_template(template_str: str, messages: list[dict[str, str]]) -> str:
    """render gguf template."""
    from jinja2 import BaseLoader, Environment

    env = Environment(loader=BaseLoader())
    env.globals["raise_exception"] = lambda msg: (_ for _ in ()).throw(ValueError(msg))
    template = env.from_string(template_str)
    return template.render(messages=messages, add_generation_prompt=True)


def _fallback_chatml_prompt(messages: list[dict[str, str]]) -> str:
    """fallback chatml prompt."""
    parts: list[str] = []
    for msg in messages:
        parts.append(f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n")
    parts.append("<|im_start|>assistant\n")
    return "".join(parts)


# ==============================================================================
# Context budget truncation
# ==============================================================================


def _truncate_messages_to_fit(
    chat_template: str,
    messages: list[dict[str, str]],
    context_length: int,
    max_tokens: int,
    force_chatml: bool = False,
) -> tuple[list[dict[str, str]], dict]:
    """
    Truncate messages to fit within context_length.

    Strategy:
    1. Remove history messages from start (oldest first).
    2. Truncate system message context chunks from end.
    3. Keep current question intact.

    Token counting uses tiktoken cl100k_base (±15% vs Qwen3 tokenizer).
    The 100-token safety_margin absorbs this variance.
    """
    safety_margin = 100
    available_budget = context_length - max_tokens - safety_margin

    def _count_prompt(msgs: list[dict[str, str]]) -> int:
        """count prompt."""
        if force_chatml:
            return count_tokens(_fallback_chatml_prompt(msgs))
        return count_tokens(_messages_to_prompt(chat_template, msgs))

    total_tokens = _count_prompt(messages)
    truncation_info = {
        "truncated": False,
        "original_tokens": total_tokens,
        "available_budget": available_budget,
        "history_messages_removed": 0,
        "system_content_truncated": False,
    }

    if total_tokens <= available_budget:
        return messages, truncation_info

    truncation_info["truncated"] = True
    truncated_messages = [msg.copy() for msg in messages]

    # Strategy 1: remove oldest history messages
    if len(truncated_messages) > 2:
        history_start = 1
        history_end = len(truncated_messages) - 1
        for i in range(history_start, history_end):
            test_messages = [truncated_messages[0]] + truncated_messages[i + 1 :]
            test_tokens = _count_prompt(test_messages)
            if test_tokens <= available_budget:
                truncated_messages = test_messages
                truncation_info["history_messages_removed"] = i - history_start + 1
                total_tokens = _count_prompt(truncated_messages)
                if total_tokens <= available_budget:
                    truncation_info["final_tokens"] = total_tokens
                    return truncated_messages, truncation_info
                break

    # Strategy 2: truncate system message context chunks from end
    system_content = truncated_messages[0]["content"]
    context_marker = "Context:\n"
    marker_pos = system_content.find(context_marker)

    if marker_pos != -1:
        system_prompt_part = system_content[: marker_pos + len(context_marker)]
        context_part = system_content[marker_pos + len(context_marker) :]
        chunks: list[str] = []
        if context_part:
            parts = context_part.split("\n\n")
            current_chunk: list[str] = []
            for part in parts:
                if part.strip().startswith("[Source:"):
                    if current_chunk:
                        chunks.append("\n\n".join(current_chunk))
                    current_chunk = [part]
                else:
                    if current_chunk:
                        current_chunk.append(part)
                    else:
                        chunks.append(part)
            if current_chunk:
                chunks.append("\n\n".join(current_chunk))

        if chunks:
            for i in range(len(chunks) - 1, -1, -1):
                remaining_chunks = chunks[:i]
                new_context = "\n\n".join(remaining_chunks) if remaining_chunks else ""
                new_system_content = system_prompt_part + new_context
                test_messages = [
                    {"role": "system", "content": new_system_content}
                ] + truncated_messages[1:]
                test_tokens = _count_prompt(test_messages)
                if test_tokens <= available_budget:
                    truncated_messages[0]["content"] = new_system_content
                    truncation_info["system_content_truncated"] = True
                    truncation_info["chunks_removed"] = len(chunks) - len(remaining_chunks)
                    truncation_info["final_tokens"] = _count_prompt(truncated_messages)
                    return truncated_messages, truncation_info

    truncation_info["final_tokens"] = _count_prompt(truncated_messages)
    truncation_info["warning"] = "Prompt still exceeds budget after truncation"
    return truncated_messages, truncation_info


# ==============================================================================
# Stream worker — runs in a background thread
# ==============================================================================


def _run_stream_worker(
    server: object,
    messages: list[dict[str, str]],
    max_tok: int,
    temp: float,
    top_p_val: float,
    stop_seqs: list[str],
    chat_template_kwargs_override: dict[str, object] | None,
    loop: asyncio.AbstractEventLoop,
    queue: asyncio.Queue[str | object],
    exception_holder: list[BaseException],
    cancel_event: threading.Event,
    request_start: float | None = None,
    timing_state: dict[str, float | None] | None = None,
    timing_context: dict[str, object] | None = None,
) -> None:
    # Run the blocking xllamacpp generation call in a background thread.
    # Sends messages via handle_chat_completions (OpenAI-compatible chat API)
    # with stream=True. Each text chunk from the delta stream is pushed to
    # the asyncio queue via call_soon_threadsafe so the event loop is never
    # blocked.
    #
    # Response format: OpenAI chat completions streaming delta —
    #   {"choices": [{"delta": {"content": "token"}, "finish_reason": null}]}
    # Final chunk: {"choices": [{"delta": {}, "finish_reason": "stop"}]}
    #
    # Cancellation: when cancel_event is set (consumer disconnect or timeout),
    # the callback stops pushing tokens. C++ generation may continue briefly
    # until the current n_predict budget is exhausted; output is discarded.
    """run stream worker."""
    try:
        call_started_at = time.perf_counter()
        if timing_state is not None and request_start is not None:
            timing_state["submit_ms"] = (time.perf_counter() - request_start) * 1000
            timing_state["submit_at_s"] = time.time()
        runtime_frames: list[dict[str, object]] = []
        _tmpl_kwargs = _merge_chat_template_kwargs(
            _profile_chat_template_kwargs(get_profile()),
            chat_template_kwargs_override,
        )
        payload = json.dumps(
            {
                "messages": messages,
                "max_tokens": max_tok,
                "temperature": temp,
                "top_p": top_p_val,
                "stop": stop_seqs or [],
                "stream": True,
                **({"chat_template_kwargs": _tmpl_kwargs} if _tmpl_kwargs else {}),
            }
        )

        finish_reason: str | None = None

        def _callback(chunk: object) -> None:
            """callback."""
            nonlocal finish_reason
            if cancel_event.is_set():
                return

            # Parse chunk — xllamacpp delivers JSON dicts or JSON strings
            if isinstance(chunk, dict):
                data = chunk
            elif isinstance(chunk, (str, bytes)):
                raw = chunk if isinstance(chunk, str) else chunk.decode("utf-8", errors="replace")
                # Strip SSE "data: " prefix if present
                if raw.startswith("data:"):
                    raw = raw[5:].strip()
                try:
                    data = json.loads(raw)
                except (json.JSONDecodeError, ValueError):
                    # Treat the raw string as token text directly
                    if raw:
                        loop.call_soon_threadsafe(queue.put_nowait, raw)
                    return
            else:
                return

            if not isinstance(data, dict):
                return
            runtime_frames.append(data)

            # OpenAI chat completions streaming format: choices[0].delta.content
            choices = data.get("choices") or []
            choice = choices[0] if choices else {}
            delta = choice.get("delta") or {}
            token = delta.get("content") or ""

            # Some local adapters emit chat-completions chunks as choices[0].message.content
            # (without delta). Accept that shape to preserve streaming compatibility.
            if not token:
                token = (choice.get("message") or {}).get("content") or ""

            # Alternate llama.cpp completions payload uses top-level 'content'
            if not token:
                token = data.get("content") or ""

            if token and not cancel_event.is_set():
                if timing_state is not None and "runtime_first_raw_token_at_s" not in timing_state:
                    timing_state["runtime_first_raw_token_at_s"] = time.time()
                loop.call_soon_threadsafe(queue.put_nowait, token)

            # Finish reason from OpenAI format
            fr = choice.get("finish_reason")
            if fr:
                finish_reason = _normalize_finish_reason(fr)
            elif data.get("stop", False):
                # Alternate llama.cpp stop payload format
                if data.get("stopped_eos", False) or data.get("stopped_word", False):
                    finish_reason = "stop"
                elif data.get("stopped_limit", False):
                    finish_reason = "length"
                else:
                    finish_reason = _normalize_finish_reason(data.get("stop_type"))

        server.handle_chat_completions(payload, _callback)  # type: ignore[attr-defined]

        if cancel_event.is_set():
            finish_reason = "cancelled"

        loop.call_soon_threadsafe(
            queue.put_nowait,
            (StreamSignalTag.FINISH_REASON, finish_reason),
        )
        if timing_state is not None:
            timing_state["runtime_frames"] = runtime_frames

    except (
        AttributeError,
        IndexError,
        KeyError,
        RuntimeError,
        TypeError,
        UnicodeError,
        ValueError,
    ) as exc:
        exception_holder.append(exc)
    finally:
        elapsed_ms = (time.perf_counter() - call_started_at) * 1000.0
        runtime_metrics = _extract_runtime_metrics_from_chunks(
            runtime_frames,
            total_elapsed_ms=elapsed_ms,
            token_count=None,
        )
        _record_runtime_call_probe(
            call_kind="generate_stream",
            max_tokens=max_tok,
            enable_thinking=bool(_tmpl_kwargs.get("enable_thinking", False)),
            duration_ms=elapsed_ms,
            prompt_token_count=(
                int(runtime_metrics["prompt_token_count"])
                if isinstance(runtime_metrics.get("prompt_token_count"), int)
                else None
            ),
            completion_token_count=(
                int(runtime_metrics["completion_token_count"])
                if isinstance(runtime_metrics.get("completion_token_count"), int)
                else None
            ),
            runtime_metrics=runtime_metrics,
            status="error" if exception_holder else "ok",
            error=str(exception_holder[-1]) if exception_holder else None,
        )
        with suppress(RuntimeError):
            loop.call_soon_threadsafe(queue.put_nowait, _STREAM_END)


# ==============================================================================
# XllamaCppProvider — lazy-loading xllamacpp wrapper
# ==============================================================================


class XllamaCppProvider:
    # Wraps an xllamacpp Server with lazy loading, automatic download,
    # and async streaming generation. Configured for Apple Metal GPU by default.

    """XllamaCppProvider model."""

    def __init__(self, model_filename: str | None = None, *, model_dir: Path | None = None) -> None:
        """init  ."""
        self._server: object | None = None
        self._chat_template: str = ""
        self._model_filename_override = str(model_filename or "").strip() or None
        self._model_dir_override = model_dir

    # -- Internal server accessor ---------------------------------------------

    @property
    def server(self) -> object | None:
        """server."""
        return self._server

    @server.setter
    def server(self, value: object | None) -> None:
        """server."""
        self._server = value

    @property
    def chat_template(self) -> str:
        """chat template."""
        return self._chat_template

    @chat_template.setter
    def chat_template(self, value: str) -> None:
        """chat template."""
        self._chat_template = value

    @property
    def _loaded_server(self) -> object:
        """loaded server."""
        if self._server is None:
            self._load_model()
        return self._server  # type: ignore[return-value]

    # -- Public state ----------------------------------------------------------

    @property
    def is_loaded(self) -> bool:
        """is loaded."""
        return self._server is not None

    def unload(self) -> None:
        """Unload the model and free GPU memory."""
        if self._server is not None:
            del self._server
            self._server = None
            self._chat_template = ""
            import gc

            gc.collect()
            log.debug("llm_model_unloaded")

    def count_tokens(self, text: str) -> int:
        # Count tokens using tiktoken cl100k_base (±15% vs Qwen3 tokenizer).
        # Used by the RAG pipeline for context budget management.
        """count tokens."""
        return count_tokens(text)

    # -- Model path -----------------------------------------------------------

    def get_model_path(self) -> Path:
        """get model path."""
        model_filename = self._model_filename_override or settings.llm_model_filename
        model_dir = self._model_dir_override or settings.models_dir
        return model_dir / model_filename

    def _get_model_path(self) -> Path:
        """get model path."""
        return self.get_model_path()

    # -- Model loading --------------------------------------------------------

    def load_model(self, model_filename: str | None = None) -> None:
        """
        Load the GGUF model via xllamacpp.

        When llm_local_only is True, only load from models_dir; never download.
        When False, download from Hugging Face if the file is missing.

        Args:
            model_filename: Optional filename for profile lookup. If None, uses
                            settings.llm_model_filename. Allows the model path
                            and profile to be resolved independently.
        """
        if self._server is not None:
            self.unload()

        profile_filename = (
            model_filename
            if model_filename is not None
            else self._model_filename_override or settings.llm_model_filename
        )
        model_dir = self._model_dir_override or settings.models_dir
        model_path = model_dir / profile_filename

        if not model_path.exists():
            local_only = settings.full_privacy or settings.llm_local_only
            if local_only:
                raise LLMError(
                    f"LLM model not found at {model_path}. "
                    "Place your GGUF file in the models directory "
                    f"({model_dir}) or turn off Full Privacy Mode (Settings) "
                    "or set INFORMITY_FULL_PRIVACY=false to allow download."
                )
            log.info("model_not_found_locally", path=str(model_path), filename=profile_filename)
            self._download_model(model_path)

        _check_generation_model_memory_headroom(model_path=model_path, stage="load_model")

        profile = get_profile_for_filename(profile_filename)
        profile_ctx_len = int(profile.context_length)
        configured_ctx_len = int(getattr(settings, "llm_context_length", 0) or 0)
        ctx_len = get_effective_context_length(profile)

        log.info(
            "loading_llm_model",
            path=str(model_path),
            context_length=ctx_len,
            profile_context_length=profile_ctx_len,
            configured_context_length=configured_ctx_len if configured_ctx_len > 0 else None,
            n_batch=256,
            n_threads=settings.llm_cpu_threads,
        )
        load_memory_before = capture_resource_snapshot()

        start = time.perf_counter()

        try:
            from xllamacpp import CommonParams, Server  # type: ignore[import-untyped]

            params = CommonParams()
            params.model.path = str(model_path)
            params.n_ctx = ctx_len
            params.n_gpu_layers = -1  # Offload all layers to Metal GPU
            params.n_batch = 256
            # Reduce peak CPU during prompt prefill (lowered from 512 to reduce fan noise; raise if
            # TTFT regresses)
            params.cpuparams.n_threads = settings.llm_cpu_threads  # Cap CPU threads
            params.cpuparams_batch.n_threads = settings.llm_cpu_threads
            params.port = _pick_free_loopback_port()

            # Read chat template from GGUF metadata before constructing Server,
            # while we still have direct file access.
            self._chat_template = _read_gguf_chat_template(model_path)

            # Suppress C-layer verbosity: per-request slot/timing logs written
            # directly to stdout by the llama.cpp server layer.
            params.verbosity = -1

            # Suppress C-layer init output (Metal init, model loading progress,
            # n_ctx warnings) by redirecting OS-level fd 1/2 to /dev/null.
            # Python's redirect_stderr only covers sys.stderr; C stdio uses fd
            # numbers directly, so an fd-level redirect is required.
            _devnull_fd = os.open(os.devnull, os.O_WRONLY)
            _saved_out, _saved_err = os.dup(1), os.dup(2)
            os.dup2(_devnull_fd, 1)
            os.dup2(_devnull_fd, 2)
            try:
                self._server = Server(params)
            finally:
                os.dup2(_saved_out, 1)
                os.dup2(_saved_err, 2)
                os.close(_devnull_fd)
                os.close(_saved_out)
                os.close(_saved_err)

        except ImportError as exc:
            raise LLMError(f"xllamacpp is not installed: {exc}") from exc
        except AttributeError as exc:
            raise LLMError(f"xllamacpp parameter mapping failed — API mismatch: {exc}") from exc
        except ValueError as exc:
            raise LLMError(f"Invalid model configuration: {exc}") from exc
        except RuntimeError as exc:
            raise LLMError(f'Failed to load LLM model "{model_path.name}": {exc}') from exc

        elapsed_ms = (time.perf_counter() - start) * 1000
        load_memory_after = capture_resource_snapshot()
        log.info(
            "llm_model_loaded",
            model=model_path.name,
            elapsed_ms=round(elapsed_ms, 1),
            chat_template_found=bool(self._chat_template),
            available_ram_mb_before=load_memory_before.get("system_memory_available_mb"),
            available_ram_mb_after=load_memory_after.get("system_memory_available_mb"),
            process_rss_mb_before=load_memory_before.get("process_rss_mb"),
            process_rss_mb_after=load_memory_after.get("process_rss_mb"),
        )

    # -- Model download -------------------------------------------------------

    def download_model(
        self,
        target_path: Path,
        repo_id: str | None = None,
        filename: str | None = None,
        revision: str | None = None,
        expected_sha256: str | None = None,
        progress_callback: Callable[[int, int | None, float], None] | None = None,
        cancel_event: threading.Event | None = None,
    ) -> None:
        """download model."""
        repo = repo_id or settings.llm_hf_repo
        fname = filename or target_path.name
        log.info(
            "downloading_llm_model",
            repo=repo,
            filename=fname,
            revision=revision,
            target=str(target_path),
        )
        download_gguf_model(
            spec=GGUFModelSpec(
                repo_id=repo,
                filename=fname,
                expected_sha256=expected_sha256,
                revision=revision,
                model_label="llm",
            ),
            target_path=target_path,
            progress_callback=progress_callback,
            cancel_event=cancel_event,
        )
        remove_models_dir_cache()

    def _load_model(self, model_filename: str | None = None) -> None:
        """load model."""
        self.load_model(model_filename=model_filename)

    def _download_model(
        self,
        target_path: Path,
        repo_id: str | None = None,
        filename: str | None = None,
        revision: str | None = None,
        expected_sha256: str | None = None,
        progress_callback: Callable[[int, int | None, float], None] | None = None,
        cancel_event: threading.Event | None = None,
    ) -> None:
        """download model."""
        self.download_model(
            target_path=target_path,
            repo_id=repo_id,
            filename=filename,
            revision=revision,
            expected_sha256=expected_sha256,
            progress_callback=progress_callback,
            cancel_event=cancel_event,
        )

    # -- Synchronous chat completion ------------------------------------------

    def chat_complete(
        self,
        messages: list[dict],
        max_tokens: int = 400,
        temperature: float = 0.0,
        stop: list[str] | None = None,
        response_format: dict | None = None,
        chat_template_kwargs_override: dict[str, object] | None = None,
    ) -> dict:
        """
        Synchronous (blocking) chat completion via xllamacpp.

        Uses server.handle_chat_completions with stream=False. Returns a dict
        compatible with the OpenAI chat completions format:
        {'choices': [{'message': {'content': '...'}}]}.

        Intended for internal callers (classifier, planner, warmup) that need
        deterministic, non-streaming responses.

        Raises:
            LLMError: If inference fails.
        """
        server = self._loaded_server
        start = time.perf_counter()

        _tmpl_kwargs = _merge_chat_template_kwargs(
            _profile_chat_template_kwargs(get_profile()),
            chat_template_kwargs_override,
        )
        payload_dict: dict = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stop": stop or [],
            "stream": False,
        }
        if response_format is not None:
            payload_dict["response_format"] = response_format
        if _tmpl_kwargs:
            payload_dict["chat_template_kwargs"] = _tmpl_kwargs
        payload = json.dumps(payload_dict)

        collected: list[dict] = []

        def _cb(chunk: object) -> None:
            """cb."""
            if isinstance(chunk, dict):
                collected.append(chunk)
            elif isinstance(chunk, (str, bytes)):
                raw = chunk if isinstance(chunk, str) else chunk.decode("utf-8", errors="replace")
                if raw.startswith("data:"):
                    raw = raw[5:].strip()
                with suppress(json.JSONDecodeError, ValueError):
                    collected.append(json.loads(raw))

        try:
            server.handle_chat_completions(payload, _cb)  # type: ignore[attr-defined]
        except Exception as exc:
            _record_runtime_call_probe(
                call_kind="chat_complete",
                max_tokens=max_tokens,
                enable_thinking=bool(_tmpl_kwargs.get("enable_thinking", False)),
                duration_ms=(time.perf_counter() - start) * 1000.0,
                prompt_token_count=None,
                completion_token_count=None,
                runtime_metrics=None,
                status="error",
                error=str(exc),
            )
            raise LLMError(f"Chat completion inference failed: {exc}") from exc

        # Assemble content from collected chunks.
        # Non-streaming: {'choices': [{'message': {'content': '...'}}]}
        # Streaming delta: {'choices': [{'delta': {'content': '...'}}]}
        content_parts: list[str] = []
        for chunk in collected:
            for choice in chunk.get("choices", []):
                msg = choice.get("message", {})
                if msg.get("content"):
                    content_parts.append(msg["content"])
                delta = choice.get("delta", {})
                if delta.get("content"):
                    content_parts.append(delta["content"])

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        runtime_metrics = _extract_runtime_metrics_from_chunks(
            collected,
            prompt_token_estimate=count_tokens(_messages_to_prompt(self._chat_template, messages)),
            total_elapsed_ms=elapsed_ms,
            token_count=None,
        )
        _record_runtime_call_probe(
            call_kind="chat_complete",
            max_tokens=max_tokens,
            enable_thinking=bool(_tmpl_kwargs.get("enable_thinking", False)),
            duration_ms=elapsed_ms,
            prompt_token_count=(
                int(runtime_metrics["prompt_token_count"])
                if isinstance(runtime_metrics.get("prompt_token_count"), int)
                else None
            ),
            completion_token_count=(
                int(runtime_metrics["completion_token_count"])
                if isinstance(runtime_metrics.get("completion_token_count"), int)
                else None
            ),
            runtime_metrics=runtime_metrics,
            status="ok",
        )
        log.info(
            "llm_chat_complete_completed",
            provider="local_gguf",
            model=self._model_filename_override or settings.llm_model_filename,
            max_tokens=max_tokens,
            enable_thinking=bool(_tmpl_kwargs.get("enable_thinking", False)),
            prompt_token_count=runtime_metrics.get("prompt_token_count"),
            prompt_token_count_source=runtime_metrics.get("prompt_token_count_source"),
            cached_token_count=runtime_metrics.get("cached_token_count"),
            prefix_cache_hit=runtime_metrics.get("prefix_cache_hit"),
            prompt_eval_tokens_per_second=runtime_metrics.get("prompt_eval_tokens_per_second"),
            generation_tokens_per_second=runtime_metrics.get("generation_tokens_per_second"),
            prompt_eval_duration_ms=runtime_metrics.get("prompt_eval_duration_ms"),
            generation_duration_ms=runtime_metrics.get("generation_duration_ms"),
            elapsed_ms=round(elapsed_ms, 1),
        )
        return {"choices": [{"message": {"content": "".join(content_parts)}}]}

    # -- Streaming generation -------------------------------------------------

    async def generate_stream(
        self,
        messages: list[dict[str, str]],
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        stop: list[str] | None = None,
        force_chatml: bool = False,
        timeout_seconds: float | None = None,
        chat_template_kwargs_override: dict[str, object] | None = None,
        timing_context: dict[str, object] | None = None,
    ) -> AsyncGenerator[str | tuple[str, object]]:
        # Stream generated tokens one at a time as an async generator.
        # Sends messages via handle_chat_completions in a background thread.
        # Tokens (OpenAI chat streaming delta format) are delivered to the
        # asyncio event loop via a thread-safe queue.
        #
        # Args:
        #   messages:        Chat messages (system, user, assistant turns).
        #   max_tokens:      Maximum tokens to generate. Defaults to config value.
        #   temperature:     Sampling temperature. Defaults to config value.
        #   top_p:           Nucleus sampling. Defaults to 1.0.
        #   stop:            Stop sequences to halt generation.
        #   force_chatml:    When True, use ChatML format for token-budget
        #                    estimation in _truncate_messages_to_fit only.
        #                    Has no effect on actual generation — the server
        #                    applies the GGUF template internally via
        #                    handle_chat_completions regardless of this flag.
        #                    Reasoning suppression is controlled by /no_think
        #                    in the user message (model_adapter.prepare_messages).
        #   timeout_seconds: Wall-clock generation timeout. Defaults to 120s.
        #
        # Yields:
        #   str — individual token strings as generated.
        #   tuple[str, dict] — ('__timeout__', ...) marker on timeout.
        #
        # Raises:
        #   LLMError: If messages are empty or generation fails.
        """generate stream."""
        if not messages:
            raise LLMError("Cannot generate from empty messages")

        max_tok = max_tokens if max_tokens is not None else settings.llm_max_tokens
        temp = temperature if temperature is not None else settings.llm_temperature
        top_p_val = 1.0 if top_p is None else top_p
        stop_seqs = stop if stop is not None else []
        wall_clock = 120.0 if timeout_seconds is None else float(timeout_seconds)

        profile = get_profile()
        context_len = get_effective_context_length(profile)
        merged_template_kwargs = _merge_chat_template_kwargs(
            _profile_chat_template_kwargs(profile),
            chat_template_kwargs_override,
        )
        think_enabled = bool(merged_template_kwargs.get("enable_thinking", False))
        _check_generation_model_memory_headroom(model_path=self.get_model_path(), stage="generate_stream")
        server = self._loaded_server

        truncated_messages, truncation_info = _truncate_messages_to_fit(
            chat_template=self._chat_template,
            messages=messages,
            context_length=context_len,
            max_tokens=max_tok,
            force_chatml=force_chatml,
        )

        if truncation_info["truncated"]:
            log.warning(
                "prompt_truncated",
                original_tokens=truncation_info["original_tokens"],
                final_tokens=truncation_info.get(
                    "final_tokens", truncation_info["original_tokens"]
                ),
                available_budget=truncation_info["available_budget"],
                history_messages_removed=truncation_info.get("history_messages_removed", 0),
                system_content_truncated=truncation_info.get("system_content_truncated", False),
                chunks_removed=truncation_info.get("chunks_removed", 0),
                warning=truncation_info.get("warning"),
            )

        messages = truncated_messages

        log.debug(
            "llm_streaming",
            messages_count=len(messages),
            max_tokens=max_tok,
            temperature=temp,
            top_p=top_p_val,
            timeout_seconds=wall_clock,
            context_length=context_len,
        )

        start = time.perf_counter()
        _record_generation_memory_snapshot(point="generation_start", timing_context=timing_context)
        token_count = 0
        first_token_ms: float | None = None
        first_token_at_s: float | None = None
        total_text_parts: list[str] = []
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[str | object] = asyncio.Queue()
        exception_holder: list[BaseException] = []
        cancel_event = threading.Event()
        timing_state: dict[str, object | None] = {"submit_ms": None, "submit_at_s": None}

        request_context = copy_context()
        worker = threading.Thread(
            target=request_context.run,
            args=(
                _run_stream_worker,
                server,
                messages,
                max_tok,
                temp,
                top_p_val,
                stop_seqs,
                chat_template_kwargs_override,
                loop,
                queue,
                exception_holder,
                cancel_event,
                start,
                timing_state,
                timing_context,
            ),
            name="llm-stream-worker",
            daemon=True,
        )
        worker.start()

        finish_reason: str | None = None
        timeout_occurred = False
        timeout_reason: str | None = None
        first_token_deadline_seconds = _resolve_first_token_deadline_seconds(
            wall_clock=wall_clock,
            profile_tps=float(getattr(profile, "generation_tokens_per_second", 12.0) or 12.0),
        )

        stripper = ThinkStrip()

        try:
            async for event_kind, payload in _stream_from_queue(
                queue,
                cancel_event,
                start=start,
                wall_clock=wall_clock,
                first_token_deadline_seconds=first_token_deadline_seconds,
                stripper=stripper,
                timing_context=timing_context,
            ):
                if event_kind == "timeout":
                    timeout_payload = payload if isinstance(payload, dict) else {}
                    timeout_reason_value = str(
                        normalize_timeout_reason(timeout_payload.get("reason"))
                    )
                    if timeout_reason_value == TimeoutReason.FIRST_TOKEN_WATCHDOG_TIMEOUT.value:
                        log.warning(
                            "llm_stream_first_token_watchdog_timeout",
                            elapsed_seconds=timeout_payload.get("elapsed_seconds"),
                            first_token_deadline_seconds=round(first_token_deadline_seconds, 1),
                            timeout_seconds=timeout_payload.get("timeout_seconds"),
                            tokens_generated=token_count,
                            msg=(
                                "No first token observed before watchdog deadline; stopping"
                                "generation"
                            ),
                        )
                    else:
                        log.warning(
                            "llm_stream_wall_clock_timeout",
                            elapsed_seconds=timeout_payload.get("elapsed_seconds"),
                            timeout_seconds=timeout_payload.get("timeout_seconds"),
                            tokens_generated=token_count,
                            msg="Hard timeout reached; stopping generation",
                        )
                    cancel_event.set()
                    timeout_occurred = True
                    timeout_reason = timeout_reason_value or TimeoutReason.UNKNOWN_TIMEOUT.value
                    break
                if event_kind == "finish_reason":
                    finish_reason = payload if isinstance(payload, str) else None
                    continue

                emit_text = str(payload)
                if first_token_ms is None:
                    first_token_ms = (time.perf_counter() - start) * 1000
                    first_token_at_s = time.time()
                    if timing_context is not None:
                        timing_context["runtime_first_token_at_s"] = first_token_at_s
                    _record_generation_memory_snapshot(
                        point="first_token",
                        timing_context=timing_context,
                    )
                token_count += 1
                total_text_parts.append(emit_text)
                yield emit_text

            if timeout_occurred:
                _record_generation_memory_snapshot(
                    point="watchdog_timeout",
                    timing_context=timing_context,
                )
                timeout_notice = (
                    f"\n\n[Response truncated: generation time limit ({int(wall_clock)}s) reached]"
                )
                total_text_parts.append(timeout_notice)
                yield timeout_notice
                yield (
                    StreamSignalTag.TIMEOUT,
                    {
                        "reason": timeout_reason or TimeoutReason.UNKNOWN_TIMEOUT.value,
                        "elapsed_seconds": round(time.perf_counter() - start, 1),
                        "timeout_seconds": wall_clock,
                    },
                )

            if token_count == 0 and not timeout_occurred and not cancel_event.is_set():
                log.warning(
                    "llm_stream_empty_completion_local_fallback",
                    msg="Streaming returned zero output; attempting non-stream completion fallback",
                )
                fallback_response = self.chat_complete(
                    messages=messages,
                    max_tokens=max_tok,
                    temperature=temp,
                    stop=stop_seqs or None,
                )
                fallback_text = str(
                    (
                        (fallback_response.get("choices") or [{}])[0]
                        .get("message", {})
                        .get("content")
                    )
                    or "",
                )
                if fallback_text:
                    fallback_stripper = ThinkStrip()
                    cleaned_fallback = (
                        fallback_stripper.feed(fallback_text) + fallback_stripper.flush()
                    )
                    cleaned_fallback = cleaned_fallback.strip()
                    if cleaned_fallback:
                        if first_token_ms is None:
                            first_token_ms = (time.perf_counter() - start) * 1000
                        token_count += 1
                        total_text_parts.append(cleaned_fallback)
                        yield cleaned_fallback
                if token_count == 0:
                    raise LLMError("Local model returned no response tokens")

            if token_count > 0 and not timeout_occurred and not exception_holder:
                _record_generation_memory_snapshot(
                    point="healthy_completion",
                    timing_context=timing_context,
                )

            if exception_holder:
                exc = exception_holder[0]
                if not isinstance(exc, Exception):
                    raise exc
                raise LLMError(f"LLM streaming failed: {exc}") from exc

        except RuntimeError as exc:  # pylint: disable=try-except-raise
            _raise_llm_streaming_failed(exc)

        except GeneratorExit:
            cancel_event.set()
            log.debug(
                "llm_stream_cancelled",
                tokens_generated=token_count,
                msg="Stream cancelled (GeneratorExit); worker signaled to stop",
            )
            return

        except asyncio.CancelledError:
            cancel_event.set()
            log.debug(
                "llm_stream_cancelled",
                tokens_generated=token_count,
                msg="Stream cancelled (CancelledError); worker signaled to stop",
            )
            raise

        finally:
            # Always attempt worker cleanup, including cancellation paths.
            # Do not block the event loop on long native worker joins.
            # On stop/cancel we signal the worker and move on quickly so the
            # chat request can emit terminal events and unregister promptly.
            if worker.is_alive():
                cancel_event.set()
                await asyncio.to_thread(worker.join, 0.25)
                if worker.is_alive():
                    log.info(
                        "llm_stream_worker_detached",
                        msg="Worker still running after cancellation signal; leaving it detached",
                        cancelled=cancel_event.is_set(),
                        timeout_occurred=timeout_occurred,
                    )

            elapsed_ms = (time.perf_counter() - start) * 1000
            request_submit_ms = timing_state.get("submit_ms")
            request_submit_at_s = timing_state.get("submit_at_s")
            runtime_frames = (
                timing_state.get("runtime_frames") if isinstance(timing_state.get("runtime_frames"), list) else []
            )
            runtime_metrics = _extract_runtime_metrics_from_chunks(
                runtime_frames if isinstance(runtime_frames, list) else [],
                prompt_token_estimate=count_tokens(_messages_to_prompt(self._chat_template, messages)),
                total_elapsed_ms=elapsed_ms,
                token_count=token_count,
            )
            timing_state["runtime_metrics"] = runtime_metrics
            if timing_context is not None:
                timing_context["payload_sent_to_model_runtime_at_s"] = (
                    request_submit_at_s if isinstance(request_submit_at_s, (int, float)) else None
                )
                timing_context["runtime_metrics"] = runtime_metrics
                if first_token_at_s is not None:
                    timing_context["runtime_first_token_at_s"] = first_token_at_s
            queue_wait_ms: float | None = None
            if isinstance(request_submit_ms, (int, float)) and first_token_ms is not None:
                queue_wait_ms = max(0.0, first_token_ms - float(request_submit_ms))
            log.info(
                "llm_stream_completed",
                messages_count=len(messages),
                tokens=token_count,
                output_length=len("".join(total_text_parts)),
                elapsed_ms=round(elapsed_ms, 1),
                first_token_ms=round(first_token_ms, 1) if first_token_ms is not None else None,
                max_tokens=max_tok,
                enable_thinking=think_enabled,
                prompt_token_count=runtime_metrics.get("prompt_token_count"),
                prompt_token_count_source=runtime_metrics.get("prompt_token_count_source"),
                cached_token_count=runtime_metrics.get("cached_token_count"),
                prefix_cache_hit=runtime_metrics.get("prefix_cache_hit"),
                prompt_eval_tokens_per_second=runtime_metrics.get("prompt_eval_tokens_per_second"),
                generation_tokens_per_second=runtime_metrics.get("generation_tokens_per_second"),
                finish_reason=finish_reason,
                cancelled=cancel_event.is_set(),
                timeout_occurred=timeout_occurred,
                timeout_reason=timeout_reason,
                provider="local_gguf",
            )
        yield (
            StreamSignalTag.STREAM_SUMMARY,
            {
                "token_count": token_count,
                "first_token_ms": round(first_token_ms, 1) if first_token_ms is not None else None,
                "total_elapsed_ms": round(elapsed_ms, 1),
                "submit_ms": round(request_submit_ms, 1) if isinstance(request_submit_ms, (int, float)) else None,
                "queue_wait_ms": round(queue_wait_ms, 1) if queue_wait_ms is not None else None,
                "submit_at_s": request_submit_at_s if isinstance(request_submit_at_s, (int, float)) else None,
                "first_token_at_s": first_token_at_s,
                "timeout_reason": timeout_reason,
                "finish_reason": finish_reason,
                "runtime_metrics": runtime_metrics,
            },
        )


class OllamaProvider:
    """Ollama-backed provider using /api/chat compatible streaming."""

    def __init__(self, model_id: str | None = None) -> None:
        """init  ."""
        self._base_url = (
            str(
                getattr(settings, "ollama_base_url", DEFAULT_OLLAMA_BASE_URL)
                or DEFAULT_OLLAMA_BASE_URL
            )
            .strip()
            .rstrip("/")
        )
        self._timeout_seconds = float(getattr(settings, "ollama_timeout_seconds", 120.0) or 120.0)
        self._model_id_override = str(model_id or "").strip().lower() or None

    @property
    def is_loaded(self) -> bool:
        # Ollama manages model lifecycle in its daemon process.
        """is loaded."""
        return True

    def unload(self) -> None:
        """unload."""
        return

    def count_tokens(self, text: str) -> int:
        """count tokens."""
        return count_tokens(text)

    def _get_model_path(self) -> Path:
        """get model path."""
        if self._model_id_override:
            alias_filenames = get_model_alias_filenames(self._model_id_override)
            if alias_filenames:
                return settings.models_dir / alias_filenames[0]
        return settings.models_dir / settings.llm_model_filename

    def _download_model(
        self,
        target_path: Path,
        repo_id: str | None = None,
        filename: str | None = None,
        revision: str | None = None,
        expected_sha256: str | None = None,
        progress_callback: Callable[[int, int | None, float], None] | None = None,
        cancel_event: threading.Event | None = None,
    ) -> None:
        """download model."""
        _ = (
            target_path,
            repo_id,
            filename,
            revision,
            expected_sha256,
            progress_callback,
            cancel_event,
        )
        raise LLMError("Ollama provider does not support local GGUF download")

    def _resolve_model(self) -> str:
        """resolve model."""
        model = self._model_id_override or str(getattr(settings, "llm_model_id", "") or "").strip()
        if model:
            return model
        raise LLMError("Ollama provider requires llm_model_id to be set")

    def _post_chat(
        self,
        *,
        payload: dict,
        stream: bool,
    ) -> dict | list[dict]:
        """post chat."""
        req = urllib.request.Request(
            url=f"{self._base_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self._timeout_seconds) as resp:  # noqa: S310
                if stream:
                    events: list[dict] = []
                    for raw_line in resp:
                        line = raw_line.decode("utf-8", errors="replace").strip()
                        if not line:
                            continue
                        with suppress(json.JSONDecodeError, ValueError):
                            data = json.loads(line)
                            if isinstance(data, dict):
                                events.append(data)
                    return events
                body = resp.read().decode("utf-8", errors="replace")
                parsed = json.loads(body) if body else {}
                if not isinstance(parsed, dict):
                    raise LLMError("Invalid Ollama response format")
                return parsed
        except urllib.error.HTTPError as exc:
            detail = ""
            with suppress(Exception):
                detail = exc.read().decode("utf-8", errors="replace").strip()
            raise LLMError(f"Ollama HTTP error ({exc.code}): {detail or exc.reason}") from exc
        except urllib.error.URLError as exc:
            raise LLMError(f"Ollama connection failed: {exc.reason}") from exc
        except TimeoutError as exc:
            raise LLMError("Ollama request timed out") from exc
        except json.JSONDecodeError as exc:
            raise LLMError(f"Invalid Ollama JSON response: {exc}") from exc

    def chat_complete(
        self,
        messages: list[dict],
        max_tokens: int = 400,
        temperature: float = 0.0,
        stop: list[str] | None = None,
        response_format: dict | None = None,
        chat_template_kwargs_override: dict[str, object] | None = None,
    ) -> dict:
        """chat complete."""
        start = time.perf_counter()
        model = self._resolve_model()
        merged_template_kwargs = _merge_chat_template_kwargs(
            _profile_chat_template_kwargs(get_profile()),
            chat_template_kwargs_override,
        )
        think_enabled = bool(merged_template_kwargs.get("enable_thinking", False))
        options: dict[str, object] = {
            "num_predict": int(max_tokens),
            "temperature": float(temperature),
        }
        if stop:
            options["stop"] = stop
        payload: dict[str, object] = {
            "model": model,
            "messages": messages,
            "stream": False,
            "think": think_enabled,
            "options": options,
        }
        if response_format is not None:
            payload["format"] = response_format

        try:
            parsed = self._post_chat(payload=payload, stream=False)
        except Exception as exc:
            _record_runtime_call_probe(
                call_kind="chat_complete",
                max_tokens=max_tokens,
                enable_thinking=think_enabled,
                duration_ms=(time.perf_counter() - start) * 1000.0,
                prompt_token_count=None,
                completion_token_count=None,
                runtime_metrics=None,
                status="error",
                error=str(exc),
            )
            raise
        if not isinstance(parsed, dict):
            raise LLMError("Invalid Ollama response format")
        content = str((parsed.get("message") or {}).get("content") or "")
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        runtime_metrics = _extract_runtime_metrics_from_chunks(
            [parsed],
            prompt_token_estimate=None,
            total_elapsed_ms=elapsed_ms,
            token_count=None,
        )
        _record_runtime_call_probe(
            call_kind="chat_complete",
            max_tokens=max_tokens,
            enable_thinking=think_enabled,
            duration_ms=elapsed_ms,
            prompt_token_count=(
                int(runtime_metrics["prompt_token_count"])
                if isinstance(runtime_metrics.get("prompt_token_count"), int)
                else None
            ),
            completion_token_count=(
                int(runtime_metrics["completion_token_count"])
                if isinstance(runtime_metrics.get("completion_token_count"), int)
                else None
            ),
            runtime_metrics=runtime_metrics,
            status="ok",
        )
        log.info(
            "llm_chat_complete_completed",
            provider="ollama",
            model=model,
            max_tokens=max_tokens,
            enable_thinking=think_enabled,
            prompt_token_count=runtime_metrics.get("prompt_token_count"),
            prompt_token_count_source=runtime_metrics.get("prompt_token_count_source"),
            cached_token_count=runtime_metrics.get("cached_token_count"),
            prefix_cache_hit=runtime_metrics.get("prefix_cache_hit"),
            prompt_eval_tokens_per_second=runtime_metrics.get("prompt_eval_tokens_per_second"),
            generation_tokens_per_second=runtime_metrics.get("generation_tokens_per_second"),
            prompt_eval_duration_ms=runtime_metrics.get("prompt_eval_duration_ms"),
            generation_duration_ms=runtime_metrics.get("generation_duration_ms"),
            elapsed_ms=round(elapsed_ms, 1),
        )
        return {"choices": [{"message": {"content": content}}]}

    async def generate_stream(
        self,
        messages: list[dict[str, str]],
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        stop: list[str] | None = None,
        force_chatml: bool = False,
        timeout_seconds: float | None = None,
        chat_template_kwargs_override: dict[str, object] | None = None,
        timing_context: dict[str, object] | None = None,
    ) -> AsyncGenerator[str | tuple[str, object]]:
        """generate stream."""
        if not messages:
            raise LLMError("Cannot generate from empty messages")

        max_tok = max_tokens if max_tokens is not None else settings.llm_max_tokens
        temp = temperature if temperature is not None else settings.llm_temperature
        top_p_val = 1.0 if top_p is None else top_p
        stop_seqs = stop if stop is not None else []
        wall_clock = 120.0 if timeout_seconds is None else float(timeout_seconds)
        merged_template_kwargs = _merge_chat_template_kwargs(
            _profile_chat_template_kwargs(get_profile()),
            chat_template_kwargs_override,
        )
        think_enabled = bool(merged_template_kwargs.get("enable_thinking", False))

        profile = get_profile()
        context_len = get_effective_context_length(profile)

        truncated_messages, truncation_info = _truncate_messages_to_fit(
            chat_template="",
            messages=messages,
            context_length=context_len,
            max_tokens=max_tok,
            force_chatml=force_chatml if force_chatml else True,
        )
        if truncation_info["truncated"]:
            log.warning(
                "prompt_truncated",
                original_tokens=truncation_info["original_tokens"],
                final_tokens=truncation_info.get(
                    "final_tokens", truncation_info["original_tokens"]
                ),
                available_budget=truncation_info["available_budget"],
                history_messages_removed=truncation_info.get("history_messages_removed", 0),
                system_content_truncated=truncation_info.get("system_content_truncated", False),
                chunks_removed=truncation_info.get("chunks_removed", 0),
                warning=truncation_info.get("warning"),
            )
        messages = truncated_messages

        queue: asyncio.Queue[str | object] = asyncio.Queue()
        loop = asyncio.get_running_loop()
        exception_holder: list[BaseException] = []
        cancel_event = threading.Event()

        model = self._resolve_model()
        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            "think": think_enabled,
            "options": {
                "num_predict": int(max_tok),
                "temperature": float(temp),
                "top_p": float(top_p_val),
                **({"stop": stop_seqs} if stop_seqs else {}),
            },
        }
        raw_frame_count = 0
        parsed_frame_count = 0
        non_dict_frame_count = 0
        json_decode_error_count = 0
        empty_content_frame_count = 0
        raw_frame_samples: list[str] = []
        done_frame_snapshot: dict[str, object] | None = None
        request_submit_ms: float | None = None
        request_submit_at_s: float | None = None
        start = time.perf_counter()
        _record_generation_memory_snapshot(point="generation_start", timing_context=timing_context)

        request_context = copy_context()

        def _worker() -> None:
            """worker."""
            nonlocal request_submit_ms
            nonlocal request_submit_at_s
            nonlocal raw_frame_count, parsed_frame_count, non_dict_frame_count
            nonlocal json_decode_error_count, empty_content_frame_count
            nonlocal raw_frame_samples, done_frame_snapshot
            finish_reason: str | None = None
            req = urllib.request.Request(
                url=f"{self._base_url}/api/chat",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                request_submit_ms = (time.perf_counter() - start) * 1000
                request_submit_at_s = time.time()
                with urllib.request.urlopen(req, timeout=self._timeout_seconds) as resp:
                    # noqa: S310
                    for raw_line in resp:
                        raw_frame_count += 1
                        if cancel_event.is_set():
                            finish_reason = "cancelled"
                            break
                        line = raw_line.decode("utf-8", errors="replace").strip()
                        if len(raw_frame_samples) < 5:
                            raw_frame_samples.append(line[:300])
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                        except (json.JSONDecodeError, ValueError):
                            json_decode_error_count += 1
                            continue
                        parsed_frame_count += 1
                        if not isinstance(data, dict):
                            non_dict_frame_count += 1
                            continue
                        token = str((data.get("message") or {}).get("content") or "")
                        if token:
                            loop.call_soon_threadsafe(queue.put_nowait, token)
                        else:
                            empty_content_frame_count += 1
                        if data.get("done") is True:
                            done_frame_snapshot = {
                                "done": data.get("done"),
                                "done_reason": data.get("done_reason"),
                                "has_message": isinstance(data.get("message"), dict),
                                "content_length": len(token),
                            }
                            finish_reason = _normalize_finish_reason(
                                str(data.get("done_reason") or "stop")
                            )
                            break
            except urllib.error.HTTPError as exc:
                detail = ""
                with suppress(Exception):
                    detail = exc.read().decode("utf-8", errors="replace").strip()
                exception_holder.append(
                    LLMError(f"Ollama HTTP error ({exc.code}): {detail or exc.reason}")
                )
            except urllib.error.URLError as exc:
                exception_holder.append(LLMError(f"Ollama connection failed: {exc.reason}"))
            except TimeoutError:
                exception_holder.append(LLMError("Ollama request timed out"))
            except (
                AttributeError,
                IndexError,
                KeyError,
                OSError,
                RuntimeError,
                TypeError,
                UnicodeError,
                ValueError,
            ) as exc:
                exception_holder.append(exc)
            finally:
                loop.call_soon_threadsafe(
                    queue.put_nowait, (StreamSignalTag.FINISH_REASON, finish_reason)
                )
            with suppress(RuntimeError):
                loop.call_soon_threadsafe(queue.put_nowait, _STREAM_END)

        worker = threading.Thread(
            target=request_context.run,
            args=(_worker,),
            name="ollama-stream-worker",
            daemon=True,
        )
        worker.start()
        token_count = 0
        first_token_ms: float | None = None
        first_token_at_s: float | None = None
        total_text_parts: list[str] = []
        finish_reason: str | None = None
        timeout_occurred = False
        timeout_reason: str | None = None
        first_token_deadline_seconds = _resolve_first_token_deadline_seconds(
            wall_clock=wall_clock,
            profile_tps=float(getattr(profile, "generation_tokens_per_second", 12.0) or 12.0),
        )
        stripper = ThinkStrip()

        try:
            async for event_kind, payload in _stream_from_queue(
                queue,
                cancel_event,
                start=start,
                wall_clock=wall_clock,
                first_token_deadline_seconds=first_token_deadline_seconds,
                stripper=stripper,
                timing_context=timing_context,
            ):
                if event_kind == "timeout":
                    timeout_payload = payload if isinstance(payload, dict) else {}
                    timeout_reason_value = str(
                        normalize_timeout_reason(timeout_payload.get("reason"))
                    )
                    cancel_event.set()
                    timeout_occurred = True
                    timeout_reason = timeout_reason_value or TimeoutReason.UNKNOWN_TIMEOUT.value
                    _record_generation_memory_snapshot(
                        point="watchdog_timeout",
                        timing_context=timing_context,
                    )
                    break
                if event_kind == "finish_reason":
                    finish_reason = payload if isinstance(payload, str) else None
                    continue

                emit_text = str(payload)
                if first_token_ms is None:
                    first_token_ms = (time.perf_counter() - start) * 1000
                    if timing_context is not None:
                        first_token_at_s = time.time()
                        timing_context["runtime_first_token_at_s"] = first_token_at_s
                    _record_generation_memory_snapshot(
                        point="first_token",
                        timing_context=timing_context,
                    )
                token_count += 1
                total_text_parts.append(emit_text)
                yield emit_text

            if timeout_occurred:
                timeout_notice = (
                    f"\n\n[Response truncated: generation time limit ({int(wall_clock)}s) reached]"
                )
                total_text_parts.append(timeout_notice)
                yield timeout_notice
                yield (
                    StreamSignalTag.TIMEOUT,
                    {
                        "reason": timeout_reason or TimeoutReason.UNKNOWN_TIMEOUT.value,
                        "elapsed_seconds": round(time.perf_counter() - start, 1),
                        "timeout_seconds": wall_clock,
                    },
                )

            if exception_holder:
                exc = exception_holder[0]
                if isinstance(exc, LLMError):
                    raise exc
                raise LLMError(f"LLM streaming failed: {exc}") from exc
            if token_count == 0 and not timeout_occurred:
                log.warning(
                    "ollama_stream_empty_completion",
                    model=model,
                    raw_frame_count=raw_frame_count,
                    parsed_frame_count=parsed_frame_count,
                    non_dict_frame_count=non_dict_frame_count,
                    json_decode_error_count=json_decode_error_count,
                    empty_content_frame_count=empty_content_frame_count,
                    done_frame_snapshot=done_frame_snapshot,
                    raw_frame_samples=raw_frame_samples,
                )
                raise LLMError("Ollama returned no response tokens")
            if token_count > 0 and not timeout_occurred and not exception_holder:
                _record_generation_memory_snapshot(
                    point="healthy_completion",
                    timing_context=timing_context,
                )
        finally:
            if worker.is_alive():
                cancel_event.set()
                await asyncio.to_thread(worker.join, 0.25)

            queue_wait_ms: float | None = None
            if request_submit_ms is not None and first_token_ms is not None:
                queue_wait_ms = max(0.0, first_token_ms - request_submit_ms)
            elapsed_ms = (time.perf_counter() - start) * 1000
            runtime_metrics = _extract_runtime_metrics_from_chunks(
                [done_frame_snapshot] if done_frame_snapshot is not None else [],
                prompt_token_estimate=None,
                total_elapsed_ms=elapsed_ms,
                token_count=token_count,
            )
            if timing_context is not None:
                timing_context["payload_sent_to_model_runtime_at_s"] = request_submit_at_s
                timing_context["runtime_metrics"] = runtime_metrics
            log.info(
                "llm_stream_completed",
                provider="ollama",
                messages_count=len(messages),
                tokens=token_count,
                output_length=len("".join(total_text_parts)),
                elapsed_ms=round(elapsed_ms, 1),
                first_token_ms=round(first_token_ms, 1) if first_token_ms is not None else None,
                max_tokens=max_tok,
                enable_thinking=think_enabled,
                prompt_token_count=runtime_metrics.get("prompt_token_count"),
                prompt_token_count_source=runtime_metrics.get("prompt_token_count_source"),
                cached_token_count=runtime_metrics.get("cached_token_count"),
                prefix_cache_hit=runtime_metrics.get("prefix_cache_hit"),
                prompt_eval_tokens_per_second=runtime_metrics.get("prompt_eval_tokens_per_second"),
                generation_tokens_per_second=runtime_metrics.get("generation_tokens_per_second"),
                finish_reason=finish_reason,
                cancelled=cancel_event.is_set(),
                timeout_occurred=timeout_occurred,
                timeout_reason=timeout_reason,
            )
        yield (
            StreamSignalTag.STREAM_SUMMARY,
            {
                "token_count": token_count,
                "first_token_ms": round(first_token_ms, 1) if first_token_ms is not None else None,
                "total_elapsed_ms": round(elapsed_ms, 1),
                "submit_ms": round(request_submit_ms, 1) if request_submit_ms is not None else None,
                "queue_wait_ms": round(queue_wait_ms, 1) if queue_wait_ms is not None else None,
                "timeout_reason": timeout_reason,
                "finish_reason": finish_reason,
            },
        )


class LLMEngine:
    """
    Provider-facade for inference runtime.
    Keeps legacy LLMEngine API stable while routing calls to a concrete provider.
    """

    def __init__(
        self,
        provider_name: str | None = None,
        *,
        model_id: str | None = None,
        model_filename: str | None = None,
        model_dir: Path | None = None,
    ) -> None:
        """init  ."""
        provider_name = (
            str(provider_name or getattr(settings, "llm_provider", "local_gguf") or "local_gguf")
            .strip()
            .lower()
        )
        if provider_name == "local_gguf":
            self._provider = XllamaCppProvider(model_filename=model_filename, model_dir=model_dir)
        elif provider_name == "ollama":
            self._provider = OllamaProvider(model_id=model_id)
        else:
            raise LLMError(f"Unsupported llm_provider: {provider_name}")
        self.provider_name = provider_name

    # Backward-compatible private hooks relied on by runtime/tests.
    @property
    def _server(self) -> object | None:
        """server."""
        if isinstance(self._provider, XllamaCppProvider):
            return self._provider.server
        return None

    @_server.setter
    def _server(self, value: object | None) -> None:
        """server."""
        if isinstance(self._provider, XllamaCppProvider):
            self._provider.server = value

    @property
    def _chat_template(self) -> str:
        """chat template."""
        if isinstance(self._provider, XllamaCppProvider):
            return self._provider.chat_template
        return ""

    @_chat_template.setter
    def _chat_template(self, value: str) -> None:
        """chat template."""
        if isinstance(self._provider, XllamaCppProvider):
            self._provider.chat_template = value

    @property
    def is_loaded(self) -> bool:
        """is loaded."""
        return self._provider.is_loaded

    def unload(self) -> None:
        """unload."""
        self._provider.unload()

    def count_tokens(self, text: str) -> int:
        """count tokens."""
        return self._provider.count_tokens(text)

    def get_model_path(self) -> Path:
        """get model path."""
        return self._provider.get_model_path()

    def _get_model_path(self) -> Path:
        """get model path."""
        return self.get_model_path()

    def download_model(
        self,
        target_path: Path,
        repo_id: str | None = None,
        filename: str | None = None,
        revision: str | None = None,
        expected_sha256: str | None = None,
        progress_callback: Callable[[int, int | None, float], None] | None = None,
        cancel_event: threading.Event | None = None,
    ) -> None:
        """download model."""
        self._provider.download_model(
            target_path=target_path,
            repo_id=repo_id,
            filename=filename,
            revision=revision,
            expected_sha256=expected_sha256,
            progress_callback=progress_callback,
            cancel_event=cancel_event,
        )

    def _download_model(
        self,
        target_path: Path,
        repo_id: str | None = None,
        filename: str | None = None,
        revision: str | None = None,
        expected_sha256: str | None = None,
        progress_callback: Callable[[int, int | None, float], None] | None = None,
        cancel_event: threading.Event | None = None,
    ) -> None:
        """download model."""
        self.download_model(
            target_path=target_path,
            repo_id=repo_id,
            filename=filename,
            revision=revision,
            expected_sha256=expected_sha256,
            progress_callback=progress_callback,
            cancel_event=cancel_event,
        )

    def load_model(self, model_filename: str | None = None) -> None:
        """load model."""
        if isinstance(self._provider, XllamaCppProvider):
            self._provider.load_model(model_filename=model_filename)
            return
        raise LLMError(f'Provider "{self.provider_name}" does not support in-process model loading')

    def _load_model(self, model_filename: str | None = None) -> None:
        """load model."""
        self.load_model(model_filename=model_filename)

    def chat_complete(
        self,
        messages: list[dict],
        max_tokens: int = 400,
        temperature: float = 0.0,
        stop: list[str] | None = None,
        response_format: dict | None = None,
        chat_template_kwargs_override: dict[str, object] | None = None,
    ) -> dict:
        """chat complete."""
        return self._provider.chat_complete(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
            response_format=response_format,
            chat_template_kwargs_override=chat_template_kwargs_override,
        )

    async def generate_stream(
        self,
        messages: list[dict[str, str]],
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        stop: list[str] | None = None,
        force_chatml: bool = False,
        timeout_seconds: float | None = None,
        chat_template_kwargs_override: dict[str, object] | None = None,
        timing_context: dict[str, object] | None = None,
    ) -> AsyncGenerator[str | tuple[str, object]]:
        """generate stream."""
        async for item in self._provider.generate_stream(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
            force_chatml=force_chatml,
            timeout_seconds=timeout_seconds,
            chat_template_kwargs_override=chat_template_kwargs_override,
            timing_context=timing_context,
        ):
            yield item


# ==============================================================================
# Module-level singleton
# ==============================================================================

llm_engine = LLMEngine()
