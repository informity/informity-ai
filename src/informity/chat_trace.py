# ==============================================================================
# Informity AI — Chat Trace Logging
# Writes a per-chat, per-message trace log as a single JSON object file
# for debugging and LLM-assisted analysis of relevance and accuracy.
# Only active when settings.chat_trace_logging is True.
#
# Format: One JSON object per file (pretty-printed).
# User chats: {app_data_dir}/chats/{chat_id}/{message_id}.json
# Evaluation: {app_data_dir}/diagnostics/runs/{run_id}/traces/{chat_id}--{message_id}.json
# Each file contains one complete trace entry with timestamp, chat_id, message_id, and steps.
# ==============================================================================

from __future__ import annotations

import asyncio
import os
import tempfile
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import structlog
from structlog.contextvars import get_contextvars

from informity.config import DirNames, get_chat_trace_logging, settings
from informity.llm.types import DiagnosticsQueryType
from informity.utils.directory_utils import ensure_directory, ensure_file_directory
from informity.utils.json_utils import serialize_trace

# ==============================================================================
# Logger
# ==============================================================================

log = structlog.get_logger(__name__)

# Trace sanitization constants
MAX_TRACE_SECTIONS = 50  # Maximum number of sections to include in trace sanitization
MAX_TRACE_STRING_LENGTH = 8000  # Maximum string length before truncation in trace logs
MAX_TRACE_LIST_ITEM_LENGTH = 2000  # Maximum length of list items in trace logs
TRACE_SCHEMA_NAME = 'informity.chat_trace'
TRACE_SCHEMA_VERSION = 1
TRACE_SUMMARY_SCHEMA_NAME = 'informity.chat_trace.summary'
TRACE_SUMMARY_SCHEMA_VERSION = 1
_TRACE_PRUNE_INTERVAL_SECONDS = 3600.0
_SECONDS_PER_DAY = 86400
_TRACE_REDACTION_MODE_DEFAULT = 'minimal'
_TRACE_REDACTION_MODE_OPTIONS = {'off', 'minimal', 'strict'}
_TRACE_RETENTION_DAYS_DEFAULT = 30
_SENSITIVE_TRACE_KEYS = (
    'question',
    'answer',
    'content',
    'prompt',
    'chunk',
    'source',
    'message',
)

# ==============================================================================
# Lock for serializing trace file writes (avoid interleaved blocks)
# ==============================================================================

_trace_write_lock: asyncio.Lock | None = None
_last_trace_prune_ts: dict[str, float] = {}
_trace_prune_state_lock = threading.Lock()


def _get_trace_lock() -> asyncio.Lock:
    global _trace_write_lock
    if _trace_write_lock is None:
        _trace_write_lock = asyncio.Lock()
    return _trace_write_lock


def _get_redaction_mode() -> str:
    mode = str(getattr(settings, 'chat_trace_redaction_mode', _TRACE_REDACTION_MODE_DEFAULT)).strip().lower()
    if mode in _TRACE_REDACTION_MODE_OPTIONS:
        return mode
    return _TRACE_REDACTION_MODE_DEFAULT


def _is_sensitive_key(key: str) -> bool:
    key_lower = key.strip().lower()
    return any(marker in key_lower for marker in _SENSITIVE_TRACE_KEYS)


def _sanitize_string(value: str, key: str, mode: str) -> str:
    if mode == 'off':
        return value if len(value) <= MAX_TRACE_STRING_LENGTH else value[:MAX_TRACE_STRING_LENGTH] + '... (truncated)'
    if not _is_sensitive_key(key):
        return value if len(value) <= MAX_TRACE_STRING_LENGTH else value[:MAX_TRACE_STRING_LENGTH] + '... (truncated)'
    if mode == 'strict':
        return f'[REDACTED length={len(value)}]'
    # minimal
    preview_limit = min(500, MAX_TRACE_STRING_LENGTH)
    preview = value[:preview_limit]
    return preview + (f'... [REDACTED_TAIL length={len(value)}]' if len(value) > preview_limit else '')


def _get_trace_retention_days(chat_type: str) -> int:
    if chat_type == 'evaluation':
        return int(getattr(settings, 'chat_trace_evaluation_retention_days', _TRACE_RETENTION_DAYS_DEFAULT))
    return int(getattr(settings, 'chat_trace_user_retention_days', _TRACE_RETENTION_DAYS_DEFAULT))


async def _maybe_prune_traces(chat_type: str, base_dir: Path) -> None:
    retention_days = _get_trace_retention_days(chat_type)
    if retention_days <= 0 or not base_dir.exists():
        return

    now_mono = time.monotonic()
    base_key = str(base_dir)
    with _trace_prune_state_lock:
        last_pruned_at = _last_trace_prune_ts.get(base_key, 0.0)
        if (now_mono - last_pruned_at) < _TRACE_PRUNE_INTERVAL_SECONDS:
            return
        _last_trace_prune_ts[base_key] = now_mono

    cutoff_ts = (datetime.now(UTC).timestamp() - (retention_days * _SECONDS_PER_DAY))

    def _prune_sync() -> int:
        deleted_count = 0
        if chat_type == 'evaluation':
            # Retain only trace files under runs/*/traces/.
            json_files = [p for p in base_dir.rglob('*.json') if '/traces/' in str(p).replace('\\', '/')]
        else:
            json_files = list(base_dir.rglob('*.json'))
        for file_path in json_files:
            try:
                if file_path.stat().st_mtime < cutoff_ts:
                    file_path.unlink(missing_ok=True)
                    deleted_count += 1
            except OSError:
                continue
        return deleted_count

    deleted = await asyncio.to_thread(_prune_sync)
    if deleted > 0:
        log.info(
            'trace_retention_pruned',
            chat_type=chat_type,
            base_dir=str(base_dir),
            retention_days=retention_days,
            files_deleted=deleted,
        )

# ==============================================================================
# Trace writer protocol and implementation
# ==============================================================================


@runtime_checkable
class TraceWriter(Protocol):
    """Protocol for recording trace steps. Flush is async and done via flush_trace_writer()."""

    def record(self, step: str, data: dict[str, Any]) -> None:
        """Record one trace step (e.g. 'request', 'embed', 'vector_search')."""
        ...

    def get_summary_envelope(self) -> dict[str, Any]:
        """Return stable trace summary envelope for diagnostics consumers."""
        ...


class _ChatTraceWriter:
    """Collects trace steps for one chat message and writes one JSON object to the log file."""

    def __init__(self, chat_id: str, message_id: str, chat_type: str = 'user', run_id: str | None = None) -> None:
        self._chat_id  = chat_id
        self._message_id = message_id
        self._chat_type = chat_type  # 'user' or 'evaluation'
        self._run_id = run_id  # Optional run_id for evaluation runs
        self._steps: list[tuple[str, dict[str, Any]]] = []
        self._started_at = datetime.now(UTC).isoformat()
        # Snapshot correlation context at creation time so trace survives contextvars cleanup.
        self._context = dict(get_contextvars())

    def record(self, step: str, data: dict[str, Any]) -> None:
        self._steps.append((step, data))

    def get_sections(self) -> dict[str, Any]:
        """
        Get all recorded trace sections as a dictionary.

        Returns:
            Dictionary mapping step names to their data
        """
        sections: dict[str, Any] = {}
        for step_name, payload in self._steps:
            sections[step_name] = payload
        return sections

    def _get_latest_step_payload(self, step_name: str) -> dict[str, Any]:
        for current_step_name, payload in reversed(self._steps):
            if current_step_name == step_name and isinstance(payload, dict):
                return payload
        return {}

    def _coerce_non_negative_int(self, value: Any) -> int | None:
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, int):
            return value if value >= 0 else 0
        if isinstance(value, float):
            as_int = int(value)
            return as_int if as_int >= 0 else 0
        return None

    def _coerce_non_negative_float(self, value: Any) -> float | None:
        if isinstance(value, bool):
            return float(value)
        if isinstance(value, (int, float)):
            as_float = float(value)
            return as_float if as_float >= 0 else 0.0
        return None

    def _coerce_string(self, value: Any) -> str | None:
        if isinstance(value, str):
            stripped = value.strip()
            return stripped if stripped else None
        return None

    def _coerce_resource_snapshot(self, value: Any) -> dict[str, Any] | None:
        if not isinstance(value, dict):
            return None
        snapshot: dict[str, Any] = {}
        for key in (
            'captured_at_epoch_ms',
            'system_cpu_percent',
            'process_cpu_percent',
            'process_rss_mb',
            'process_vms_mb',
            'system_memory_used_percent',
            'system_memory_available_mb',
            'system_memory_used_mb',
            'logical_cpu_count',
            'capture_error',
        ):
            item = value.get(key)
            if item is not None:
                snapshot[key] = item
        return snapshot if snapshot else None

    def _coerce_resource_metrics(self, value: Any) -> dict[str, Any] | None:
        if not isinstance(value, dict):
            return None
        before_snapshot = self._coerce_resource_snapshot(value.get('before'))
        after_snapshot = self._coerce_resource_snapshot(value.get('after'))
        delta_value = value.get('delta')
        delta_payload = delta_value if isinstance(delta_value, dict) and delta_value else None
        payload: dict[str, Any] = {}
        if before_snapshot is not None:
            payload['before'] = before_snapshot
        if after_snapshot is not None:
            payload['after'] = after_snapshot
        if delta_payload is not None:
            payload['delta'] = delta_payload
        return payload if payload else None

    def _build_summary_envelope(self) -> dict[str, Any]:
        request = self._get_latest_step_payload('request')
        intent = self._get_latest_step_payload('intent')
        retrieval = self._get_latest_step_payload('retrieval')
        llm = self._get_latest_step_payload('llm')
        sources = self._get_latest_step_payload('sources')
        response = self._get_latest_step_payload('response')
        response_cancelled = self._get_latest_step_payload('response_cancelled')
        response_error = self._get_latest_step_payload('response_error')

        intent_value = self._coerce_string(intent.get('intent'))
        subtype_value = self._coerce_string(intent.get('subtype'))
        query_type = self._coerce_string(intent.get('query_type')) or DiagnosticsQueryType.UNKNOWN.value

        raw_chunks_count = self._coerce_non_negative_int(retrieval.get('raw_chunks_count')) or 0
        matching_files = self._coerce_non_negative_int(retrieval.get('matching_files'))
        files_covered_after_fallback = self._coerce_non_negative_int(retrieval.get('files_covered_after_fallback'))

        llm_total_elapsed_ms = self._coerce_non_negative_float(llm.get('total_elapsed_ms'))
        llm_token_count = self._coerce_non_negative_int(llm.get('token_count'))

        sources_count = self._coerce_non_negative_int(sources.get('count'))
        response_sources_count = self._coerce_non_negative_int(response.get('sources_count'))
        effective_sources_count = (
            sources_count
            if sources_count is not None
            else (response_sources_count if response_sources_count is not None else 0)
        )

        answer_length = self._coerce_non_negative_int(response.get('answer_length')) or 0
        display_answer_length = self._coerce_non_negative_int(response.get('display_answer_length')) or 0

        response_error_text = self._coerce_string(response_error.get('error'))
        cancelled_stopped_by_user = bool(response_cancelled.get('stopped_by_user')) if response_cancelled else False
        request_resource_snapshot = self._coerce_resource_snapshot(request.get('resource_snapshot'))
        response_resource_metrics = self._coerce_resource_metrics(response.get('resource_metrics'))
        cancelled_resource_metrics = self._coerce_resource_metrics(response_cancelled.get('resource_metrics'))
        errored_resource_metrics = self._coerce_resource_metrics(response_error.get('resource_metrics'))
        effective_resource_metrics = (
            response_resource_metrics
            or cancelled_resource_metrics
            or errored_resource_metrics
        )

        return {
            'schema': TRACE_SUMMARY_SCHEMA_NAME,
            'summary_version': TRACE_SUMMARY_SCHEMA_VERSION,
            'intent': {
                'intent': intent_value,
                'subtype': subtype_value,
                'query_type': query_type,
            },
            'retrieval': {
                'raw_chunks_count': raw_chunks_count,
                'matching_files': matching_files,
                'files_covered_after_fallback': files_covered_after_fallback,
            },
            'llm': {
                'total_elapsed_ms': llm_total_elapsed_ms,
                'token_count': llm_token_count,
            },
            'sources': {
                'count': effective_sources_count,
            },
            'response': {
                'answer_length': answer_length,
                'display_answer_length': display_answer_length,
                'sources_count': response_sources_count,
            },
            'status': {
                'has_response_error': response_error_text is not None,
                'response_error': response_error_text,
                'response_cancelled': bool(response_cancelled),
                'stopped_by_user': cancelled_stopped_by_user,
            },
            'diagnostics': {
                'query_type': query_type,
                'raw_chunks_count': raw_chunks_count,
                'sources_count': effective_sources_count,
                'generation_seconds': (llm_total_elapsed_ms / 1000.0) if llm_total_elapsed_ms is not None else None,
                'answer_length': answer_length,
                'resource_snapshot_start': request_resource_snapshot,
                'resource_metrics': effective_resource_metrics,
            },
        }

    def get_summary_envelope(self) -> dict[str, Any]:
        return self._build_summary_envelope()

    def _build_steps(self) -> list[dict[str, Any]]:
        # Ordered step history; avoids information loss when a step name repeats.
        steps: list[dict[str, Any]] = []
        for idx, (step_name, payload) in enumerate(self._steps):
            steps.append({
                'index': idx,
                'name': step_name,
                'data': _sanitize_for_trace(payload),
            })
        return steps

    def _format_json(self) -> dict[str, Any]:
        """Format trace data as a JSON-serializable dict."""
        steps = self._build_steps()

        return {
            'trace_version': TRACE_SCHEMA_VERSION,
            'schema': TRACE_SCHEMA_NAME,
            'started_at': self._started_at,
            'flushed_at': datetime.now(UTC).isoformat(),
            'type': self._chat_type,  # 'user' or 'evaluation'
            'chat_id': self._chat_id,
            'message_id': self._message_id,
            'run_id': self._run_id,
            'correlation': {
                'request_id': self._context.get('request_id'),
                'operation_id': self._context.get('operation_id'),
                'operation_type': self._context.get('operation_type'),
            },
            'summary': self._build_summary_envelope(),
            'step_count': len(steps),
            'steps': steps,
        }

    async def flush(self) -> None:
        # Evaluation chats must have a run_id - skip if missing
        if self._chat_type == 'evaluation' and not self._run_id:
            log.warning(
                'evaluation_trace_skipped_no_run_id',
                chat_id=self._chat_id,
                message_id=self._message_id,
            )
            return

        # For evaluation runs with run_id, always write a trace file (even if no steps)
        # so every query×model produces a trace for diagnostics analysis.
        # User traces only flush when at least one step is recorded.
        if not self._steps and self._chat_type != 'evaluation':
            return

        if self._chat_type == 'evaluation':
            # Evaluation runs: write to runs/{run_id}/traces/
            diagnostics_dir = settings.diagnostics_dir or (settings.app_data_dir / DirNames.DIAGNOSTICS)
            base_dir = diagnostics_dir / DirNames.RUNS / self._run_id / DirNames.TRACES
            prune_base_dir = diagnostics_dir / DirNames.RUNS
            # Use chat_id--message_id format for trace filename
            trace_path = base_dir / f'{self._chat_id}--{self._message_id}.json'
        else:
            # User chats go to app_data_dir/chats/{chat_id}/{message_id}.json.
            # run_id may still be present in metadata for correlation.
            base_dir = settings.app_data_dir / DirNames.CHAT_LOGS
            prune_base_dir = base_dir
            chat_dir = base_dir / self._chat_id
            ensure_directory(chat_dir)
            trace_path = chat_dir / f'{self._message_id}.json'

        # Ensure directory exists
        ensure_file_directory(trace_path)
        await _maybe_prune_traces(self._chat_type, prune_base_dir)
        trace_data = self._format_json()
        lock = _get_trace_lock()

        def _write_json() -> None:
            # Atomic write: write temp file then rename into place.
            ensure_file_directory(trace_path)
            json_text = serialize_trace(trace_data) + '\n'
            with tempfile.NamedTemporaryFile(
                mode='w',
                encoding='utf-8',
                delete=False,
                dir=str(trace_path.parent),
                prefix=f'.{trace_path.name}.',
                suffix='.tmp',
            ) as tmp:
                tmp.write(json_text)
                tmp.flush()
                os.fsync(tmp.fileno())
                tmp_path = Path(tmp.name)
            os.replace(tmp_path, trace_path)

        async with lock:
            try:
                await asyncio.to_thread(_write_json)
            except OSError as exc:
                log.warning(
                    'trace_write_failed',
                    chat_id = self._chat_id,
                    path    = str(trace_path),
                    error   = str(exc),
                )


def _sanitize_for_trace(data: dict[str, Any], mode: str | None = None) -> dict[str, Any]:
    effective_mode = mode or _get_redaction_mode()
    out: dict[str, Any] = {}
    for k, v in data.items():
        if v is None:
            out[k] = None
        elif isinstance(v, (bool, int, float)):
            out[k] = v
        elif isinstance(v, str):
            out[k] = _sanitize_string(v, k, effective_mode)
        elif isinstance(v, (list, tuple)):
            out[k] = _sanitize_list(v, mode=effective_mode)
        elif isinstance(v, dict):
            out[k] = _sanitize_for_trace(v, mode=effective_mode)
        else:
            out[k] = str(v)
    return out


def _sanitize_list(items: list[Any] | tuple[Any, ...], mode: str | None = None) -> list[Any]:
    effective_mode = mode or _get_redaction_mode()
    result: list[Any] = []
    for i, item in enumerate(items):
        if i >= MAX_TRACE_SECTIONS:
            result.append('... (list truncated)')
            break
        if isinstance(item, dict):
            result.append(_sanitize_for_trace(item, mode=effective_mode))
        elif isinstance(item, (list, tuple)):
            result.append(_sanitize_list(item, mode=effective_mode))
        elif isinstance(item, str):
            result.append(
                item if len(item) <= MAX_TRACE_LIST_ITEM_LENGTH else item[:MAX_TRACE_LIST_ITEM_LENGTH] + '...'
            )
        elif isinstance(item, (int, float, bool)) or item is None:
            result.append(item)
        else:
            result.append(str(item))
    return result


# ==============================================================================
# Public API
# ==============================================================================


def get_trace_writer(chat_id: str, message_id: str, chat_type: str = 'user', run_id: str | None = None) -> TraceWriter | None:
    """
    Return a trace writer for this chat if chat_trace_logging is enabled.
    Otherwise return None. Uses persisted config file so the setting is
    respected as soon as the user saves (checkbox state).

    For evaluation type, trace logging is automatically enabled regardless
    of the chat_trace_logging setting.

    Args:
        chat_id: Unique chat identifier
        message_id: Unique message identifier
        chat_type: 'user' (default) or 'evaluation'
        run_id: Optional run ID. For evaluation runs this controls trace location.
            For user runs this is persisted in trace metadata for correlation.
    """
    # Auto-enable trace logging for evaluation runs
    if chat_type == 'evaluation':
        return _ChatTraceWriter(chat_id, message_id, chat_type='evaluation', run_id=run_id)
    if not get_chat_trace_logging():
        return None
    return _ChatTraceWriter(chat_id, message_id, chat_type='user', run_id=run_id)


async def flush_trace_writer(writer: TraceWriter | None) -> None:
    """If writer is a _ChatTraceWriter, await its flush(). No-op if None."""
    if writer is None:
        return
    if isinstance(writer, _ChatTraceWriter):
        await writer.flush()
