from __future__ import annotations

import hashlib
import time
from datetime import UTC, datetime

import aiosqlite
import structlog

from informity.db.sqlite import (
    LOG_EVENT_CHANNELS,
    LOG_EVENT_TYPES,
    LOG_EVENTS_MAX_ROWS_DEFAULT,
    LOG_EVENTS_RETENTION_DAYS_DEFAULT,
    get_connection,
    insert_log_event,
    prune_log_events,
)

log = structlog.get_logger(__name__)
_LOG_PRUNE_INTERVAL_SECONDS = 600.0
_last_log_prune_ts = 0.0

_CANONICAL_EVENTS: dict[str, tuple[str, str]] = {
    'scan_started': ('application', 'info'),
    'scan_completed': ('application', 'info'),
    'scan_failed': ('errors', 'error'),
    'scan_cancelled': ('errors', 'warning'),
    'indexing_timeout': ('errors', 'warning'),
    'mcp_server_started': ('integrations', 'info'),
    'mcp_server_stopped': ('integrations', 'info'),
    'mcp_server_failed': ('integrations', 'error'),
    'mcp_scope_denied': ('integrations', 'warning'),
    'mcp_auth_failed': ('integrations', 'warning'),
    'mcp_policy_violation': ('integrations', 'warning'),
    'index_refresh_started': ('application', 'info'),
    'index_refresh_completed': ('application', 'info'),
    'index_refresh_failed': ('errors', 'error'),
    'database_compaction_failed': ('errors', 'error'),
    'ollama_unavailable': ('integrations', 'error'),
}

_NOISE_SUBSTRINGS = (
    ' 200 ok',
    'http 200',
    'status=200',
    'status 200',
)


def _build_event_id(
    *,
    event_name: str,
    source: str,
    message: str,
    correlation_id: str | None,
    bucket_seconds: int | None,
    created_at: datetime,
) -> str:
    if bucket_seconds and bucket_seconds > 0:
        epoch = int(created_at.timestamp())
        bucket = epoch - (epoch % bucket_seconds)
    else:
        bucket = int(created_at.timestamp() * 1000)
    identity = f'{event_name}|{source}|{correlation_id or "none"}|{bucket}|{message}'
    digest = hashlib.sha256(identity.encode('utf-8')).hexdigest()[:16]
    return f'{event_name}:{digest}'


async def emit_log_event(
    *,
    event_name: str,
    source: str,
    message: str,
    channel: str | None = None,
    event_type: str | None = None,
    details: dict[str, object] | None = None,
    scope: str | None = None,
    correlation_id: str | None = None,
    file_id: int | None = None,
    scan_id: int | None = None,
    created_by: str | None = None,
    dedupe_bucket_seconds: int | None = None,
    db: aiosqlite.Connection | None = None,
) -> None:
    message_value = str(message or '').strip()
    if not message_value:
        return
    lowered = f' {message_value.lower()}'
    if any(marker in lowered for marker in _NOISE_SUBSTRINGS):
        return

    canonical = _CANONICAL_EVENTS.get(event_name)
    resolved_channel = channel or (canonical[0] if canonical else None)
    resolved_type = event_type or (canonical[1] if canonical else None)
    if resolved_channel not in LOG_EVENT_CHANNELS or resolved_type not in LOG_EVENT_TYPES:
        log.debug(
            'log_event_skip_invalid_taxonomy',
            event_name=event_name,
            channel=resolved_channel,
            event_type=resolved_type,
        )
        return

    created_at = datetime.now(UTC)
    event_id = _build_event_id(
        event_name=event_name,
        source=source,
        message=message,
        correlation_id=correlation_id,
        bucket_seconds=dedupe_bucket_seconds,
        created_at=created_at,
    )

    owned_connection = db is None
    conn = db
    try:
        if conn is None:
            conn = await get_connection()
        await insert_log_event(
            conn,
            event_id=event_id,
            created_at=created_at,
            channel=resolved_channel,
            event_type=resolved_type,
            event_name=event_name,
            source=source,
            message=message_value,
            details=details,
            scope=scope,
            correlation_id=correlation_id,
            file_id=file_id,
            scan_id=scan_id,
            created_by=created_by,
        )
    except (aiosqlite.Error, RuntimeError, ValueError, TypeError, OSError) as exc:
        log.warning(
            'log_event_emit_failed',
            event_name=event_name,
            error=str(exc),
        )
    finally:
        if owned_connection and conn is not None:
            await conn.close()

    global _last_log_prune_ts
    now_mono = time.monotonic()
    if now_mono - _last_log_prune_ts < _LOG_PRUNE_INTERVAL_SECONDS:
        return
    _last_log_prune_ts = now_mono
    try:
        prune_conn = await get_connection()
        await prune_log_events(
            prune_conn,
            retention_days=LOG_EVENTS_RETENTION_DAYS_DEFAULT,
            max_rows=LOG_EVENTS_MAX_ROWS_DEFAULT,
        )
        await prune_conn.close()
    except (aiosqlite.Error, RuntimeError, ValueError, TypeError, OSError):
        pass
