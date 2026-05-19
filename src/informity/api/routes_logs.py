# ==============================================================================
# Informity AI — Logs API Routes
# User-facing logs endpoint (DB-backed, cursor-paginated).
# ==============================================================================

import base64
import binascii
import json
from datetime import datetime

import aiosqlite
from fastapi import APIRouter, Depends, HTTPException, Query

from informity.api.schemas import LogEventsResponse
from informity.db.sqlite import get_db, get_log_events

router = APIRouter(tags=['logs'])


def _encode_cursor(*, created_at: str, row_id: int) -> str:
    payload = {'created_at': created_at, 'id': row_id}
    raw = json.dumps(payload, separators=(',', ':'), ensure_ascii=True).encode('utf-8')
    return base64.urlsafe_b64encode(raw).decode('ascii')


def _decode_cursor(value: str) -> tuple[str, int]:
    try:
        decoded = base64.urlsafe_b64decode(value.encode('ascii')).decode('utf-8')
        payload = json.loads(decoded)
        created_at = str(payload.get('created_at') or '').strip()
        row_id = int(payload.get('id'))
        if not created_at or row_id < 1:
            raise ValueError('invalid cursor payload')
        return created_at, row_id
    except (ValueError, TypeError, json.JSONDecodeError, binascii.Error) as exc:
        raise HTTPException(status_code=400, detail='Invalid cursor') from exc


def _format_timestamp(value: str) -> str:
    raw = str(value or '').strip()
    if not raw:
        return ''
    try:
        normalized = raw.replace('Z', '+00:00')
        dt = datetime.fromisoformat(normalized)
        return dt.astimezone().strftime('%Y-%m-%d %H:%M:%S')
    except ValueError:
        return raw


@router.get('/api/logs/events', response_model=LogEventsResponse)
async def list_log_events(
    channel: str = Query(..., pattern='^(application|errors|integrations)$'),
    limit: int = Query(50, ge=1, le=200),
    cursor: str | None = Query(None),
    event_type: str | None = Query(None, pattern='^(debug|info|warning|error|critical)$'),
    source: str | None = Query(None),
    db: aiosqlite.Connection = Depends(get_db),
) -> LogEventsResponse:
    cursor_created_at: str | None = None
    cursor_id: int | None = None
    if cursor:
        cursor_created_at, cursor_id = _decode_cursor(cursor)

    try:
        # Fetch one extra row to determine has_more without extra count query.
        rows = await get_log_events(
            db,
            channel=channel,
            limit=limit + 1,
            cursor_created_at=cursor_created_at,
            cursor_id=cursor_id,
            event_type=event_type,
            source=source,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    has_more = len(rows) > limit
    visible = rows[:limit]
    next_cursor: str | None = None
    if has_more and visible:
        tail = visible[-1]
        next_cursor = _encode_cursor(created_at=str(tail['created_at']), row_id=int(tail['id']))

    items = [
        {
            'id': int(row['id']),
            'timestamp': _format_timestamp(str(row['created_at'])),
            'created_at': str(row['created_at']),
            'channel': str(row['channel']),
            'event_type': str(row['event_type']),
            'event_name': str(row['event_name']),
            'source': str(row['source']),
            'message': str(row['message']),
            'scope': row.get('scope'),
            'details': row.get('details'),
        }
        for row in visible
    ]

    return LogEventsResponse(items=items, next_cursor=next_cursor, has_more=has_more)
