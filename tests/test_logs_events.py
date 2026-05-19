import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from informity.config import settings
from informity.db.sqlite import (
    get_connection,
    get_log_events,
    init_db,
    insert_log_event,
    prune_log_events,
)
from informity.log_events import emit_log_event


@pytest.mark.asyncio
async def test_log_events_table_created_for_existing_v3_database(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    db_path = tmp_path / 'logs-migration-v3.db'
    monkeypatch.setattr(settings, 'db_path', db_path)

    conn = sqlite3.connect(str(db_path))
    conn.execute('CREATE TABLE schema_version (version INTEGER PRIMARY KEY)')
    conn.execute('INSERT INTO schema_version (version) VALUES (3)')
    conn.commit()
    conn.close()

    await init_db()

    db = await get_connection()
    try:
        schema_row = await (await db.execute('SELECT version FROM schema_version LIMIT 1')).fetchone()
        assert schema_row is not None
        assert int(schema_row['version']) == 4

        table_row = await (
            await db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='log_events'")
        ).fetchone()
        assert table_row is not None
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_emit_log_event_drops_transport_noise_and_dedupes_bucket(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    db_path = tmp_path / 'logs-dedupe.db'
    monkeypatch.setattr(settings, 'db_path', db_path)
    await init_db()

    db = await get_connection()
    try:
        await emit_log_event(
            event_name='scan_completed',
            source='Scanner',
            message='HTTP 200 OK',
            db=db,
        )
        row = await (await db.execute('SELECT COUNT(*) AS cnt FROM log_events')).fetchone()
        assert row is not None
        assert int(row['cnt']) == 0

        await emit_log_event(
            event_name='scan_completed',
            source='Scanner',
            message='Scan completed. 10 files checked.',
            correlation_id='scan:1',
            dedupe_bucket_seconds=60,
            db=db,
        )
        await emit_log_event(
            event_name='scan_completed',
            source='Scanner',
            message='Scan completed. 10 files checked.',
            correlation_id='scan:1',
            dedupe_bucket_seconds=60,
            db=db,
        )
        row = await (await db.execute('SELECT COUNT(*) AS cnt FROM log_events')).fetchone()
        assert row is not None
        assert int(row['cnt']) == 1
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_prune_log_events_enforces_age_and_row_cap(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    db_path = tmp_path / 'logs-prune.db'
    monkeypatch.setattr(settings, 'db_path', db_path)
    await init_db()

    db = await get_connection()
    try:
        old_ts = datetime.now(UTC) - timedelta(days=45)
        await insert_log_event(
            db,
            event_id='old-1',
            created_at=old_ts,
            channel='application',
            event_type='info',
            event_name='scan_started',
            source='Scanner',
            message='old event',
        )

        base = datetime.now(UTC)
        for idx in range(5):
            await insert_log_event(
                db,
                event_id=f'new-{idx}',
                created_at=base - timedelta(seconds=idx),
                channel='application',
                event_type='info',
                event_name='scan_started',
                source='Scanner',
                message=f'new event {idx}',
            )

        result = await prune_log_events(db, retention_days=30, max_rows=3)
        assert int(result['deleted_by_age']) >= 1
        assert int(result['deleted_by_count']) >= 2

        row = await (await db.execute('SELECT COUNT(*) AS cnt FROM log_events')).fetchone()
        assert row is not None
        assert int(row['cnt']) == 3

        rows = await (
            await db.execute('SELECT event_id FROM log_events ORDER BY created_at DESC, id DESC')
        ).fetchall()
        kept = [str(item['event_id']) for item in rows]
        assert kept == ['new-0', 'new-1', 'new-2']
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_log_events_insert_and_cursor_pagination(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    db_path = tmp_path / 'logs-events.db'
    monkeypatch.setattr(settings, 'db_path', db_path)
    await init_db()

    db = await get_connection()
    try:
        now = datetime.now(UTC)
        for idx in range(3):
            await insert_log_event(
                db,
                event_id=f'ev-{idx}',
                created_at=now - timedelta(seconds=idx),
                channel='application',
                event_type='info',
                event_name='scan_completed',
                source='Scanner',
                message=f'scan event {idx}',
            )

        first_page = await get_log_events(db, channel='application', limit=2)
        assert len(first_page) == 2
        assert first_page[0]['message'] == 'scan event 0'
        assert first_page[1]['message'] == 'scan event 1'

        tail = first_page[-1]
        second_page = await get_log_events(
            db,
            channel='application',
            limit=2,
            cursor_created_at=str(tail['created_at']),
            cursor_id=int(tail['id']),
        )
        assert len(second_page) == 1
        assert second_page[0]['message'] == 'scan event 2'
    finally:
        await db.close()
