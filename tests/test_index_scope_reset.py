from datetime import UTC, datetime
from pathlib import Path

import pytest

from informity.config import settings
from informity.db.models import FileCategory, IndexedFile
from informity.db.sqlite import (
    get_connection,
    get_index_scope_counts,
    init_db,
    insert_file,
    reset_index_data_scope,
)


@pytest.mark.asyncio
async def test_get_index_scope_counts_groups_by_provider_and_entity_type(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    db_path = tmp_path / 'index-scope-counts.db'
    monkeypatch.setattr(settings, 'db_path', db_path)

    await init_db()
    db = await get_connection()
    try:
        now = datetime.now(UTC)
        await insert_file(
            db,
            IndexedFile(
                source_provider='filesystem',
                entity_type='file',
                source_item_id='/docs/a.txt',
                path='/docs/a.txt',
                filename='a.txt',
                extension='.txt',
                size_bytes=10,
                content_hash='hash-a',
                extracted_text_preview='a',
                category=FileCategory.PLAINTEXT,
                modified_at=now,
                indexed_at=now,
            ),
        )
        await insert_file(
            db,
            IndexedFile(
                source_provider='mail.apple',
                entity_type='mail',
                source_item_id='msg-1',
                path='source://mail.apple/mail/msg-1',
                filename='msg-1',
                extension='.txt',
                size_bytes=20,
                content_hash='hash-b',
                extracted_text_preview='b',
                category=FileCategory.PLAINTEXT,
                modified_at=now,
                indexed_at=now,
            ),
        )

        counts = await get_index_scope_counts(db)
        assert len(counts) == 2
        assert counts[0]['source_provider'] == 'filesystem'
        assert counts[0]['entity_type'] == 'file'
        assert counts[0]['files_count'] == 1
        assert counts[1]['source_provider'] == 'mail.apple'
        assert counts[1]['entity_type'] == 'mail'
        assert counts[1]['files_count'] == 1
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_reset_index_data_scope_deletes_only_target_scope(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    db_path = tmp_path / 'index-scope-reset.db'
    monkeypatch.setattr(settings, 'db_path', db_path)

    await init_db()
    db = await get_connection()
    try:
        now = datetime.now(UTC)
        await insert_file(
            db,
            IndexedFile(
                source_provider='filesystem',
                entity_type='file',
                source_item_id='/docs/a.txt',
                path='/docs/a.txt',
                filename='a.txt',
                extension='.txt',
                size_bytes=10,
                content_hash='hash-a',
                extracted_text_preview='a',
                category=FileCategory.PLAINTEXT,
                modified_at=now,
                indexed_at=now,
            ),
        )
        await insert_file(
            db,
            IndexedFile(
                source_provider='mail.apple',
                entity_type='mail',
                source_item_id='msg-1',
                path='source://mail.apple/mail/msg-1',
                filename='msg-1',
                extension='.txt',
                size_bytes=20,
                content_hash='hash-b',
                extracted_text_preview='b',
                category=FileCategory.PLAINTEXT,
                modified_at=now,
                indexed_at=now,
            ),
        )

        result = await reset_index_data_scope(
            db,
            source_provider='mail.apple',
            entity_type='mail',
        )
        assert result['files_deleted'] == 1

        counts = await get_index_scope_counts(db)
        assert len(counts) == 1
        assert counts[0]['source_provider'] == 'filesystem'
        assert counts[0]['entity_type'] == 'file'
        assert counts[0]['files_count'] == 1
    finally:
        await db.close()
