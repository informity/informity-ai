from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from informity.config import settings
from informity.db.models import FileCategory, IndexedFile
from informity.db.sqlite import (
    get_connection,
    init_db,
    insert_chat_upload_attachment,
    insert_file,
    reset_all_data,
)


@pytest.mark.asyncio
async def test_reset_all_data_reports_upload_purge_counts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    db_path = tmp_path / 'reset-upload-counts.db'
    monkeypatch.setattr(settings, 'db_path', db_path)

    await init_db()
    db = await get_connection()
    try:
        await insert_chat_upload_attachment(
            db,
            upload_id='upload-1',
            chat_id='chat-1',
            filename_at_upload='a.txt',
            size_bytes=10,
            state='ready',
        )
        now = datetime.now(UTC)
        file = await insert_file(
            db,
            IndexedFile(
                source_provider='upload.local',
                entity_type='file',
                source_item_id='upload-1',
                path='/tmp/a.txt',
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
        assert file.id is not None
        chunk_cursor = await db.execute(
            """
            INSERT INTO chunks (file_id, chunk_index, content, token_count)
            VALUES (?, ?, ?, ?)
            """,
            (int(file.id), 0, 'content', 1),
        )
        await db.commit()
        chunk_id = int(chunk_cursor.lastrowid or 0)
        assert chunk_id > 0
        await db.execute(
            """
            INSERT INTO vec_chunks (
                chunk_id, file_id, file_path, chunk_text, vector, year, filename, extension, category
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (chunk_id, int(file.id), file.path, 'content', b'\x00', 2026, file.filename, file.extension, file.category.value),
        )
        await db.commit()

        counts = await reset_all_data(db)
        assert counts['upload_attachments'] == 1
        assert counts['upload_files'] == 1
        assert counts['upload_chunks'] == 1
        assert counts['upload_vectors'] == 1
    finally:
        await db.close()
