from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from informity.config import settings
from informity.db.models import FileCategory, IndexedFile
from informity.db.sqlite import (
    get_chat_upload_attachment_by_upload_id,
    get_connection,
    init_db,
    insert_chat_upload_attachment,
    insert_file,
    update_chat_upload_attachment_state,
)


@pytest.mark.asyncio
async def test_chat_upload_attachment_persists_content_hash(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    db_path = tmp_path / 'chat-upload-content-hash.db'
    monkeypatch.setattr(settings, 'db_path', db_path)

    await init_db()
    db = await get_connection()
    try:
        upload_id = 'upload-1'
        chat_id = 'chat-1'
        attachment = await insert_chat_upload_attachment(
            db,
            upload_id=upload_id,
            chat_id=chat_id,
            filename_at_upload='a.txt',
            size_bytes=10,
            state='uploading',
        )
        assert attachment.content_hash is None

        now = datetime.now(UTC)
        indexed = await insert_file(
            db,
            IndexedFile(
                source_provider='upload.local',
                entity_type='file',
                source_item_id=upload_id,
                path='/tmp/a.txt',
                filename='a.txt',
                extension='.txt',
                size_bytes=10,
                content_hash='abc123hash',
                extracted_text_preview='a',
                category=FileCategory.PLAINTEXT,
                modified_at=now,
                indexed_at=now,
            ),
        )
        assert indexed.id is not None
        await update_chat_upload_attachment_state(
            db,
            upload_id=upload_id,
            chat_id=chat_id,
            state='ready',
            file_id=indexed.id,
            content_hash=indexed.content_hash,
        )
        loaded = await get_chat_upload_attachment_by_upload_id(
            db,
            upload_id=upload_id,
            chat_id=chat_id,
        )
        assert loaded is not None
        assert loaded.content_hash == 'abc123hash'
    finally:
        await db.close()
