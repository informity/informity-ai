from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from fastapi import HTTPException

from informity.api import routes_chat
from informity.config import settings
from informity.db.models import FileCategory, IndexedFile
from informity.db.sqlite import (
    get_chat_upload_attachment_by_upload_id,
    get_connection,
    init_db,
    insert_chat_upload_attachment,
    insert_file,
)


@pytest.mark.asyncio
async def test_delete_chat_upload_prunes_empty_chat_upload_dir(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    app_data_dir = tmp_path / 'app-data'
    app_data_dir.mkdir(parents=True, exist_ok=True)
    db_path = tmp_path / 'delete-upload-success.db'
    monkeypatch.setattr(settings, 'app_data_dir', app_data_dir)
    monkeypatch.setattr(settings, 'db_path', db_path)

    await init_db()
    db = await get_connection()
    try:
        chat_id = 'chat-upload-delete-success'
        upload_id = 'upload-delete-success'
        filename = 'sample.txt'
        await insert_chat_upload_attachment(
            db,
            upload_id=upload_id,
            chat_id=chat_id,
            filename_at_upload=filename,
            size_bytes=10,
            state='ready',
        )
        upload_dir = routes_chat._upload_file_dir(chat_id, upload_id)
        upload_dir.mkdir(parents=True, exist_ok=True)
        (upload_dir / filename).write_text('sample', encoding='utf-8')

        result = await routes_chat.delete_chat_upload(upload_id=upload_id, chat_id=chat_id, db=db)
        assert result['deleted'] is True
        assert result['fallback_to_scanned_documents'] is True
        assert not routes_chat._upload_chat_dir(chat_id).exists()

        attachment = await get_chat_upload_attachment_by_upload_id(db, upload_id=upload_id, chat_id=chat_id)
        assert attachment is not None
        assert attachment.state == 'deleted'
        assert attachment.removed_at is not None
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_delete_chat_upload_marks_failed_when_index_record_delete_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    app_data_dir = tmp_path / 'app-data'
    app_data_dir.mkdir(parents=True, exist_ok=True)
    db_path = tmp_path / 'delete-upload-fail.db'
    monkeypatch.setattr(settings, 'app_data_dir', app_data_dir)
    monkeypatch.setattr(settings, 'db_path', db_path)

    await init_db()
    db = await get_connection()
    try:
        chat_id = 'chat-upload-delete-fail'
        upload_id = 'upload-delete-fail'
        filename = 'sample.txt'
        now = datetime.now(UTC)
        indexed_file = await insert_file(
            db,
            IndexedFile(
                source_provider='upload.local',
                entity_type='file',
                source_item_id=upload_id,
                path=f'/tmp/{filename}',
                filename=filename,
                extension='.txt',
                size_bytes=10,
                content_hash='hash-delete-fail',
                extracted_text_preview='sample',
                category=FileCategory.PLAINTEXT,
                modified_at=now,
                indexed_at=now,
            ),
        )
        assert indexed_file.id is not None
        await insert_chat_upload_attachment(
            db,
            upload_id=upload_id,
            chat_id=chat_id,
            filename_at_upload=filename,
            size_bytes=10,
            state='ready',
            file_id=int(indexed_file.id),
            content_hash=indexed_file.content_hash,
        )
        upload_dir = routes_chat._upload_file_dir(chat_id, upload_id)
        upload_dir.mkdir(parents=True, exist_ok=True)
        (upload_dir / filename).write_text('sample', encoding='utf-8')

        async def _remove_file_always_fails(*args, **kwargs) -> bool:
            return False

        monkeypatch.setattr(routes_chat, 'remove_file', _remove_file_always_fails)

        with pytest.raises(HTTPException) as exc_info:
            await routes_chat.delete_chat_upload(upload_id=upload_id, chat_id=chat_id, db=db)
        assert exc_info.value.status_code == 500

        attachment = await get_chat_upload_attachment_by_upload_id(db, upload_id=upload_id, chat_id=chat_id)
        assert attachment is not None
        assert attachment.state == 'failed'
        assert attachment.removed_at is None
    finally:
        await db.close()
