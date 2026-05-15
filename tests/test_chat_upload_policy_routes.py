from __future__ import annotations

from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path

import pytest
from fastapi import HTTPException, UploadFile
from starlette.datastructures import Headers

from informity.api import routes_chat
from informity.api.schemas import ChatRequest
from informity.config import settings
from informity.db.models import ChatMessage, ChatRole, FileCategory, IndexedFile
from informity.db.sqlite import (
    get_chat,
    get_connection,
    init_db,
    insert_chat_message,
    insert_chat_upload_attachment,
    insert_file,
)


async def _insert_indexed_file(db, *, path: str, filename: str, content_hash: str) -> int:
    now = datetime.now(UTC)
    indexed_file = await insert_file(
        db,
        IndexedFile(
            source_provider='local.fs',
            entity_type='file',
            source_item_id=path,
            path=path,
            filename=filename,
            extension='.txt',
            size_bytes=64,
            content_hash=content_hash,
            extracted_text_preview='preview',
            category=FileCategory.PLAINTEXT,
            modified_at=now,
            indexed_at=now,
        ),
    )
    assert indexed_file.id is not None
    return int(indexed_file.id)


@pytest.mark.asyncio
async def test_chat_rejects_uploads_in_assistant_mode(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(settings, 'app_data_dir', tmp_path / 'app-data')
    monkeypatch.setattr(settings, 'db_path', tmp_path / 'policy-assistant.db')
    await init_db()
    db = await get_connection()
    try:
        chat_id = 'policy-chat-assistant'
        await insert_chat_upload_attachment(
            db,
            upload_id='upload-assistant',
            chat_id=chat_id,
            filename_at_upload='assistant.txt',
            size_bytes=10,
            state='ready',
        )
        with pytest.raises(HTTPException) as exc_info:
            await routes_chat.chat(
                request=ChatRequest(message='hello', chat_id=chat_id, mode='assistant'),
                db=db,
            )
        assert exc_info.value.status_code == 409
        assert exc_info.value.detail == 'Uploaded files are available only in Researcher mode.'
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_upload_endpoint_rejects_uploads_for_assistant_locked_chat(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(settings, 'app_data_dir', tmp_path / 'app-data')
    monkeypatch.setattr(settings, 'db_path', tmp_path / 'policy-upload-assistant-locked.db')
    await init_db()
    db = await get_connection()
    try:
        chat_id = 'policy-upload-assistant-locked'
        await insert_chat_message(
            db,
            ChatMessage(
                chat_id=chat_id,
                role=ChatRole.USER,
                content='Assistant-only chat',
                chat_mode='assistant',
            ),
        )
        upload = UploadFile(
            filename='note.txt',
            file=BytesIO(b'hello world'),
            headers=Headers({'content-type': 'text/plain'}),
        )

        with pytest.raises(HTTPException) as exc_info:
            await routes_chat.upload_chat_file(
                chat_id=chat_id,
                file=upload,
                db=db,
            )
        assert exc_info.value.status_code == 409
        assert exc_info.value.detail == 'Uploaded files are available only in Researcher mode.'
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_chat_rejects_mixed_library_scope_and_uploads(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(settings, 'app_data_dir', tmp_path / 'app-data')
    monkeypatch.setattr(settings, 'db_path', tmp_path / 'policy-mixed.db')
    await init_db()
    db = await get_connection()
    try:
        chat_id = 'policy-chat-mixed'
        file_id = await _insert_indexed_file(
            db,
            path='/tmp/library-scope.txt',
            filename='library-scope.txt',
            content_hash='policy-mixed-library-hash',
        )
        await insert_chat_upload_attachment(
            db,
            upload_id='upload-mixed',
            chat_id=chat_id,
            filename_at_upload='upload.txt',
            size_bytes=20,
            state='ready',
        )
        with pytest.raises(HTTPException) as exc_info:
            await routes_chat.chat(
                request=ChatRequest(
                    message='mixed scope request',
                    chat_id=chat_id,
                    mode='researcher',
                    scoped_file_ids=[file_id],
                ),
                db=db,
            )
        assert exc_info.value.status_code == 409
        assert exc_info.value.detail == 'Mixed library scope and chat uploads are not supported in one turn.'
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_chat_rejects_scoped_file_ids_in_assistant_mode(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(settings, 'app_data_dir', tmp_path / 'app-data')
    monkeypatch.setattr(settings, 'db_path', tmp_path / 'policy-assistant-scoped-file.db')
    await init_db()
    db = await get_connection()
    try:
        file_id = await _insert_indexed_file(
            db,
            path='/tmp/policy-assistant-scope.txt',
            filename='policy-assistant-scope.txt',
            content_hash='policy-assistant-scoped-hash',
        )
        with pytest.raises(HTTPException) as exc_info:
            await routes_chat.chat(
                request=ChatRequest(
                    message='answer from this file',
                    chat_id='policy-chat-assistant-scoped-file',
                    mode='assistant',
                    scoped_file_ids=[file_id],
                ),
                db=db,
            )
        assert exc_info.value.status_code == 409
        assert exc_info.value.detail == 'Scoped file retrieval is available only in Researcher mode.'
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_chat_explicit_scoped_upload_ids_require_existing_uploads(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(settings, 'app_data_dir', tmp_path / 'app-data')
    monkeypatch.setattr(settings, 'db_path', tmp_path / 'policy-no-uploads.db')
    await init_db()
    db = await get_connection()
    try:
        with pytest.raises(HTTPException) as exc_info:
            await routes_chat.chat(
                request=ChatRequest(
                    message='use selected upload',
                    chat_id='policy-chat-no-uploads',
                    mode='researcher',
                    scoped_upload_ids=['missing-upload'],
                ),
                db=db,
            )
        assert exc_info.value.status_code == 404
        assert exc_info.value.detail == 'No uploaded files found for this chat.'
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_chat_explicit_scoped_upload_ids_reject_indexing_upload(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(settings, 'app_data_dir', tmp_path / 'app-data')
    monkeypatch.setattr(settings, 'db_path', tmp_path / 'policy-indexing.db')
    await init_db()
    db = await get_connection()
    try:
        chat_id = 'policy-chat-indexing'
        await insert_chat_upload_attachment(
            db,
            upload_id='upload-indexing',
            chat_id=chat_id,
            filename_at_upload='indexing.txt',
            size_bytes=20,
            state='indexing',
        )
        upload_dir = routes_chat._upload_file_dir(chat_id, 'upload-indexing')
        upload_dir.mkdir(parents=True, exist_ok=True)
        (upload_dir / 'indexing.txt').write_text('still indexing', encoding='utf-8')
        with pytest.raises(HTTPException) as exc_info:
            await routes_chat.chat(
                request=ChatRequest(
                    message='use indexing upload',
                    chat_id=chat_id,
                    mode='researcher',
                    scoped_upload_ids=['upload-indexing'],
                ),
                db=db,
            )
        assert exc_info.value.status_code == 409
        assert exc_info.value.detail == 'Selected uploaded file is still indexing. Please retry in a moment.'
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_chat_explicit_scoped_upload_ids_reject_missing_upload_ids(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(settings, 'app_data_dir', tmp_path / 'app-data')
    monkeypatch.setattr(settings, 'db_path', tmp_path / 'policy-missing-upload-id.db')
    await init_db()
    db = await get_connection()
    try:
        chat_id = 'policy-chat-missing-upload-id'
        await insert_chat_upload_attachment(
            db,
            upload_id='upload-existing',
            chat_id=chat_id,
            filename_at_upload='existing.txt',
            size_bytes=20,
            state='ready',
        )
        with pytest.raises(HTTPException) as exc_info:
            await routes_chat.chat(
                request=ChatRequest(
                    message='use missing upload id',
                    chat_id=chat_id,
                    mode='researcher',
                    scoped_upload_ids=['upload-missing'],
                ),
                db=db,
            )
        assert exc_info.value.status_code == 404
        assert exc_info.value.detail == 'Upload not found or inactive: upload-missing.'
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_chat_allows_follow_up_when_role_is_omitted_but_chat_role_is_locked(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(settings, 'app_data_dir', tmp_path / 'app-data')
    monkeypatch.setattr(settings, 'db_path', tmp_path / 'policy-role-lock-omitted.db')
    await init_db()
    db = await get_connection()
    try:
        chat_id = 'policy-chat-role-lock-omitted'
        await insert_chat_message(
            db,
            ChatMessage(
                chat_id=chat_id,
                role=ChatRole.USER,
                content='Initial legal question',
                chat_mode='assistant',
                role_id='legal',
            ),
        )

        def _raise_after_role_lock(**_kwargs):
            raise RuntimeError('after_role_lock')

        monkeypatch.setattr(routes_chat, 'resolve_retrieval_context_scope_key', _raise_after_role_lock)

        with pytest.raises(RuntimeError, match='after_role_lock'):
            await routes_chat.chat(
                request=ChatRequest(
                    message='Follow-up question without explicit role id',
                    chat_id=chat_id,
                    mode='assistant',
                    role_id=None,
                ),
                db=db,
            )
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_chat_allows_follow_up_when_user_role_missing_but_assistant_role_locked(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(settings, 'app_data_dir', tmp_path / 'app-data')
    monkeypatch.setattr(settings, 'db_path', tmp_path / 'policy-role-lock-assistant-fallback.db')
    await init_db()
    db = await get_connection()
    try:
        chat_id = 'policy-chat-role-lock-assistant-fallback'
        await insert_chat_message(
            db,
            ChatMessage(
                chat_id=chat_id,
                role=ChatRole.USER,
                content='Initial question without role metadata',
                chat_mode='assistant',
                role_id=None,
            ),
        )
        await insert_chat_message(
            db,
            ChatMessage(
                chat_id=chat_id,
                role=ChatRole.ASSISTANT,
                content='Assistant reply tagged legal',
                chat_mode='assistant',
                role_id='legal',
            ),
        )

        def _raise_after_role_lock(**_kwargs):
            raise RuntimeError('after_role_lock_assistant_fallback')

        monkeypatch.setattr(routes_chat, 'resolve_retrieval_context_scope_key', _raise_after_role_lock)

        with pytest.raises(RuntimeError, match='after_role_lock_assistant_fallback'):
            await routes_chat.chat(
                request=ChatRequest(
                    message='Follow-up with legal role should be accepted',
                    chat_id=chat_id,
                    mode='assistant',
                    role_id='legal',
                ),
                db=db,
            )
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_chat_coerces_follow_up_mode_to_locked_chat_mode(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(settings, 'app_data_dir', tmp_path / 'app-data')
    monkeypatch.setattr(settings, 'db_path', tmp_path / 'policy-mode-lock-coerce.db')
    await init_db()
    db = await get_connection()
    try:
        chat_id = 'policy-chat-mode-lock-coerce'
        await insert_chat_message(
            db,
            ChatMessage(
                chat_id=chat_id,
                role=ChatRole.USER,
                content='Initial researcher question',
                chat_mode='researcher',
                role_id='legal',
            ),
        )
        await routes_chat.chat(
            request=ChatRequest(
                message='Follow-up tries to switch mode',
                chat_id=chat_id,
                mode='assistant',
                role_id='legal',
            ),
            db=db,
        )
        history = await get_chat(db, chat_id)
        persisted_user = [msg for msg in history if msg.role == ChatRole.USER and not bool(msg.is_internal)]
        assert len(persisted_user) >= 2
        assert persisted_user[-1].chat_mode == 'researcher'
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_get_chat_preserves_role_id_from_db_rows(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(settings, 'app_data_dir', tmp_path / 'app-data')
    monkeypatch.setattr(settings, 'db_path', tmp_path / 'policy-role-row-preserve.db')
    await init_db()
    db = await get_connection()
    try:
        chat_id = 'policy-chat-role-row-preserve'
        await insert_chat_message(
            db,
            ChatMessage(
                chat_id=chat_id,
                role=ChatRole.USER,
                content='Initial legal question',
                chat_mode='assistant',
                role_id='legal',
            ),
        )
        history = await get_chat(db, chat_id)
        assert len(history) == 1
        assert history[0].role_id == 'legal'
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_chat_coerces_follow_up_role_to_locked_chat_role(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(settings, 'app_data_dir', tmp_path / 'app-data')
    monkeypatch.setattr(settings, 'db_path', tmp_path / 'policy-role-lock-coerce.db')
    await init_db()
    db = await get_connection()
    try:
        chat_id = 'policy-chat-role-lock-coerce'
        await insert_chat_message(
            db,
            ChatMessage(
                chat_id=chat_id,
                role=ChatRole.USER,
                content='Initial legal question',
                chat_mode='assistant',
                role_id='legal',
            ),
        )

        def _raise_after_role_lock(**_kwargs):
            raise RuntimeError('after_role_lock_coerce')

        monkeypatch.setattr(routes_chat, 'resolve_retrieval_context_scope_key', _raise_after_role_lock)

        with pytest.raises(RuntimeError, match='after_role_lock_coerce'):
            await routes_chat.chat(
                request=ChatRequest(
                    message='Follow-up tries to switch role',
                    chat_id=chat_id,
                    mode='assistant',
                    role_id='financial',
                ),
                db=db,
            )
    finally:
        await db.close()
