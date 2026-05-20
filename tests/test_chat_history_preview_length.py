from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from informity.config import settings
from informity.db.models import ChatMessage, ChatRole
from informity.db.sqlite import get_chats, get_connection, init_db, insert_chat_message


@pytest.mark.asyncio
async def test_get_chats_keeps_full_preview_text_without_backend_truncation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(settings, 'app_data_dir', tmp_path / 'app-data')
    monkeypatch.setattr(settings, 'db_path', tmp_path / 'history-preview-length.db')
    await init_db()
    db = await get_connection()
    try:
        chat_id = 'history-preview-length-chat'
        long_user = 'U' * 260
        long_assistant = 'A' * 320
        user = await insert_chat_message(
            db,
            ChatMessage(
                chat_id=chat_id,
                role=ChatRole.USER,
                content=long_user,
            ),
        )
        assistant = await insert_chat_message(
            db,
            ChatMessage(
                chat_id=chat_id,
                role=ChatRole.ASSISTANT,
                content=long_assistant,
            ),
        )
        assert user.id is not None and assistant.id is not None
        base = datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC)
        await db.execute('UPDATE chat_messages SET created_at = ? WHERE id = ?', (base.isoformat(), int(user.id)))
        await db.execute(
            'UPDATE chat_messages SET created_at = ? WHERE id = ?',
            ((base + timedelta(seconds=1)).isoformat(), int(assistant.id)),
        )
        await db.commit()

        chats = await get_chats(db, limit=10, offset=0)
        assert chats
        chat = next(item for item in chats if item['chat_id'] == chat_id)
        assert chat['first_user_message'] == long_user
        assert chat['last_message_preview'] == long_assistant
    finally:
        await db.close()
