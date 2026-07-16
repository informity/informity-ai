"""Test module for tests test chat message translation."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

from informity.api import routes_chat
from informity.api.schemas import ChatMessageTranslateRequest
from informity.config import settings
from informity.db.models import ChatMessage
from informity.db.sqlite import get_chat, get_connection, init_db, insert_chat_message
from informity.llm.types import ChatRole


async def _setup_translation_db(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Internal helper for setup translation db."""
    monkeypatch.setattr(settings, "app_data_dir", tmp_path / "app-data")
    monkeypatch.setattr(settings, "db_path", tmp_path / "chat-translation.db")
    await init_db()
    return await get_connection()


@pytest.mark.asyncio
async def test_translate_chat_message_persists_translation_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Test translate chat message persists translation metadata."""
    db = await _setup_translation_db(monkeypatch, tmp_path)
    try:
        chat_id = "chat-translation-1"
        await insert_chat_message(
            db,
            ChatMessage(
                chat_id=chat_id,
                role=ChatRole.USER,
                content="Please translate this answer.",
            ),
        )
        source_message = await insert_chat_message(
            db,
            ChatMessage(
                chat_id=chat_id,
                role=ChatRole.ASSISTANT,
                content="Hello world.",
                sources=[{"file_id": 1, "filename": "example.txt"}],
                chat_mode="assistant",
                specialization_id="legal",
            ),
        )

        translate_mock = AsyncMock(return_value=("Bonjour le monde.", "stop"))
        monkeypatch.setattr(routes_chat, "translate_chat_message_text", translate_mock)

        response = await routes_chat.translate_chat_message(
            chat_id=chat_id,
            message_id=int(source_message.id or 0),
            request=ChatMessageTranslateRequest(target_language="French", tone="formal"),
            db=db,
        )

        assert response["chat_id"] == chat_id
        assert response["source_message_id"] == int(source_message.id or 0)
        assert response["reused_existing_translation"] is False
        assert response["target_language"] == "French"
        assert response["tone"] == "formal"
        assert int(response["translated_message_id"]) > 0
        assert translate_mock.await_count == 1

        translated_payload = response["translated_message"]
        assert translated_payload["content"] == "Bonjour le monde."
        assert translated_payload["translated_from_message_id"] == int(source_message.id or 0)
        assert translated_payload["translation_language"] == "French"
        assert translated_payload["translation_tone"] == "formal"
        assert translated_payload["translation_is_stale"] is False

        history = await get_chat(db, chat_id)
        assert [message.role for message in history] == [
            ChatRole.USER,
            ChatRole.ASSISTANT,
            ChatRole.ASSISTANT,
        ]
        translated_message = history[-1]
        assert translated_message.translated_from_message_id == int(source_message.id or 0)
        assert translated_message.translation_language == "French"
        assert translated_message.translation_tone == "formal"
        assert translated_message.translation_source_hash
        assert translated_message.translation_is_stale is False
        assert translated_message.sources == [{"file_id": 1, "filename": "example.txt"}]
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_translate_chat_message_reuses_existing_translation_and_rejects_translated_reply(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Test translate chat message reuses existing translation and rejects translated reply."""
    db = await _setup_translation_db(monkeypatch, tmp_path)
    try:
        chat_id = "chat-translation-2"
        await insert_chat_message(
            db,
            ChatMessage(
                chat_id=chat_id,
                role=ChatRole.USER,
                content="Please translate this answer.",
            ),
        )
        source_message = await insert_chat_message(
            db,
            ChatMessage(
                chat_id=chat_id,
                role=ChatRole.ASSISTANT,
                content="Hello world.",
                chat_mode="assistant",
            ),
        )

        translate_mock = AsyncMock(return_value=("Bonjour le monde.", "stop"))
        monkeypatch.setattr(routes_chat, "translate_chat_message_text", translate_mock)

        first_response = await routes_chat.translate_chat_message(
            chat_id=chat_id,
            message_id=int(source_message.id or 0),
            request=ChatMessageTranslateRequest(target_language="French", tone="natural"),
            db=db,
        )
        second_response = await routes_chat.translate_chat_message(
            chat_id=chat_id,
            message_id=int(source_message.id or 0),
            request=ChatMessageTranslateRequest(target_language="French", tone="natural"),
            db=db,
        )

        assert first_response["reused_existing_translation"] is False
        assert second_response["reused_existing_translation"] is True
        assert second_response["translated_message_id"] == first_response["translated_message_id"]
        assert translate_mock.await_count == 1

        history = await get_chat(db, chat_id)
        translations = [
            message
            for message in history
            if message.translated_from_message_id == int(source_message.id or 0)
        ]
        assert len(translations) == 1

        with pytest.raises(HTTPException) as exc_info:
            await routes_chat.translate_chat_message(
                chat_id=chat_id,
                message_id=int(first_response["translated_message_id"]),
                request=ChatMessageTranslateRequest(target_language="French", tone="natural"),
                db=db,
            )

        assert exc_info.value.status_code == 409
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_translated_messages_are_marked_stale_when_source_content_changes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Test translated messages are marked stale when source content changes."""
    db = await _setup_translation_db(monkeypatch, tmp_path)
    try:
        chat_id = "chat-translation-3"
        await insert_chat_message(
            db,
            ChatMessage(
                chat_id=chat_id,
                role=ChatRole.USER,
                content="Please translate this answer.",
            ),
        )
        source_message = await insert_chat_message(
            db,
            ChatMessage(
                chat_id=chat_id,
                role=ChatRole.ASSISTANT,
                content="Hello world.",
                chat_mode="assistant",
            ),
        )

        translate_mock = AsyncMock(return_value=("Bonjour le monde.", "stop"))
        monkeypatch.setattr(routes_chat, "translate_chat_message_text", translate_mock)

        await routes_chat.translate_chat_message(
            chat_id=chat_id,
            message_id=int(source_message.id or 0),
            request=ChatMessageTranslateRequest(target_language="French", tone="natural"),
            db=db,
        )

        await db.execute(
            "UPDATE chat_messages SET content = ? WHERE id = ?",
            ("Hello world. Updated source text.", int(source_message.id or 0)),
        )
        await db.commit()

        payload = await routes_chat.get_chat_messages(chat_id=chat_id, db=db)
        translated_payload = next(
            (
                message
                for message in payload["messages"]
                if message.get("translated_from_message_id") == int(source_message.id or 0)
            ),
            None,
        )

        assert translated_payload is not None
        assert translated_payload["translation_is_stale"] is True
    finally:
        await db.close()
