from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

from informity.api import routes_chat
from informity.db.models import ChatMessage


def _messages() -> list[ChatMessage]:
    return [
        ChatMessage(chat_id='chat-1', role='user', content='Summarize this'),
        ChatMessage(chat_id='chat-1', role='assistant', content='## Answer\n\nDone.', id=11),
    ]


@pytest.mark.asyncio
async def test_export_chat_markdown_returns_full_chat_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(routes_chat, 'get_chat', AsyncMock(return_value=_messages()))
    monkeypatch.setattr(routes_chat, 'get_chat_title', AsyncMock(return_value='Export Chat'))

    payload = await routes_chat.export_chat_markdown(
        chat_id='chat-1',
        scope='full_chat',
        message_id=None,
        include_frontmatter=True,
        template='full_transcript',
        db=object(),  # type: ignore[arg-type]
    )

    assert payload['chat_id'] == 'chat-1'
    assert payload['scope'] == 'full_chat'
    assert payload['include_frontmatter'] is True
    assert payload['template'] == 'full_transcript'
    assert isinstance(payload['markdown'], str) and len(payload['markdown']) > 0
    assert str(payload['filename']).endswith('.md')


@pytest.mark.asyncio
async def test_export_chat_unified_returns_markdown_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(routes_chat, 'get_chat', AsyncMock(return_value=_messages()))
    monkeypatch.setattr(routes_chat, 'get_chat_title', AsyncMock(return_value='Export Chat'))

    payload = await routes_chat.export_chat(
        chat_id='chat-1',
        scope='full_chat',
        message_id=None,
        include_frontmatter=False,
        template='full_transcript',
        format='markdown',
        db=object(),  # type: ignore[arg-type]
    )

    assert payload['chat_id'] == 'chat-1'
    assert payload['scope'] == 'full_chat'
    assert payload['format'] == 'markdown'
    assert payload['mime_type'] == 'text/markdown; charset=utf-8'
    assert isinstance(payload['content'], str) and len(payload['content']) > 0
    assert str(payload['filename']).endswith('.md')


@pytest.mark.asyncio
async def test_export_chat_unified_pdf_not_implemented(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(routes_chat, 'get_chat', AsyncMock(return_value=_messages()))
    monkeypatch.setattr(routes_chat, 'get_chat_title', AsyncMock(return_value='Export Chat'))

    with pytest.raises(HTTPException) as exc_info:
        await routes_chat.export_chat(
            chat_id='chat-1',
            scope='full_chat',
            message_id=None,
            include_frontmatter=False,
            template='full_transcript',
            format='pdf',
            db=object(),  # type: ignore[arg-type]
        )

    assert exc_info.value.status_code == 501


@pytest.mark.asyncio
async def test_export_chat_markdown_returns_current_answer_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(routes_chat, 'get_chat', AsyncMock(return_value=_messages()))
    monkeypatch.setattr(routes_chat, 'get_chat_title', AsyncMock(return_value='Export Chat'))

    payload = await routes_chat.export_chat_markdown(
        chat_id='chat-1',
        scope='current_answer',
        message_id=11,
        include_frontmatter=False,
        template='concise_summary',
        db=object(),  # type: ignore[arg-type]
    )

    assert payload['chat_id'] == 'chat-1'
    assert payload['scope'] == 'current_answer'
    assert payload['message_id'] == 11
    assert payload['template'] == 'concise_summary'
    assert payload['include_frontmatter'] is False
    assert '## Answer' in str(payload['markdown'])


@pytest.mark.asyncio
async def test_export_chat_markdown_rejects_invalid_scope(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(routes_chat, 'get_chat', AsyncMock(return_value=_messages()))

    with pytest.raises(HTTPException) as exc_info:
        await routes_chat.export_chat_markdown(
            chat_id='chat-1',
            scope='bad_scope',
            message_id=None,
            include_frontmatter=False,
            template='full_transcript',
            db=object(),  # type: ignore[arg-type]
        )

    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_export_chat_markdown_returns_404_for_missing_chat(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(routes_chat, 'get_chat', AsyncMock(return_value=[]))

    with pytest.raises(HTTPException) as exc_info:
        await routes_chat.export_chat_markdown(
            chat_id='missing',
            scope='full_chat',
            message_id=None,
            include_frontmatter=False,
            template='full_transcript',
            db=object(),  # type: ignore[arg-type]
        )

    assert exc_info.value.status_code == 404


def test_resolve_markdown_export_payload_returns_404_for_missing_assistant_message() -> None:
    with pytest.raises(HTTPException) as exc_info:
        routes_chat._resolve_markdown_export_payload(
            chat_id='chat-1',
            chat_title='Title',
            messages=_messages(),
            scope='current_answer',
            message_id=999,
            include_frontmatter=False,
            template='concise_summary',
        )

    assert exc_info.value.status_code == 404
