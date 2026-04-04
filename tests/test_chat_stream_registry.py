import asyncio

import pytest

from informity.api.chat_stream_registry import ChatStreamRegistry


@pytest.mark.asyncio
async def test_mark_stopped_by_request_id_is_idempotent() -> None:
    registry = ChatStreamRegistry()
    stop_event = asyncio.Event()

    await registry.register(
        stream_id='stream-1',
        chat_id='chat-1',
        request_id='request-1',
        stop_event=stop_event,
    )

    first = await registry.mark_stopped_by_user(
        stream_id=None,
        request_id='request-1',
        chat_id='chat-1',
    )
    second = await registry.mark_stopped_by_user(
        stream_id=None,
        request_id='request-1',
        chat_id='chat-1',
    )

    assert first.status == 'stopped_now'
    assert second.status == 'already_terminal'
    assert stop_event.is_set() is True


@pytest.mark.asyncio
async def test_mark_stopped_by_stream_id_preserved() -> None:
    registry = ChatStreamRegistry()
    stop_event = asyncio.Event()

    await registry.register(
        stream_id='stream-2',
        chat_id='chat-2',
        request_id='request-2',
        stop_event=stop_event,
    )

    status = await registry.mark_stopped_by_user(
        stream_id='stream-2',
        request_id=None,
        chat_id='chat-2',
    )

    assert status.status == 'stopped_now'
    assert stop_event.is_set() is True
    assert registry.is_stopped_by_user('stream-2') is True


@pytest.mark.asyncio
async def test_mark_stopped_by_user_requires_matching_chat_id() -> None:
    registry = ChatStreamRegistry()
    stop_event = asyncio.Event()

    await registry.register(
        stream_id='stream-3',
        chat_id='chat-3',
        request_id='request-3',
        stop_event=stop_event,
    )

    status = await registry.mark_stopped_by_user(
        stream_id=None,
        request_id='request-3',
        chat_id='wrong-chat',
    )

    assert status.status == 'not_found'
    assert stop_event.is_set() is False


@pytest.mark.asyncio
async def test_stale_request_id_does_not_stop_newer_request() -> None:
    registry = ChatStreamRegistry()
    old_event = asyncio.Event()
    new_event = asyncio.Event()

    await registry.register(
        stream_id='stream-old',
        chat_id='chat-4',
        request_id='request-old',
        stop_event=old_event,
    )
    await registry.unregister('stream-old')
    await registry.register(
        stream_id='stream-new',
        chat_id='chat-4',
        request_id='request-new',
        stop_event=new_event,
    )

    stale_status = await registry.mark_stopped_by_user(
        stream_id=None,
        request_id='request-old',
        chat_id='chat-4',
    )

    assert stale_status.status == 'not_found'
    assert old_event.is_set() is False
    assert new_event.is_set() is False
    assert registry.is_stopped_by_user('stream-new') is False


@pytest.mark.asyncio
async def test_mark_stopped_by_user_cancels_active_task() -> None:
    registry = ChatStreamRegistry()
    gate = asyncio.Event()

    async def _run_forever() -> None:
        await gate.wait()

    task = asyncio.create_task(_run_forever())
    stop_event = asyncio.Event()
    await registry.register(
        stream_id='stream-task',
        chat_id='chat-task',
        request_id='request-task',
        stop_event=stop_event,
        task=task,
    )

    status = await registry.mark_stopped_by_user(
        stream_id='stream-task',
        request_id=None,
        chat_id='chat-task',
    )

    assert status.status == 'stopped_now'
    assert stop_event.is_set() is True
    await asyncio.sleep(0)
    assert task.cancelled() is True
