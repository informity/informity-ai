from __future__ import annotations

import asyncio
import json
import sys
import threading
import time
from contextlib import suppress
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from informity.llm.engine import (
    _STREAM_END,
    LLMEngine,
    _run_stream_worker,
    _truncate_messages_to_fit,
)


def test_truncate_messages_removes_history_before_system_content() -> None:
    messages = [
        {'role': 'system', 'content': 'You are helpful.\n\nContext:\n[Source: 1] baseline context.'},
        {'role': 'user', 'content': 'old question ' * 500},
        {'role': 'assistant', 'content': 'old answer ' * 500},
        {'role': 'user', 'content': 'current question'},
    ]

    truncated, info = _truncate_messages_to_fit(
        chat_template='',
        messages=messages,
        context_length=700,
        max_tokens=50,
        force_chatml=True,
    )

    assert info['truncated'] is True
    assert info['history_messages_removed'] >= 1
    assert truncated[-1]['content'] == 'current question'
    assert len(truncated) < len(messages)


def test_truncate_messages_truncates_system_context_chunks_when_needed() -> None:
    system_content = (
        'Rules for answering.\n\nContext:\n'
        '[Source: 1] ' + ('alpha ' * 100) + '\n\n'
        '[Source: 2] ' + ('beta ' * 100) + '\n\n'
        '[Source: 3] ' + ('gamma ' * 100)
    )
    messages = [
        {'role': 'system', 'content': system_content},
        {'role': 'user', 'content': 'What changed?'},
    ]

    truncated, info = _truncate_messages_to_fit(
        chat_template='',
        messages=messages,
        context_length=400,
        max_tokens=50,
        force_chatml=True,
    )

    assert info['truncated'] is True
    assert info['system_content_truncated'] is True
    assert info.get('chunks_removed', 0) >= 1
    assert '[Source: 3]' not in truncated[0]['content']


@pytest.mark.asyncio
async def test_generate_stream_emits_timeout_notice_and_marker(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = LLMEngine()
    engine._server = object()  # type: ignore[assignment]

    monkeypatch.setattr('informity.llm.engine.get_profile', lambda: SimpleNamespace(context_length=4096))
    monkeypatch.setattr(
        'informity.llm.engine._truncate_messages_to_fit',
        lambda **kwargs: (
            kwargs['messages'],
            {
                'truncated': False,
                'original_tokens': 1,
                'available_budget': 3900,
            },
        ),
    )

    def _silent_worker(
        server, messages, max_tok, temp, top_p_val, stop_seqs,  # type: ignore[no-untyped-def]
        loop, queue, exception_holder, cancel_event,
    ) -> None:
        _ = (server, messages, max_tok, temp, top_p_val, stop_seqs, loop, queue, exception_holder)
        while not cancel_event.is_set():
            time.sleep(0.01)

    monkeypatch.setattr('informity.llm.engine._run_stream_worker', _silent_worker)

    outputs: list[object] = []
    async for item in engine.generate_stream(
        messages=[{'role': 'user', 'content': 'hello'}],
        timeout_seconds=0.05,
    ):
        outputs.append(item)

    assert any(isinstance(item, str) and 'Response truncated: generation time limit' in item for item in outputs)
    timeout_markers = [item for item in outputs if isinstance(item, tuple) and item[0] == '__timeout__']
    assert timeout_markers


def _make_stream_worker_that_emits(tokens: list[str]):
    """Return a _run_stream_worker replacement that emits the given token strings."""
    def _worker(server, messages, max_tok, temp, top_p_val, stop_seqs,  # type: ignore[no-untyped-def]
                loop, queue, exception_holder, cancel_event) -> None:
        for token in tokens:
            if cancel_event.is_set():
                break
            loop.call_soon_threadsafe(queue.put_nowait, token)
        loop.call_soon_threadsafe(queue.put_nowait, ('__finish_reason__', 'stop'))
        loop.call_soon_threadsafe(queue.put_nowait, _STREAM_END)
    return _worker


def _common_engine_monkeypatches(monkeypatch: pytest.MonkeyPatch, worker_fn) -> LLMEngine:  # type: ignore[no-untyped-def]
    engine = LLMEngine()
    engine._server = object()  # type: ignore[assignment]
    monkeypatch.setattr('informity.llm.engine.get_profile', lambda: SimpleNamespace(context_length=4096))
    monkeypatch.setattr(
        'informity.llm.engine._truncate_messages_to_fit',
        lambda **kwargs: (kwargs['messages'], {'truncated': False, 'original_tokens': 1, 'available_budget': 3900}),
    )
    monkeypatch.setattr('informity.llm.engine._run_stream_worker', worker_fn)
    return engine


@pytest.mark.asyncio
async def test_generate_stream_flushes_partial_buffer_at_stream_end(monkeypatch: pytest.MonkeyPatch) -> None:
    # The think-block filter keeps up to 6 chars buffered to detect split '<think>' tags.
    # On normal stream end (_STREAM_END), those buffered chars must be flushed or the
    # last word of short answers is silently dropped.
    engine = _common_engine_monkeypatches(
        monkeypatch,
        _make_stream_worker_that_emits(['Hello', ' world']),
    )
    output = ''.join(
        item for item in [i async for i in engine.generate_stream(
            messages=[{'role': 'user', 'content': 'hi'}],
        )]
        if isinstance(item, str)
    )
    assert output == 'Hello world', f'Expected full answer, got {output!r}'


@pytest.mark.asyncio
async def test_generate_stream_strips_think_block_from_output(monkeypatch: pytest.MonkeyPatch) -> None:
    # Qwen3 reasoning-enabled queries prefix the answer with <think>...</think>.
    # generate_stream must strip the think block; only the answer text is yielded.
    tokens = ['<think>', 'thinking content here', '</think>', 'The answer is 42.']
    engine = _common_engine_monkeypatches(monkeypatch, _make_stream_worker_that_emits(tokens))
    output = ''.join(
        item for item in [i async for i in engine.generate_stream(
            messages=[{'role': 'user', 'content': 'question'}],
        )]
        if isinstance(item, str)
    )
    assert '<think>' not in output
    assert 'thinking content here' not in output
    assert 'The answer is 42.' in output


@pytest.mark.asyncio
async def test_generate_stream_no_think_block_passes_through(monkeypatch: pytest.MonkeyPatch) -> None:
    # When no think block is present (reasoning disabled via /no_think), every
    # token must pass through to the consumer unmodified.
    tokens = ['The', ' answer', ' is', ' 42.']
    engine = _common_engine_monkeypatches(monkeypatch, _make_stream_worker_that_emits(tokens))
    output = ''.join(
        item for item in [i async for i in engine.generate_stream(
            messages=[{'role': 'user', 'content': 'question'}],
        )]
        if isinstance(item, str)
    )
    assert output == 'The answer is 42.'


@pytest.mark.asyncio
async def test_generate_stream_cancellation_cleans_up_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    worker_stopped = threading.Event()

    def _blocking_worker(
        server, messages, max_tok, temp, top_p_val, stop_seqs,  # type: ignore[no-untyped-def]
        loop, queue, exception_holder, cancel_event,
    ) -> None:
        _ = (server, messages, max_tok, temp, top_p_val, stop_seqs, exception_holder)
        try:
            while not cancel_event.is_set():
                time.sleep(0.01)
        finally:
            worker_stopped.set()
            loop.call_soon_threadsafe(queue.put_nowait, _STREAM_END)

    engine = _common_engine_monkeypatches(monkeypatch, _blocking_worker)
    stream = engine.generate_stream(messages=[{'role': 'user', 'content': 'cancel me'}], timeout_seconds=10.0)
    next_item = asyncio.create_task(stream.__anext__())

    await asyncio.sleep(0.05)
    next_item.cancel()
    with pytest.raises(asyncio.CancelledError):
        await next_item

    with suppress(Exception):
        await stream.aclose()

    assert worker_stopped.wait(timeout=1.0), 'Expected worker to stop after stream cancellation'


@pytest.mark.asyncio
async def test_stream_worker_includes_chat_template_kwargs(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_payload: dict[str, object] = {}

    class _FakeServer:
        def handle_chat_completions(self, payload: str, callback) -> None:  # type: ignore[no-untyped-def]
            nonlocal captured_payload
            captured_payload = json.loads(payload)
            callback({'choices': [{'delta': {}, 'finish_reason': 'stop'}]})

    monkeypatch.setattr(
        'informity.llm.model_adapter.get_profile',
        lambda: SimpleNamespace(chat_template_kwargs={'enable_thinking': False}),
    )

    queue: asyncio.Queue[str | object] = asyncio.Queue()
    exception_holder: list[BaseException] = []
    _run_stream_worker(
        server=_FakeServer(),
        messages=[{'role': 'user', 'content': 'Hi'}],
        max_tok=32,
        temp=0.0,
        top_p_val=1.0,
        stop_seqs=[],
        loop=asyncio.get_running_loop(),
        queue=queue,
        exception_holder=exception_holder,
        cancel_event=threading.Event(),
    )

    assert exception_holder == []
    assert captured_payload['chat_template_kwargs'] == {'enable_thinking': False}


def test_chat_complete_includes_chat_template_kwargs(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_payload: dict[str, object] = {}

    class _FakeServer:
        def handle_chat_completions(self, payload: str, callback) -> None:  # type: ignore[no-untyped-def]
            nonlocal captured_payload
            captured_payload = json.loads(payload)
            callback({'choices': [{'message': {'content': 'ok'}}]})

    monkeypatch.setattr(
        'informity.llm.model_adapter.get_profile',
        lambda: SimpleNamespace(chat_template_kwargs={'enable_thinking': False}),
    )

    engine = LLMEngine()
    engine._server = _FakeServer()  # type: ignore[assignment]
    response = engine.chat_complete(
        messages=[{'role': 'user', 'content': 'Hello'}],
        max_tokens=16,
        temperature=0.0,
    )

    assert response['choices'][0]['message']['content'] == 'ok'
    assert captured_payload['chat_template_kwargs'] == {'enable_thinking': False}


def test_download_model_uses_httpx_stream_api(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class _FakeResponse:
        status_code = 200
        headers = {'Content-Length': '5'}

        def raise_for_status(self) -> None:
            return None

        def iter_bytes(self, chunk_size: int):  # type: ignore[no-untyped-def]
            _ = chunk_size
            yield b'he'
            yield b'llo'

    class _FakeStreamContext:
        def __enter__(self) -> _FakeResponse:
            return _FakeResponse()

        def __exit__(self, exc_type, exc, tb) -> bool:  # type: ignore[no-untyped-def]
            return False

    class _FakeSession:
        called = False

        def stream(self, method: str, url: str, headers: dict, timeout):  # type: ignore[no-untyped-def]
            self.called = True
            assert method == 'GET'
            assert isinstance(url, str) and url
            assert isinstance(headers, dict)
            assert timeout == (10, 60)
            return _FakeStreamContext()

    fake_session = _FakeSession()
    fake_hf_module = SimpleNamespace(hf_hub_url=lambda **kwargs: 'https://example.invalid/model.gguf')
    fake_hf_utils = SimpleNamespace(
        build_hf_headers=lambda: {},
        get_session=lambda: fake_session,
    )
    monkeypatch.setitem(sys.modules, 'huggingface_hub', fake_hf_module)
    monkeypatch.setitem(sys.modules, 'huggingface_hub.utils', fake_hf_utils)
    monkeypatch.setattr('informity.config.configure_hf_environment', lambda **kwargs: None)
    monkeypatch.setattr('informity.llm.engine.remove_models_dir_cache', lambda: None)

    engine = LLMEngine()
    target = tmp_path / 'model.gguf'
    progress_calls: list[tuple[int, int | None, float]] = []

    engine._download_model(
        target_path=target,
        repo_id='repo/test',
        filename='model.gguf',
        progress_callback=lambda done, total, speed: progress_calls.append((done, total, speed)),
    )

    assert fake_session.called is True
    assert target.read_bytes() == b'hello'
    assert progress_calls
    assert progress_calls[-1][0] == 5
    assert progress_calls[-1][1] == 5
