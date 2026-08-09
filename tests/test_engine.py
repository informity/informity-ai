"""Test module for tests test engine."""

# pylint: disable=line-too-long
# pylint: disable=unused-argument

from __future__ import annotations

import asyncio
import json
import sys
import threading
import time
import urllib.error
from contextlib import suppress
from contextvars import copy_context
from pathlib import Path
from types import SimpleNamespace

import pytest

from informity.llm.engine import (
    _STREAM_END,
    LLMEngine,
    StreamSignalTag,
    _run_stream_worker,
    reset_runtime_call_probe_context,
    set_runtime_call_probe_context,
    _truncate_messages_to_fit,
)


def _force_local_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    """Internal helper for force local provider."""
    monkeypatch.setattr("informity.llm.engine.settings.llm_provider", "local_gguf")


def test_truncate_messages_removes_history_before_system_content() -> None:
    """Test truncate messages removes history before system content."""
    messages = [
        {
            "role": "system",
            "content": "You are helpful.\n\nContext:\n[Source: 1] baseline context.",
        },
        {"role": "user", "content": "old question " * 500},
        {"role": "assistant", "content": "old answer " * 500},
        {"role": "user", "content": "current question"},
    ]

    truncated, info = _truncate_messages_to_fit(
        chat_template="",
        messages=messages,
        context_length=700,
        max_tokens=50,
        force_chatml=True,
    )

    assert info["truncated"] is True
    assert info["history_messages_removed"] >= 1
    assert truncated[-1]["content"] == "current question"
    assert len(truncated) < len(messages)


def test_truncate_messages_truncates_system_context_chunks_when_needed() -> None:
    """Test truncate messages truncates system context chunks when needed."""
    system_content = (
        "Rules for answering.\n\nContext:\n"
        "[Source: 1] " + ("alpha " * 100) + "\n\n"
        "[Source: 2] " + ("beta " * 100) + "\n\n"
        "[Source: 3] " + ("gamma " * 100)
    )
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": "What changed?"},
    ]

    truncated, info = _truncate_messages_to_fit(
        chat_template="",
        messages=messages,
        context_length=400,
        max_tokens=50,
        force_chatml=True,
    )

    assert info["truncated"] is True
    assert info["system_content_truncated"] is True
    assert info.get("chunks_removed", 0) >= 1
    assert "[Source: 3]" not in truncated[0]["content"]


@pytest.mark.asyncio
async def test_generate_stream_emits_timeout_notice_and_marker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test generate stream emits timeout notice and marker."""
    _force_local_provider(monkeypatch)
    engine = LLMEngine()
    engine._server = object()  # type: ignore[assignment]

    monkeypatch.setattr(
        "informity.llm.engine.get_profile", lambda: SimpleNamespace(context_length=4096)
    )
    monkeypatch.setattr(
        "informity.llm.engine._truncate_messages_to_fit",
        lambda **kwargs: (
            kwargs["messages"],
            {
                "truncated": False,
                "original_tokens": 1,
                "available_budget": 3900,
            },
        ),
    )

    def _silent_worker(
        server,
        messages,
        max_tok,
        temp,
        top_p_val,
        stop_seqs,  # type: ignore[no-untyped-def]
        *args,
    ) -> None:
        """Internal helper for silent worker."""
        remaining = args[1:] if len(args) >= 5 and args[0] is None else args
        loop = remaining[0]
        queue = remaining[1]
        exception_holder = remaining[2]
        cancel_event = remaining[3]
        _ = (server, messages, max_tok, temp, top_p_val, stop_seqs, loop, queue, exception_holder)
        while not cancel_event.is_set():
            time.sleep(0.01)

    monkeypatch.setattr("informity.llm.engine._run_stream_worker", _silent_worker)

    outputs: list[object] = []
    async for item in engine.generate_stream(
        messages=[{"role": "user", "content": "hello"}],
        timeout_seconds=0.05,
    ):
        outputs.append(item)

    assert any(
        isinstance(item, str) and "Response truncated: generation time limit" in item
        for item in outputs
    )
    timeout_markers = [
        item for item in outputs if isinstance(item, tuple) and item[0] == "__timeout__"
    ]
    assert timeout_markers


def _make_stream_worker_that_emits(tokens: list[str]):
    """Return a _run_stream_worker replacement that emits the given token strings."""

    def _worker(
        server,
        messages,
        max_tok,
        temp,
        top_p_val,
        stop_seqs,  # type: ignore[no-untyped-def]
        *args,
    ) -> None:
        """Internal helper for worker."""
        remaining = args[1:] if len(args) >= 5 and args[0] is None else args
        loop = remaining[0]
        queue = remaining[1]
        cancel_event = remaining[3]
        for token in tokens:
            if cancel_event.is_set():
                break
            loop.call_soon_threadsafe(queue.put_nowait, token)
        loop.call_soon_threadsafe(queue.put_nowait, ("__finish_reason__", "stop"))
        loop.call_soon_threadsafe(queue.put_nowait, _STREAM_END)

    return _worker


def _common_engine_monkeypatches(monkeypatch: pytest.MonkeyPatch, worker_fn) -> LLMEngine:  # type: ignore[no-untyped-def]
    """Internal helper for common engine monkeypatches."""
    _force_local_provider(monkeypatch)
    engine = LLMEngine()
    engine._server = object()  # type: ignore[assignment]
    monkeypatch.setattr(
        "informity.llm.engine.get_profile", lambda: SimpleNamespace(context_length=4096)
    )
    monkeypatch.setattr(
        "informity.llm.engine._truncate_messages_to_fit",
        lambda **kwargs: (
            kwargs["messages"],
            {"truncated": False, "original_tokens": 1, "available_budget": 3900},
        ),
    )
    monkeypatch.setattr("informity.llm.engine._run_stream_worker", worker_fn)
    return engine


@pytest.mark.asyncio
async def test_generate_stream_flushes_partial_buffer_at_stream_end(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The think-block filter keeps up to 6 chars buffered to detect split '<think>' tags.
    # On normal stream end (_STREAM_END), those buffered chars must be flushed or the
    # last word of short answers is silently dropped.
    """Test generate stream flushes partial buffer at stream end."""
    engine = _common_engine_monkeypatches(
        monkeypatch,
        _make_stream_worker_that_emits(["Hello", " world"]),
    )
    output = "".join(
        item
        for item in [
            i
            async for i in engine.generate_stream(
                messages=[{"role": "user", "content": "hi"}],
            )
        ]
        if isinstance(item, str)
    )
    assert output == "Hello world", f"Expected full answer, got {output!r}"


@pytest.mark.asyncio
async def test_generate_stream_strips_think_block_from_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Qwen3 reasoning-enabled queries prefix the answer with <think>...</think>.
    # generate_stream must strip the think block; only the answer text is yielded.
    """Test generate stream strips think block from output."""
    tokens = ["<think>", "thinking content here", "</think>", "The answer is 42."]
    engine = _common_engine_monkeypatches(monkeypatch, _make_stream_worker_that_emits(tokens))
    output = "".join(
        item
        for item in [
            i
            async for i in engine.generate_stream(
                messages=[{"role": "user", "content": "question"}],
            )
        ]
        if isinstance(item, str)
    )
    assert "<think>" not in output
    assert "thinking content here" not in output
    assert "The answer is 42." in output


@pytest.mark.asyncio
async def test_generate_stream_no_think_block_passes_through(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # When no think block is present (reasoning disabled via /no_think), every
    # token must pass through to the consumer unmodified.
    """Test generate stream no think block passes through."""
    tokens = ["The", " answer", " is", " 42."]
    engine = _common_engine_monkeypatches(monkeypatch, _make_stream_worker_that_emits(tokens))
    output = "".join(
        item
        for item in [
            i
            async for i in engine.generate_stream(
                messages=[{"role": "user", "content": "question"}],
            )
        ]
        if isinstance(item, str)
    )
    assert output == "The answer is 42."


@pytest.mark.asyncio
async def test_generate_stream_cancellation_cleans_up_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test generate stream cancellation cleans up worker."""
    worker_stopped = threading.Event()

    def _blocking_worker(
        server,
        messages,
        max_tok,
        temp,
        top_p_val,
        stop_seqs,  # type: ignore[no-untyped-def]
        *args,
    ) -> None:
        """Internal helper for blocking worker."""
        remaining = args[1:] if len(args) >= 5 and args[0] is None else args
        loop = remaining[0]
        queue = remaining[1]
        exception_holder = remaining[2]
        cancel_event = remaining[3]
        _ = (server, messages, max_tok, temp, top_p_val, stop_seqs, exception_holder)
        try:
            while not cancel_event.is_set():
                time.sleep(0.01)
        finally:
            worker_stopped.set()
            loop.call_soon_threadsafe(queue.put_nowait, _STREAM_END)

    engine = _common_engine_monkeypatches(monkeypatch, _blocking_worker)
    stream = engine.generate_stream(
        messages=[{"role": "user", "content": "cancel me"}], timeout_seconds=10.0
    )
    next_item = asyncio.create_task(stream.__anext__())

    await asyncio.sleep(0.05)
    next_item.cancel()
    with pytest.raises(asyncio.CancelledError):
        await next_item

    with suppress(Exception):
        await stream.aclose()

    assert worker_stopped.wait(timeout=1.0), "Expected worker to stop after stream cancellation"


@pytest.mark.asyncio
async def test_generate_stream_uses_effective_context_length(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test generate stream uses effective context length."""
    _force_local_provider(monkeypatch)
    captured: dict[str, int] = {}

    def _capture_truncate(**kwargs):  # type: ignore[no-untyped-def]
        """Internal helper for capture truncate."""
        captured["context_length"] = int(kwargs["context_length"])
        return kwargs["messages"], {
            "truncated": False,
            "original_tokens": 1,
            "available_budget": 3900,
        }

    engine = LLMEngine()
    engine._server = object()  # type: ignore[assignment]
    monkeypatch.setattr(
        "informity.llm.engine.get_profile",
        lambda: SimpleNamespace(context_length=24576, generation_tokens_per_second=12.0),
    )
    monkeypatch.setattr("informity.llm.engine.settings.llm_context_length", 8192)
    monkeypatch.setattr("informity.llm.engine._truncate_messages_to_fit", _capture_truncate)
    monkeypatch.setattr(
        "informity.llm.engine._run_stream_worker",
        _make_stream_worker_that_emits(["ok"]),
    )

    _ = [
        item async for item in engine.generate_stream(messages=[{"role": "user", "content": "hi"}])
    ]
    assert captured.get("context_length") == 8192


@pytest.mark.asyncio
async def test_stream_worker_includes_chat_template_kwargs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test stream worker includes chat template kwargs."""
    captured_payload: dict[str, object] = {}

    class _FakeServer:
        def handle_chat_completions(self, payload: str, callback) -> None:  # type: ignore[no-untyped-def]
            """Handle chat completions."""
            nonlocal captured_payload
            captured_payload = json.loads(payload)
            callback({"choices": [{"delta": {}, "finish_reason": "stop"}]})

    monkeypatch.setattr(
        "informity.llm.model_adapter.get_profile",
        lambda: SimpleNamespace(chat_template_kwargs={"enable_thinking": False}),
    )

    queue: asyncio.Queue[str | object] = asyncio.Queue()
    exception_holder: list[BaseException] = []
    _run_stream_worker(
        server=_FakeServer(),
        messages=[{"role": "user", "content": "Hi"}],
        max_tok=32,
        temp=0.0,
        top_p_val=1.0,
        stop_seqs=[],
        loop=asyncio.get_running_loop(),
        queue=queue,
        exception_holder=exception_holder,
        cancel_event=threading.Event(),
        runtime_probe_record=None,
    )

    assert not exception_holder
    assert captured_payload["chat_template_kwargs"] == {"enable_thinking": False}


@pytest.mark.asyncio
async def test_stream_worker_accepts_choice_message_content_chunk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test stream worker accepts choice message content chunk."""
    seen_payload: dict[str, object] = {}

    class _FakeServer:
        def handle_chat_completions(self, payload: str, callback) -> None:  # type: ignore[no-untyped-def]
            """Handle chat completions."""
            nonlocal seen_payload
            seen_payload = json.loads(payload)
            callback({"choices": [{"message": {"content": "Hello"}}]})
            callback({"choices": [{"finish_reason": "stop"}]})

    monkeypatch.setattr(
        "informity.llm.model_adapter.get_profile",
        lambda: SimpleNamespace(chat_template_kwargs={}),
    )

    queue: asyncio.Queue[str | object] = asyncio.Queue()
    exception_holder: list[BaseException] = []
    _run_stream_worker(
        server=_FakeServer(),
        messages=[{"role": "user", "content": "Hi"}],
        max_tok=16,
        temp=0.0,
        top_p_val=1.0,
        stop_seqs=[],
        loop=asyncio.get_running_loop(),
        queue=queue,
        exception_holder=exception_holder,
        cancel_event=threading.Event(),
        runtime_probe_record=None,
    )

    assert not exception_holder
    assert seen_payload["stream"] is True

    emitted: list[str | tuple[str, object]] = []
    while True:
        item = await asyncio.wait_for(queue.get(), timeout=1.0)
        if item is _STREAM_END:
            break
        if isinstance(item, tuple) and len(item) == 2 and item[0] == StreamSignalTag.FINISH_REASON:
            emitted.append(item)
            continue
        emitted.append(str(item))

    assert "Hello" in emitted
    assert (StreamSignalTag.FINISH_REASON, "stop") in emitted


@pytest.mark.asyncio
async def test_stream_worker_propagates_runtime_probe_context_into_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test stream worker propagates runtime probe context into thread."""
    _force_local_provider(monkeypatch)
    probe_context: dict[str, object] = {"runtime_call_probe_enabled": True}
    token = set_runtime_call_probe_context(probe_context)

    class _FakeServer:
        def handle_chat_completions(self, payload: str, callback) -> None:  # type: ignore[no-untyped-def]
            """Handle chat completions."""
            _ = payload
            callback({"choices": [{"delta": {"content": "Hello"}}]})
            callback({"choices": [{"finish_reason": "stop"}]})

    queue: asyncio.Queue[str | object] = asyncio.Queue()
    exception_holder: list[BaseException] = []
    cancel_event = threading.Event()
    thread_context = copy_context()

    try:
        thread = threading.Thread(
            target=thread_context.run,
            args=(
                _run_stream_worker,
                _FakeServer(),
                [{"role": "user", "content": "hi"}],
                8,
                0.0,
                1.0,
                [],
                None,
                asyncio.get_running_loop(),
                queue,
                exception_holder,
                cancel_event,
                None,
                None,
                None,
                None,
                "test-worker",
            ),
            daemon=True,
        )
        thread.start()
        await asyncio.to_thread(thread.join)
    finally:
        reset_runtime_call_probe_context(token)

    assert not exception_holder
    records = probe_context.get("runtime_call_records")
    assert isinstance(records, list)
    assert len(records) == 1
    assert records[0]["call_kind"] == "generate_stream"
    assert records[0]["caller"] == "test-worker"



@pytest.mark.asyncio
async def test_generate_stream_forwards_probe_context_to_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test generate stream forwards probe context to provider."""
    _force_local_provider(monkeypatch)
    engine = LLMEngine()

    probe_context: dict[str, object] = {"runtime_call_probe_enabled": True}
    seen: dict[str, object | None] = {"probe_context": None}

    class _FakeProvider:
        async def generate_stream(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            seen["probe_context"] = kwargs.get("probe_context")
            yield "hello"

    engine._provider = _FakeProvider()  # type: ignore[assignment]
    monkeypatch.setattr(
        "informity.llm.engine.get_profile",
        lambda: SimpleNamespace(context_length=4096),
    )
    monkeypatch.setattr(
        "informity.llm.engine._truncate_messages_to_fit",
        lambda **kwargs: (
            kwargs["messages"],
            {"truncated": False, "original_tokens": 1, "available_budget": 3900},
        ),
    )

    emitted = [
        item
        async for item in engine.generate_stream(
            messages=[{"role": "user", "content": "hi"}],
            probe_context=probe_context,
        )
    ]

    assert emitted == ["hello"]
    assert seen["probe_context"] is probe_context
@pytest.mark.asyncio
async def test_generate_stream_emits_summary_timings_for_local_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test generate stream emits summary timings for local provider."""
    _force_local_provider(monkeypatch)
    monkeypatch.setattr(
        "informity.llm.engine.get_profile",
        lambda: SimpleNamespace(
            context_length=4096,
            generation_tokens_per_second=12.0,
            chat_template_kwargs={},
        ),
    )
    monkeypatch.setattr(
        "informity.llm.engine._truncate_messages_to_fit",
        lambda **kwargs: (
            kwargs["messages"],
            {"truncated": False, "original_tokens": 1, "available_budget": 3900},
        ),
    )

    class _FakeServer:
        def handle_chat_completions(self, payload: str, callback) -> None:  # type: ignore[no-untyped-def]
            """Handle chat completions."""
            _ = payload
            callback({"choices": [{"delta": {"content": "Hello"}}]})
            callback({"choices": [{"finish_reason": "stop"}]})

    engine = LLMEngine()
    engine._server = _FakeServer()  # type: ignore[assignment]

    summary = None
    parts: list[str] = []
    async for item in engine.generate_stream(messages=[{"role": "user", "content": "hi"}]):
        if isinstance(item, tuple) and item[0] == StreamSignalTag.STREAM_SUMMARY:
            summary = item[1]
            continue
        if isinstance(item, str):
            parts.append(item)

    assert "".join(parts) == "Hello"
    assert summary is not None
    assert summary["submit_ms"] is not None
    assert summary["queue_wait_ms"] is not None
    assert summary["first_token_ms"] is not None
    assert summary["total_elapsed_ms"] >= summary["first_token_ms"]


def test_chat_complete_includes_chat_template_kwargs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test chat complete includes chat template kwargs."""
    _force_local_provider(monkeypatch)
    captured_payload: dict[str, object] = {}

    class _FakeServer:
        def handle_chat_completions(self, payload: str, callback) -> None:  # type: ignore[no-untyped-def]
            """Handle chat completions."""
            nonlocal captured_payload
            captured_payload = json.loads(payload)
            callback({"choices": [{"message": {"content": "ok"}}]})

    monkeypatch.setattr(
        "informity.llm.model_adapter.get_profile",
        lambda: SimpleNamespace(chat_template_kwargs={"enable_thinking": False}),
    )

    engine = LLMEngine()
    engine._server = _FakeServer()  # type: ignore[assignment]
    response = engine.chat_complete(
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=16,
        temperature=0.0,
    )

    assert response["choices"][0]["message"]["content"] == "ok"
    assert captured_payload["chat_template_kwargs"] == {"enable_thinking": False}


def test_download_model_uses_httpx_stream_api(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test download model uses httpx stream api."""
    _force_local_provider(monkeypatch)

    class _FakeResponse:
        status_code = 200
        headers = {"Content-Length": "5"}

        def raise_for_status(self) -> None:
            """Raise for status."""
            return None

        def iter_bytes(self, chunk_size: int):  # type: ignore[no-untyped-def]
            """Iter bytes."""
            _ = chunk_size
            yield b"he"
            yield b"llo"

    class _FakeStreamContext:
        def __enter__(self) -> _FakeResponse:
            """Enter the context manager."""
            return _FakeResponse()

        def __exit__(self, _exc_type, _exc, _tb) -> bool:  # type: ignore[no-untyped-def]
            """Exit the context manager."""
            return False

    class _FakeSession:
        called = False

        def stream(self, method: str, url: str, headers: dict, timeout):  # type: ignore[no-untyped-def]
            """Stream."""
            self.called = True
            assert method == "GET"
            assert isinstance(url, str) and url
            assert isinstance(headers, dict)
            assert timeout == (10, 60)
            return _FakeStreamContext()

    fake_session = _FakeSession()
    fake_hf_module = SimpleNamespace(
        hf_hub_url=lambda **kwargs: "https://example.invalid/model.gguf"
    )
    fake_hf_utils = SimpleNamespace(
        build_hf_headers=lambda: {},
        get_session=lambda: fake_session,
    )
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf_module)
    monkeypatch.setitem(sys.modules, "huggingface_hub.utils", fake_hf_utils)
    monkeypatch.setattr("informity.config.configure_hf_environment", lambda **kwargs: None)
    monkeypatch.setattr("informity.llm.engine.remove_models_dir_cache", lambda: None)

    engine = LLMEngine()
    target = tmp_path / "model.gguf"
    progress_calls: list[tuple[int, int | None, float]] = []

    engine._download_model(
        target_path=target,
        repo_id="repo/test",
        filename="model.gguf",
        progress_callback=lambda done, total, speed: progress_calls.append((done, total, speed)),
    )

    assert fake_session.called is True
    assert target.read_bytes() == b"hello"
    assert progress_calls
    assert progress_calls[-1][0] == 5
    assert progress_calls[-1][1] == 5


def test_load_model_caps_context_length_to_configured_limit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test load model caps context length to configured limit."""
    _force_local_provider(monkeypatch)
    captured: dict[str, object] = {}

    class _FakeServer:
        def __init__(self, params) -> None:  # type: ignore[no-untyped-def]
            """Initialize the instance."""
            captured["n_ctx"] = getattr(params, "n_ctx", None)
            captured["n_threads"] = getattr(getattr(params, "cpuparams", None), "n_threads", None)

    class _FakeCommonParams:
        def __init__(self) -> None:
            """Initialize the instance."""
            self.model = SimpleNamespace(path="")
            self.n_ctx = 0
            self.n_gpu_layers = 0
            self.n_batch = 0
            self.cpuparams = SimpleNamespace(n_threads=0)
            self.cpuparams_batch = SimpleNamespace(n_threads=0)
            self.verbosity = 0

    monkeypatch.setitem(
        sys.modules,
        "xllamacpp",
        SimpleNamespace(CommonParams=_FakeCommonParams, Server=_FakeServer),
    )
    monkeypatch.setattr("informity.llm.engine._read_gguf_chat_template", lambda _p: None)
    monkeypatch.setattr(
        "informity.llm.engine.get_profile_for_filename",
        lambda _name: SimpleNamespace(context_length=24576),
    )

    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"gguf")

    monkeypatch.setattr("informity.llm.engine.settings.models_dir", tmp_path)
    monkeypatch.setattr("informity.llm.engine.settings.llm_model_filename", "model.gguf")
    monkeypatch.setattr("informity.llm.engine.settings.llm_context_length", 8192)
    monkeypatch.setattr("informity.llm.engine.settings.llm_cpu_threads", 4)

    engine = LLMEngine()
    engine._load_model()

    assert captured["n_ctx"] == 8192
    assert captured["n_threads"] == 4


def test_ollama_chat_complete_maps_response(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test ollama chat complete maps response."""
    monkeypatch.setattr("informity.llm.engine.settings.llm_provider", "ollama")
    monkeypatch.setattr("informity.llm.engine.settings.llm_model_id", "qwen3:14b")
    monkeypatch.setattr("informity.llm.engine.settings.ollama_base_url", "http://127.0.0.1:11434")
    monkeypatch.setattr("informity.llm.engine.settings.ollama_timeout_seconds", 5.0)

    class _Resp:
        def __enter__(self) -> _Resp:
            """Enter the context manager."""
            return self

        def __exit__(self, _exc_type, _exc, _tb) -> bool:  # type: ignore[no-untyped-def]
            """Exit the context manager."""
            return False

        def read(self) -> bytes:
            """Read."""
            return json.dumps({"message": {"content": "hello from ollama"}}).encode("utf-8")

    captured_payload: dict[str, object] = {}

    def _fake_urlopen(req, *_args, **_kwargs):  # type: ignore[no-untyped-def]
        """Internal helper for fake urlopen."""
        nonlocal captured_payload
        captured_payload = json.loads(req.data.decode("utf-8"))
        return _Resp()

    monkeypatch.setattr("informity.llm.engine.urllib.request.urlopen", _fake_urlopen)

    engine = LLMEngine()
    response = engine.chat_complete(
        messages=[{"role": "user", "content": "hi"}], max_tokens=32, temperature=0.1
    )
    assert response["choices"][0]["message"]["content"] == "hello from ollama"
    assert captured_payload.get("think") is False


@pytest.mark.asyncio
async def test_ollama_generate_stream_maps_tokens_and_finish(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test ollama generate stream maps tokens and finish."""
    monkeypatch.setattr("informity.llm.engine.settings.llm_provider", "ollama")
    monkeypatch.setattr("informity.llm.engine.settings.llm_model_id", "qwen3:14b")
    monkeypatch.setattr("informity.llm.engine.settings.ollama_base_url", "http://127.0.0.1:11434")
    monkeypatch.setattr("informity.llm.engine.settings.ollama_timeout_seconds", 5.0)
    monkeypatch.setattr(
        "informity.llm.engine.get_profile",
        lambda: SimpleNamespace(
            context_length=4096,
            generation_tokens_per_second=12.0,
            chat_template_kwargs={},
        ),
    )

    class _StreamResp:
        def __enter__(self) -> _StreamResp:
            """Enter the context manager."""
            return self

        def __exit__(self, _exc_type, _exc, _tb) -> bool:  # type: ignore[no-untyped-def]
            """Exit the context manager."""
            return False

        def __iter__(self):  # type: ignore[no-untyped-def]
            """Internal helper for iter  ."""
            yield json.dumps({"message": {"content": "Hel"}}).encode("utf-8")
            yield json.dumps({"message": {"content": "lo"}}).encode("utf-8")
            yield json.dumps({"done": True, "done_reason": "stop"}).encode("utf-8")

    captured_payload: dict[str, object] = {}

    def _fake_urlopen(req, *_args, **_kwargs):  # type: ignore[no-untyped-def]
        """Internal helper for fake urlopen."""
        nonlocal captured_payload
        captured_payload = json.loads(req.data.decode("utf-8"))
        return _StreamResp()

    monkeypatch.setattr("informity.llm.engine.urllib.request.urlopen", _fake_urlopen)

    engine = LLMEngine()
    out = "".join(
        item
        for item in [
            i async for i in engine.generate_stream(messages=[{"role": "user", "content": "hi"}])
        ]
        if isinstance(item, str)
    )
    assert out == "Hello"
    assert captured_payload.get("think") is False


@pytest.mark.asyncio
async def test_ollama_generate_stream_emits_summary_timings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test ollama generate stream emits summary timings."""
    monkeypatch.setattr("informity.llm.engine.settings.llm_provider", "ollama")
    monkeypatch.setattr("informity.llm.engine.settings.llm_model_id", "qwen3:14b")
    monkeypatch.setattr("informity.llm.engine.settings.ollama_base_url", "http://127.0.0.1:11434")
    monkeypatch.setattr("informity.llm.engine.settings.ollama_timeout_seconds", 5.0)
    monkeypatch.setattr(
        "informity.llm.engine.get_profile",
        lambda: SimpleNamespace(
            context_length=4096,
            generation_tokens_per_second=12.0,
            chat_template_kwargs={},
        ),
    )

    class _StreamResp:
        def __enter__(self) -> _StreamResp:
            """Enter the context manager."""
            return self

        def __exit__(self, _exc_type, _exc, _tb) -> bool:  # type: ignore[no-untyped-def]
            """Exit the context manager."""
            return False

        def __iter__(self):  # type: ignore[no-untyped-def]
            """Internal helper for iter."""
            yield json.dumps({"message": {"content": "Hel"}}).encode("utf-8")
            yield json.dumps({"message": {"content": "lo"}}).encode("utf-8")
            yield json.dumps({"done": True, "done_reason": "stop"}).encode("utf-8")

    monkeypatch.setattr(
        "informity.llm.engine.urllib.request.urlopen", lambda *_args, **_kwargs: _StreamResp()
    )

    engine = LLMEngine()
    summary = None
    async for item in engine.generate_stream(messages=[{"role": "user", "content": "hi"}]):
        if isinstance(item, tuple) and item[0] == "__stream_summary__":
            summary = item[1]

    assert summary is not None
    assert summary["submit_ms"] is not None
    assert summary["queue_wait_ms"] is not None
    assert summary["queue_wait_ms"] >= 0
    assert summary["first_token_ms"] is not None
    assert summary["total_elapsed_ms"] >= summary["first_token_ms"]


@pytest.mark.asyncio
async def test_ollama_generate_stream_raises_on_empty_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test ollama generate stream raises on empty output."""
    monkeypatch.setattr("informity.llm.engine.settings.llm_provider", "ollama")
    monkeypatch.setattr("informity.llm.engine.settings.llm_model_id", "qwen3:14b")
    monkeypatch.setattr("informity.llm.engine.settings.ollama_base_url", "http://127.0.0.1:11434")
    monkeypatch.setattr("informity.llm.engine.settings.ollama_timeout_seconds", 5.0)
    monkeypatch.setattr(
        "informity.llm.engine.get_profile",
        lambda: SimpleNamespace(context_length=4096, generation_tokens_per_second=12.0),
    )

    class _StreamResp:
        def __enter__(self) -> _StreamResp:
            """Enter the context manager."""
            return self

        def __exit__(self, _exc_type, _exc, _tb) -> bool:  # type: ignore[no-untyped-def]
            """Exit the context manager."""
            return False

        def __iter__(self):  # type: ignore[no-untyped-def]
            """Internal helper for iter  ."""
            yield json.dumps({"done": True, "done_reason": "stop"}).encode("utf-8")

    monkeypatch.setattr(
        "informity.llm.engine.urllib.request.urlopen", lambda *_args, **_kwargs: _StreamResp()
    )

    engine = LLMEngine()
    with pytest.raises(Exception, match="Ollama returned no response tokens"):
        async for _ in engine.generate_stream(messages=[{"role": "user", "content": "hi"}]):
            pass


def test_ollama_chat_complete_connection_error_mapped(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test ollama chat complete connection error mapped."""
    monkeypatch.setattr("informity.llm.engine.settings.llm_provider", "ollama")
    monkeypatch.setattr("informity.llm.engine.settings.llm_model_id", "qwen3:14b")
    monkeypatch.setattr(
        "informity.llm.engine.urllib.request.urlopen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            urllib.error.URLError("connection refused")
        ),
    )

    engine = LLMEngine()
    with pytest.raises(Exception, match="Ollama connection failed"):
        _ = engine.chat_complete(messages=[{"role": "user", "content": "hi"}])
