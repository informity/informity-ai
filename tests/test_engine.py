from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

from informity.llm.engine import LLMEngine, _truncate_messages_to_fit


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
        server, chat_template, messages, max_tok, temp, top_p_val, stop_seqs,  # type: ignore[no-untyped-def]
        loop, queue, exception_holder, cancel_event, force_chatml=False,
    ) -> None:
        _ = (server, chat_template, messages, max_tok, temp, top_p_val, stop_seqs, loop, queue, exception_holder, force_chatml)
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
