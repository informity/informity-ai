from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

from informity.llm.engine import LLMEngine, _truncate_messages_to_fit


class _TokenCountingModel:
    metadata: dict[str, str] = {}

    def tokenize(self, data: bytes, add_bos: bool = False, special: bool = False) -> list[str]:
        _ = (add_bos, special)
        return data.decode('utf-8').split()


def test_truncate_messages_removes_history_before_system_content() -> None:
    model = _TokenCountingModel()
    messages = [
        {'role': 'system', 'content': 'You are helpful.\n\nContext:\n[Source: 1] baseline context.'},
        {'role': 'user', 'content': 'old question ' * 45},
        {'role': 'assistant', 'content': 'old answer ' * 45},
        {'role': 'user', 'content': 'current question'},
    ]

    truncated, info = _truncate_messages_to_fit(
        model=model,  # type: ignore[arg-type]
        messages=messages,
        context_length=260,
        max_tokens=100,
        force_chatml=True,
    )

    assert info['truncated'] is True
    assert info['history_messages_removed'] >= 1
    assert truncated[-1]['content'] == 'current question'
    assert len(truncated) < len(messages)


def test_truncate_messages_truncates_system_context_chunks_when_needed() -> None:
    model = _TokenCountingModel()
    system_content = (
        'Rules for answering.\n\nContext:\n'
        '[Source: 1] ' + ('alpha ' * 35) + '\n\n'
        '[Source: 2] ' + ('beta ' * 35) + '\n\n'
        '[Source: 3] ' + ('gamma ' * 35)
    )
    messages = [
        {'role': 'system', 'content': system_content},
        {'role': 'user', 'content': 'What changed?'},
    ]

    truncated, info = _truncate_messages_to_fit(
        model=model,  # type: ignore[arg-type]
        messages=messages,
        context_length=320,
        max_tokens=170,
        force_chatml=True,
    )

    assert info['truncated'] is True
    assert info['system_content_truncated'] is True
    assert info.get('chunks_removed', 0) >= 1
    assert '[Source: 3]' not in truncated[0]['content']


@pytest.mark.asyncio
async def test_generate_stream_emits_timeout_notice_and_marker(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = LLMEngine()
    engine._model = object()  # type: ignore[assignment]

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
        model, messages, max_tok, temp, top_p_val, stop_seqs, loop, queue, exception_holder, cancel_event,
        min_tokens=0, force_chatml=False, extra_eos_tokens=(),  # type: ignore[no-untyped-def]
    ) -> None:
        _ = (model, messages, max_tok, temp, top_p_val, stop_seqs, loop, queue, exception_holder, min_tokens, force_chatml, extra_eos_tokens)
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
