from __future__ import annotations

from typing import Any

import pytest

from informity.llm.rag_runtime import generation_stream as _generation_stream


@pytest.mark.asyncio
async def test_generation_stream_emits_checkpoint_with_query_type_and_summary() -> None:
    async def _fake_stream_llm(*_args: Any, **_kwargs: Any):
        yield 'Token one.'
        yield ' Token two.'

    events: list[str | tuple[str, object]] = []
    summary = None
    async for item in _generation_stream.stream_generation_with_budget(
        messages=[{'role': 'user', 'content': 'test'}],
        max_tokens=256,
        temperature=0.1,
        top_p=0.9,
        timeout_seconds=0,
        stop_sequences=[],
        fit_to_budget_enabled=False,
        stream_soft_limit_ratio=0.8,
        soft_closeout_allowed=False,
        checkpoint_query_type='focused',
        dedupe_insufficient_context_after_stream=False,
        insufficient_context_response='insufficient',
        applied_degradations=[],
        output_contract_plan=None,
        collapse_duplicate_message_fn=lambda value: (value, False),
        stream_llm_fn=_fake_stream_llm,
    ):
        if isinstance(item, tuple) and item[0] == _generation_stream.STREAM_SUMMARY_EVENT:
            summary = item[1]
            continue
        events.append(item)

    checkpoint_events = [
        item[1]
        for item in events
        if isinstance(item, tuple) and len(item) == 2 and item[0] == '__budget_checkpoint__'
    ]
    assert checkpoint_events
    assert all(isinstance(payload, dict) and payload.get('query_type') == 'focused' for payload in checkpoint_events)

    token_events = [event for event in events if isinstance(event, str)]
    assert token_events == ['Token one.', ' Token two.']
    assert isinstance(summary, _generation_stream.StreamExecutionSummary)
    assert summary.token_count == 2
    assert summary.completion_mode == 'complete'


@pytest.mark.asyncio
async def test_generation_stream_does_not_enforce_contract_shape_in_stream_path() -> None:
    async def _fake_stream_llm(*_args: Any, **_kwargs: Any):
        yield '## Executive Summary\nAll requested comparisons are listed.'

    events: list[str | tuple[str, object]] = []
    async for item in _generation_stream.stream_generation_with_budget(
        messages=[{'role': 'user', 'content': 'test'}],
        max_tokens=256,
        temperature=0.1,
        top_p=0.9,
        timeout_seconds=120,
        stop_sequences=[],
        fit_to_budget_enabled=False,
        stream_soft_limit_ratio=0.8,
        soft_closeout_allowed=False,
        checkpoint_query_type='coverage',
        dedupe_insufficient_context_after_stream=False,
        insufficient_context_response='insufficient',
        applied_degradations=[],
        output_contract_plan={
            'requires_missing_evidence_callout': True,
            'min_year_subsections': 3,
            'expected_years': [2021, 2022, 2023],
            'required_terms': ['evidence'],
            'enforce_required_terms': True,
            'required_headings': ['Scope', 'Method', 'Findings'],
            'enforce_required_headings': True,
            'required_table_columns': ['Group', 'Years Covered'],
            'enforce_required_table': True,
        },
        collapse_duplicate_message_fn=lambda value: (value, False),
        stream_llm_fn=_fake_stream_llm,
    ):
        if isinstance(item, tuple) and item[0] == _generation_stream.STREAM_SUMMARY_EVENT:
            continue
        events.append(item)

    merged = ''.join(part for part in events if isinstance(part, str))
    assert '2021' not in merged
    assert '2022' not in merged
    assert 'Required Terms:' not in merged
    assert '## Scope' not in merged
    assert '| Group | Years Covered |' not in merged
    assert 'Missing Evidence:' not in merged


