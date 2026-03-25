from __future__ import annotations

from typing import Any

import pytest

from informity.llm.rag_runtime import generation_closeout as _generation_closeout
from informity.llm.rag_runtime import generation_stream as _generation_stream
from informity.llm.rag_runtime import generation_terminal as _generation_terminal


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
async def test_generation_stream_enforces_missing_evidence_callout_when_required() -> None:
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
        output_contract_plan={'requires_missing_evidence_callout': True},
        collapse_duplicate_message_fn=lambda value: (value, False),
        stream_llm_fn=_fake_stream_llm,
    ):
        if isinstance(item, tuple) and item[0] == _generation_stream.STREAM_SUMMARY_EVENT:
            continue
        events.append(item)

    merged = ''.join(part for part in events if isinstance(part, str))
    assert 'Missing Evidence:' in merged


@pytest.mark.asyncio
async def test_generation_stream_enforces_min_year_subsections_when_required() -> None:
    async def _fake_stream_llm(*_args: Any, **_kwargs: Any):
        yield '## Findings by Year\n### 2021\n- Evidence: sample.'

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
            'min_year_subsections': 2,
            'expected_years': [2021, 2022],
        },
        collapse_duplicate_message_fn=lambda value: (value, False),
        stream_llm_fn=_fake_stream_llm,
    ):
        if isinstance(item, tuple) and item[0] == _generation_stream.STREAM_SUMMARY_EVENT:
            continue
        events.append(item)

    merged = ''.join(part for part in events if isinstance(part, str))
    assert '2021' in merged
    assert '2022' in merged
    assert 'Missing Evidence:' in merged


@pytest.mark.asyncio
async def test_generation_stream_enforces_required_terms_when_enabled() -> None:
    async def _fake_stream_llm(*_args: Any, **_kwargs: Any):
        yield 'Summary of findings by year and source documents.'

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
            'required_terms': ['evidence'],
            'enforce_required_terms': True,
        },
        collapse_duplicate_message_fn=lambda value: (value, False),
        stream_llm_fn=_fake_stream_llm,
    ):
        if isinstance(item, tuple) and item[0] == _generation_stream.STREAM_SUMMARY_EVENT:
            continue
        events.append(item)

    merged = ''.join(part for part in events if isinstance(part, str))
    assert 'Required Terms: evidence.' in merged


@pytest.mark.asyncio
async def test_generation_stream_skips_required_term_enforcement_when_disabled() -> None:
    async def _fake_stream_llm(*_args: Any, **_kwargs: Any):
        yield 'Summary of findings by year and source documents.'

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
            'required_terms': ['evidence'],
            'enforce_required_terms': False,
        },
        collapse_duplicate_message_fn=lambda value: (value, False),
        stream_llm_fn=_fake_stream_llm,
    ):
        if isinstance(item, tuple) and item[0] == _generation_stream.STREAM_SUMMARY_EVENT:
            continue
        events.append(item)

    merged = ''.join(part for part in events if isinstance(part, str))
    assert 'Required Terms: evidence.' not in merged


def test_generation_closeout_metrics_payload_contract_shape() -> None:
    payload = _generation_closeout.build_generation_metrics_payload(
        query_type='focused',
        timeout_seconds=120,
        retrieval_elapsed_ms=42.34,
        prompt_elapsed_ms=11.11,
        first_token_ms=123.45,
        llm_elapsed_ms=456.78,
        timeout_reason=None,
        checkpoints_hit=[60, 80],
        completion_mode='complete',
        preflight_projected_seconds=8.9,
        preflight_ratio=0.21,
        post_retrieval_projected_seconds=9.8,
        post_retrieval_ratio=0.33,
        fit_to_budget_rollout_stage='stage1',
        fit_to_budget_enabled=True,
        fit_to_budget_sample_count=100,
        fit_to_budget_timeout_rate=0.02,
        fit_to_budget_first_token_p95_ms=900.0,
        fit_to_budget_completion_p95_seconds=12.5,
        applied_degradations=[],
        fallback_events=[],
        has_remaining_scope=False,
        stream_recovery_reason=None,
    )
    assert payload['generation_skipped'] is False
    assert payload['query_type'] == 'focused'
    assert payload['first_token_latency_ms'] == 123.5
    assert payload['soft_budget_checkpoints_hit'] == [60, 80]


def test_generation_terminal_builds_generation_skipped_payload_and_limited_sources() -> None:
    payload = _generation_terminal.build_generation_skipped_metrics_payload(
        query_type='coverage',
        timeout_seconds=90,
        retrieval_elapsed_ms=55.55,
        preflight_projected_seconds=20.0,
        preflight_ratio=0.5,
        applied_degradations=[],
        fallback_events=[],
        has_remaining_scope=True,
        validation_gates={'retrieval_relevance_gate': False},
    )
    assert payload['generation_skipped'] is True
    assert payload['suggested_completion_mode'] == 'complete'
    assert payload['validation_gates'] == {'retrieval_relevance_gate': False}

    sources = _generation_terminal.build_limited_fallback_sources(
        chunks=[
            {'filename': 'a.txt', 'file_path': '/a.txt', 'chunk_text': 'A', 'score': 1.0},
            {'filename': 'b.txt', 'file_path': '/b.txt', 'chunk_text': 'B', 'score': 2.0},
            {'filename': 'c.txt', 'file_path': '/c.txt', 'chunk_text': 'C', 'score': 3.0},
        ],
        limit=2,
        truncate_preview_fn=lambda text: text,
        normalize_relevance_score_fn=lambda value: float(value),
    )
    assert len(sources) == 2
    assert [source.filename for source in sources] == ['a.txt', 'b.txt']
