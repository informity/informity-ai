"""
Invariant: OutputContract failure MUST NOT mutate continuation control fields.

When _evaluate_output_contract returns passed=False, the StreamExecutionSummary must:
  - has_remaining_scope is False   (no silent continuation trigger)
  - completion_mode == 'complete'  (not 'scoped_complete')
  - output_contract_check['passed'] is False   (failure is visible in trace)
  - applied_degradations contains 'strict_output_contract_incomplete'  (failure is logged)
"""

from __future__ import annotations

from collections.abc import AsyncGenerator

from informity.llm.rag_runtime.generation_stream import (
    STREAM_SUMMARY_EVENT,
    StreamExecutionSummary,
    stream_generation_with_budget,
)
from informity.llm.rag_runtime.strict_output_contract import _build_output_contract_plan


async def _mock_stream_llm_missing_heading(
    messages: list,
    *,
    max_tokens: int,
    temperature: float,
    top_p: float,
    timeout_seconds: int,
    stop_sequences: list,
) -> AsyncGenerator[str, None]:
    """Yields a response that satisfies only the first required heading, not the second."""
    tokens = ['## ', 'Scope\n', 'Some scope details here.\n']
    for token in tokens:
        yield token


async def _mock_stream_llm_complete(
    messages: list,
    *,
    max_tokens: int,
    temperature: float,
    top_p: float,
    timeout_seconds: int,
    stop_sequences: list,
) -> AsyncGenerator[str, None]:
    """Yields a response that satisfies all required headings."""
    tokens = ['## ', 'Scope\n', 'Details.\n\n', '## ', 'Method\n', 'More details.\n']
    for token in tokens:
        yield token


def _no_op_collapse(answer: str) -> tuple[str, bool]:
    return answer, False


async def _collect_summary(gen: AsyncGenerator) -> tuple[list[str], StreamExecutionSummary]:
    tokens: list[str] = []
    summary: StreamExecutionSummary | None = None
    async for item in gen:
        if isinstance(item, tuple) and item[0] == STREAM_SUMMARY_EVENT:
            summary = item[1]
        elif isinstance(item, str):
            tokens.append(item)
    assert summary is not None, 'Stream did not emit a StreamExecutionSummary'
    return tokens, summary


def _make_plan(headings: list[str]):
    requirements = ['use the required headings exactly and in the requested order'] + [
        f'include heading: {h}' for h in headings
    ]
    return _build_output_contract_plan(
        question=' '.join(f'{i+1}) {h}' for i, h in enumerate(headings)),
        format_requirements=requirements,
    )


async def test_contract_failure_does_not_set_has_remaining_scope() -> None:
    """Core invariant: has_remaining_scope must be False when only contract check fails."""
    applied_degradations: list[dict] = []
    plan = _make_plan(['Scope', 'Method'])

    _, summary = await _collect_summary(
        stream_generation_with_budget(
            messages=[{'role': 'user', 'content': 'q'}],
            max_tokens=256,
            temperature=0.0,
            top_p=1.0,
            timeout_seconds=30,
            stop_sequences=[],
            fit_to_budget_enabled=False,
            stream_soft_limit_ratio=0.8,
            soft_closeout_allowed=False,
            checkpoint_query_type=None,
            dedupe_insufficient_context_after_stream=False,
            insufficient_context_response='',
            applied_degradations=applied_degradations,
            output_contract_plan=plan,
            collapse_duplicate_message_fn=_no_op_collapse,
            stream_llm_fn=_mock_stream_llm_missing_heading,
        )
    )

    assert summary.has_remaining_scope is False, (
        'Contract failure must not set has_remaining_scope=True'
    )


async def test_contract_failure_does_not_set_scoped_complete() -> None:
    """completion_mode must remain 'complete' when only the output contract check fails."""
    applied_degradations: list[dict] = []
    plan = _make_plan(['Scope', 'Method'])

    _, summary = await _collect_summary(
        stream_generation_with_budget(
            messages=[{'role': 'user', 'content': 'q'}],
            max_tokens=256,
            temperature=0.0,
            top_p=1.0,
            timeout_seconds=30,
            stop_sequences=[],
            fit_to_budget_enabled=False,
            stream_soft_limit_ratio=0.8,
            soft_closeout_allowed=False,
            checkpoint_query_type=None,
            dedupe_insufficient_context_after_stream=False,
            insufficient_context_response='',
            applied_degradations=applied_degradations,
            output_contract_plan=plan,
            collapse_duplicate_message_fn=_no_op_collapse,
            stream_llm_fn=_mock_stream_llm_missing_heading,
        )
    )

    assert summary.completion_mode == 'complete', (
        f"Expected 'complete', got '{summary.completion_mode}' — contract failure must not set scoped_complete"
    )


async def test_contract_failure_surfaces_in_trace_fields() -> None:
    """Failure must be visible in output_contract_check and applied_degradations (trace/metrics only)."""
    applied_degradations: list[dict] = []
    plan = _make_plan(['Scope', 'Method'])

    _, summary = await _collect_summary(
        stream_generation_with_budget(
            messages=[{'role': 'user', 'content': 'q'}],
            max_tokens=256,
            temperature=0.0,
            top_p=1.0,
            timeout_seconds=30,
            stop_sequences=[],
            fit_to_budget_enabled=False,
            stream_soft_limit_ratio=0.8,
            soft_closeout_allowed=False,
            checkpoint_query_type=None,
            dedupe_insufficient_context_after_stream=False,
            insufficient_context_response='',
            applied_degradations=applied_degradations,
            output_contract_plan=plan,
            collapse_duplicate_message_fn=_no_op_collapse,
            stream_llm_fn=_mock_stream_llm_missing_heading,
        )
    )

    assert summary.output_contract_check.get('passed') is False, (
        'Contract failure must be recorded in output_contract_check'
    )
    degradation_steps = [d.get('step') for d in applied_degradations]
    assert 'strict_output_contract_incomplete' in degradation_steps, (
        'Contract failure must append strict_output_contract_incomplete degradation'
    )


async def test_contract_pass_leaves_continuation_fields_unchanged() -> None:
    """Sanity: when the contract passes, has_remaining_scope stays False and mode is 'complete'."""
    applied_degradations: list[dict] = []
    plan = _make_plan(['Scope', 'Method'])

    _, summary = await _collect_summary(
        stream_generation_with_budget(
            messages=[{'role': 'user', 'content': 'q'}],
            max_tokens=256,
            temperature=0.0,
            top_p=1.0,
            timeout_seconds=30,
            stop_sequences=[],
            fit_to_budget_enabled=False,
            stream_soft_limit_ratio=0.8,
            soft_closeout_allowed=False,
            checkpoint_query_type=None,
            dedupe_insufficient_context_after_stream=False,
            insufficient_context_response='',
            applied_degradations=applied_degradations,
            output_contract_plan=plan,
            collapse_duplicate_message_fn=_no_op_collapse,
            stream_llm_fn=_mock_stream_llm_complete,
        )
    )

    assert summary.output_contract_check.get('passed') is True
    assert summary.has_remaining_scope is False
    assert summary.completion_mode == 'complete'
    degradation_steps = [d.get('step') for d in applied_degradations]
    assert 'strict_output_contract_incomplete' not in degradation_steps
