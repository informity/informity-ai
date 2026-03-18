from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from informity.chat_trace import _ChatTraceWriter


# ==============================================================================
# Summary envelope: plan section
# ==============================================================================

def test_summary_envelope_plan_section_all_none_for_non_planned_query() -> None:
    writer = _ChatTraceWriter(chat_id='c1', message_id='m1')
    writer.record('request', {'question': 'What is revenue?', 'response_mode': 'balanced'})
    envelope = writer.get_summary_envelope()
    plan = envelope.get('plan', {})
    assert plan['answer_sections_count'] is None
    assert plan['steps_requested'] is None
    assert plan['steps_executed'] is None
    assert plan['steps_empty'] is None
    assert plan['aggregation_mode'] is None
    assert plan['output_shape'] is None
    assert plan['planner_latency_ms'] is None


def test_summary_envelope_plan_section_populated_for_planned_query() -> None:
    writer = _ChatTraceWriter(chat_id='c2', message_id='m2')
    writer.record('plan', {
        'answer_sections_count': 3,
        'steps_requested': 2,
        'aggregation_mode': 'compare',
        'output_shape': 'hybrid',
        'planner_latency_ms': 142.5,
    })
    writer.record('multi_step_retrieval', {
        'steps_executed': 2,
        'steps_empty': 0,
    })
    envelope = writer.get_summary_envelope()
    plan = envelope.get('plan', {})
    assert plan['answer_sections_count'] == 3
    assert plan['steps_requested'] == 2
    assert plan['steps_executed'] == 2
    assert plan['steps_empty'] == 0
    assert plan['aggregation_mode'] == 'compare'
    assert plan['output_shape'] == 'hybrid'
    assert plan['planner_latency_ms'] == 142.5


def test_summary_envelope_plan_section_present_in_all_traces() -> None:
    writer = _ChatTraceWriter(chat_id='c3', message_id='m3')
    envelope = writer.get_summary_envelope()
    assert 'plan' in envelope


def test_summary_envelope_existing_fields_unaffected_by_plan_addition() -> None:
    writer = _ChatTraceWriter(chat_id='c4', message_id='m4')
    writer.record('retrieval', {'raw_chunks_count': 8, 'matching_files': 3})
    writer.record('llm', {'total_elapsed_ms': 1200.0, 'token_count': 250})
    envelope = writer.get_summary_envelope()
    assert envelope['retrieval']['raw_chunks_count'] == 8
    assert envelope['llm']['total_elapsed_ms'] == 1200.0
    assert 'plan' in envelope


@pytest.mark.asyncio
async def test_flush_warns_and_skips_evaluation_trace_when_run_id_missing_even_without_steps() -> None:
    writer = _ChatTraceWriter(chat_id='chat-eval', message_id='msg-1', chat_type='evaluation', run_id=None)

    with patch('informity.chat_trace.log.warning') as warning_mock, \
         patch('informity.chat_trace._maybe_prune_traces', new_callable=AsyncMock) as prune_mock:
        await writer.flush()

    warning_mock.assert_called_once()
    prune_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_flush_noops_for_empty_user_trace_without_warning() -> None:
    writer = _ChatTraceWriter(chat_id='chat-user', message_id='msg-2', chat_type='user', run_id=None)

    with patch('informity.chat_trace.log.warning') as warning_mock, \
         patch('informity.chat_trace._maybe_prune_traces', new_callable=AsyncMock) as prune_mock:
        await writer.flush()

    warning_mock.assert_not_called()
    prune_mock.assert_not_awaited()
