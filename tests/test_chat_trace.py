from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from informity.chat_trace import _ChatTraceWriter

# ==============================================================================
# Summary envelope: no planning artifacts
# ==============================================================================

def test_summary_envelope_has_no_plan_section() -> None:
    writer = _ChatTraceWriter(chat_id='c1', message_id='m1')
    writer.record('request', {'question': 'What is revenue?'})
    envelope = writer.get_summary_envelope()
    assert 'plan' not in envelope


def test_summary_envelope_existing_fields_unaffected_by_plan_addition() -> None:
    writer = _ChatTraceWriter(chat_id='c2', message_id='m2')
    writer.record('retrieval', {'raw_chunks_count': 8, 'matching_files': 3})
    writer.record('llm', {'total_elapsed_ms': 1200.0, 'token_count': 250})
    envelope = writer.get_summary_envelope()
    assert envelope['retrieval']['raw_chunks_count'] == 8
    assert envelope['llm']['total_elapsed_ms'] == 1200.0
    assert 'plan' not in envelope


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
