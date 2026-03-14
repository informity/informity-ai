from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from informity.chat_trace import _ChatTraceWriter


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
