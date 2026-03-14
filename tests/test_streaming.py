# ==============================================================================
# Informity AI — Streaming Tests
# Tests LLM streaming functionality
# ==============================================================================

from unittest.mock import MagicMock, patch

import pytest

from informity.llm.streaming import stream_llm


@pytest.mark.asyncio
async def test_stream_llm_calls_engine():
    # Should call llm_engine.generate_stream
    messages = [{'role': 'user', 'content': 'test'}]

    async def mock_stream():
        yield 'token1'
        yield 'token2'
        yield 'token3'

    with patch('informity.llm.streaming.llm_engine') as mock_engine:
        mock_engine.generate_stream = MagicMock(return_value=mock_stream())

        tokens = []
        async for token in stream_llm(messages, max_tokens=100):
            tokens.append(token)

        # Should have called generate_stream
        mock_engine.generate_stream.assert_called_once()
        call_kwargs = mock_engine.generate_stream.call_args[1]
        assert call_kwargs['max_tokens'] == 100
        assert call_kwargs['temperature'] == 0.1  # default

    assert len(tokens) == 3
    assert tokens == ['token1', 'token2', 'token3']


@pytest.mark.asyncio
async def test_stream_llm_passes_parameters():
    # Should pass max_tokens, temperature, timeout, stop_sequences
    messages = [{'role': 'user', 'content': 'test'}]

    async def mock_stream():
        yield 'token'

    with patch('informity.llm.streaming.llm_engine') as mock_engine:
        mock_engine.generate_stream = MagicMock(return_value=mock_stream())

        async for _ in stream_llm(
            messages,
            max_tokens=2048,
            temperature=0.5,
            timeout_seconds=30.0,
            stop_sequences=['<|im_end|>'],
        ):
            pass

        call_kwargs = mock_engine.generate_stream.call_args[1]
        assert call_kwargs['max_tokens'] == 2048
        assert call_kwargs['temperature'] == 0.5
        assert call_kwargs['timeout_seconds'] == 30.0
        assert '<|im_end|>' in call_kwargs['stop']


@pytest.mark.asyncio
async def test_stream_llm_empty_stop_sequences():
    # Should handle empty stop_sequences
    messages = [{'role': 'user', 'content': 'test'}]

    async def mock_stream():
        yield 'token'

    with patch('informity.llm.streaming.llm_engine') as mock_engine:
        mock_engine.generate_stream = MagicMock(return_value=mock_stream())

        async for _ in stream_llm(messages, stop_sequences=None):
            pass

        call_kwargs = mock_engine.generate_stream.call_args[1]
        assert call_kwargs['stop'] == []
