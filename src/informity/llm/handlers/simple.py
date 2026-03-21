# ==============================================================================
# Informity AI — Simple Query Handler
# Handles conversational queries (greetings, clarifications, off-topic) without retrieval
# ==============================================================================

import asyncio
import time
from collections.abc import AsyncGenerator

import aiosqlite
import structlog

from informity.api.schemas import ChatSourceReference
from informity.config import settings
from informity.db.models import ChatMessage
from informity.llm.model_adapter import get_profile
from informity.llm.query_classifier import QueryClassification
from informity.llm.streaming import stream_llm

log = structlog.get_logger(__name__)
_HANDLER_RUNTIME_EXCEPTIONS = (RuntimeError, ValueError, TypeError, OSError, asyncio.TimeoutError)

# Minimal system prompt for simple queries (no document context)
_SIMPLE_SYSTEM_PROMPT = """You are a helpful AI assistant. Answer questions conversationally and helpfully.

If asked about document search or file indexing capabilities, explain that you can:
- Answer questions about indexed documents
- Search for specific information across files
- List and enumerate files by metadata (year, category, file type)
- Compare and analyze content across multiple documents
- Describe capabilities in general terms only; do not invent technical limits or internal implementation details

Keep responses concise and friendly."""


class SimpleHandler:
    """
    Handler for simple/conversational queries.

    Skips retrieval entirely and uses LLM directly with minimal context.
    """

    def matches(self, classification: QueryClassification) -> bool:
        """Match simple/conversational queries."""
        return classification.intent == 'simple'

    async def handle(
        self,
        question:       str,
        classification: QueryClassification,
        history:        list[ChatMessage] | None,
        db:             aiosqlite.Connection,
        trace:          object | None,
        diagnostics_context: dict[str, object] | None = None,
    ) -> AsyncGenerator[str | list[ChatSourceReference] | tuple[str, object]]:
        """
        Handle simple query by using LLM directly without retrieval.

        Uses minimal system prompt and simple query settings (lower token budget,
        shorter timeout) since no document context is needed.
        """
        try:
            profile = get_profile()
            query_type = 'simple'

            if trace is not None:
                trace.record('intent', {
                    'model_profile':     profile.name,
                    'intent':            classification.intent,
                    'query_type':        query_type,
                    'simple_mode':       True,
                    'db_attached':       db is not None,
                })

            # Build minimal messages (system prompt + history + question)
            messages = [{'role': 'system', 'content': _SIMPLE_SYSTEM_PROMPT}]

            # Add history (truncate if needed)
            if history:
                history_limit = settings.chat_history_messages
                for msg in history[-history_limit:]:  # Last N messages (configurable)
                    messages.append({'role': msg.role, 'content': msg.content})

            # Add current question
            messages.append({'role': 'user', 'content': question})

            # Prepare messages according to model profile (handles prompt format, etc.)
            messages = profile.prepare_messages(messages, query_type)

            if trace is not None:
                trace.record('prompt', {
                    'messages_count':    len(messages),
                    'context_chunks':    0,  # No context for simple queries
                    'history_messages':   len(history) if history else 0,
                    'reasoning_enabled':  False,  # Simple queries never use reasoning
                })

            # Get model profile settings for simple queries
            max_tokens = profile.get_max_tokens(query_type)
            timeout_seconds = profile.get_timeout_seconds(query_type)
            stop_sequences = profile.get_stop_sequences(reasoning_enabled=False)

            # Stream response
            llm_start = time.perf_counter()
            token_count = 0
            async for token in stream_llm(
                messages,
                max_tokens=max_tokens,
                temperature=profile.temperature,
                top_p=profile.top_p,
                timeout_seconds=timeout_seconds,
                stop_sequences=stop_sequences,
            ):
                token_count += 1
                yield token

            llm_elapsed_ms = (time.perf_counter() - llm_start) * 1000
            if trace is not None:
                trace.record('llm', {
                    'token_count':       token_count,
                    'max_tokens':        max_tokens,
                    'total_elapsed_ms':  round(llm_elapsed_ms, 1),
                    'model_profile':     profile.name,
                })

            # Simple queries have no sources
            sources: list[ChatSourceReference] = []
            if trace is not None:
                trace.record('sources', {
                    'count':   0,
                    'sources': [],
                })

            yield ('__metrics__', {
                'query_type': 'simple',
                'raw_chunks_count': 0,
            })
            yield sources

        except _HANDLER_RUNTIME_EXCEPTIONS as exc:
            log.error('simple_handler_failed', error=str(exc), exc_info=True)
            yield f"Error: {exc}"
            yield []
