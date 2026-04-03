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
from informity.db.models import ChatMessage
from informity.llm.model_adapter import get_profile
from informity.llm.prompt_builder import build_messages, resolve_history_limit
from informity.llm.query_classifier import QueryClassification
from informity.llm.streaming import stream_llm
from informity.llm.types import QueryType, StreamSignalTag

log = structlog.get_logger(__name__)
_HANDLER_RUNTIME_EXCEPTIONS = (RuntimeError, ValueError, TypeError, OSError, asyncio.TimeoutError)

# Assistant mode system prompt (no corpus/index access)
_ASSISTANT_SYSTEM_PROMPT = """You are a helpful AI assistant. Answer conversationally, clearly, and directly.

You have no access to indexed documents, local files, or any private corpus unless the user explicitly provides content in this chat.
If asked to search files or cite corpus evidence, explain briefly that this is direct assistant chat without document retrieval.

Keep responses concise."""

# Researcher-simple prompt remains corpus-aware for non-RAG simple replies.
_RESEARCHER_SIMPLE_SYSTEM_PROMPT = """You are a helpful AI assistant. Answer questions conversationally and helpfully.

You have access to a private document corpus.
Answer conversationally and directly. You do not need to cite documents for casual or conversational replies.
If asked about document search capabilities, describe them accurately but briefly.

Keep responses concise."""


class SimpleHandler:
    """
    Handler for simple/conversational queries.

    Skips retrieval entirely and uses LLM directly with minimal context.
    """

    def matches(self, classification: QueryClassification) -> bool:
        """Match simple/conversational queries."""
        return classification.intent == QueryType.SIMPLE

    async def handle(
        self,
        question:       str,
        classification: QueryClassification,
        history:        list[ChatMessage] | None,
        db:             aiosqlite.Connection,
        trace:          object | None,
        diagnostics_context: dict[str, object] | None = None,
        chat_mode: str | None = None,
    ) -> AsyncGenerator[str | list[ChatSourceReference] | tuple[str, object]]:
        """
        Handle simple query by using LLM directly without retrieval.

        Uses minimal system prompt and simple query settings (lower token budget,
        shorter timeout) since no document context is needed.
        """
        try:
            profile = get_profile()
            query_type = QueryType.SIMPLE
            normalized_chat_mode = str(chat_mode or '').strip().lower()
            system_prompt = _ASSISTANT_SYSTEM_PROMPT if normalized_chat_mode == 'assistant' else _RESEARCHER_SIMPLE_SYSTEM_PROMPT

            if trace is not None:
                trace.record('intent', {
                    'model_profile':     profile.name,
                    'intent':            classification.intent,
                    'query_type':        query_type,
                    'simple_mode':       True,
                    'chat_mode':         normalized_chat_mode or 'researcher',
                    'db_attached':       db is not None,
                })

            # Build messages via shared prompt-builder path so assistant/researcher
            # simple chats also benefit from token-budget-aware history trimming.
            messages = build_messages(
                question=question,
                context_chunks=[],
                history=history,
                model_profile=profile,
                system_prompt=system_prompt,
                chat_mode=normalized_chat_mode,
            )

            # Prepare messages according to model profile (handles prompt format, etc.)
            messages = profile.prepare_messages(messages, query_type)

            if trace is not None:
                trace.record('prompt', {
                    'messages_count':    len(messages),
                    'context_chunks':    0,  # No context for simple queries
                    'history_messages':   len(history) if history else 0,
                    'effective_history_limit': resolve_history_limit(normalized_chat_mode),
                    'chat_mode': normalized_chat_mode or 'researcher',
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

            yield (StreamSignalTag.METRICS, {
                'query_type': QueryType.SIMPLE,
                'raw_chunks_count': 0,
            })
            yield sources

        except _HANDLER_RUNTIME_EXCEPTIONS as exc:
            log.error('simple_handler_failed', error=str(exc), exc_info=True)
            yield f"Error: {exc}"
            yield []
