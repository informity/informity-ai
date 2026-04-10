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
from informity.llm.chat_mode import is_assistant_mode, resolve_chat_mode
from informity.llm.metrics_payload import build_metrics_payload
from informity.llm.model_adapter import get_profile
from informity.llm.prompt_builder import build_messages, resolve_history_limit
from informity.llm.query_classifier import QueryClassification
from informity.llm.streaming import stream_llm
from informity.llm.system_prompts import (
    SIMPLE_ASSISTANT_SYSTEM_PROMPT,
    SIMPLE_ASSISTANT_WEB_SEARCH_SYNTHESIS_PROMPT,
    SIMPLE_RESEARCHER_SYSTEM_PROMPT,
)
from informity.llm.types import QueryType, StreamSignalTag
from informity.llm.user_messages import get_web_search_status_message
from informity.llm.web_search import format_search_context, search_web

log = structlog.get_logger(__name__)
_HANDLER_RUNTIME_EXCEPTIONS = (RuntimeError, ValueError, TypeError, OSError, asyncio.TimeoutError)


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
        chat_web_search_enabled: bool = False,
        chat_web_search_privacy_override: bool = False,
    ) -> AsyncGenerator[str | list[ChatSourceReference] | tuple[str, object]]:
        """
        Handle simple query by using LLM directly without retrieval.

        Uses minimal system prompt and simple query settings (lower token budget,
        shorter timeout) since no document context is needed.
        """
        try:
            profile = get_profile()
            query_type = QueryType.SIMPLE
            normalized_chat_mode = resolve_chat_mode(chat_mode)
            system_prompt = (
                SIMPLE_ASSISTANT_SYSTEM_PROMPT
                if is_assistant_mode(normalized_chat_mode)
                else SIMPLE_RESEARCHER_SYSTEM_PROMPT
            )
            allow_assistant_web_search = (
                is_assistant_mode(normalized_chat_mode)
                and bool(chat_web_search_enabled)
                and (not bool(settings.full_privacy) or bool(chat_web_search_privacy_override))
                and bool(str(settings.tavily_api_key or '').strip())
            )

            if trace is not None:
                trace.record('intent', {
                    'model_profile':     profile.name,
                    'intent':            classification.intent,
                    'query_type':        query_type,
                    'simple_mode':       True,
                    'chat_mode':         normalized_chat_mode or 'researcher',
                    'db_attached':       db is not None,
                })

            # Get model profile settings for simple queries
            max_tokens = profile.get_max_tokens(query_type)
            timeout_seconds = profile.get_timeout_seconds(query_type)
            stop_sequences = profile.get_stop_sequences(reasoning_enabled=False)

            llm_start = time.perf_counter()
            token_count = 0
            web_search_used = False

            response_question = question
            response_system_prompt = system_prompt
            should_use_web_search = (
                allow_assistant_web_search
                and bool(chat_web_search_enabled)
            )
            web_search_tokens_used: int | None = None
            web_search_tokens_limit: int | None = None
            web_search_tokens_label: str | None = None
            web_search_status: str | None = None

            if should_use_web_search:
                web_search_used = True
                yield (
                    StreamSignalTag.SEARCHING_STATUS,
                    {'message': 'Searching the web...'},
                )
                web_outcome = await asyncio.to_thread(
                    search_web,
                    question,
                    allow_privacy_override=bool(chat_web_search_privacy_override),
                )
                web_search_status = str(web_outcome.status or '').strip() or 'ok'
                web_search_tokens_used = web_outcome.usage_used
                web_search_tokens_limit = web_outcome.usage_limit
                if (
                    isinstance(web_search_tokens_used, int)
                    and isinstance(web_search_tokens_limit, int)
                    and web_search_tokens_limit > 0
                ):
                    web_search_tokens_label = f'{web_search_tokens_used}/{web_search_tokens_limit}'
                if web_search_status != 'ok':
                    fallback_message = get_web_search_status_message(web_search_status)
                    if trace is not None:
                        trace.record('web_search', {
                            'status': web_search_status,
                            'tokens_used': web_search_tokens_used,
                            'tokens_limit': web_search_tokens_limit,
                            'result_count': 0,
                        })
                    yield (
                        StreamSignalTag.METRICS,
                        build_metrics_payload(
                            query_type=QueryType.SIMPLE,
                            raw_chunks_count=0,
                            web_search_used=True,
                            web_search_status=web_search_status,
                            web_search_tokens_used=web_search_tokens_used,
                            web_search_tokens_limit=web_search_tokens_limit,
                            web_search_tokens_label=web_search_tokens_label,
                        ),
                    )
                    yield fallback_message
                    yield []
                    return
                search_context = format_search_context(web_outcome.results)
                response_question = f"{question}\n\n{search_context}"
                response_system_prompt = SIMPLE_ASSISTANT_WEB_SEARCH_SYNTHESIS_PROMPT

            # Build messages via shared prompt-builder path so assistant/researcher
            # simple chats also benefit from token-budget-aware history trimming.
            messages = build_messages(
                question=response_question,
                context_chunks=[],
                history=history,
                model_profile=profile,
                system_prompt=response_system_prompt,
                chat_mode=normalized_chat_mode,
            )
            messages = profile.prepare_messages(messages, query_type)

            if trace is not None:
                trace.record('prompt', {
                    'messages_count':    len(messages),
                    'context_chunks':    0,  # No context for simple queries
                    'history_messages':   len(history) if history else 0,
                    'effective_history_limit': resolve_history_limit(normalized_chat_mode),
                    'chat_mode': normalized_chat_mode or 'researcher',
                    'reasoning_enabled':  False,  # Simple queries never use reasoning
                    'web_search_eligible': allow_assistant_web_search,
                    'web_search_triggered': web_search_used,
                    'web_search_status': web_search_status,
                    'web_search_tokens_used': web_search_tokens_used,
                    'web_search_tokens_limit': web_search_tokens_limit,
                })

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
                    'web_search_used':   web_search_used,
                    'web_search_status': web_search_status,
                    'web_search_tokens_used': web_search_tokens_used,
                    'web_search_tokens_limit': web_search_tokens_limit,
                })

            # Simple queries have no sources
            sources: list[ChatSourceReference] = []
            if trace is not None:
                trace.record('sources', {
                    'count':   0,
                    'sources': [],
                })

            yield (
                StreamSignalTag.METRICS,
                build_metrics_payload(
                    query_type=QueryType.SIMPLE,
                    raw_chunks_count=0,
                    web_search_used=web_search_used,
                    web_search_status=web_search_status,
                    web_search_tokens_used=web_search_tokens_used,
                    web_search_tokens_limit=web_search_tokens_limit,
                    web_search_tokens_label=web_search_tokens_label,
                ),
            )
            yield sources

        except _HANDLER_RUNTIME_EXCEPTIONS as exc:
            log.error('simple_handler_failed', error=str(exc), exc_info=True)
            yield f"Error: {exc}"
            yield []
