# ==============================================================================
# Informity AI — Query Router (v2)
# Routes queries to appropriate handlers (metadata, RAG, simple)
# ==============================================================================

import asyncio
import dataclasses
from collections.abc import AsyncGenerator

import aiosqlite
import structlog

from informity.api.schemas import ChatSourceReference
from informity.db.models import ChatMessage
from informity.db.sqlite import get_chunk_count
from informity.llm.chat_mode import is_assistant_mode, resolve_chat_mode
from informity.llm.handlers.metadata import MetadataHandler
from informity.llm.handlers.rag import RAGHandler
from informity.llm.handlers.simple import SimpleHandler
from informity.llm.query_classifier import QueryClassification, classify_query
from informity.llm.types import QueryType, StreamSignalTag
from informity.llm.user_messages import EMPTY_KNOWLEDGE_BASE_RESEARCHER_MESSAGE

log = structlog.get_logger(__name__)
_ROUTER_RUNTIME_EXCEPTIONS = (aiosqlite.Error, RuntimeError, ValueError, TypeError, OSError, TimeoutError)

# Handler registry - order matters (first match wins)
_HANDLER_REGISTRY = [
    MetadataHandler(),  # Metadata queries (count, enumeration, file listing)
    SimpleHandler(),    # Simple/conversational queries (greetings, clarifications, off-topic)
    RAGHandler(),       # Focused and coverage queries (fallback - should always match)
]


async def answer_question(
    question: str,
    chat_id: str | None = None,
    history: list[ChatMessage] | None = None,
    db: aiosqlite.Connection | None = None,
    trace: object | None = None,  # TraceWriter protocol - optional, for chat trace logging
    diagnostics_context: dict[str, object] | None = None,
    classification: QueryClassification | None = None,  # If provided, skip re-classification (continuation passes)
    chat_mode: str | None = None,
    chat_web_search_enabled: bool = False,
    chat_web_search_privacy_override: bool = False,
) -> AsyncGenerator[str | list[ChatSourceReference] | tuple[str, object]]:
    """
    Query router - dispatches queries to appropriate handlers.
    """
    if db is None:
        yield "Error: No database connection provided."
        yield []
        return

    try:
        normalized_chat_mode = resolve_chat_mode(chat_mode)

        if is_assistant_mode(normalized_chat_mode):
            classify_elapsed_ms = 0.0
            base_classification = classification
            if base_classification is None:
                classify_start = asyncio.get_running_loop().time()
                base_classification = await asyncio.to_thread(classify_query, question)
                classify_elapsed_ms = (asyncio.get_running_loop().time() - classify_start) * 1000.0

            # Assistant always routes to SimpleHandler, but we preserve PromptCue
            # freshness/action signals on the forced-simple classification.
            if isinstance(base_classification, QueryClassification):
                forced_classification = dataclasses.replace(
                    base_classification,
                    intent=QueryType.SIMPLE,
                )
            else:
                forced_classification = QueryClassification(intent=QueryType.SIMPLE)
            if trace is not None:
                trace.record('classification', {
                    'query_length': len(question),
                    'intent': QueryType.SIMPLE,
                    'route_candidate': forced_classification.route_candidate,
                    'confidence': forced_classification.confidence,
                    'duration_ms': round(classify_elapsed_ms, 2),
                    'chat_mode': 'assistant',
                    'forced': True,
                    'needs_current_info': forced_classification.needs_current_info,
                    'should_check_recency': bool(
                        forced_classification.action_hints.get('should_check_recency')
                    ),
                    'mentions_time': forced_classification.mentions_time,
                })
            log.info(
                'query_classified_forced_assistant',
                intent=QueryType.SIMPLE,
                chat_mode='assistant',
            )
            handler = SimpleHandler()
            async for item in handler.handle(
                question=question,
                classification=forced_classification,
                history=history,
                db=db,
                trace=trace,
                diagnostics_context=diagnostics_context,
                chat_mode='assistant',
                chat_web_search_enabled=chat_web_search_enabled,
                chat_web_search_privacy_override=chat_web_search_privacy_override,
            ):
                yield item
            return

        # 1. Classify query (extract filters and intent)
        if classification is None:
            classify_start = asyncio.get_running_loop().time()
            classification = await asyncio.to_thread(classify_query, question)
            classify_elapsed_ms = (asyncio.get_running_loop().time() - classify_start) * 1000.0
            if trace is not None:
                trace.record('classification', {
                    'query_length': len(question),
                    'intent': classification.intent,
                    'route_candidate': classification.route_candidate,
                    'confidence': classification.confidence,
                    'confidence_band': classification.confidence_band,
                    'alternatives': classification.alternatives,
                    'reason_codes': classification.reason_codes,
                    'missing_slots': classification.missing_slots,
                    'subtype': classification.subtype,
                    'group_by': classification.group_by,
                    'field_hint': classification.field_hint,
                    'source_terms': classification.source_terms,
                    'has_multi_year_scope': classification.has_multi_year_scope,
                    'year_filter': classification.year_filter,
                    'category_filter': classification.category_filter,
                    'file_type_filter': classification.file_type_filter,
                    'filename_filter': classification.filename_filter,
                    'duration_ms': round(classify_elapsed_ms, 2),
                    'chat_mode': normalized_chat_mode or 'researcher',
                })
            log.info(
                'query_classified',
                intent=classification.intent,
                route_candidate=classification.route_candidate,
                confidence=classification.confidence,
                year_filter=classification.year_filter,
                category_filter=classification.category_filter,
                chat_mode=normalized_chat_mode or 'researcher',
                duration_ms=round(classify_elapsed_ms, 1),
            )
            yield (StreamSignalTag.CLASSIFICATION, classification)
        else:
            log.info(
                'query_classified_locked',
                intent=classification.intent,
                route_candidate=classification.route_candidate,
                confidence=classification.confidence,
                chat_mode=normalized_chat_mode or 'researcher',
            )

        total_chunks = await get_chunk_count(db)
        if total_chunks == 0:
            log.info(
                'researcher_empty_index_short_circuit',
                intent=classification.intent,
                route_candidate=classification.route_candidate,
                chat_mode=normalized_chat_mode or 'researcher',
            )
            if trace is not None:
                trace.record('empty_index_gate', {
                    'query_length': len(question),
                    'intent': classification.intent,
                    'chat_mode': normalized_chat_mode or 'researcher',
                    'total_chunks': 0,
                })
            yield (
                StreamSignalTag.METRICS,
                {
                    'query_type': classification.intent,
                    'raw_chunks_count': 0,
                    'generation_skipped': True,
                    'answerability_passed': False,
                    'index_empty': True,
                },
            )
            yield EMPTY_KNOWLEDGE_BASE_RESEARCHER_MESSAGE
            yield []
            return

        # 2. Route to appropriate handler
        for handler in _HANDLER_REGISTRY:
            if handler.matches(classification):
                log.info(
                    'route_dispatched',
                    handler=type(handler).__name__,
                    intent=classification.intent,
                    route_candidate=classification.route_candidate,
                )
                async for item in handler.handle(
                    question=question,
                    classification=classification,
                    history=history,
                    db=db,
                    trace=trace,
                    diagnostics_context=diagnostics_context,
                ):
                    yield item
                return

        # Fallback: should never reach here (RAGHandler matches everything)
        log.error('no_handler_matched', intent=classification.intent)
        yield "Error: No handler matched the query. This should not happen."
        yield []

    except _ROUTER_RUNTIME_EXCEPTIONS as exc:
        log.error('answer_question_failed', error=str(exc), exc_info=True)
        yield f"Error: {exc}"
        yield []
