# ==============================================================================
# Informity AI — Query Router (v2)
# Routes queries to appropriate handlers (metadata, RAG, simple)
# ==============================================================================

import asyncio
import time
from collections.abc import AsyncGenerator

import aiosqlite
import structlog

from informity.api.schemas import ChatSourceReference
from informity.db.models import ChatMessage
from informity.llm.chat_mode import normalize_chat_mode
from informity.llm.handlers.metadata import MetadataHandler
from informity.llm.handlers.rag import RAGHandler
from informity.llm.handlers.simple import SimpleHandler
from informity.llm.query_classifier import QueryClassification, classify_query
from informity.llm.types import QueryType, StreamSignalTag

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
) -> AsyncGenerator[str | list[ChatSourceReference] | tuple[str, object]]:
    """
    Query router - dispatches queries to appropriate handlers.
    """
    if db is None:
        yield "Error: No database connection provided."
        yield []
        return

    try:
        normalized_chat_mode = normalize_chat_mode(chat_mode)

        if normalized_chat_mode == 'assistant':
            forced_classification = classification or QueryClassification(intent=QueryType.SIMPLE)
            if trace is not None:
                trace.record('classification', {
                    'query_length': len(question),
                    'intent': QueryType.SIMPLE,
                    'route_candidate': forced_classification.route_candidate,
                    'confidence': forced_classification.confidence,
                    'duration_ms': 0.0,
                    'chat_mode': 'assistant',
                    'forced': True,
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
            ):
                yield item
            return

        # 1. Classify query (extract filters and intent)
        if classification is None:
            classify_start = time.perf_counter()
            classification = await asyncio.to_thread(classify_query, question)
            classify_elapsed_ms = (time.perf_counter() - classify_start) * 1000
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
