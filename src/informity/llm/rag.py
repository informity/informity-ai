# ==============================================================================
# Informity AI — Query Router (v2)
# Routes queries to appropriate handlers (metadata, RAG, simple)
# ==============================================================================

import asyncio
import dataclasses
import re
from collections.abc import AsyncGenerator
from typing import Any

import aiosqlite
import structlog

from informity.api.error_messages import to_client_error_message
from informity.api.schemas import ChatSourceReference
from informity.db.models import ChatMessage
from informity.db.sqlite import get_chunk_count
from informity.llm.chat_mode import is_assistant_mode, resolve_chat_mode
from informity.llm.handlers.metadata import MetadataHandler
from informity.llm.handlers.rag import RAGHandler
from informity.llm.handlers.simple import SimpleHandler
from informity.llm.query_classifier import QueryClassification, classify_query
from informity.llm.types import (
    IntentProfileId,
    OutputShape,
    QuerySubtype,
    QueryType,
    StreamSignalTag,
)
from informity.llm.user_messages import EMPTY_KNOWLEDGE_BASE_RESEARCHER_MESSAGE

log = structlog.get_logger(__name__)
_ROUTER_RUNTIME_EXCEPTIONS = (aiosqlite.Error, RuntimeError, ValueError, TypeError, OSError, TimeoutError)

# Handler registry - order matters (first match wins)
_HANDLER_REGISTRY = [
    MetadataHandler(),  # Metadata queries (count, enumeration, file listing)
    SimpleHandler(),    # Simple/conversational queries (greetings, clarifications, off-topic)
    RAGHandler(),       # Focused and coverage queries (fallback - should always match)
]

_COMPOUND_RESPONSE_SEPARATOR = '\n\n---\n\n'
_COMPOUND_SECONDARY_MAX_QUERY_WORDS = 32
_COMPOUND_SECONDARY_BROAD_SCOPE_PATTERN = re.compile(
    r'\b(all|across|every|each|by\s+year|year[-\s]*by[-\s]*year|cross[-\s]*year|summarize|compare)\b',
    re.IGNORECASE,
)


def _resolve_handler_for_classification(classification: QueryClassification) -> Any | None:
    for handler in _HANDLER_REGISTRY:
        if handler.matches(classification):
            return handler
    return None


def _build_secondary_classification(classification: QueryClassification) -> QueryClassification | None:
    secondary_intent = classification.secondary_intent
    if secondary_intent is None or secondary_intent == classification.intent:
        return None
    if secondary_intent == QueryType.METADATA:
        return dataclasses.replace(
            classification,
            intent=QueryType.METADATA,
            route_candidate=IntentProfileId.METADATA_INVENTORY,
            response_shape=OutputShape.NARRATIVE_SYNTHESIS,
            secondary_intent=None,
            is_metadata_query=True,
            is_file_list_query=True,
        )
    if secondary_intent == QueryType.FOCUSED:
        return dataclasses.replace(
            classification,
            intent=QueryType.FOCUSED,
            route_candidate=IntentProfileId.TARGETED_FACT_LOOKUP,
            response_shape=OutputShape.NARRATIVE_SYNTHESIS,
            secondary_intent=None,
            is_metadata_query=False,
            is_file_list_query=False,
        )
    return None


def _should_execute_secondary_path(
    *,
    question: str,
    primary: QueryClassification,
    secondary: QueryClassification,
) -> tuple[bool, str | None]:
    # Metadata secondary work is cheap (SQL-only) and can proceed without an extra gate.
    if secondary.intent != QueryType.FOCUSED:
        return True, None
    # Secondary focused execution can invoke full RAG. Keep it constrained to
    # compact metadata-first compounds to avoid unbounded latency.
    if primary.intent != QueryType.METADATA:
        return False, 'secondary_focused_requires_metadata_primary'
    if primary.subtype == QuerySubtype.AGGREGATE_BY_PERIOD or primary.has_multi_year_scope:
        return False, 'secondary_focused_blocked_for_multi_year_scope'
    query_words = len(re.findall(r'\S+', str(question or '')))
    if query_words > _COMPOUND_SECONDARY_MAX_QUERY_WORDS:
        return False, 'secondary_focused_query_length_budget_exceeded'
    if _COMPOUND_SECONDARY_BROAD_SCOPE_PATTERN.search(str(question or '')):
        return False, 'secondary_focused_broad_scope_budget_block'
    return True, None


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
                chat_id=chat_id,
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
        primary_handler = _resolve_handler_for_classification(classification)
        if primary_handler is not None:
            secondary_classification = _build_secondary_classification(classification)
            secondary_handler = (
                _resolve_handler_for_classification(secondary_classification)
                if secondary_classification is not None else None
            )
            if secondary_handler is not None and secondary_classification is not None:
                should_run_secondary, skip_reason = _should_execute_secondary_path(
                    question=question,
                    primary=classification,
                    secondary=secondary_classification,
                )
                if not should_run_secondary:
                    log.info(
                        'route_compound_secondary_skipped_budget_gate',
                        reason=skip_reason,
                        primary_intent=classification.intent,
                        secondary_intent=secondary_classification.intent,
                    )
                    secondary_handler = None
            merged_sources: list[ChatSourceReference] = []
            if secondary_handler is None:
                log.info(
                    'route_dispatched',
                    handler=type(primary_handler).__name__,
                    intent=classification.intent,
                    route_candidate=classification.route_candidate,
                )
                async for item in primary_handler.handle(
                    question=question,
                    classification=classification,
                    history=history,
                    db=db,
                    trace=trace,
                    diagnostics_context=diagnostics_context,
                    chat_id=chat_id,
                ):
                    yield item
                return

            log.info(
                'route_dispatched_compound',
                primary_handler=type(primary_handler).__name__,
                primary_intent=classification.intent,
                secondary_handler=type(secondary_handler).__name__,
                secondary_intent=secondary_classification.intent if secondary_classification is not None else None,
                route_candidate=classification.route_candidate,
            )
            for run_index, (run_handler, run_classification) in enumerate(
                (
                    (primary_handler, classification),
                    (secondary_handler, secondary_classification),
                ),
                start=1,
            ):
                if run_classification is None:
                    continue
                secondary_separator_emitted = False
                async for item in run_handler.handle(
                    question=question,
                    classification=run_classification,
                    history=history,
                    db=db,
                    trace=trace,
                    diagnostics_context=diagnostics_context,
                    chat_id=chat_id,
                ):
                    if isinstance(item, list):
                        for source in item:
                            if source not in merged_sources:
                                merged_sources.append(source)
                        continue
                    if (
                        run_index == 2
                        and isinstance(item, tuple)
                        and len(item) == 2
                        and item[0] == StreamSignalTag.METRICS
                    ):
                        metrics_payload = item[1]
                        if isinstance(metrics_payload, dict):
                            metrics_payload = dict(metrics_payload)
                            metrics_payload['compound_secondary_intent_applied'] = True
                            yield (item[0], metrics_payload)
                            continue
                    if run_index == 2 and not secondary_separator_emitted and isinstance(item, str):
                        yield _COMPOUND_RESPONSE_SEPARATOR
                        secondary_separator_emitted = True
                    yield item
            yield merged_sources
            return

        # Fallback: should never reach here (RAGHandler matches everything)
        log.error('no_handler_matched', intent=classification.intent)
        yield "Error: No handler matched the query. This should not happen."
        yield []

    except _ROUTER_RUNTIME_EXCEPTIONS as exc:
        log.error('answer_question_failed', error=str(exc), exc_info=True)
        yield to_client_error_message(exc)
        yield []
