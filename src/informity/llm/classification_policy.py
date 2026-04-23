# ==============================================================================
# Informity AI — Classification Policy
# Shared classification helpers for route pre-classification and router fallback.
# ==============================================================================

from __future__ import annotations

import asyncio
import re
import time

from informity.db.models import ChatMessage
from informity.llm.query_classifier import QueryClassification, classify_query
from informity.llm.types import IntentProfileId, QueryType

_CHAT_SUMMARY_EXPLICIT_PATTERN = re.compile(
    r'\b('
    r'summar(?:ize|ise)\s+(?:our|this)\s+(?:chat|conversation|discussion)'
    r'|'
    r'recap\s+(?:our|this)\s+(?:chat|conversation|discussion)'
    r'|'
    r'what\s+have\s+we\s+been\s+(?:chatting|discussing)\s+about'
    r'|'
    r'(?:show|list)\s+(?:me\s+)?(?:our\s+)?(?:chat|conversation|discussion)\s+topics'
    r'|'
    r'topics?\s+(?:in|from)\s+(?:our|this)\s+(?:chat|conversation|discussion)'
    r')\b',
    re.IGNORECASE,
)

_SCOPED_DOCUMENT_REFERENCE_PATTERN = re.compile(
    r'\b('
    r'(?:this|that)\s+(?:file|document|text|record|entry|item|source|material|attachment|note|paper)'
    r'|'
    r'(?:from|in|about)\s+this\s+(?:file|document|text|record|entry|item|source|material|attachment|note|paper)'
    r')\b',
    re.IGNORECASE,
)

_DOCUMENT_CONTENT_TASK_PATTERN = re.compile(
    r'\b('
    r'topics?'
    r'|'
    r'themes?'
    r'|'
    r'key\s+points?'
    r'|'
    r'main\s+ideas?'
    r'|'
    r'summar(?:ize|ise)'
    r'|'
    r'outline'
    r'|'
    r'what\s+(?:is|are)\s+(?:the\s+)?(?:top|main|key)'
    r')\b',
    re.IGNORECASE,
)


def apply_scoped_file_chat_summary_precedence(
    *,
    question: str,
    classification: QueryClassification,
    scoped_file_active: bool,
) -> QueryClassification:
    """
    Preserve chat-summary intent generally, but when a file scope is active and
    the query clearly references "this document/file/text" content, prioritize
    scoped researcher retrieval for this turn.
    """
    if not scoped_file_active or not classification.needs_chat_history:
        return classification

    text = str(question or '').strip()
    if not text:
        return classification

    if _CHAT_SUMMARY_EXPLICIT_PATTERN.search(text):
        return classification
    if not _SCOPED_DOCUMENT_REFERENCE_PATTERN.search(text):
        return classification
    if not _DOCUMENT_CONTENT_TASK_PATTERN.search(text):
        return classification

    classification.needs_chat_history = False
    classification.intent = QueryType.FOCUSED
    classification.route_candidate = IntentProfileId.TARGETED_FACT_LOOKUP
    classification.is_metadata_query = False
    classification.is_file_list_query = False
    classification.deterministic_override = True
    if 'policy_scoped_file_document_request_precedence' not in classification.reason_codes:
        classification.reason_codes.append('policy_scoped_file_document_request_precedence')
    return classification


async def classify_query_with_timing(
    question: str,
    *,
    scoped_file_active: bool = False,
    history: list[ChatMessage] | None = None,
) -> tuple[QueryClassification, float]:
    """
    Classify a query off-thread and return classification + elapsed milliseconds.
    """
    classify_start = time.perf_counter()
    classification = await asyncio.to_thread(classify_query, question, history=history)
    classification = apply_scoped_file_chat_summary_precedence(
        question=question,
        classification=classification,
        scoped_file_active=scoped_file_active,
    )
    classify_elapsed_ms = (time.perf_counter() - classify_start) * 1000.0
    return classification, classify_elapsed_ms


def resolve_assistant_forced_classification(
    classification: QueryClassification | None,
) -> QueryClassification:
    """
    Resolve assistant-mode classification and ensure a valid default.
    """
    return classification or QueryClassification(intent=QueryType.SIMPLE)
