# ==============================================================================
# Informity AI — Classification Policy
# Thin timing and shadow-logging wrapper around query classification.
# ==============================================================================

"""Module for llm classification policy."""

from __future__ import annotations

import asyncio
import time
from contextlib import suppress

from informity.db.models import ChatMessage
from informity.llm.query_classifier import QueryClassification, classify_query
from informity.log_events import emit_log_event


async def classify_query_with_timing(
    question: str,
    *,
    history: list[ChatMessage] | None = None,
    chat_mode: str | None = None,
    scope_kind: str | None = None,
    prior_user_query: str | None = None,
) -> tuple[QueryClassification, float]:
    """
    Classify a query off-thread and return classification + elapsed milliseconds.
    """
    classify_start = time.perf_counter()
    classification = await asyncio.to_thread(
        classify_query,
        question,
        history=history,
        chat_mode=chat_mode,
        scope_kind=scope_kind,
        prior_user_query=prior_user_query,
    )
    classify_elapsed_ms = (time.perf_counter() - classify_start) * 1000.0
    with suppress(Exception):
        await emit_log_event(
            event_name="five_q_shadow",
            source="classification_policy",
            message=f"five_q shadow classification for: {str(question or '')[:120]}",
            channel="application",
            event_type="debug",
            details={
                "question": str(question or ""),
                "intent": str(classification.intent),
                "route_candidate": str(classification.route_candidate),
                "response_shape": str(classification.response_shape),
                "confidence": classification.confidence,
                "shadow_model": classification.shadow_classifier_model,
                "shadow_raw_output": classification.shadow_classifier_raw_output,
                "shadow_decision": classification.shadow_classifier_decision,
                "chat_mode": chat_mode,
                "scope_kind": scope_kind,
                "elapsed_ms": round(classify_elapsed_ms, 1),
            },
        )
    return classification, classify_elapsed_ms
