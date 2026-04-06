# ==============================================================================
# Informity AI — Classification Policy
# Shared classification helpers for route pre-classification and router fallback.
# ==============================================================================

from __future__ import annotations

import asyncio
import time

from informity.llm.query_classifier import QueryClassification, classify_query
from informity.llm.types import QueryType


async def classify_query_with_timing(question: str) -> tuple[QueryClassification, float]:
    """
    Classify a query off-thread and return classification + elapsed milliseconds.
    """
    classify_start = time.perf_counter()
    classification = await asyncio.to_thread(classify_query, question)
    classify_elapsed_ms = (time.perf_counter() - classify_start) * 1000.0
    return classification, classify_elapsed_ms


def resolve_assistant_forced_classification(
    classification: QueryClassification | None,
) -> QueryClassification:
    """
    Resolve assistant-mode classification and ensure a valid default.
    """
    return classification or QueryClassification(intent=QueryType.SIMPLE)
