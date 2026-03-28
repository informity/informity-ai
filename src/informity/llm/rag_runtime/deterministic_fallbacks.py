# ==============================================================================
# Informity AI — Deterministic Fallbacks
# Structured extraction deterministic fallback gate.
# ==============================================================================

from __future__ import annotations

from dataclasses import dataclass

import aiosqlite

from informity.api.schemas import ChatSourceReference
from informity.llm.query_classifier import QueryClassification
from informity.llm.rag_runtime import structured_numeric as _structured_numeric
from informity.llm.types import OutputShape


@dataclass
class DeterministicFallbackResult:
    kind: str
    answer: str | None = None
    sources: list[ChatSourceReference] | None = None
    structured_metrics: dict[str, object] | None = None


async def try_structured_fallback(
    *,
    question: str,
    classification: QueryClassification,
    response_shape: OutputShape,
    chunks: list[dict],
    db: aiosqlite.Connection,
    trace: object | None,
) -> DeterministicFallbackResult:
    structured_result = await _structured_numeric._try_structured_value_extraction(
        question=question,
        classification=classification,
        response_shape=response_shape,
        chunks=chunks,
        db=db,
        trace=trace,
    )
    if structured_result is not None:
        structured_answer, structured_sources, structured_metrics = structured_result
        return DeterministicFallbackResult(
            kind='structured',
            answer=structured_answer,
            sources=structured_sources,
            structured_metrics=structured_metrics,
        )

    return DeterministicFallbackResult(kind='none')
