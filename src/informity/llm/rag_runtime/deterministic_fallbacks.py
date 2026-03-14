# ==============================================================================
# Informity AI — Deterministic Fallbacks
# Strict contract composer and structured extraction deterministic fallback gate.
# ==============================================================================

from __future__ import annotations

from dataclasses import dataclass

import aiosqlite

from informity.api.schemas import ChatSourceReference
from informity.llm.query_classifier import QueryClassification
from informity.llm.rag_runtime import strict_composers as _strict_composers
from informity.llm.rag_runtime import structured_numeric as _structured_numeric


@dataclass
class DeterministicFallbackResult:
    kind: str
    answer: str | None = None
    sources: list[ChatSourceReference] | None = None
    strict_metrics: dict[str, object] | None = None
    structured_metrics: dict[str, object] | None = None


async def try_strict_or_structured_fallback(
    *,
    question: str,
    classification: QueryClassification,
    response_shape: str,
    chunks: list[dict],
    response_mode: str,
    db: aiosqlite.Connection,
    trace: object | None,
) -> DeterministicFallbackResult:
    strict_composed_result = _strict_composers.try_compose_strict_contract_answer(
        question=question,
        chunks=chunks,
        response_mode=response_mode,
    )
    if strict_composed_result is not None:
        strict_answer, strict_sources, strict_metrics = strict_composed_result
        return DeterministicFallbackResult(
            kind='strict',
            answer=strict_answer,
            sources=strict_sources,
            strict_metrics=strict_metrics,
        )

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
