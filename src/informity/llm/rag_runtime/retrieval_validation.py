import math
from typing import Literal

import aiosqlite

from informity.config import settings
from informity.db.models import ChatMessage
from informity.llm.query_classifier import QueryClassification
from informity.llm.retrieval import retrieve_chunks
from informity.llm.types import (
    ChatRole,
    FallbackReason,
    GroupBy,
    IntentProfileId,
    OutputShape,
    QuerySubtype,
    RetrievalMode,
)


def _normalize_relevance_score(raw_score: float) -> float:
    """
    Convert CrossEncoder raw logit (-10..10) to 0-1 relevance for display.
    Uses sigmoid so 0 -> 0.5, positive -> higher, negative -> lower.
    """
    try:
        numeric_score = float(raw_score)
    except (TypeError, ValueError):
        return 0.0
    try:
        return 1.0 / (1.0 + math.exp(-numeric_score))
    except OverflowError:
        return 0.0 if numeric_score < 0 else 1.0


def _evaluate_retrieval_relevance_gate(
    *,
    chunks: list[dict],
    query_type: RetrievalMode,
    route_candidate: IntentProfileId,
    has_strong_anchor: bool = False,
) -> tuple[bool, float]:
    if not chunks:
        return False, 0.0
    top_scores = [_normalize_relevance_score(chunk.get('score', 0.0)) for chunk in chunks[:3]]
    mean_score = sum(top_scores) / max(1, len(top_scores))
    if query_type == RetrievalMode.COVERAGE:
        return mean_score >= float(settings.retrieval_relevance_threshold_coverage), mean_score
    if route_candidate == IntentProfileId.STRUCTURED_FIELD_EXTRACTION:
        return mean_score >= float(settings.retrieval_relevance_threshold_structured), mean_score
    if query_type == RetrievalMode.FOCUSED and has_strong_anchor:
        # Strong metadata anchors (especially filename constraints) should not fail hard
        # when reranker scores are near-zero but retrieval returned concrete chunks.
        return True, mean_score
    return mean_score >= float(settings.retrieval_relevance_threshold_focused), mean_score


def _evaluate_source_diversity_gate(
    *,
    chunks: list[dict],
    query_type: RetrievalMode,
) -> tuple[bool, int]:
    if query_type != RetrievalMode.COVERAGE:
        return True, len({int(chunk.get('file_id', 0)) for chunk in chunks if chunk.get('file_id') is not None})
    distinct_sources = {
        int(chunk.get('file_id', 0))
        for chunk in chunks
        if chunk.get('file_id') is not None
    }
    return len(distinct_sources) >= 2, len(distinct_sources)


def _extract_prior_source_anchors(history: list[ChatMessage] | None) -> set[str]:
    if not history:
        return set()
    anchors: set[str] = set()
    for message in reversed(history):
        if message.role != ChatRole.ASSISTANT or not message.sources:
            continue
        for source in message.sources:
            if not isinstance(source, dict):
                continue
            path = str(source.get('path') or '').strip().casefold()
            filename = str(source.get('filename') or '').strip().casefold()
            if path:
                anchors.add(path)
            if filename:
                anchors.add(filename)
        if anchors:
            break
    return anchors


def _extract_prior_has_remaining_scope(history: list[ChatMessage] | None) -> bool:
    if not history:
        return False
    for message in reversed(history):
        if message.role != ChatRole.ASSISTANT:
            continue
        return bool(message.has_remaining_scope)
    return False


def _extract_last_user_question(history: list[ChatMessage] | None) -> str | None:
    if not history:
        return None
    for message in reversed(history):
        if message.role != ChatRole.USER:
            continue
        content = (message.content or '').strip()
        if content:
            return content
    return None


def _build_continuation_retrieval_query(
    *,
    question: str,
    route_candidate: IntentProfileId,
    prior_has_remaining_scope: bool,
    scope_reset_detected: bool,
    is_continuation: bool = False,
    history: list[ChatMessage] | None,
) -> str:
    normalized_question = question.strip()
    if route_candidate != IntentProfileId.CONTINUATION_OR_REFINEMENT:
        return normalized_question
    if scope_reset_detected or not prior_has_remaining_scope:
        return normalized_question
    if not normalized_question:
        return normalized_question
    if not is_continuation:
        return normalized_question
    last_user_question = _extract_last_user_question(history)
    if not last_user_question:
        return normalized_question
    return (
        f'{normalized_question}\n\n'
        f'Continue this exact prior request context:\n{last_user_question}'
    ).strip()


def _derive_continuation_source_terms(
    *,
    route_candidate: IntentProfileId,
    prior_has_remaining_scope: bool,
    scope_reset_detected: bool,
    prior_source_anchors: set[str],
) -> list[str]:
    if route_candidate != IntentProfileId.CONTINUATION_OR_REFINEMENT:
        return []
    if not prior_has_remaining_scope or scope_reset_detected or not prior_source_anchors:
        return []

    terms: list[str] = []
    seen: set[str] = set()
    for anchor in sorted(prior_source_anchors):
        normalized_anchor = anchor.strip()
        if not normalized_anchor:
            continue
        filename = normalized_anchor.rsplit('/', 1)[-1].strip()
        if not filename:
            continue
        stem = filename.rsplit('.', 1)[0].strip()
        for candidate in (filename, stem):
            if len(candidate) < 4:
                continue
            key = candidate.casefold()
            if key in seen:
                continue
            seen.add(key)
            terms.append(candidate)
            if len(terms) >= 8:
                return terms
    return terms


def _extract_current_source_keys(chunks: list[dict]) -> set[str]:
    keys: set[str] = set()
    for chunk in chunks:
        file_path = str(chunk.get('file_path') or '').strip().casefold()
        filename = str(chunk.get('filename') or '').strip().casefold()
        if file_path:
            keys.add(file_path)
        if filename:
            keys.add(filename)
    return keys


def _apply_coverage_evidence_floor_override(
    *,
    retrieval_relevance_passed: bool,
    query_type: RetrievalMode,
    subtype: QuerySubtype | None,
    group_by: GroupBy | None,
    response_shape: OutputShape,
    distinct_sources_count: int,
    chunk_count: int,
    fallback_events: list[dict[str, object]],
    route_profile_id: IntentProfileId,
    retrieval_relevance_score: float,
) -> tuple[bool, list[dict[str, object]]]:
    hard_floor_enabled = bool(settings.retrieval_coverage_evidence_floor_hard_floor_enabled)
    hard_floor_min_score = float(settings.retrieval_coverage_evidence_floor_min_score)
    schema_driven_shape = response_shape in {OutputShape.METADATA_TABLE, OutputShape.STRUCTURED_EXTRACT}
    score_clears_hard_floor = (not hard_floor_enabled) or retrieval_relevance_score >= hard_floor_min_score
    if schema_driven_shape:
        score_clears_hard_floor = True

    coverage_evidence_floor_eligible = (
        subtype == QuerySubtype.AGGREGATE_BY_PERIOD
        or group_by in {GroupBy.YEAR, GroupBy.CATEGORY, GroupBy.FILE}
        or (
            response_shape in {OutputShape.NARRATIVE_SYNTHESIS, OutputShape.METADATA_TABLE, OutputShape.HYBRID}
            and route_profile_id in {
                IntentProfileId.COMPARATIVE_ANALYSIS,
                IntentProfileId.CROSS_DOCUMENT_SYNTHESIS,
                IntentProfileId.AUDIT_OR_COMPLIANCE_BRIEF,
            }
        )
    )
    if (
        not retrieval_relevance_passed
        and query_type == RetrievalMode.COVERAGE
        and coverage_evidence_floor_eligible
        and score_clears_hard_floor
        and distinct_sources_count >= 3
        and chunk_count >= 8
    ):
        retrieval_relevance_passed = True
        fallback_events.append({
            'fallback_from': route_profile_id,
            'fallback_to': route_profile_id,
            'fallback_reason': FallbackReason.COVERAGE_EVIDENCE_FLOOR_OVERRIDE,
            'retrieval_relevance_score': round(retrieval_relevance_score, 3),
            'distinct_sources_count': distinct_sources_count,
            'group_by': group_by,
            'response_shape': response_shape,
            'subtype': subtype,
        })

    focused_structured_evidence_floor_eligible = (
        query_type == RetrievalMode.FOCUSED
        and route_profile_id == IntentProfileId.STRUCTURED_FIELD_EXTRACTION
        and response_shape == OutputShape.STRUCTURED_EXTRACT
    )
    if (
        not retrieval_relevance_passed
        and focused_structured_evidence_floor_eligible
        and distinct_sources_count >= 3
        and chunk_count >= 8
    ):
        retrieval_relevance_passed = True
        fallback_events.append({
            'fallback_from': route_profile_id,
            'fallback_to': route_profile_id,
            'fallback_reason': FallbackReason.FOCUSED_STRUCTURED_EVIDENCE_FLOOR_OVERRIDE,
            'retrieval_relevance_score': round(retrieval_relevance_score, 3),
            'distinct_sources_count': distinct_sources_count,
            'chunk_count': chunk_count,
            'response_shape': response_shape,
            'subtype': subtype,
        })
    return retrieval_relevance_passed, fallback_events


def _evaluate_continuation_anchor_gate(
    *,
    route_candidate: IntentProfileId,
    scope_reset_detected: bool,
    prior_source_anchors: set[str],
    current_source_keys: set[str],
    prior_has_remaining_scope: bool = False,
) -> tuple[bool, int]:
    anchor_overlap_count = len(prior_source_anchors.intersection(current_source_keys))
    continuation_anchor_passed = True
    if route_candidate == IntentProfileId.CONTINUATION_OR_REFINEMENT:
        if scope_reset_detected:
            continuation_anchor_passed = True
        elif not prior_source_anchors:
            continuation_anchor_passed = False
        else:
            continuation_anchor_passed = anchor_overlap_count > 0
    return continuation_anchor_passed, anchor_overlap_count


async def _retrieve_with_staged_structural_constraints(
    *,
    question: str,
    effective_top_k: int,
    profile_max_score: float | None,
    classification: QueryClassification,
    effective_query_type: RetrievalMode,
    route_profile_id: IntentProfileId,
    db: aiosqlite.Connection,
    trace: object | None,
    retrieve_fn=retrieve_chunks,
    timing_output: dict | None = None,
) -> tuple[list[dict], Literal['none', 'section', 'both', 'block']]:
    """
    Retrieve with baseline-first strategy for structural constraints.

    Contract-aligned behavior:
    1) Retrieve baseline (no block/section constraints).
    2) If baseline quality passes and constraints were requested, try constrained retrieval.
    3) If constrained quality fails, relax section first, then block filter.
    """
    has_structural_constraints = bool(classification.block_type_filter or classification.section_filter)

    baseline_chunks = await retrieve_fn(
        query=question,
        top_k=effective_top_k,
        max_score=profile_max_score,
        year_filter=classification.year_filter,
        category_filter=classification.category_filter,
        extension_filter=classification.file_type_filter,
        filename_filter=classification.filename_filter,
        block_type_filter=None,
        section_filter=None,
        query_type=effective_query_type,
        db=db,
        trace=trace,
        timing_output=timing_output,
    )
    if not baseline_chunks or not has_structural_constraints:
        return baseline_chunks, 'none'

    baseline_passed, _ = _evaluate_retrieval_relevance_gate(
        chunks=baseline_chunks,
        query_type=effective_query_type,
        route_candidate=route_profile_id,
        has_strong_anchor=bool(classification.filename_filter),
    )
    if not baseline_passed:
        return baseline_chunks, 'none'

    constrained_chunks = await retrieve_fn(
        query=question,
        top_k=effective_top_k,
        max_score=profile_max_score,
        year_filter=classification.year_filter,
        category_filter=classification.category_filter,
        extension_filter=classification.file_type_filter,
        filename_filter=classification.filename_filter,
        block_type_filter=classification.block_type_filter,
        section_filter=classification.section_filter,
        query_type=effective_query_type,
        db=db,
        trace=trace,
        timing_output=timing_output,
    )
    if not constrained_chunks:
        return baseline_chunks, 'both'

    constrained_passed, _ = _evaluate_retrieval_relevance_gate(
        chunks=constrained_chunks,
        query_type=effective_query_type,
        route_candidate=route_profile_id,
        has_strong_anchor=bool(classification.filename_filter),
    )
    if constrained_passed:
        return constrained_chunks, 'none'

    # First relaxation: remove section hint, keep block filter.
    if classification.section_filter:
        section_relaxed_chunks = await retrieve_fn(
            query=question,
            top_k=effective_top_k,
            max_score=profile_max_score,
            year_filter=classification.year_filter,
            category_filter=classification.category_filter,
            extension_filter=classification.file_type_filter,
            filename_filter=classification.filename_filter,
            block_type_filter=classification.block_type_filter,
            section_filter=None,
            query_type=effective_query_type,
            db=db,
            trace=trace,
            timing_output=timing_output,
        )
        if section_relaxed_chunks:
            section_relaxed_passed, _ = _evaluate_retrieval_relevance_gate(
                chunks=section_relaxed_chunks,
                query_type=effective_query_type,
                route_candidate=route_profile_id,
                has_strong_anchor=bool(classification.filename_filter),
            )
            if section_relaxed_passed:
                return section_relaxed_chunks, 'section'

    # Second relaxation: drop both structural constraints.
    return baseline_chunks, 'both' if classification.section_filter else 'block'
