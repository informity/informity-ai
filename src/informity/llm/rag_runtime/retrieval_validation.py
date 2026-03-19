import math

import aiosqlite

from informity.config import settings
from informity.db.models import ChatMessage
from informity.llm.query_classifier import QueryClassification
from informity.llm.retrieval import retrieve_chunks


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
    query_type: str,
    route_candidate: str,
    has_strong_anchor: bool = False,
) -> tuple[bool, float]:
    if not chunks:
        return False, 0.0
    top_scores = [_normalize_relevance_score(chunk.get('score', 0.0)) for chunk in chunks[:3]]
    mean_score = sum(top_scores) / max(1, len(top_scores))
    if query_type == 'coverage':
        return mean_score >= float(settings.retrieval_relevance_threshold_coverage), mean_score
    if route_candidate == 'structured_field_extraction':
        return mean_score >= float(settings.retrieval_relevance_threshold_structured), mean_score
    if query_type == 'focused' and has_strong_anchor:
        # Strong metadata anchors (especially filename constraints) should not fail hard
        # when reranker scores are near-zero but retrieval returned concrete chunks.
        return True, mean_score
    return mean_score >= float(settings.retrieval_relevance_threshold_focused), mean_score


def _evaluate_source_diversity_gate(
    *,
    chunks: list[dict],
    query_type: str,
) -> tuple[bool, int]:
    if query_type != 'coverage':
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
        if message.role != 'assistant' or not message.sources:
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
        if message.role != 'assistant':
            continue
        return bool(message.has_remaining_scope)
    return False


def _extract_last_user_question(history: list[ChatMessage] | None) -> str | None:
    if not history:
        return None
    for message in reversed(history):
        if message.role != 'user':
            continue
        content = (message.content or '').strip()
        if content:
            return content
    return None


def _build_continuation_retrieval_query(
    *,
    question: str,
    route_candidate: str,
    prior_has_remaining_scope: bool,
    scope_reset_detected: bool,
    is_continuation: bool = False,
    history: list[ChatMessage] | None,
) -> str:
    normalized_question = question.strip()
    if route_candidate != 'continuation_or_refinement':
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
    route_candidate: str,
    prior_has_remaining_scope: bool,
    scope_reset_detected: bool,
    prior_source_anchors: set[str],
) -> list[str]:
    if route_candidate != 'continuation_or_refinement':
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
    query_type: str,
    subtype: str | None,
    group_by: str | None,
    response_shape: str,
    distinct_sources_count: int,
    chunk_count: int,
    fallback_events: list[dict[str, object]],
    route_profile_id: str,
    retrieval_relevance_score: float,
) -> tuple[bool, list[dict[str, object]]]:
    hard_floor_enabled = bool(settings.retrieval_coverage_evidence_floor_hard_floor_enabled)
    hard_floor_min_score = float(settings.retrieval_coverage_evidence_floor_min_score)
    score_clears_hard_floor = (not hard_floor_enabled) or retrieval_relevance_score >= hard_floor_min_score

    coverage_evidence_floor_eligible = (
        subtype == 'aggregate_by_period'
        or group_by in {'year', 'category', 'file'}
        or (
            response_shape == 'narrative_synthesis'
            and route_profile_id in {'comparative_analysis', 'cross_document_synthesis', 'audit_or_compliance_brief'}
        )
    )
    if (
        not retrieval_relevance_passed
        and query_type == 'coverage'
        and coverage_evidence_floor_eligible
        and score_clears_hard_floor
        and distinct_sources_count >= 3
        and chunk_count >= 8
    ):
        retrieval_relevance_passed = True
        fallback_events.append({
            'fallback_from': route_profile_id,
            'fallback_to': route_profile_id,
            'fallback_reason': 'coverage_evidence_floor_override',
            'retrieval_relevance_score': round(retrieval_relevance_score, 3),
            'distinct_sources_count': distinct_sources_count,
            'group_by': group_by,
            'response_shape': response_shape,
            'subtype': subtype,
        })

    focused_structured_evidence_floor_eligible = (
        query_type == 'focused'
        and route_profile_id == 'structured_field_extraction'
        and response_shape == 'structured_extract'
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
            'fallback_reason': 'focused_structured_evidence_floor_override',
            'retrieval_relevance_score': round(retrieval_relevance_score, 3),
            'distinct_sources_count': distinct_sources_count,
            'chunk_count': chunk_count,
            'response_shape': response_shape,
            'subtype': subtype,
        })
    return retrieval_relevance_passed, fallback_events


def _evaluate_continuation_anchor_gate(
    *,
    route_candidate: str,
    scope_reset_detected: bool,
    prior_source_anchors: set[str],
    current_source_keys: set[str],
    prior_has_remaining_scope: bool = False,
) -> tuple[bool, int]:
    anchor_overlap_count = len(prior_source_anchors.intersection(current_source_keys))
    continuation_anchor_passed = True
    if route_candidate == 'continuation_or_refinement':
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
    effective_query_type: str,
    route_profile_id: str,
    db: aiosqlite.Connection,
    trace: object | None,
    retrieve_fn=retrieve_chunks,
    timing_output: dict | None = None,
) -> tuple[list[dict], str]:
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
        source_terms_filter=classification.source_terms,
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
        has_strong_anchor=bool(
            classification.filename_filter
            or (classification.year_filter is not None and classification.source_terms)
        ),
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
        source_terms_filter=classification.source_terms,
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
        has_strong_anchor=bool(
            classification.filename_filter
            or (classification.year_filter is not None and classification.source_terms)
        ),
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
            source_terms_filter=classification.source_terms,
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
                has_strong_anchor=bool(
                    classification.filename_filter
                    or (classification.year_filter is not None and classification.source_terms)
                ),
            )
            if section_relaxed_passed:
                return section_relaxed_chunks, 'section'

    # Second relaxation: drop both structural constraints.
    return baseline_chunks, 'both' if classification.section_filter else 'block'
