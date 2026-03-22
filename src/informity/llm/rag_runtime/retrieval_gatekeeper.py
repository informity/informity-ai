# ==============================================================================
# Informity AI — Retrieval Gatekeeper
# Validation-gate recovery path (profile fallback) for RAG retrieval.
# Widened-retry (second pass with relaxed constraints) has been removed per
# Item 6 — gates are single-pass checks. quality_score is logged on every query.
# ==============================================================================

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

import aiosqlite

from informity.llm.query_classifier import QueryClassification
from informity.llm.rag_runtime import retrieval_validation as _retrieval_validation


@dataclass
class ValidationRecoveryResult:
    chunks: list[dict]
    effective_query_type: str
    effective_top_k: int
    retrieval_relevance_passed: bool
    source_diversity_passed: bool
    continuation_anchor_passed: bool
    retrieval_relevance_score: float
    distinct_sources_count: int
    anchor_overlap_count: int
    validation_gates: dict[str, bool]
    fallback_events: list[dict[str, object]]
    original_query_type: str
    # Weighted retrieval quality score: mean(top_3_reranker_scores) × file_diversity_factor.
    # High score → confident answer tone; low score → hedged answer tone.
    # Zero indicates no retrievable evidence.
    quality_score: float


def _compute_quality_score(
    *,
    retrieval_relevance_score: float,
    distinct_sources_count: int,
    chunk_count: int,
) -> float:
    """
    Compute weighted retrieval quality score.

    quality_score = mean(top_3_reranker_scores) × file_diversity_factor
    file_diversity_factor = distinct_sources / max(chunk_count, 1), capped at 1.0.

    Returns a value in [0.0, 1.0].
    """
    if chunk_count == 0:
        return 0.0
    file_diversity_factor = min(distinct_sources_count / chunk_count, 1.0)
    return retrieval_relevance_score * file_diversity_factor


async def run_validation_recovery_when_failed(
    *,
    chunks: list[dict],
    effective_query_type: str,
    effective_top_k: int,
    retrieval_relevance_passed: bool,
    source_diversity_passed: bool,
    continuation_anchor_passed: bool,
    retrieval_relevance_score: float,
    distinct_sources_count: int,
    anchor_overlap_count: int,
    validation_gates: dict[str, bool],
    fallback_events: list[dict[str, object]],
    classification: QueryClassification,
    effective_response_shape: str,
    selected_policy_profile_id: str,
    selected_policy_fallback_target_route: str,
    scope_reset_detected: bool,
    prior_source_anchors: set[str],
    prior_has_remaining_scope: bool,
    retrieval_question: str,
    retrieval_filename_filter: str | None,
    profile_rag_max_score: float | None,
    db: aiosqlite.Connection,
    trace: object | None,
    retrieve_fn: Callable[..., Awaitable[list[dict]]],
    get_retrieval_top_k_fn: Callable[..., int],
    get_intent_profile_policy_fn: Callable[..., object],
) -> ValidationRecoveryResult:
    quality_score = _compute_quality_score(
        retrieval_relevance_score=retrieval_relevance_score,
        distinct_sources_count=distinct_sources_count,
        chunk_count=len(chunks),
    )
    if all(validation_gates.values()):
        return ValidationRecoveryResult(
            chunks=chunks,
            effective_query_type=effective_query_type,
            effective_top_k=effective_top_k,
            retrieval_relevance_passed=retrieval_relevance_passed,
            source_diversity_passed=source_diversity_passed,
            continuation_anchor_passed=continuation_anchor_passed,
            retrieval_relevance_score=retrieval_relevance_score,
            distinct_sources_count=distinct_sources_count,
            anchor_overlap_count=anchor_overlap_count,
            validation_gates=validation_gates,
            fallback_events=fallback_events,
            original_query_type=effective_query_type,
            quality_score=quality_score,
        )

    # Preserve the original query type before the fallback may change it.
    # The fallback profile (e.g. clarification_or_disambiguation) may prefer 'focused'
    # retrieval, which would change effective_query_type to 'focused'. The coverage
    # floor override requires query_type == 'coverage', so we must pass the original
    # query type when evaluating the override — the user's intent has not changed.
    original_query_type = effective_query_type

    has_strong_anchor = bool(
        classification.filename_filter
    )
    fallback_profile = get_intent_profile_policy_fn(selected_policy_fallback_target_route)
    fallback_events.append({
        'fallback_from': selected_policy_profile_id,
        'fallback_to': fallback_profile.profile_id,
        'fallback_reason': 'validation_gate_failed',
        'validation_gates': validation_gates,
    })
    fallback_chunks = await retrieve_fn(
        query=retrieval_question,
        top_k=get_retrieval_top_k_fn(fallback_profile.preferred_retrieval_mode),
        max_score=profile_rag_max_score,
        year_filter=classification.year_filter,
        category_filter=classification.category_filter,
        extension_filter=classification.file_type_filter,
        filename_filter=retrieval_filename_filter,
        block_type_filter=None,
        section_filter=None,
        query_type=fallback_profile.preferred_retrieval_mode,
        db=db,
        trace=trace,
    )
    if fallback_chunks:
        chunks = fallback_chunks
        effective_query_type = fallback_profile.preferred_retrieval_mode
        effective_top_k = min(effective_top_k, len(chunks))
        retrieval_relevance_passed, retrieval_relevance_score = _retrieval_validation._evaluate_retrieval_relevance_gate(
            chunks=chunks,
            query_type=effective_query_type,
            route_candidate=fallback_profile.profile_id,
            has_strong_anchor=has_strong_anchor,
        )
        source_diversity_passed, distinct_sources_count = _retrieval_validation._evaluate_source_diversity_gate(
            chunks=chunks,
            query_type=effective_query_type,
        )
        retrieval_relevance_passed, fallback_events = _retrieval_validation._apply_coverage_evidence_floor_override(
            retrieval_relevance_passed=retrieval_relevance_passed,
            # Use the original query type so coverage floor eligibility is not lost when the
            # fallback profile switches to a focused-mode route (e.g. clarification_or_disambiguation).
            query_type=original_query_type,
            subtype=classification.subtype,
            group_by=classification.group_by,
            # Use effective_response_shape rather than classification.response_shape: the
            # classifier may produce 'structured_extract' for listing-type coverage queries
            # (e.g. "names of people", "dates across documents"), but execution_plan.py
            # normalises it to 'narrative_synthesis'. The floor override eligibility check
            # requires 'narrative_synthesis', so we must use the normalised shape.
            response_shape=effective_response_shape,
            distinct_sources_count=distinct_sources_count,
            chunk_count=len(chunks),
            fallback_events=fallback_events,
            # Use the original route profile so coverage floor eligibility is preserved
            # even after the fallback profile changes to clarification_or_disambiguation.
            route_profile_id=selected_policy_profile_id,
            retrieval_relevance_score=retrieval_relevance_score,
        )
        # Escalation case: the fallback escalated the query from focused to coverage mode
        # (e.g. targeted_fact_lookup -> cross_document_synthesis). The original floor override
        # above uses the focused original_query_type and doesn't fire. Re-evaluate using the
        # fallback profile's coverage eligibility so that year-filtered or broad-scope queries
        # can pass through when sufficient evidence is present.
        if not retrieval_relevance_passed and original_query_type != effective_query_type and effective_query_type == 'coverage':
            retrieval_relevance_passed, fallback_events = _retrieval_validation._apply_coverage_evidence_floor_override(
                retrieval_relevance_passed=retrieval_relevance_passed,
                query_type=effective_query_type,
                subtype=classification.subtype,
                group_by=classification.group_by,
                response_shape=effective_response_shape,
                distinct_sources_count=distinct_sources_count,
                chunk_count=len(chunks),
                fallback_events=fallback_events,
                route_profile_id=fallback_profile.profile_id,
                retrieval_relevance_score=retrieval_relevance_score,
            )
        current_source_keys = _retrieval_validation._extract_current_source_keys(chunks)
        continuation_anchor_passed, anchor_overlap_count = _retrieval_validation._evaluate_continuation_anchor_gate(
            route_candidate=classification.route_candidate,
            scope_reset_detected=scope_reset_detected,
            prior_source_anchors=prior_source_anchors,
            current_source_keys=current_source_keys,
            prior_has_remaining_scope=prior_has_remaining_scope,
        )
        validation_gates = {
            'retrieval_relevance_gate': retrieval_relevance_passed,
            'source_diversity_gate': source_diversity_passed,
            'continuation_anchor_gate': continuation_anchor_passed,
        }
        quality_score = _compute_quality_score(
            retrieval_relevance_score=retrieval_relevance_score,
            distinct_sources_count=distinct_sources_count,
            chunk_count=len(chunks),
        )

    return ValidationRecoveryResult(
        chunks=chunks,
        effective_query_type=effective_query_type,
        effective_top_k=effective_top_k,
        retrieval_relevance_passed=retrieval_relevance_passed,
        source_diversity_passed=source_diversity_passed,
        continuation_anchor_passed=continuation_anchor_passed,
        retrieval_relevance_score=retrieval_relevance_score,
        distinct_sources_count=distinct_sources_count,
        anchor_overlap_count=anchor_overlap_count,
        validation_gates=validation_gates,
        fallback_events=fallback_events,
        original_query_type=original_query_type,
        quality_score=quality_score,
    )
