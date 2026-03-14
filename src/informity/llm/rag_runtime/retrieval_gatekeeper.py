# ==============================================================================
# Informity AI — Retrieval Gatekeeper
# Validation-gate recovery path (fallback + widened retry) for RAG retrieval.
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
    selected_policy_profile_id: str,
    selected_policy_fallback_target_route: str,
    source_terms_for_retrieval: list[str],
    scope_reset_detected: bool,
    prior_source_anchors: set[str],
    prior_has_remaining_scope: bool,
    retrieval_question: str,
    retrieval_filename_filter: str | None,
    response_mode_used: str,
    profile_rag_max_score: float | None,
    db: aiosqlite.Connection,
    trace: object | None,
    retrieve_fn: Callable[..., Awaitable[list[dict]]],
    get_retrieval_top_k_fn: Callable[..., int],
    get_intent_profile_policy_fn: Callable[..., object],
    compute_widened_retry_top_k_fn: Callable[..., int],
) -> ValidationRecoveryResult:
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
        )

    has_strong_anchor = bool(
        classification.filename_filter
        or (classification.year_filter is not None and source_terms_for_retrieval)
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
        top_k=get_retrieval_top_k_fn(
            fallback_profile.preferred_retrieval_mode,
            response_mode=response_mode_used,
        ),
        max_score=profile_rag_max_score,
        year_filter=classification.year_filter,
        category_filter=classification.category_filter,
        extension_filter=classification.file_type_filter,
        filename_filter=retrieval_filename_filter,
        source_terms_filter=source_terms_for_retrieval,
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
            query_type=effective_query_type,
            subtype=classification.subtype,
            group_by=classification.group_by,
            response_shape=classification.response_shape,
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
    if not all(validation_gates.values()):
        widened_retry_top_k = compute_widened_retry_top_k_fn(
            current_top_k=effective_top_k,
            query_type=effective_query_type,
            response_mode=response_mode_used,
        )
        fallback_events.append({
            'fallback_from': fallback_profile.profile_id,
            'fallback_to': fallback_profile.profile_id,
            'fallback_reason': 'validation_gate_failed_widened_retry',
            'widened_top_k': widened_retry_top_k,
        })
        widened_chunks = await retrieve_fn(
            query=retrieval_question,
            top_k=widened_retry_top_k,
            max_score=None,
            year_filter=classification.year_filter,
            category_filter=classification.category_filter,
            extension_filter=classification.file_type_filter,
            filename_filter=retrieval_filename_filter,
            source_terms_filter=source_terms_for_retrieval,
            block_type_filter=None,
            section_filter=None,
            query_type=effective_query_type,
            db=db,
            trace=trace,
        )
        if widened_chunks:
            chunks = widened_chunks
            effective_top_k = min(widened_retry_top_k, len(chunks))
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
                query_type=effective_query_type,
                subtype=classification.subtype,
                group_by=classification.group_by,
                response_shape=classification.response_shape,
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
    )
