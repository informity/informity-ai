# ==============================================================================
# Informity AI — RAG Retrieval Plan
# Continuation-aware retrieval context and initial retrieval attempt extraction.
# ==============================================================================

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

import aiosqlite

from informity.db.models import ChatMessage
from informity.llm.query_classifier import QueryClassification
from informity.llm.rag_runtime import retrieval_validation as _retrieval_validation


@dataclass
class RetrievalContext:
    scope_reset_detected: bool
    prior_source_anchors: set[str]
    prior_has_remaining_scope: bool
    continuation_source_terms: list[str]
    source_terms_for_retrieval: list[str]
    retrieval_question: str


@dataclass
class InitialRetrievalResult:
    chunks: list[dict]
    constraint_relaxation_applied: str
    effective_query_type: str
    effective_top_k: int
    fallback_events: list[dict[str, object]]
    retrieval_elapsed_ms: float
    retrieve_timing: dict  # Per-stage timing: embed_ms, vector_search_ms, rerank_ms


def build_retrieval_context(
    *,
    question: str,
    classification: QueryClassification,
    history: list[ChatMessage] | None,
) -> RetrievalContext:
    scope_reset_detected = classification.is_scope_reset
    prior_source_anchors = _retrieval_validation._extract_prior_source_anchors(history)
    prior_has_remaining_scope = _retrieval_validation._extract_prior_has_remaining_scope(history)
    continuation_source_terms = _retrieval_validation._derive_continuation_source_terms(
        route_candidate=classification.route_candidate,
        prior_has_remaining_scope=prior_has_remaining_scope,
        scope_reset_detected=scope_reset_detected,
        prior_source_anchors=prior_source_anchors,
    )
    source_terms_for_retrieval: list[str] = list(classification.source_terms or [])
    for term in continuation_source_terms:
        if term not in source_terms_for_retrieval:
            source_terms_for_retrieval.append(term)
    retrieval_question = _retrieval_validation._build_continuation_retrieval_query(
        question=question,
        route_candidate=classification.route_candidate,
        prior_has_remaining_scope=prior_has_remaining_scope,
        scope_reset_detected=scope_reset_detected,
        is_continuation=classification.is_continuation,
        history=history,
    )
    return RetrievalContext(
        scope_reset_detected=scope_reset_detected,
        prior_source_anchors=prior_source_anchors,
        prior_has_remaining_scope=prior_has_remaining_scope,
        continuation_source_terms=continuation_source_terms,
        source_terms_for_retrieval=source_terms_for_retrieval,
        retrieval_question=retrieval_question,
    )


async def run_initial_retrieval_plan(
    *,
    retrieval_question: str,
    classification: QueryClassification,
    selected_policy_profile_id: str,
    effective_query_type: str,
    effective_top_k: int,
    profile_rag_max_score: float | None,
    source_terms_for_retrieval: list[str],
    continuation_source_terms: list[str],
    prior_has_remaining_scope: bool,
    scope_reset_detected: bool,
    retrieval_filename_filter: str | None,
    db: aiosqlite.Connection,
    trace: object | None,
    fallback_events: list[dict[str, object]],
    retrieve_with_constraints_fn: Callable[..., Awaitable[tuple[list[dict], str]]],
    retrieve_fn: Callable[..., Awaitable[list[dict]]],
) -> InitialRetrievalResult:
    retrieval_start = time.perf_counter()
    retrieve_timing: dict = {}
    retrieval_classification = QueryClassification(
        intent=classification.intent,
        response_shape=classification.response_shape,
        route_candidate=classification.route_candidate,
        confidence=classification.confidence,
        alternatives=classification.alternatives,
        reason_codes=classification.reason_codes,
        missing_slots=classification.missing_slots,
        subtype=classification.subtype,
        group_by=classification.group_by,
        field_hint=classification.field_hint,
        source_terms=source_terms_for_retrieval,
        year_filter=classification.year_filter,
        category_filter=classification.category_filter,
        file_type_filter=classification.file_type_filter,
        filename_filter=retrieval_filename_filter,
        block_type_filter=classification.block_type_filter,
        section_filter=classification.section_filter,
        is_metadata_query=classification.is_metadata_query,
        is_file_list_query=classification.is_file_list_query,
    )
    anchor_bias_enabled = (
        classification.route_candidate == 'continuation_or_refinement'
        and prior_has_remaining_scope
        and not scope_reset_detected
        and bool(continuation_source_terms)
    )
    constraint_relaxation_applied = 'none'
    chunks: list[dict] = []
    if anchor_bias_enabled:
        anchor_classification = QueryClassification(
            intent=classification.intent,
            response_shape=classification.response_shape,
            route_candidate=classification.route_candidate,
            confidence=classification.confidence,
            alternatives=classification.alternatives,
            reason_codes=classification.reason_codes,
            missing_slots=classification.missing_slots,
            subtype=classification.subtype,
            group_by=classification.group_by,
            field_hint=classification.field_hint,
            source_terms=continuation_source_terms,
            year_filter=classification.year_filter,
            category_filter=classification.category_filter,
            file_type_filter=classification.file_type_filter,
            filename_filter=retrieval_filename_filter,
            block_type_filter=classification.block_type_filter,
            section_filter=classification.section_filter,
            is_metadata_query=classification.is_metadata_query,
            is_file_list_query=classification.is_file_list_query,
        )
        chunks, constraint_relaxation_applied = await retrieve_with_constraints_fn(
            question=retrieval_question,
            effective_top_k=effective_top_k,
            profile_max_score=profile_rag_max_score,
            classification=anchor_classification,
            effective_query_type=effective_query_type,
            route_profile_id=selected_policy_profile_id,
            db=db,
            trace=trace,
            retrieve_fn=retrieve_fn,
            timing_output=retrieve_timing,
        )
        if chunks:
            fallback_events.append({
                'fallback_from': selected_policy_profile_id,
                'fallback_to': selected_policy_profile_id,
                'fallback_reason': 'continuation_anchor_bias_applied',
            })

    if not chunks:
        chunks, constraint_relaxation_applied = await retrieve_with_constraints_fn(
            question=retrieval_question,
            effective_top_k=effective_top_k,
            profile_max_score=profile_rag_max_score,
            classification=retrieval_classification,
            effective_query_type=effective_query_type,
            route_profile_id=selected_policy_profile_id,
            db=db,
            trace=trace,
            retrieve_fn=retrieve_fn,
            timing_output=retrieve_timing,
        )
    source_terms_relaxed_retry_eligible = (
        len(chunks) <= 1
        and bool(source_terms_for_retrieval)
        and retrieval_filename_filter is None
    )
    if source_terms_relaxed_retry_eligible:
        fallback_events.append({
            'fallback_from': selected_policy_profile_id,
            'fallback_to': selected_policy_profile_id,
            'fallback_reason': 'retry_without_source_terms_filter',
            'source_terms_count': len(source_terms_for_retrieval),
            'initial_chunk_count': len(chunks),
        })
        relaxed_classification = QueryClassification(
            intent=classification.intent,
            response_shape=classification.response_shape,
            route_candidate=classification.route_candidate,
            confidence=classification.confidence,
            alternatives=classification.alternatives,
            reason_codes=classification.reason_codes,
            missing_slots=classification.missing_slots,
            subtype=classification.subtype,
            group_by=classification.group_by,
            field_hint=classification.field_hint,
            source_terms=[],
            year_filter=classification.year_filter,
            category_filter=classification.category_filter,
            file_type_filter=classification.file_type_filter,
            filename_filter=retrieval_filename_filter,
            block_type_filter=classification.block_type_filter,
            section_filter=classification.section_filter,
            is_metadata_query=classification.is_metadata_query,
            is_file_list_query=classification.is_file_list_query,
        )
        relaxed_chunks, relaxed_constraint_relaxation = await retrieve_with_constraints_fn(
            question=retrieval_question,
            effective_top_k=effective_top_k,
            profile_max_score=profile_rag_max_score,
            classification=relaxed_classification,
            effective_query_type=effective_query_type,
            route_profile_id=selected_policy_profile_id,
            db=db,
            trace=trace,
            retrieve_fn=retrieve_fn,
            timing_output=retrieve_timing,
        )
        if relaxed_chunks:
            chunks = relaxed_chunks
            constraint_relaxation_applied = (
                f'{relaxed_constraint_relaxation}|source_terms_relaxed'
                if relaxed_constraint_relaxation != 'none'
                else 'source_terms_relaxed'
            )
    if not chunks and profile_rag_max_score is not None:
        fallback_events.append({
            'fallback_from': selected_policy_profile_id,
            'fallback_to': selected_policy_profile_id,
            'fallback_reason': 'retry_without_max_score',
        })
        chunks, constraint_relaxation_applied = await retrieve_with_constraints_fn(
            question=retrieval_question,
            effective_top_k=effective_top_k,
            profile_max_score=None,
            classification=retrieval_classification,
            effective_query_type=effective_query_type,
            route_profile_id=selected_policy_profile_id,
            db=db,
            trace=trace,
            retrieve_fn=retrieve_fn,
            timing_output=retrieve_timing,
        )
    retrieval_elapsed_ms = (time.perf_counter() - retrieval_start) * 1000
    return InitialRetrievalResult(
        chunks=chunks,
        constraint_relaxation_applied=constraint_relaxation_applied,
        effective_query_type=effective_query_type,
        effective_top_k=effective_top_k,
        fallback_events=fallback_events,
        retrieval_elapsed_ms=retrieval_elapsed_ms,
        retrieve_timing=retrieve_timing,
    )
