# ==============================================================================
# Informity AI — RAG Execution Plan
# Pre-retrieval orchestration and budget planning extracted from RAG handler.
# ==============================================================================

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

import aiosqlite

from informity.llm.fit_to_budget_tuning import resolve_fit_to_budget_policy
from informity.llm.intent_profiles import IntentProfilePolicy, get_intent_profile_policy
from informity.llm.model_adapter import ModelProfile, get_profile, get_retrieval_top_k
from informity.llm.query_classifier import QueryClassification

_VALID_RESPONSE_MODES = {'balanced', 'analysis', 'research'}


@dataclass
class RAGExecutionPlan:
    profile: ModelProfile
    selected_policy: IntentProfilePolicy
    effective_response_shape: str
    response_mode_used: str
    retrieval_top_k: int
    timeout_seconds: int
    max_tokens: int
    reasoning_enabled: bool
    mode_adjustments_applied: list[dict[str, object]]
    diagnostics_min_words: int | None
    policy: object
    effective_query_type: str
    effective_top_k: int
    effective_reasoning_enabled: bool
    effective_max_tokens: int
    fit_to_budget_enabled: bool
    output_constraints: dict[str, int]
    applied_degradations: list[dict[str, object]]
    strict_ordered_mode: bool
    fallback_events: list[dict[str, object]]


async def build_execution_plan(
    *,
    question: str,
    classification: QueryClassification,
    response_mode: str | None,
    diagnostics_context: dict[str, object] | None,
    db: aiosqlite.Connection,
    resolve_fit_to_budget_policy_fn: Callable[..., Awaitable[object]] = resolve_fit_to_budget_policy,
) -> RAGExecutionPlan:
    profile = get_profile()
    selected_policy = get_intent_profile_policy(classification.route_candidate)
    query_type = selected_policy.preferred_retrieval_mode
    fallback_events: list[dict[str, object]] = []
    effective_response_shape = classification.response_shape
    if effective_response_shape not in selected_policy.allowed_output_shapes:
        fallback_events.append({
            'fallback_from': classification.response_shape,
            'fallback_to': 'narrative_synthesis',
            'fallback_reason': 'response_shape_not_allowed_for_profile',
        })
        effective_response_shape = 'narrative_synthesis'
    if classification.confidence_band == 'low':
        fallback_events.append({
            'fallback_from': classification.route_candidate,
            'fallback_to': 'clarification_or_disambiguation',
            'fallback_reason': 'low_confidence_route_guard',
        })
        selected_policy = get_intent_profile_policy('clarification_or_disambiguation')
        query_type = selected_policy.preferred_retrieval_mode

    response_mode_used = str(response_mode or 'balanced').strip().lower()
    if response_mode_used not in _VALID_RESPONSE_MODES:
        response_mode_used = 'balanced'

    retrieval_top_k = get_retrieval_top_k(query_type, response_mode=response_mode_used)
    timeout_seconds = profile.get_mode_timeout_seconds(query_type, response_mode_used)
    max_tokens = profile.get_mode_max_tokens(query_type, response_mode_used)
    reasoning_enabled = profile.get_reasoning_enabled(query_type)

    mode_adjustments_applied: list[dict[str, object]] = []
    diagnostics_min_words: int | None = None
    if isinstance(diagnostics_context, dict):
        min_words_value = diagnostics_context.get('output_shape_min_words')
        if isinstance(min_words_value, int) and min_words_value > 0:
            diagnostics_min_words = min_words_value
    if response_mode_used in {'analysis', 'research'}:
        mode_adjustments_applied.append({
            'step': f'{response_mode_used}_profile_budget_applied',
            'top_k': {'to': retrieval_top_k},
            'timeout_seconds': {'to': timeout_seconds},
            'max_tokens': {'to': max_tokens},
        })

    policy = await resolve_fit_to_budget_policy_fn(
        db=db,
        query_type=query_type,
        timeout_seconds=timeout_seconds,
        response_mode=response_mode_used,
    )
    effective_query_type = query_type
    effective_top_k = retrieval_top_k
    effective_reasoning_enabled = reasoning_enabled
    effective_max_tokens = max_tokens
    fit_to_budget_enabled = policy.enabled
    output_constraints: dict[str, int] = {}
    applied_degradations: list[dict[str, object]] = []
    strict_ordered_mode = False

    return RAGExecutionPlan(
        profile=profile,
        selected_policy=selected_policy,
        effective_response_shape=effective_response_shape,
        response_mode_used=response_mode_used,
        retrieval_top_k=retrieval_top_k,
        timeout_seconds=timeout_seconds,
        max_tokens=max_tokens,
        reasoning_enabled=reasoning_enabled,
        mode_adjustments_applied=mode_adjustments_applied,
        diagnostics_min_words=diagnostics_min_words,
        policy=policy,
        effective_query_type=effective_query_type,
        effective_top_k=effective_top_k,
        effective_reasoning_enabled=effective_reasoning_enabled,
        effective_max_tokens=effective_max_tokens,
        fit_to_budget_enabled=fit_to_budget_enabled,
        output_constraints=output_constraints,
        applied_degradations=applied_degradations,
        strict_ordered_mode=strict_ordered_mode,
        fallback_events=fallback_events,
    )
