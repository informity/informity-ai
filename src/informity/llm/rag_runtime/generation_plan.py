# ==============================================================================
# Informity AI — Generation Plan
# Post-retrieval budget shaping and prompt preparation for RAG generation.
# ==============================================================================

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

from informity.llm.rag_runtime import generation_runtime as _generation_runtime


@dataclass
class GenerationPromptPlan:
    chunks: list[dict]
    effective_query_type: str
    effective_top_k: int
    effective_reasoning_enabled: bool
    effective_max_tokens: int
    timeout_seconds: int
    context_chars: int
    applied_degradations: list[dict[str, object]]
    post_retrieval_projected_seconds: float
    post_retrieval_ratio: float
    pre_closeout_quality_check_passed: bool
    pre_closeout_quality_check_reason: str | None
    output_constraints: dict[str, int]
    format_requirements: list[str]
    output_contract_plan: object | None
    messages: list[dict[str, str]]
    prompt_elapsed_ms: float
    diagnostics_depth_constraints_applied: bool


def build_generation_prompt_plan(
    *,
    question: str,
    chunks: list[dict],
    history: list | None,
    profile_name: str,
    profile_prepare_messages_fn: Callable[[list[dict[str, str]], str], list[dict[str, str]]],
    build_messages_fn: Callable[..., list[dict[str, str]]],
    fit_to_budget_enabled: bool,
    policy_soft_top_k_threshold: float,
    policy_soft_coverage_to_focused_threshold: float,
    policy_soft_output_cap_threshold: float,
    retrieval_precloseout_min_relevance_score: float,
    retrieval_relevance_score: float,
    subtype: str | None,
    focused_max_tokens: int,
    focused_timeout_seconds: int,
    applied_degradations: list[dict[str, object]],
    min_output_budget_floor: int | None,
    output_constraints: dict[str, int],
    effective_query_type: str,
    effective_top_k: int,
    effective_reasoning_enabled: bool,
    effective_max_tokens: int,
    timeout_seconds: int,
    response_mode: str,
    route_candidate: str,
    dedupe_prompt_chunks_fn: Callable[[list[dict]], list[dict]],
    derive_format_requirements_fn: Callable[[str], list[str]],
    skip_precloseout_quality_check: bool = False,
) -> GenerationPromptPlan:
    prompt_start = time.perf_counter()
    chunks = dedupe_prompt_chunks_fn(chunks)
    (
        chunks,
        effective_query_type,
        effective_top_k,
        effective_reasoning_enabled,
        effective_max_tokens,
        timeout_seconds,
        context_chars,
        applied_degradations,
        post_retrieval_projected_seconds,
        post_retrieval_ratio,
    ) = _generation_runtime._apply_post_retrieval_budget_degradations(
        fit_to_budget_enabled=fit_to_budget_enabled,
        policy_soft_top_k_threshold=policy_soft_top_k_threshold,
        policy_soft_coverage_to_focused_threshold=policy_soft_coverage_to_focused_threshold,
        profile_name=profile_name,
        question_length=len(question),
        query_type=effective_query_type,
        timeout_seconds=timeout_seconds,
        top_k=effective_top_k,
        reasoning_enabled=effective_reasoning_enabled,
        max_tokens=effective_max_tokens,
        chunks=chunks,
        subtype=subtype,
        focused_max_tokens=focused_max_tokens,
        focused_timeout_seconds=focused_timeout_seconds,
        applied_degradations=applied_degradations,
        route_candidate=route_candidate,
        min_output_budget_floor=min_output_budget_floor,
    )

    pre_closeout_quality_check_passed = True
    pre_closeout_quality_check_reason: str | None = None
    if (
        not skip_precloseout_quality_check
        and fit_to_budget_enabled
        and effective_query_type == 'focused'
        and post_retrieval_ratio >= policy_soft_output_cap_threshold
        and retrieval_relevance_score < float(retrieval_precloseout_min_relevance_score)
    ):
        pre_closeout_quality_check_passed = False
        pre_closeout_quality_check_reason = 'insufficient_relevance_under_budget_pressure'

    (
        format_requirements,
        output_constraints,
        effective_max_tokens,
        effective_reasoning_enabled,
        chunks,
        applied_degradations,
    ) = _generation_runtime._apply_strict_format_prompt_controls(
        question=question,
        chunks=chunks,
        query_type=effective_query_type,
        output_constraints=output_constraints,
        max_tokens=effective_max_tokens,
        reasoning_enabled=effective_reasoning_enabled,
        response_mode=response_mode,
        derive_format_requirements_fn=derive_format_requirements_fn,
        applied_degradations=applied_degradations,
        min_output_budget_floor=min_output_budget_floor,
    )
    diagnostics_depth_constraints_applied = any(
        str(item.get('step') or '').startswith('diagnostics_')
        for item in applied_degradations
    )
    # Phase 1 reset: no output-contract derived requirements in runtime prompt.
    output_contract_plan = None
    format_requirements = []
    output_constraints = {}
    messages = build_messages_fn(
        question,
        chunks,
        history,
        output_constraints=None,
        format_requirements=None,
        response_mode=response_mode,
    )
    messages = profile_prepare_messages_fn(messages, effective_query_type)
    prompt_elapsed_ms = (time.perf_counter() - prompt_start) * 1000

    return GenerationPromptPlan(
        chunks=chunks,
        effective_query_type=effective_query_type,
        effective_top_k=effective_top_k,
        effective_reasoning_enabled=effective_reasoning_enabled,
        effective_max_tokens=effective_max_tokens,
        timeout_seconds=timeout_seconds,
        context_chars=context_chars,
        applied_degradations=applied_degradations,
        post_retrieval_projected_seconds=post_retrieval_projected_seconds,
        post_retrieval_ratio=post_retrieval_ratio,
        pre_closeout_quality_check_passed=pre_closeout_quality_check_passed,
        pre_closeout_quality_check_reason=pre_closeout_quality_check_reason,
        output_constraints=output_constraints,
        format_requirements=format_requirements,
        output_contract_plan=output_contract_plan,
        messages=messages,
        prompt_elapsed_ms=prompt_elapsed_ms,
        diagnostics_depth_constraints_applied=diagnostics_depth_constraints_applied,
    )
