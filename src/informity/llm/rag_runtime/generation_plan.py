# ==============================================================================
# Informity AI — Generation Plan
# Post-retrieval budget shaping and prompt preparation for RAG generation.
# ==============================================================================

from __future__ import annotations

import re
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
    route_candidate: str,
    dedupe_prompt_chunks_fn: Callable[[list[dict]], list[dict]],
    derive_format_requirements_fn: Callable[[str, dict[str, bool] | None], list[str]],
    action_hints: dict[str, bool] | None = None,
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
        derive_format_requirements_fn=derive_format_requirements_fn,
        action_hints=action_hints,
        applied_degradations=applied_degradations,
        min_output_budget_floor=min_output_budget_floor,
    )
    diagnostics_depth_constraints_applied = any(
        str(item.get('step') or '').startswith('diagnostics_')
        for item in applied_degradations
    )
    requires_missing_evidence_callout = any(
        'missing evidence' in str(requirement or '').casefold()
        for requirement in format_requirements
    )
    min_year_subsections = 0
    if any('at least 2 distinct year subsections' in str(requirement or '').casefold() for requirement in format_requirements):
        min_year_subsections = 2
    context_years = sorted({
        int(chunk.get('year'))
        for chunk in chunks
        if isinstance(chunk.get('year'), int)
    })
    output_contract_plan_data: dict[str, object] = {}
    required_terms: list[str] = []
    required_headings: list[str] = []
    enforce_heading_order = False
    for requirement in format_requirements:
        heading_match = re.match(r'^\s*include\s+heading\s*:\s*(.+?)\s*$', str(requirement or ''), flags=re.IGNORECASE)
        if heading_match is not None:
            heading = str(heading_match.group(1) or '').strip()
            if heading and heading.casefold() not in {item.casefold() for item in required_headings}:
                required_headings.append(heading)
            continue
        if 'requested order' in str(requirement or '').casefold():
            enforce_heading_order = True
        match = re.match(r'^\s*include\s+term\s*:\s*(.+?)\s*$', str(requirement or ''), flags=re.IGNORECASE)
        if match is None:
            continue
        term = str(match.group(1) or '').strip().casefold()
        if not term or term in required_terms:
            continue
        required_terms.append(term)
    if requires_missing_evidence_callout:
        output_contract_plan_data['requires_missing_evidence_callout'] = True
    if min_year_subsections > 0:
        output_contract_plan_data['min_year_subsections'] = min_year_subsections
        output_contract_plan_data['expected_years'] = context_years
    if required_headings:
        output_contract_plan_data['required_headings'] = required_headings
        output_contract_plan_data['enforce_required_headings'] = True
        output_contract_plan_data['enforce_heading_order'] = enforce_heading_order
    if required_terms:
        output_contract_plan_data['required_terms'] = required_terms
        output_contract_plan_data['enforce_required_terms'] = True
    output_contract_plan = output_contract_plan_data or None
    messages = build_messages_fn(
        question,
        chunks,
        history,
        output_constraints=output_constraints,
        format_requirements=format_requirements,
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
