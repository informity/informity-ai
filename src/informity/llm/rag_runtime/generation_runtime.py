# ==============================================================================
# Informity AI — RAG Generation Runtime Policy Helpers
# Budget estimation + strict-format output controls extracted from handler
# ==============================================================================

from informity.llm.model_adapter import get_profile_tokens_per_second
from informity.llm.rag_runtime import strict_output_contract as _strict_output_contract

_STRICT_ORDERED_MAX_TOKENS = 1800
_STRICT_ORDERED_MAX_WORDS = 420      # ~2 pages; keeps structured answers concise — see completed/output-length-control-strategies.md
_STRICT_ORDERED_MAX_ROWS = 18        # table row cap; beyond 18 rows strict-format answers exceed practical budget
_STRICT_ORDERED_MAX_CONTEXT_CHUNKS = 8
_STRICT_ORDERED_MAX_CONTEXT_CHARS = 2800
_STRICT_ORDERED_MAX_CHUNK_CHARS = 900
_STRICT_ORDERED_PRE_RETRIEVAL_TOP_K_CAP = 8
_STRICT_ORDERED_PRE_RETRIEVAL_TIMEOUT_CAP_SECONDS = 75
_STRICT_ORDERED_PRE_RETRIEVAL_MAX_TOKENS_CAP = 1800
_COVERAGE_ROUTE_TOPK_GUARD_IDS = {
    'cross_document_synthesis',
    'comparative_analysis',
    'audit_or_compliance_brief',
}
_FOCUSED_ROUTE_DEGRADATION_GUARD_IDS = {
    'targeted_fact_lookup',
    'structured_field_extraction',
}
_STRICT_ORDERED_TIMEOUT_AWARE_MIN_MAX_TOKENS = 720
_STRICT_COMPLEX_ORDERED_MAX_WORDS = 520  # higher cap for multi-section complex formats — see completed/output-length-control-strategies.md
_STRICT_COMPLEX_ORDERED_MAX_ROWS = 24    # proportionally higher row cap for complex ordered formats
_STRICT_COMPLEX_ORDERED_MAX_CONTEXT_CHUNKS = 8
_STRICT_COMPLEX_ORDERED_MAX_CONTEXT_CHARS = 3200
_STRICT_COMPLEX_ORDERED_MAX_CHUNK_CHARS = 1100
_STRICT_COMPLEX_ORDERED_PRE_RETRIEVAL_TOP_K_CAP = 10
_STRICT_COMPLEX_ORDERED_PRE_RETRIEVAL_TIMEOUT_CAP_SECONDS = 120
_STRICT_COMPLEX_ORDERED_PRE_RETRIEVAL_MAX_TOKENS_CAP = 1800
_STRICT_COMPLEX_ORDERED_TIMEOUT_AWARE_MIN_MAX_TOKENS = 900
_FOCUSED_POST_RETRIEVAL_MAX_CONTEXT_CHARS = 5200
_FOCUSED_POST_RETRIEVAL_MAX_CHUNK_CHARS = 1200
_COVERAGE_PREFILL_MAX_CONTEXT_CHARS = 20000
_COVERAGE_PREFILL_MAX_CHUNK_CHARS = 2200
_DIAGNOSTICS_DEPTH_CONTEXT_CHARS_FLOOR = 22000
_SOURCE_SCOPED_COVERAGE_TOP_K_CAP = 14
_SOURCE_SCOPED_COVERAGE_TIMEOUT_CAP_SECONDS = 220
_SOURCE_SCOPED_COVERAGE_MAX_TOKENS_CAP = 1600


def _apply_source_scoped_coverage_guard(
    *,
    query_type: str,
    route_candidate: str,
    source_terms: list[str],
    timeout_seconds: int,
    top_k: int,
    reasoning_enabled: bool,
    max_tokens: int,
    applied_degradations: list[dict[str, object]],
) -> tuple[int, int, bool, int, list[dict[str, object]]]:
    if query_type != 'coverage' or route_candidate != 'cross_document_synthesis' or not source_terms:
        return timeout_seconds, top_k, reasoning_enabled, max_tokens, applied_degradations

    adjusted_timeout_seconds = timeout_seconds
    adjusted_top_k = top_k
    adjusted_reasoning_enabled = reasoning_enabled
    adjusted_max_tokens = max_tokens

    if adjusted_top_k > _SOURCE_SCOPED_COVERAGE_TOP_K_CAP:
        applied_degradations.append({
            'step': 'source_scoped_coverage_top_k_cap',
            'from': adjusted_top_k,
            'to': _SOURCE_SCOPED_COVERAGE_TOP_K_CAP,
            'reason': 'source_scoped_coverage_latency_guard',
        })
        adjusted_top_k = _SOURCE_SCOPED_COVERAGE_TOP_K_CAP

    if adjusted_timeout_seconds > _SOURCE_SCOPED_COVERAGE_TIMEOUT_CAP_SECONDS:
        applied_degradations.append({
            'step': 'source_scoped_coverage_timeout_cap',
            'from': adjusted_timeout_seconds,
            'to': _SOURCE_SCOPED_COVERAGE_TIMEOUT_CAP_SECONDS,
            'reason': 'source_scoped_coverage_latency_guard',
        })
        adjusted_timeout_seconds = _SOURCE_SCOPED_COVERAGE_TIMEOUT_CAP_SECONDS

    if adjusted_max_tokens > _SOURCE_SCOPED_COVERAGE_MAX_TOKENS_CAP:
        applied_degradations.append({
            'step': 'source_scoped_coverage_max_tokens_cap',
            'from': adjusted_max_tokens,
            'to': _SOURCE_SCOPED_COVERAGE_MAX_TOKENS_CAP,
            'reason': 'source_scoped_coverage_latency_guard',
        })
        adjusted_max_tokens = _SOURCE_SCOPED_COVERAGE_MAX_TOKENS_CAP

    return (
        adjusted_timeout_seconds,
        adjusted_top_k,
        adjusted_reasoning_enabled,
        adjusted_max_tokens,
        applied_degradations,
    )


def _has_remaining_scope(
    *,
    timeout_reason: str | None,
    stream_recovery_reason: str | None,
    generation_skipped: bool,
    applied_degradations: list[dict[str, object]],
) -> bool:
    if timeout_reason is not None or stream_recovery_reason is not None or generation_skipped:
        return True
    scope_reduction_steps = {
        'coverage_to_focused_subset',
        'reduce_top_k',
        'reduce_context_chunks',
    }
    for degradation in applied_degradations:
        step = str(degradation.get('step') or '')
        if step in scope_reduction_steps:
            return True
    return False


def _should_apply_soft_stream_closeout(format_requirements: list[str]) -> bool:
    # Strict section-ordered outputs should not be cut off by soft closeout.
    return not any(
        _strict_output_contract._is_ordered_headings_requirement(requirement)
        for requirement in format_requirements
    )


def _count_required_headings(format_requirements: list[str]) -> int:
    return sum(
        1
        for requirement in format_requirements
        if _strict_output_contract._extract_heading_from_requirement(requirement) is not None
    )


def _is_complex_strict_ordered_contract(format_requirements: list[str]) -> bool:
    heading_count = _count_required_headings(format_requirements)
    max_bullet_depth = max(
        (
            _strict_output_contract._extract_bullet_depth_requirement(requirement) or 0
            for requirement in format_requirements
        ),
        default=0,
    )
    has_missing_evidence_callout = any(
        _strict_output_contract._is_missing_evidence_requirement(requirement)
        for requirement in format_requirements
    )
    return heading_count >= 6 or max_bullet_depth >= 3 or has_missing_evidence_callout


def _augment_strict_ordered_format_requirements(format_requirements: list[str]) -> list[str]:
    has_strict_order = any(
        _strict_output_contract._is_ordered_headings_requirement(item)
        for item in format_requirements
    )
    if not has_strict_order:
        return format_requirements
    extra_requirements = [
        'ensure every required heading appears in the answer, even if some sections are brief',
        'prioritize breadth before depth: cover all required headings first, then add concise details',
    ]
    requires_three_level_bullets = any(
        (_strict_output_contract._extract_bullet_depth_requirement(item) or 0) >= 3
        for item in format_requirements
    )
    if requires_three_level_bullets:
        extra_requirements.append(
            'when nested bullets are requested, include at least one explicit 3-level chain (parent -> child -> grandchild)'
        )
        extra_requirements.append(
            'use this nesting shape where requested: "- Parent\\n  - Child\\n    - Grandchild" (not flat bullets)'
        )
    merged = list(format_requirements)
    for requirement in extra_requirements:
        if requirement not in merged:
            merged.append(requirement)
    return merged


def _apply_strict_ordered_output_budget(
    *,
    format_requirements: list[str],
    query_type: str,
    output_constraints: dict[str, int],
    max_tokens: int,
    reasoning_enabled: bool,
    strict_contract_complexity: bool = False,
    response_mode: str = 'analysis',
) -> tuple[dict[str, int], int, bool, dict[str, object] | None]:
    has_strict_order = any(
        _strict_output_contract._is_ordered_headings_requirement(item)
        for item in format_requirements
    )
    if not has_strict_order:
        return output_constraints, max_tokens, reasoning_enabled, None

    normalized_mode = str(response_mode or 'analysis').strip().lower()
    # Research mode should preserve strict structure but not force short-output caps.
    # Depth limits are controlled by profile budgets in research mode.
    if normalized_mode == 'research':
        return output_constraints, max_tokens, reasoning_enabled, None

    adjusted_constraints = dict(output_constraints)
    old_max_words = adjusted_constraints.get('max_words')
    old_max_rows = adjusted_constraints.get('max_rows')
    heading_count = _count_required_headings(format_requirements)
    strict_max_words = _STRICT_ORDERED_MAX_WORDS if query_type == 'coverage' else 340
    strict_max_rows = _STRICT_ORDERED_MAX_ROWS if query_type == 'coverage' else 14
    if query_type == 'coverage' and strict_contract_complexity:
        strict_max_words = _STRICT_COMPLEX_ORDERED_MAX_WORDS
        strict_max_rows = _STRICT_COMPLEX_ORDERED_MAX_ROWS

    if not isinstance(old_max_words, int):
        adjusted_constraints['max_words'] = strict_max_words
    if not isinstance(old_max_rows, int):
        adjusted_constraints['max_rows'] = strict_max_rows

    old_max_tokens = max_tokens
    strict_max_tokens_cap = _STRICT_ORDERED_MAX_TOKENS if query_type == 'coverage' else 900
    if query_type == 'coverage' and strict_contract_complexity:
        strict_max_tokens_cap = _STRICT_COMPLEX_ORDERED_PRE_RETRIEVAL_MAX_TOKENS_CAP
    if response_mode in {'analysis', 'research'}:
        strict_max_tokens_cap = min(
            2600 if query_type == 'coverage' else 1600,
            strict_max_tokens_cap + 280,
        )
    adjusted_max_tokens = min(max_tokens, strict_max_tokens_cap)
    adjusted_reasoning_enabled = False

    if (
        adjusted_constraints == output_constraints
        and adjusted_max_tokens == old_max_tokens
        and adjusted_reasoning_enabled == reasoning_enabled
    ):
        return output_constraints, max_tokens, reasoning_enabled, None

    return adjusted_constraints, adjusted_max_tokens, adjusted_reasoning_enabled, {
        'step': 'strict_ordered_section_budget',
        'required_heading_count': heading_count,
        'from_max_tokens': old_max_tokens,
        'to_max_tokens': adjusted_max_tokens,
        'from_reasoning_enabled': reasoning_enabled,
        'to_reasoning_enabled': adjusted_reasoning_enabled,
        'from_constraints': output_constraints,
        'to_constraints': adjusted_constraints,
        'reason': 'strict_ordered_heading_requirements',
    }


def _estimate_tokens_per_second(profile_name: str) -> float:
    # Deterministic profile-driven runtime estimate for budget planning.
    return get_profile_tokens_per_second(profile_name)


def _estimate_budget_ratio(
    *,
    profile_name: str,
    query_type: str,
    timeout_seconds: int,
    question_length: int,
    context_chunks: int,
    context_chars: int,
    top_k: int,
    reasoning_enabled: bool,
    max_tokens: int,
) -> tuple[float, float]:
    # Deterministic wall-clock estimate from query+context+profile features.
    # Calibration update (2.1): focused traces show underestimation when prompts include
    # dense structured context. Keep this deterministic and query-type based.
    default_chars_per_chunk = 1200 if query_type == 'focused' else 950
    effective_context_chars = context_chars if context_chars > 0 else context_chunks * default_chars_per_chunk
    retrieval_seconds = 0.35 + (top_k * 0.06) + (0.5 if query_type == 'coverage' else 0.35)
    prompt_seconds = 0.25 + (effective_context_chars / 9000.0) + (min(question_length, 1500) / 2800.0)
    projected_output_tokens = float(max_tokens)
    if not reasoning_enabled:
        projected_output_tokens *= 0.78
    generation_seconds = projected_output_tokens / _estimate_tokens_per_second(profile_name)
    if query_type == 'focused':
        generation_seconds *= 1.25
    projected_total_seconds = retrieval_seconds + prompt_seconds + generation_seconds
    timeout = float(max(timeout_seconds, 1))
    return projected_total_seconds, projected_total_seconds / timeout


def _apply_strict_format_prompt_controls(
    *,
    question: str,
    chunks: list[dict],
    query_type: str,
    output_constraints: dict[str, int],
    max_tokens: int,
    reasoning_enabled: bool,
    response_mode: str,
    derive_format_requirements_fn,
    applied_degradations: list[dict[str, object]],
    min_output_budget_floor: int | None = None,
) -> tuple[list[str], dict[str, int], int, bool, list[dict], list[dict[str, object]]]:
    normalized_mode = str(response_mode or 'analysis').strip().lower()
    format_requirements = derive_format_requirements_fn(question)
    format_requirements = _augment_strict_ordered_format_requirements(format_requirements)
    has_strict_ordered_headings = any(
        _strict_output_contract._is_ordered_headings_requirement(item)
        for item in format_requirements
    )
    strict_contract_complexity = _is_complex_strict_ordered_contract(format_requirements)
    output_constraints, max_tokens, reasoning_enabled, strict_budget_degradation = (
        _apply_strict_ordered_output_budget(
            format_requirements=format_requirements,
            query_type=query_type,
            output_constraints=output_constraints,
            max_tokens=max_tokens,
            reasoning_enabled=reasoning_enabled,
            strict_contract_complexity=strict_contract_complexity,
            response_mode=response_mode,
        )
    )
    if strict_budget_degradation is not None:
        applied_degradations.append(strict_budget_degradation)
    strict_context_chunks_cap = _STRICT_ORDERED_MAX_CONTEXT_CHUNKS
    strict_context_chars_cap = _STRICT_ORDERED_MAX_CONTEXT_CHARS
    strict_chunk_chars_cap = _STRICT_ORDERED_MAX_CHUNK_CHARS
    if query_type == 'coverage' and strict_contract_complexity:
        strict_context_chunks_cap = _STRICT_COMPLEX_ORDERED_MAX_CONTEXT_CHUNKS
        strict_context_chars_cap = _STRICT_COMPLEX_ORDERED_MAX_CONTEXT_CHARS
        strict_chunk_chars_cap = _STRICT_COMPLEX_ORDERED_MAX_CHUNK_CHARS
    if normalized_mode == 'research':
        strict_context_chunks_cap = max(strict_context_chunks_cap, 14)
        strict_context_chars_cap = max(strict_context_chars_cap, 8000)
        strict_chunk_chars_cap = max(strict_chunk_chars_cap, 1600)

    if has_strict_ordered_headings and len(chunks) > strict_context_chunks_cap:
        old_chunk_count = len(chunks)
        chunks = chunks[:strict_context_chunks_cap]
        applied_degradations.append({
            'step': 'strict_ordered_context_cap',
            'from': old_chunk_count,
            'to': len(chunks),
            'reason': 'strict_ordered_heading_requirements',
        })
    if has_strict_ordered_headings and chunks:
        old_total_chars = sum(len(str(chunk.get('chunk_text', '') or '')) for chunk in chunks)
        remaining_char_budget = strict_context_chars_cap
        trimmed_chunks: list[dict] = []
        for chunk in chunks:
            if remaining_char_budget <= 0:
                break
            chunk_text = str(chunk.get('chunk_text', '') or '')
            if not chunk_text:
                continue
            allowed_chars = min(strict_chunk_chars_cap, remaining_char_budget)
            trimmed_text = chunk_text[:allowed_chars]
            updated_chunk = dict(chunk)
            updated_chunk['chunk_text'] = trimmed_text
            trimmed_chunks.append(updated_chunk)
            remaining_char_budget -= len(trimmed_text)
        if trimmed_chunks:
            chunks = trimmed_chunks
        new_total_chars = sum(len(str(chunk.get('chunk_text', '') or '')) for chunk in chunks)
        if new_total_chars < old_total_chars:
            applied_degradations.append({
                'step': 'strict_ordered_context_chars_cap',
                'from_total_chars': old_total_chars,
                'to_total_chars': new_total_chars,
                'max_total_chars': strict_context_chars_cap,
                'max_chunk_chars': strict_chunk_chars_cap,
                'reason': 'strict_ordered_heading_requirements',
            })
    if (
        has_strict_ordered_headings
        and isinstance(min_output_budget_floor, int)
        and min_output_budget_floor >= 800
    ):
        floor_words = int(min_output_budget_floor)
        current_max_words = output_constraints.get('max_words')
        if not isinstance(current_max_words, int) or current_max_words < floor_words:
            output_constraints['max_words'] = floor_words
            applied_degradations.append({
                'step': 'diagnostics_min_output_budget_floor',
                'from_max_words': current_max_words,
                'to_max_words': floor_words,
                'reason': 'diagnostics_output_shape_min_words',
            })
        min_token_floor = min(4096, max(1800, floor_words + 400))
        if max_tokens < min_token_floor:
            old_max_tokens = max_tokens
            max_tokens = min_token_floor
            applied_degradations.append({
                'step': 'diagnostics_min_token_floor',
                'from_max_tokens': old_max_tokens,
                'to_max_tokens': max_tokens,
                'reason': 'diagnostics_output_shape_min_words',
            })
    return (
        format_requirements,
        output_constraints,
        max_tokens,
        reasoning_enabled,
        chunks,
        applied_degradations,
    )


def _apply_strict_pre_retrieval_guard(
    *,
    question: str,
    query_type: str,
    timeout_seconds: int,
    top_k: int,
    reasoning_enabled: bool,
    max_tokens: int,
    applied_degradations: list[dict[str, object]],
    derive_format_requirements_fn,
    profile_name: str = '',
    response_mode: str = 'analysis',
) -> tuple[int, int, bool, int, list[dict[str, object]], bool]:
    format_requirements = derive_format_requirements_fn(question)
    has_strict_order = any(
        _strict_output_contract._is_ordered_headings_requirement(item)
        for item in format_requirements
    )
    if not has_strict_order:
        return timeout_seconds, top_k, reasoning_enabled, max_tokens, applied_degradations, False

    normalized_mode = str(response_mode or 'analysis').strip().lower()
    if normalized_mode == 'research':
        # Keep strict-ordered mode semantics (for contract handling), but do not
        # preemptively degrade retrieval/generation budgets in research mode.
        return timeout_seconds, top_k, reasoning_enabled, max_tokens, applied_degradations, True

    strict_contract_complexity = _is_complex_strict_ordered_contract(format_requirements)

    adjusted_timeout_seconds = timeout_seconds
    adjusted_top_k = top_k
    adjusted_reasoning_enabled = reasoning_enabled
    adjusted_max_tokens = max_tokens

    strict_top_k_cap = _STRICT_ORDERED_PRE_RETRIEVAL_TOP_K_CAP if query_type == 'coverage' else 6
    strict_timeout_cap = _STRICT_ORDERED_PRE_RETRIEVAL_TIMEOUT_CAP_SECONDS if query_type == 'coverage' else 60
    strict_max_tokens_cap = _STRICT_ORDERED_PRE_RETRIEVAL_MAX_TOKENS_CAP if query_type == 'coverage' else 700
    timeout_aware_min_max_tokens = _STRICT_ORDERED_TIMEOUT_AWARE_MIN_MAX_TOKENS
    generation_budget_share = 0.58 if query_type == 'coverage' else 0.5
    if query_type == 'coverage' and strict_contract_complexity:
        strict_top_k_cap = _STRICT_COMPLEX_ORDERED_PRE_RETRIEVAL_TOP_K_CAP
        strict_timeout_cap = _STRICT_COMPLEX_ORDERED_PRE_RETRIEVAL_TIMEOUT_CAP_SECONDS
        strict_max_tokens_cap = _STRICT_COMPLEX_ORDERED_PRE_RETRIEVAL_MAX_TOKENS_CAP
        timeout_aware_min_max_tokens = _STRICT_COMPLEX_ORDERED_TIMEOUT_AWARE_MIN_MAX_TOKENS
        generation_budget_share = 0.58
    if response_mode in {'analysis', 'research'}:
        strict_top_k_cap = min(
            14 if query_type == 'coverage' else 8,
            strict_top_k_cap + 2,
        )
        strict_timeout_cap = min(210, int(strict_timeout_cap * 1.45))
        strict_max_tokens_cap = min(
            2800 if query_type == 'coverage' else 1800,
            strict_max_tokens_cap + 380,
        )
        timeout_aware_min_max_tokens = min(
            strict_max_tokens_cap,
            timeout_aware_min_max_tokens + 180,
        )
        generation_budget_share = min(0.78, generation_budget_share + 0.10)

    if adjusted_top_k > strict_top_k_cap:
        applied_degradations.append({
            'step': 'strict_pre_retrieval_top_k_cap',
            'from': adjusted_top_k,
            'to': strict_top_k_cap,
            'reason': 'strict_ordered_heading_requirements',
        })
        adjusted_top_k = strict_top_k_cap

    if adjusted_timeout_seconds > strict_timeout_cap:
        applied_degradations.append({
            'step': 'strict_pre_retrieval_timeout_cap',
            'from': adjusted_timeout_seconds,
            'to': strict_timeout_cap,
            'reason': 'strict_ordered_heading_requirements',
        })
        adjusted_timeout_seconds = strict_timeout_cap

    # Keep strict ordered outputs within a bounded generation window for short query timeouts.
    # This is deterministic (timeout + model profile only) and avoids query-specific behavior.
    tokens_per_second = _estimate_tokens_per_second(profile_name)
    timeout_aware_cap = int(
        (float(max(adjusted_timeout_seconds, 1)) * generation_budget_share * tokens_per_second) / 0.78
    )
    timeout_aware_cap = max(
        timeout_aware_min_max_tokens,
        min(max(adjusted_max_tokens, strict_max_tokens_cap), timeout_aware_cap),
    )
    if adjusted_max_tokens > timeout_aware_cap:
        applied_degradations.append({
            'step': 'strict_pre_retrieval_timeout_aware_max_tokens_cap',
            'from': adjusted_max_tokens,
            'to': timeout_aware_cap,
            'timeout_seconds': adjusted_timeout_seconds,
            'profile_name': profile_name,
            'reason': 'strict_ordered_timeout_budget',
        })
        adjusted_max_tokens = timeout_aware_cap

    if adjusted_reasoning_enabled:
        applied_degradations.append({
            'step': 'strict_pre_retrieval_disable_reasoning',
            'from': True,
            'to': False,
            'reason': 'strict_ordered_heading_requirements',
        })
        adjusted_reasoning_enabled = False

    return (
        adjusted_timeout_seconds,
        adjusted_top_k,
        adjusted_reasoning_enabled,
        adjusted_max_tokens,
        applied_degradations,
        True,
    )


def _apply_preflight_budget_degradations(
    *,
    fit_to_budget_enabled: bool,
    policy_soft_top_k_threshold: float,
    policy_soft_reasoning_threshold: float,
    policy_soft_output_cap_threshold: float,
    policy_soft_coverage_to_focused_threshold: float,
    profile_name: str,
    question_length: int,
    query_type: str,
    timeout_seconds: int,
    top_k: int,
    reasoning_enabled: bool,
    max_tokens: int,
    subtype: str | None,
    focused_max_tokens: int,
    focused_timeout_seconds: int,
    output_constraints: dict[str, int],
    applied_degradations: list[dict[str, object]],
    route_candidate: str | None = None,
    response_mode: str = 'analysis',
    strict_ordered_mode: bool = False,
) -> tuple[str, int, bool, int, int, dict[str, int], list[dict[str, object]], float, float]:
    effective_query_type = query_type
    effective_top_k = top_k
    effective_reasoning_enabled = reasoning_enabled
    effective_max_tokens = max_tokens
    effective_timeout_seconds = timeout_seconds
    effective_output_constraints = dict(output_constraints)

    projected_seconds, ratio = _estimate_budget_ratio(
        profile_name=profile_name,
        query_type=effective_query_type,
        timeout_seconds=effective_timeout_seconds,
        question_length=question_length,
        context_chunks=effective_top_k,
        context_chars=0,
        top_k=effective_top_k,
        reasoning_enabled=effective_reasoning_enabled,
        max_tokens=effective_max_tokens,
    )

    coverage_route_topk_guard_enabled = (
        effective_query_type == 'coverage'
        and str(route_candidate or '').strip() in _COVERAGE_ROUTE_TOPK_GUARD_IDS
    )
    focused_route_degradation_guard_enabled = (
        effective_query_type == 'focused'
        and str(route_candidate or '').strip() in _FOCUSED_ROUTE_DEGRADATION_GUARD_IDS
    )

    if (
        fit_to_budget_enabled
        and ratio >= policy_soft_top_k_threshold
        and effective_top_k > 6
        and not coverage_route_topk_guard_enabled
        and not focused_route_degradation_guard_enabled
    ):
        old_top_k = effective_top_k
        effective_top_k = max(6, int(effective_top_k * 0.7))
        applied_degradations.append({
            'step': 'reduce_top_k',
            'from': old_top_k,
            'to': effective_top_k,
            'reason': 'preflight_ratio_high',
        })

    projected_seconds, ratio = _estimate_budget_ratio(
        profile_name=profile_name,
        query_type=effective_query_type,
        timeout_seconds=effective_timeout_seconds,
        question_length=question_length,
        context_chunks=effective_top_k,
        context_chars=0,
        top_k=effective_top_k,
        reasoning_enabled=effective_reasoning_enabled,
        max_tokens=effective_max_tokens,
    )

    effective_reasoning_threshold = policy_soft_reasoning_threshold
    if effective_query_type == 'focused':
        effective_reasoning_threshold = min(effective_reasoning_threshold, 0.72)
    if (
        fit_to_budget_enabled
        and ratio >= effective_reasoning_threshold
        and effective_reasoning_enabled
        and not coverage_route_topk_guard_enabled
        and not focused_route_degradation_guard_enabled
    ):
        effective_reasoning_enabled = False
        applied_degradations.append({
            'step': 'disable_reasoning',
            'reason': (
                'preflight_ratio_high_focused_guard'
                if effective_query_type == 'focused'
                else 'preflight_ratio_high'
            ),
        })

    projected_seconds, ratio = _estimate_budget_ratio(
        profile_name=profile_name,
        query_type=effective_query_type,
        timeout_seconds=effective_timeout_seconds,
        question_length=question_length,
        context_chunks=effective_top_k,
        context_chars=0,
        top_k=effective_top_k,
        reasoning_enabled=effective_reasoning_enabled,
        max_tokens=effective_max_tokens,
    )

    effective_output_cap_threshold = policy_soft_output_cap_threshold
    if effective_query_type == 'focused':
        effective_output_cap_threshold = min(effective_output_cap_threshold, 0.78)
    if fit_to_budget_enabled and ratio >= effective_output_cap_threshold:
        if response_mode == 'research':
            effective_output_constraints = {
                'max_sections': 10 if effective_query_type == 'coverage' else 8,
                'max_rows': 36 if effective_query_type == 'coverage' else 20,
                'max_words': 2200 if effective_query_type == 'coverage' else 1400,
            }
        else:
            effective_output_constraints = {
                'max_sections': 8 if effective_query_type == 'coverage' else 4,
                'max_rows': 30 if effective_query_type == 'coverage' else 12,
                'max_words': 900 if effective_query_type == 'coverage' else 420,
            }
        if strict_ordered_mode:
            # Strict-ordered contracts already carry explicit structure and often a prompt-level word budget.
            # Avoid adding a second synthetic max_words cap that can cause contract truncation failures.
            effective_output_constraints.pop('max_words', None)
        old_max_tokens = effective_max_tokens
        if response_mode == 'research':
            research_cap = 4096 if effective_query_type == 'coverage' else 3072
            effective_max_tokens = min(effective_max_tokens, research_cap)
        else:
            effective_max_tokens = min(
                effective_max_tokens,
                1800 if effective_query_type == 'coverage' else 900,
            )
        applied_degradations.append({
            'step': 'cap_output_structure',
            'from_max_tokens': old_max_tokens,
            'to_max_tokens': effective_max_tokens,
            'constraints': effective_output_constraints,
            'reason': (
                'preflight_ratio_high_focused_guard'
                if effective_query_type == 'focused'
                else 'preflight_ratio_high'
            ),
        })

    projected_seconds, ratio = _estimate_budget_ratio(
        profile_name=profile_name,
        query_type=effective_query_type,
        timeout_seconds=effective_timeout_seconds,
        question_length=question_length,
        context_chunks=effective_top_k,
        context_chars=0,
        top_k=effective_top_k,
        reasoning_enabled=effective_reasoning_enabled,
        max_tokens=effective_max_tokens,
    )

    if (
        fit_to_budget_enabled
        and ratio >= policy_soft_coverage_to_focused_threshold
        and effective_query_type == 'coverage'
        and subtype != 'aggregate_by_period'
        and not coverage_route_topk_guard_enabled
    ):
        old_top_k = effective_top_k
        effective_query_type = 'focused'
        effective_top_k = max(6, min(effective_top_k, int(effective_top_k * 0.75)))
        effective_max_tokens = min(effective_max_tokens, focused_max_tokens)
        effective_timeout_seconds = focused_timeout_seconds
        applied_degradations.append({
            'step': 'coverage_to_focused_subset',
            'from_query_type': 'coverage',
            'to_query_type': 'focused',
            'from_top_k': old_top_k,
            'to_top_k': effective_top_k,
            'reason': 'preflight_ratio_critical',
        })

    projected_seconds, ratio = _estimate_budget_ratio(
        profile_name=profile_name,
        query_type=effective_query_type,
        timeout_seconds=effective_timeout_seconds,
        question_length=question_length,
        context_chunks=effective_top_k,
        context_chars=0,
        top_k=effective_top_k,
        reasoning_enabled=effective_reasoning_enabled,
        max_tokens=effective_max_tokens,
    )

    return (
        effective_query_type,
        effective_top_k,
        effective_reasoning_enabled,
        effective_max_tokens,
        effective_timeout_seconds,
        effective_output_constraints,
        applied_degradations,
        projected_seconds,
        ratio,
    )


def _apply_post_retrieval_budget_degradations(
    *,
    fit_to_budget_enabled: bool,
    policy_soft_top_k_threshold: float,
    policy_soft_coverage_to_focused_threshold: float,
    profile_name: str,
    question_length: int,
    query_type: str,
    timeout_seconds: int,
    top_k: int,
    reasoning_enabled: bool,
    max_tokens: int,
    chunks: list[dict],
    subtype: str | None,
    focused_max_tokens: int,
    focused_timeout_seconds: int,
    applied_degradations: list[dict[str, object]],
    route_candidate: str | None = None,
    min_output_budget_floor: int | None = None,
) -> tuple[list[dict], str, int, bool, int, int, int, list[dict[str, object]], float, float]:
    effective_chunks = list(chunks)
    effective_query_type = query_type
    effective_top_k = top_k
    effective_reasoning_enabled = reasoning_enabled
    effective_max_tokens = max_tokens
    effective_timeout_seconds = timeout_seconds
    coverage_route_topk_guard_enabled = (
        effective_query_type == 'coverage'
        and str(route_candidate or '').strip() in _COVERAGE_ROUTE_TOPK_GUARD_IDS
    )
    context_chars = sum(len(str(chunk.get('chunk_text', ''))) for chunk in effective_chunks)

    projected_seconds, ratio = _estimate_budget_ratio(
        profile_name=profile_name,
        query_type=effective_query_type,
        timeout_seconds=effective_timeout_seconds,
        question_length=question_length,
        context_chunks=len(effective_chunks),
        context_chars=context_chars,
        top_k=effective_top_k,
        reasoning_enabled=effective_reasoning_enabled,
        max_tokens=effective_max_tokens,
    )

    diagnostics_depth_mode = isinstance(min_output_budget_floor, int) and min_output_budget_floor >= 800
    if diagnostics_depth_mode:
        applied_degradations.append({
            'step': 'diagnostics_depth_constraints_applied',
            'min_output_budget_floor': int(min_output_budget_floor),
            'reason': 'diagnostics_output_shape_min_words',
        })

    if (
        fit_to_budget_enabled
        and ratio >= policy_soft_top_k_threshold
        and len(effective_chunks) > 4
        and not diagnostics_depth_mode
        and not coverage_route_topk_guard_enabled
    ):
        old_chunk_count = len(effective_chunks)
        reduced_count = max(4, int(old_chunk_count * 0.6))
        if reduced_count < old_chunk_count:
            effective_chunks = effective_chunks[:reduced_count]
            effective_top_k = min(effective_top_k, reduced_count)
            context_chars = sum(len(str(chunk.get('chunk_text', ''))) for chunk in effective_chunks)
            applied_degradations.append({
                'step': 'reduce_context_chunks',
                'from': old_chunk_count,
                'to': reduced_count,
                'reason': 'post_retrieval_ratio_high',
            })

    if fit_to_budget_enabled and effective_query_type == 'focused' and context_chars > _FOCUSED_POST_RETRIEVAL_MAX_CONTEXT_CHARS:
        old_context_chars = context_chars
        remaining_char_budget = _FOCUSED_POST_RETRIEVAL_MAX_CONTEXT_CHARS
        trimmed_chunks: list[dict] = []
        for chunk in effective_chunks:
            if remaining_char_budget <= 0:
                break
            chunk_text = str(chunk.get('chunk_text', '') or '')
            if not chunk_text:
                continue
            allowed_chars = min(_FOCUSED_POST_RETRIEVAL_MAX_CHUNK_CHARS, remaining_char_budget)
            trimmed_text = chunk_text[:allowed_chars]
            updated_chunk = dict(chunk)
            updated_chunk['chunk_text'] = trimmed_text
            trimmed_chunks.append(updated_chunk)
            remaining_char_budget -= len(trimmed_text)
        if trimmed_chunks:
            effective_chunks = trimmed_chunks
            effective_top_k = min(effective_top_k, len(effective_chunks))
            context_chars = sum(len(str(chunk.get('chunk_text', ''))) for chunk in effective_chunks)
            applied_degradations.append({
                'step': 'focused_context_chars_cap',
                'from_total_chars': old_context_chars,
                'to_total_chars': context_chars,
                'max_total_chars': _FOCUSED_POST_RETRIEVAL_MAX_CONTEXT_CHARS,
                'max_chunk_chars': _FOCUSED_POST_RETRIEVAL_MAX_CHUNK_CHARS,
                'reason': 'focused_first_token_latency_guard',
            })

    coverage_context_cap = _COVERAGE_PREFILL_MAX_CONTEXT_CHARS
    if coverage_route_topk_guard_enabled:
        coverage_context_cap = max(coverage_context_cap, 26000)
    if diagnostics_depth_mode:
        coverage_context_cap = max(coverage_context_cap, _DIAGNOSTICS_DEPTH_CONTEXT_CHARS_FLOOR)
    if effective_query_type == 'coverage' and context_chars > coverage_context_cap:
        old_context_chars = context_chars
        remaining_char_budget = coverage_context_cap
        trimmed_chunks: list[dict] = []
        for chunk in effective_chunks:
            if remaining_char_budget <= 0:
                break
            chunk_text = str(chunk.get('chunk_text', '') or '')
            if not chunk_text:
                continue
            allowed_chars = min(_COVERAGE_PREFILL_MAX_CHUNK_CHARS, remaining_char_budget)
            trimmed_text = chunk_text[:allowed_chars]
            updated_chunk = dict(chunk)
            updated_chunk['chunk_text'] = trimmed_text
            trimmed_chunks.append(updated_chunk)
            remaining_char_budget -= len(trimmed_text)
        if trimmed_chunks:
            effective_chunks = trimmed_chunks
            effective_top_k = min(effective_top_k, len(effective_chunks))
            context_chars = sum(len(str(chunk.get('chunk_text', ''))) for chunk in effective_chunks)
            applied_degradations.append({
                'step': 'coverage_prefill_context_chars_cap',
                'from_total_chars': old_context_chars,
                'to_total_chars': context_chars,
                'max_total_chars': coverage_context_cap,
                'max_chunk_chars': _COVERAGE_PREFILL_MAX_CHUNK_CHARS,
                'reason': 'coverage_first_token_latency_guard',
            })

    projected_seconds, ratio = _estimate_budget_ratio(
        profile_name=profile_name,
        query_type=effective_query_type,
        timeout_seconds=effective_timeout_seconds,
        question_length=question_length,
        context_chunks=len(effective_chunks),
        context_chars=context_chars,
        top_k=effective_top_k,
        reasoning_enabled=effective_reasoning_enabled,
        max_tokens=effective_max_tokens,
    )

    if (
        fit_to_budget_enabled
        and ratio >= policy_soft_coverage_to_focused_threshold
        and effective_query_type == 'coverage'
        and len(effective_chunks) > 8
        and subtype != 'aggregate_by_period'
        and not coverage_route_topk_guard_enabled
    ):
        old_chunk_count = len(effective_chunks)
        effective_chunks = effective_chunks[:8]
        effective_query_type = 'focused'
        effective_top_k = min(effective_top_k, 8)
        effective_max_tokens = min(effective_max_tokens, focused_max_tokens)
        effective_timeout_seconds = focused_timeout_seconds
        context_chars = sum(len(str(chunk.get('chunk_text', ''))) for chunk in effective_chunks)
        applied_degradations.append({
            'step': 'coverage_to_focused_subset',
            'from_query_type': 'coverage',
            'to_query_type': 'focused',
            'from_chunk_count': old_chunk_count,
            'to_chunk_count': len(effective_chunks),
            'reason': 'post_retrieval_ratio_critical',
        })

    projected_seconds, ratio = _estimate_budget_ratio(
        profile_name=profile_name,
        query_type=effective_query_type,
        timeout_seconds=effective_timeout_seconds,
        question_length=question_length,
        context_chunks=len(effective_chunks),
        context_chars=context_chars,
        top_k=effective_top_k,
        reasoning_enabled=effective_reasoning_enabled,
        max_tokens=effective_max_tokens,
    )

    return (
        effective_chunks,
        effective_query_type,
        effective_top_k,
        effective_reasoning_enabled,
        effective_max_tokens,
        effective_timeout_seconds,
        context_chars,
        applied_degradations,
        projected_seconds,
        ratio,
    )
