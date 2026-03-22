# ==============================================================================
# Informity AI — RAG Generation Runtime Helpers
# Runtime budget degradations and strict-format shaping.
# ==============================================================================

import re

from informity.llm.model_adapter import get_profile_tokens_per_second


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
    return timeout_seconds, top_k, reasoning_enabled, max_tokens, applied_degradations


def _has_remaining_scope(
    *,
    timeout_reason: str | None,
    stream_recovery_reason: str | None,
    generation_skipped: bool,
    applied_degradations: list[dict[str, object]],
) -> bool:
    return bool(timeout_reason is not None or stream_recovery_reason is not None or generation_skipped)


def _should_apply_soft_stream_closeout(format_requirements: list[str]) -> bool:
    joined = ' '.join(str(item or '').casefold() for item in format_requirements)
    return 'required headings exactly' not in joined


def _augment_strict_ordered_format_requirements(format_requirements: list[str]) -> list[str]:
    return list(format_requirements or [])


def _apply_strict_ordered_output_budget(
    *,
    format_requirements: list[str],
    query_type: str,
    output_constraints: dict[str, int],
    max_tokens: int,
    reasoning_enabled: bool,
    strict_contract_complexity: bool = False,
) -> tuple[dict[str, int], int, bool, dict[str, object] | None]:
    # Compatibility shim: strict ordered budgeting removed from runtime.
    return dict(output_constraints), max_tokens, reasoning_enabled, None


def _estimate_tokens_per_second(profile_name: str) -> float:
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
    default_chars_per_chunk = 1200 if query_type == 'focused' else 950
    effective_context_chars = context_chars if context_chars > 0 else context_chunks * default_chars_per_chunk
    retrieval_seconds = 0.35 + (top_k * 0.06) + (0.5 if query_type == 'coverage' else 0.35)
    prompt_seconds = 0.25 + (effective_context_chars / 9000.0) + (min(question_length, 1500) / 2800.0)
    generation_seconds = float(max_tokens) / _estimate_tokens_per_second(profile_name)
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
    derive_format_requirements_fn,
    applied_degradations: list[dict[str, object]],
    min_output_budget_floor: int | None = None,
) -> tuple[list[str], dict[str, int], int, bool, list[dict], list[dict[str, object]]]:
    format_requirements = list(derive_format_requirements_fn(question) or [])
    constraints = dict(output_constraints or {})

    max_words_match = re.search(
        r'(?:<=?|at\s+most|max(?:imum)?|less than or equal to)\s*(\d+)\s*words?\b',
        question,
        flags=re.IGNORECASE,
    )
    if max_words_match:
        parsed_max_words = int(max_words_match.group(1))
        if parsed_max_words > 0:
            constraints['max_words'] = parsed_max_words

    exact_bullets_match = re.search(
        r'\bexactly\s+(\d+)\s+(?:numbered\s+)?(?:top-level\s+)?bullets?\b',
        question,
        flags=re.IGNORECASE,
    )
    if exact_bullets_match:
        parsed_bullets = int(exact_bullets_match.group(1))
        if parsed_bullets > 0:
            constraints['exact_top_level_bullets'] = parsed_bullets

    return format_requirements, constraints, max_tokens, reasoning_enabled, chunks, applied_degradations


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
) -> tuple[int, int, bool, int, list[dict[str, object]], bool]:
    return timeout_seconds, top_k, reasoning_enabled, max_tokens, applied_degradations, False


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
    strict_ordered_mode: bool = False,
) -> tuple[str, int, bool, int, int, dict[str, int], list[dict[str, object]], float, float]:
    projected_seconds, ratio = _estimate_budget_ratio(
        profile_name=profile_name,
        query_type=query_type,
        timeout_seconds=timeout_seconds,
        question_length=question_length,
        context_chunks=top_k,
        context_chars=0,
        top_k=top_k,
        reasoning_enabled=reasoning_enabled,
        max_tokens=max_tokens,
    )
    return (
        query_type,
        top_k,
        reasoning_enabled,
        max_tokens,
        timeout_seconds,
        {},
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
    context_chars = sum(len(str(chunk.get('chunk_text', ''))) for chunk in chunks)
    projected_seconds, ratio = _estimate_budget_ratio(
        profile_name=profile_name,
        query_type=query_type,
        timeout_seconds=timeout_seconds,
        question_length=question_length,
        context_chunks=len(chunks),
        context_chars=context_chars,
        top_k=top_k,
        reasoning_enabled=reasoning_enabled,
        max_tokens=max_tokens,
    )
    return (
        list(chunks),
        query_type,
        top_k,
        reasoning_enabled,
        max_tokens,
        timeout_seconds,
        context_chars,
        applied_degradations,
        projected_seconds,
        ratio,
    )
