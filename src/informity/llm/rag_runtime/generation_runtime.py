# ==============================================================================
# Informity AI — RAG Generation Runtime Helpers
# Runtime budget degradations and strict-format shaping.
# ==============================================================================

import re

from informity.llm.types import TimeoutReason


def _has_remaining_scope(
    *,
    timeout_reason: TimeoutReason | str | None,
    stream_recovery_reason: str | None,
    generation_skipped: bool,
    applied_degradations: list[dict[str, object]],
) -> bool:
    terminal_timeout_reasons = {TimeoutReason.QUEUE_WAIT_TIMEOUT, TimeoutReason.FIRST_TOKEN_WATCHDOG_TIMEOUT}
    normalized_timeout_reason = str(timeout_reason or '').strip().lower()
    if normalized_timeout_reason in {reason.value for reason in terminal_timeout_reasons}:
        return False
    return bool(timeout_reason is not None or stream_recovery_reason is not None or generation_skipped)


def _should_apply_soft_stream_closeout(format_requirements: list[str]) -> bool:
    joined = ' '.join(str(item or '').casefold() for item in format_requirements)
    return 'required headings exactly' not in joined


def _apply_strict_format_prompt_controls(
    *,
    question: str,
    chunks: list[dict],
    output_constraints: dict[str, int],
    max_tokens: int,
    reasoning_enabled: bool,
    derive_format_requirements_fn,
    action_hints: dict[str, bool] | None,
    applied_degradations: list[dict[str, object]],
) -> tuple[list[str], dict[str, int], int, bool, list[dict], list[dict[str, object]]]:
    format_requirements = list(derive_format_requirements_fn(question, action_hints) or [])
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
