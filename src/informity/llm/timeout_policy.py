# ==============================================================================
# Informity AI — Timeout Policy
# Shared timeout normalization and terminal-timeout semantics.
# ==============================================================================

"""Module for llm timeout policy."""

from __future__ import annotations

from informity.llm.types import TimeoutReason

_TERMINAL_TIMEOUT_REASONS = {
    TimeoutReason.QUEUE_WAIT_TIMEOUT,
    TimeoutReason.FIRST_TOKEN_WATCHDOG_TIMEOUT,
}


def normalize_timeout_reason(raw_reason: object) -> TimeoutReason | str:
    """Normalize timeout reason."""
    normalized = str(raw_reason or TimeoutReason.UNKNOWN_TIMEOUT.value).strip().lower()
    try:
        return TimeoutReason(normalized)
    except ValueError:
        return normalized


def is_terminal_timeout_reason(reason: TimeoutReason | str | None) -> bool:
    """Is terminal timeout reason."""
    normalized = str(reason or "").strip().lower()
    return normalized in {timeout_reason.value for timeout_reason in _TERMINAL_TIMEOUT_REASONS}
