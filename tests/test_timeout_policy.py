from __future__ import annotations

from informity.llm.timeout_policy import is_terminal_timeout_reason, normalize_timeout_reason
from informity.llm.types import TimeoutReason


def test_normalize_timeout_reason_maps_enum_values() -> None:
    reason = normalize_timeout_reason('queue_wait_timeout')
    assert reason == TimeoutReason.QUEUE_WAIT_TIMEOUT


def test_normalize_timeout_reason_preserves_unknown_reason_string() -> None:
    reason = normalize_timeout_reason('some_new_timeout')
    assert reason == 'some_new_timeout'


def test_is_terminal_timeout_reason_for_queue_wait() -> None:
    assert is_terminal_timeout_reason(TimeoutReason.QUEUE_WAIT_TIMEOUT) is True
    assert is_terminal_timeout_reason('first_token_watchdog_timeout') is True
    assert is_terminal_timeout_reason('wall_clock_limit') is False
