from informity.api.routes_scan import (
    SCAN_FILE_TIMEOUT_MAX_SECONDS,
    SCAN_FILE_TIMEOUT_MIN_SECONDS,
    _clamp_scan_file_timeout_seconds,
)


def test_clamp_scan_file_timeout_seconds_respects_min_bound() -> None:
    assert _clamp_scan_file_timeout_seconds(-5) == SCAN_FILE_TIMEOUT_MIN_SECONDS


def test_clamp_scan_file_timeout_seconds_keeps_in_range_values() -> None:
    assert _clamp_scan_file_timeout_seconds(123) == 123


def test_clamp_scan_file_timeout_seconds_respects_max_bound() -> None:
    assert _clamp_scan_file_timeout_seconds(9999) == SCAN_FILE_TIMEOUT_MAX_SECONDS
