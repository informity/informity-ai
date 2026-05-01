from datetime import UTC, datetime
from pathlib import Path

from informity.api.routes_scan import (
    _PLAINTEXT_TIMEOUT_CAP_SECONDS,
    SCAN_FILE_TIMEOUT_MAX_SECONDS,
    SCAN_FILE_TIMEOUT_MIN_SECONDS,
    _clamp_scan_file_timeout_seconds,
    _resolve_scan_timeout_seconds_for_file,
)
from informity.scanner.crawler import ScannedFile


def test_clamp_scan_file_timeout_seconds_respects_min_bound() -> None:
    assert _clamp_scan_file_timeout_seconds(-5) == SCAN_FILE_TIMEOUT_MIN_SECONDS
    assert _clamp_scan_file_timeout_seconds(0) == SCAN_FILE_TIMEOUT_MIN_SECONDS


def test_clamp_scan_file_timeout_seconds_keeps_in_range_values() -> None:
    assert _clamp_scan_file_timeout_seconds(123) == 123


def test_clamp_scan_file_timeout_seconds_respects_max_bound() -> None:
    assert _clamp_scan_file_timeout_seconds(9999) == SCAN_FILE_TIMEOUT_MAX_SECONDS


def test_resolve_scan_timeout_seconds_caps_plaintext(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr('informity.api.routes_scan.resolve_timeout_seconds', lambda *args, **kwargs: 600)
    sf = ScannedFile(
        path=Path('/tmp/example.txt'),
        filename='example.txt',
        extension='.txt',
        size_bytes=1234,
        content_hash='abc',
        modified_at=datetime.now(UTC),
    )
    assert _resolve_scan_timeout_seconds_for_file(sf) == _PLAINTEXT_TIMEOUT_CAP_SECONDS


def test_resolve_scan_timeout_seconds_non_plaintext_unchanged(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr('informity.api.routes_scan.resolve_timeout_seconds', lambda *args, **kwargs: 480)
    sf = ScannedFile(
        path=Path('/tmp/example.pdf'),
        filename='example.pdf',
        extension='.pdf',
        size_bytes=1234,
        content_hash='abc',
        modified_at=datetime.now(UTC),
    )
    assert _resolve_scan_timeout_seconds_for_file(sf) == 480
