from datetime import UTC, datetime
from pathlib import Path

from informity.api.routes_scan import (
    _PLAINTEXT_TIMEOUT_CAP_SECONDS,
    _resolve_scan_timeout_seconds_for_file,
)
from informity.scanner.crawler import ScannedFile


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
