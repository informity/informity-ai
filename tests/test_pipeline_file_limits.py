from datetime import UTC, datetime
from pathlib import Path

import pytest

from informity.indexer.pipeline import index_file
from informity.scanner.crawler import ScannedFile


class _ExplodingExtractor:
    supported_extensions = ['.pdf', '.txt']

    def can_handle(self, path: Path) -> bool:
        return True

    def extract(self, path: Path):  # type: ignore[no-untyped-def]
        raise AssertionError('extract() should not be called for oversized scanned files')


@pytest.mark.asyncio
async def test_index_file_scanned_file_too_large_skips_extraction(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr('informity.indexer.pipeline.get_max_file_size_bytes', lambda: 1024)
    scanned = ScannedFile(
        path=Path('/tmp/huge.pdf'),
        filename='huge.pdf',
        extension='.pdf',
        size_bytes=20 * 1024,
        content_hash='deadbeef',
        modified_at=datetime.now(UTC),
    )

    result = await index_file(
        db=object(),  # not used because function exits before DB access
        file_path_or_scanned=scanned,
        extractor=_ExplodingExtractor(),
    )

    assert result.success is False
    assert result.error_code == 'file_too_large'
    assert result.retryable is False


@pytest.mark.asyncio
async def test_index_file_pathological_text_line_skips_extraction(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    text_path = tmp_path / 'pathological.txt'
    text_path.write_text(('A' * 250_000) + '\nsmall\n', encoding='utf-8')
    monkeypatch.setattr('informity.indexer.pipeline.get_max_file_size_bytes', lambda: 1024 * 1024 * 100)

    result = await index_file(
        db=object(),  # not used because function exits before DB access
        file_path_or_scanned=text_path,
        extractor=_ExplodingExtractor(),
    )

    assert result.success is False
    assert result.error_code == 'text_pathological_line_length'
    assert result.retryable is False
