from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

from informity.api.routes_scan import open_file
from informity.api.schemas import OpenFileRequest
from informity.db.models import FileCategory, IndexedFile


@pytest.mark.asyncio
async def test_open_file_rejects_non_indexed_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    test_file = tmp_path / 'note.txt'
    test_file.write_text('hello', encoding='utf-8')

    monkeypatch.setattr('informity.api.routes_scan.get_file_by_path', AsyncMock(return_value=None))

    with pytest.raises(HTTPException) as exc_info:
        await open_file(OpenFileRequest(path=str(test_file)), db=MagicMock())

    assert exc_info.value.status_code == 403


@pytest.mark.asyncio
async def test_open_file_allows_indexed_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    test_file = tmp_path / 'note.txt'
    test_file.write_text('hello', encoding='utf-8')

    indexed = IndexedFile(
        id=1,
        path=str(test_file.resolve()),
        filename='note.txt',
        extension='.txt',
        size_bytes=5,
        content_hash='hash',
        extracted_text_preview='hello',
        category=FileCategory.PLAINTEXT,
        modified_at=datetime.now(UTC),
    )
    monkeypatch.setattr('informity.api.routes_scan.get_file_by_path', AsyncMock(return_value=indexed))
    run_mock = MagicMock()
    monkeypatch.setattr('informity.api.routes_scan.subprocess.run', run_mock)

    result = await open_file(OpenFileRequest(path=str(test_file.resolve())), db=MagicMock())
    assert result['opened'] is True
    run_mock.assert_called_once()
