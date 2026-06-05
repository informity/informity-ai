from __future__ import annotations

from types import SimpleNamespace

import pytest

from informity.api import routes_scan
from informity.translate_policy import TRANSLATE_PROVIDER
from informity.upload_policy import UPLOAD_PROVIDER
from informity.utils.path_utils import normalize_path


class _DummyDB:
    pass


@pytest.mark.asyncio
async def test_list_files_excludes_upload_and_translate_local(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, object]] = []

    async def _fake_get_files(db, **kwargs):
        _ = db
        calls.append(kwargs)
        return [], 0

    monkeypatch.setattr(routes_scan, 'get_files', _fake_get_files)

    response = await routes_scan.list_files(
        category=None,
        extension=None,
        search=None,
        tag=None,
        sort='indexed_at',
        order='desc',
        offset=0,
        limit=50,
        db=_DummyDB(),
    )

    assert response.total == 0
    assert calls == [
        {
            'category': None,
            'extensions': None,
            'search': None,
            'tag': None,
            'excluded_source_providers': [UPLOAD_PROVIDER, TRANSLATE_PROVIDER],
            'sort_by': 'indexed_at',
            'order': 'desc',
            'offset': 0,
            'limit': 50,
        }
    ]


@pytest.mark.asyncio
async def test_persist_file_result_clears_failure_on_success(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, dict[str, object]]] = []

    async def _fake_clear_file_failure(db, **kwargs):
        _ = db
        calls.append(('clear', kwargs))

    async def _fake_record_file_failure(db, **kwargs):
        _ = db
        calls.append(('record', kwargs))

    monkeypatch.setattr(routes_scan, 'clear_file_failure', _fake_clear_file_failure)
    monkeypatch.setattr(routes_scan, 'record_file_failure', _fake_record_file_failure)

    result = SimpleNamespace(success=True, error_code='x', error='bad', retryable=False)
    scanned = SimpleNamespace(path='/tmp/example.pdf', content_hash='hash-1')
    normalized_path = str(normalize_path(scanned.path, expand_user=False))

    await routes_scan._persist_file_result(
        _DummyDB(),
        result,
        scanned=scanned,
        source_provider='filesystem',
        entity_type='file',
    )

    assert calls == [
        (
            'clear',
                {
                    'source_provider': 'filesystem',
                    'entity_type': 'file',
                    'source_item_id': normalized_path,
                },
            )
        ]


@pytest.mark.asyncio
async def test_persist_file_result_records_failure_on_error(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, dict[str, object]]] = []

    async def _fake_clear_file_failure(db, **kwargs):
        _ = db
        calls.append(('clear', kwargs))

    async def _fake_record_file_failure(db, **kwargs):
        _ = db
        calls.append(('record', kwargs))

    monkeypatch.setattr(routes_scan, 'clear_file_failure', _fake_clear_file_failure)
    monkeypatch.setattr(routes_scan, 'record_file_failure', _fake_record_file_failure)

    result = SimpleNamespace(success=False, error_code='pdf_invalid_or_corrupt', error='bad pdf', retryable=False)
    scanned = SimpleNamespace(path='/tmp/example.pdf', content_hash='hash-1')
    normalized_path = str(normalize_path(scanned.path, expand_user=False))

    await routes_scan._persist_file_result(
        _DummyDB(),
        result,
        scanned=scanned,
        source_provider='filesystem',
        entity_type='file',
    )

    assert calls == [
        (
            'record',
                {
                    'source_provider': 'filesystem',
                    'entity_type': 'file',
                    'source_item_id': normalized_path,
                    'path': normalized_path,
                    'content_hash': 'hash-1',
                    'error_code': 'pdf_invalid_or_corrupt',
                    'error_message': 'bad pdf',
                'retryable': False,
            },
        )
    ]
