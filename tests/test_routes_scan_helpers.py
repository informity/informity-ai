from __future__ import annotations

from types import SimpleNamespace

import pytest

from informity.api import routes_scan
from informity.utils.path_utils import normalize_path


class _DummyDB:
    pass


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
