from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import informity.api.operation_state as op_state
import informity.api.routes_scan as routes_scan


@pytest.mark.asyncio
async def test_file_reindex_task_marks_failed_on_unhandled_exception(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    async with op_state._FILE_REINDEX_STATE_LOCK:
        op_state._FILE_REINDEX_OPERATIONS.clear()
        op_state._FILE_REINDEX_RUNNING_BY_FILE_ID.clear()

    operation, _ = await op_state.begin_file_reindex_operation(file_id=207, filename='large.txt')

    class _DummyDb:
        async def close(self) -> None:
            return None

    async def _fake_get_connection() -> _DummyDb:
        return _DummyDb()

    monkeypatch.setattr('informity.db.sqlite.get_connection', _fake_get_connection)

    file_path = tmp_path / 'large.txt'
    file_path.write_text('x')
    file_row = SimpleNamespace(
        path=str(file_path),
        filename='large.txt',
        source_provider='filesystem',
        entity_type='file',
        source_item_id='src-item-1',
    )

    monkeypatch.setattr(routes_scan, 'get_file_by_id', AsyncMock(return_value=file_row))
    monkeypatch.setattr(routes_scan, 'scanned_file_for_path', lambda _path: SimpleNamespace(size_bytes=1))
    monkeypatch.setattr(routes_scan, 'clear_file_failure', AsyncMock())
    monkeypatch.setattr(routes_scan, 'rebuild_term_dictionary', AsyncMock())

    async def _boom(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError('boom')

    monkeypatch.setattr(routes_scan, 'reindex_file', _boom)

    await routes_scan._run_file_reindex_task(
        operation_id=operation['operation_id'],
        file_id=207,
    )

    updated = await op_state.get_file_reindex_operation(operation['operation_id'])
    assert updated is not None
    assert updated['status'] == 'failed'
    assert updated['error'] is not None
    assert 'boom' in updated['error']
    assert await op_state.get_running_file_reindex_count() == 0
