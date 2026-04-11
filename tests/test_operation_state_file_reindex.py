import pytest

import informity.api.operation_state as op_state


@pytest.mark.asyncio
async def test_file_reindex_operation_dedupes_by_file_id() -> None:
    async with op_state._FILE_REINDEX_STATE_LOCK:
        op_state._FILE_REINDEX_OPERATIONS.clear()
        op_state._FILE_REINDEX_RUNNING_BY_FILE_ID.clear()

    first, first_is_new = await op_state.begin_file_reindex_operation(file_id=17, filename='a.txt')
    second, second_is_new = await op_state.begin_file_reindex_operation(file_id=17, filename='a.txt')

    assert first_is_new is True
    assert second_is_new is False
    assert first['operation_id'] == second['operation_id']
    assert first['status'] == 'running'
    assert await op_state.get_running_file_reindex_count() == 1


@pytest.mark.asyncio
async def test_file_reindex_operation_transitions_to_completed() -> None:
    async with op_state._FILE_REINDEX_STATE_LOCK:
        op_state._FILE_REINDEX_OPERATIONS.clear()
        op_state._FILE_REINDEX_RUNNING_BY_FILE_ID.clear()

    operation, _ = await op_state.begin_file_reindex_operation(file_id=99, filename='b.txt')
    completed = await op_state.complete_file_reindex_operation(
        operation['operation_id'],
        status='completed',
        chunks_created=42,
    )

    assert completed is not None
    assert completed['status'] == 'completed'
    assert completed['chunks_created'] == 42
    assert completed['completed_at'] is not None
    assert await op_state.get_running_file_reindex_count() == 0

    fetched = await op_state.get_file_reindex_operation(operation['operation_id'])
    assert fetched is not None
    assert fetched['status'] == 'completed'
    assert fetched['chunks_created'] == 42

