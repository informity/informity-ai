import asyncio
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

import informity.api.operation_state as op_state
from informity.db.models import ScanRecord, ScanStatus


@pytest.mark.asyncio
async def test_try_begin_reset_is_atomic() -> None:
    await op_state.finish_reset(result=None)

    async def _attempt_begin() -> bool:
        return await op_state.try_begin_reset()

    first, second = await asyncio.gather(_attempt_begin(), _attempt_begin())

    assert [first, second].count(True) == 1
    assert [first, second].count(False) == 1

    await op_state.finish_reset(result=None)


@pytest.mark.asyncio
async def test_resolve_running_scan_blocks_when_recent_without_force() -> None:
    running = ScanRecord(
        id=7,
        started_at=datetime.now(UTC) - timedelta(seconds=5),
        status=ScanStatus.RUNNING,
        files_scanned=2,
        files_indexed=1,
        errors=0,
    )

    with (
        patch.object(op_state, 'get_latest_scan', new=AsyncMock(return_value=running)),
        patch.object(op_state, 'update_scan_record', new=AsyncMock()) as mock_update,
        pytest.raises(HTTPException) as exc_info,
    ):
        await op_state.resolve_running_scan(
            db=MagicMock(),
            force=False,
            operation='scan',
        )

    assert exc_info.value.status_code == 409
    mock_update.assert_not_called()


@pytest.mark.asyncio
async def test_resolve_running_scan_force_cancels_running_scan() -> None:
    running = ScanRecord(
        id=9,
        started_at=datetime.now(UTC) - timedelta(seconds=2),
        status=ScanStatus.RUNNING,
        files_scanned=3,
        files_indexed=1,
        errors=0,
    )

    with patch.object(op_state, 'get_latest_scan', new=AsyncMock(return_value=running)), \
         patch.object(op_state, 'update_scan_record', new=AsyncMock()) as mock_update, \
         patch.object(op_state, 'request_scan_cancel', new=AsyncMock()) as mock_request_cancel:
        await op_state.resolve_running_scan(
            db=MagicMock(),
            force=True,
            operation='rebuild',
        )

    mock_update.assert_called_once()
    mock_request_cancel.assert_awaited_once_with(running.id)
    updated_record = mock_update.await_args.args[1]
    assert updated_record.status == ScanStatus.CANCELLED
    assert updated_record.id == running.id


@pytest.mark.asyncio
async def test_scan_cancel_request_lifecycle() -> None:
    scan_id = 1234
    await op_state.clear_scan_cancel(scan_id)
    assert await op_state.is_scan_cancel_requested(scan_id) is False

    await op_state.request_scan_cancel(scan_id)
    assert await op_state.is_scan_cancel_requested(scan_id) is True

    await op_state.clear_scan_cancel(scan_id)
    assert await op_state.is_scan_cancel_requested(scan_id) is False
