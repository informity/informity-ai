# ==============================================================================
# Informity AI — Operation State
# Module-level flags for long-running background operations (reset, etc.)
# shared across route modules. Used to prevent concurrent conflicting operations.
# Also contains shared helpers for scan/rebuild lifecycle management.
# ==============================================================================

import asyncio
from collections import OrderedDict
from datetime import UTC, datetime
from typing import Literal
from uuid import uuid4

import aiosqlite
import structlog
from fastapi import HTTPException

from informity.config import settings
from informity.db.models import ScanRecord, ScanStatus
from informity.db.sqlite import get_latest_scan, update_scan_record

# ==============================================================================
# Logger
# ==============================================================================

log = structlog.get_logger(__name__)

# ==============================================================================
# Module-level flags
# ==============================================================================

# Reset in progress: set at start of _run_reset_task, cleared when done.
reset_in_progress: bool = False

# Last reset result, set when reset completes. Cleared at start of next reset.
last_reset_result: dict | None = None

# Minimum stale scan threshold floor in seconds.
_MIN_STALE_SCAN_THRESHOLD_SECONDS: int = 30

# Global lock for mutable reset state transitions.
_RESET_STATE_LOCK = asyncio.Lock()

# Global lock for scan/rebuild/reset conflict checks and scan record creation.
_SCAN_OPERATION_LOCK = asyncio.Lock()

# Global lock for any index-mutating file ingestion work (scan/rebuild/watcher).
# This prevents races between "compare with DB" and subsequent inserts/updates.
_INGESTION_LOCK = asyncio.Lock()
_SCAN_CANCEL_LOCK = asyncio.Lock()
_SCAN_CANCEL_REQUESTS: set[int] = set()
_FILE_REINDEX_STATE_LOCK = asyncio.Lock()
_FILE_REINDEX_MAX_HISTORY = 256

FileReindexStatus = Literal['running', 'completed', 'failed']


class FileReindexOperation(dict):
    # Typed dict-like payload for per-file reindex operation status.
    operation_id: str
    operation_type: str
    file_id: int
    filename: str
    status: FileReindexStatus
    started_at: str
    completed_at: str | None
    error: str | None
    chunks_created: int | None


_FILE_REINDEX_OPERATIONS: 'OrderedDict[str, FileReindexOperation]' = OrderedDict()
_FILE_REINDEX_RUNNING_BY_FILE_ID: dict[int, str] = {}


async def is_reset_in_progress() -> bool:
    # Return whether reset is currently running.
    async with _RESET_STATE_LOCK:
        return reset_in_progress


async def get_reset_state_snapshot() -> tuple[bool, dict | None]:
    # Return a coherent snapshot of reset flags.
    async with _RESET_STATE_LOCK:
        return reset_in_progress, last_reset_result


async def try_begin_reset() -> bool:
    # Atomically set reset_in_progress=True if not already set.
    # Returns True when this call started the reset, False otherwise.
    global reset_in_progress, last_reset_result
    async with _RESET_STATE_LOCK:
        if reset_in_progress:
            return False
        reset_in_progress = True
        last_reset_result = None
        return True


async def finish_reset(result: dict | None) -> None:
    # Atomically publish reset result and clear in-progress flag.
    global reset_in_progress, last_reset_result
    async with _RESET_STATE_LOCK:
        last_reset_result = result
        reset_in_progress = False


def get_scan_operation_lock() -> asyncio.Lock:
    # Shared lock for scan/rebuild/reset conflict resolution.
    return _SCAN_OPERATION_LOCK


def get_ingestion_lock() -> asyncio.Lock:
    # Shared lock for all index-mutating ingestion operations.
    return _INGESTION_LOCK


async def request_scan_cancel(scan_id: int) -> None:
    # Mark a running scan as cancellation-requested.
    async with _SCAN_CANCEL_LOCK:
        _SCAN_CANCEL_REQUESTS.add(int(scan_id))


async def is_scan_cancel_requested(scan_id: int) -> bool:
    # Check whether cancellation was requested for this scan id.
    async with _SCAN_CANCEL_LOCK:
        return int(scan_id) in _SCAN_CANCEL_REQUESTS


async def clear_scan_cancel(scan_id: int) -> None:
    # Clear cancellation request for this scan id.
    async with _SCAN_CANCEL_LOCK:
        _SCAN_CANCEL_REQUESTS.discard(int(scan_id))


async def begin_file_reindex_operation(*, file_id: int, filename: str) -> tuple[FileReindexOperation, bool]:
    # Begin a reindex operation or return existing running operation for this file.
    now_iso = datetime.now(UTC).isoformat()
    async with _FILE_REINDEX_STATE_LOCK:
        existing_operation_id = _FILE_REINDEX_RUNNING_BY_FILE_ID.get(int(file_id))
        if existing_operation_id:
            existing = _FILE_REINDEX_OPERATIONS.get(existing_operation_id)
            if existing is not None:
                return existing, False
            _FILE_REINDEX_RUNNING_BY_FILE_ID.pop(int(file_id), None)

        operation_id = f'file-reindex-{int(file_id)}-{uuid4().hex[:10]}'
        operation: FileReindexOperation = {
            'operation_id': operation_id,
            'operation_type': 'file_reindex',
            'file_id': int(file_id),
            'filename': filename,
            'status': 'running',
            'started_at': now_iso,
            'completed_at': None,
            'error': None,
            'chunks_created': None,
        }
        _FILE_REINDEX_OPERATIONS[operation_id] = operation
        _FILE_REINDEX_RUNNING_BY_FILE_ID[int(file_id)] = operation_id
        _prune_file_reindex_history_locked()
        return operation, True


async def complete_file_reindex_operation(
    operation_id: str,
    *,
    status: FileReindexStatus,
    error: str | None = None,
    chunks_created: int | None = None,
) -> FileReindexOperation | None:
    # Mark file reindex operation as terminal.
    async with _FILE_REINDEX_STATE_LOCK:
        op = _FILE_REINDEX_OPERATIONS.get(operation_id)
        if op is None:
            return None
        op['status'] = status
        op['completed_at'] = datetime.now(UTC).isoformat()
        op['error'] = error
        op['chunks_created'] = chunks_created
        if _FILE_REINDEX_RUNNING_BY_FILE_ID.get(op['file_id']) == operation_id:
            _FILE_REINDEX_RUNNING_BY_FILE_ID.pop(op['file_id'], None)
        return op


async def get_file_reindex_operation(operation_id: str) -> FileReindexOperation | None:
    # Get one file reindex operation by id.
    async with _FILE_REINDEX_STATE_LOCK:
        op = _FILE_REINDEX_OPERATIONS.get(operation_id)
        if op is None:
            return None
        return dict(op)


async def list_file_reindex_operations(*, status: FileReindexStatus | None = None) -> list[FileReindexOperation]:
    # List file reindex operations filtered by optional status.
    async with _FILE_REINDEX_STATE_LOCK:
        ops = list(_FILE_REINDEX_OPERATIONS.values())
        if status is not None:
            ops = [op for op in ops if op['status'] == status]
        # newest first
        return [dict(op) for op in reversed(ops)]


async def get_running_file_reindex_count() -> int:
    # Count currently running file reindex operations.
    async with _FILE_REINDEX_STATE_LOCK:
        return len(_FILE_REINDEX_RUNNING_BY_FILE_ID)


def _prune_file_reindex_history_locked() -> None:
    # Keep bounded in-memory history.
    if len(_FILE_REINDEX_OPERATIONS) <= _FILE_REINDEX_MAX_HISTORY:
        return
    overflow = len(_FILE_REINDEX_OPERATIONS) - _FILE_REINDEX_MAX_HISTORY
    removed = 0
    while removed < overflow and _FILE_REINDEX_OPERATIONS:
        op_id, op = _FILE_REINDEX_OPERATIONS.popitem(last=False)
        if op['status'] == 'running':
            # Never evict running operations.
            _FILE_REINDEX_OPERATIONS[op_id] = op
            break
        _FILE_REINDEX_RUNNING_BY_FILE_ID.pop(op['file_id'], None)
        removed += 1


# ==============================================================================
# Shared scan lifecycle helpers
# ==============================================================================

async def resolve_running_scan(
    db: aiosqlite.Connection,
    force: bool,
    operation: str = 'scan',
) -> None:
    # Check for a currently running scan and resolve it based on force flag
    # and age. Raises HTTPException(409) if a recent scan is running and
    # force=False.
    #
    # Args:
    #   db:        Active database connection.
    #   force:     When True, cancel a running scan regardless of age.
    #   operation: Label for log messages ('scan' or 'rebuild').
    latest = await get_latest_scan(db)
    if latest is None or latest.status != ScanStatus.RUNNING:
        return

    now = datetime.now(UTC)
    age_seconds = (now - latest.started_at).total_seconds()
    stale_threshold_seconds = max(
        _MIN_STALE_SCAN_THRESHOLD_SECONDS,
        int(settings.scan_stale_threshold_seconds),
    )

    if force:
        await request_scan_cancel(latest.id)
        log.info(
            'scan_force_canceling_running',
            operation=operation,
            scan_id=latest.id, age_seconds=age_seconds,
        )
        cancelled = ScanRecord(
            id            = latest.id,
            started_at    = latest.started_at,
            status        = ScanStatus.CANCELLED,
            files_scanned = latest.files_scanned,
            files_indexed = latest.files_indexed,
            errors        = latest.errors,
            completed_at  = now,
        )
        await update_scan_record(db, cancelled)
        log.info('running_scan_cancelled', operation=operation, scan_id=latest.id)
    elif age_seconds > stale_threshold_seconds:
        log.warning(
            'scan_stale_detected',
            operation=operation,
            scan_id=latest.id,
            age_seconds=age_seconds,
            stale_threshold_seconds=stale_threshold_seconds,
        )
        stale = ScanRecord(
            id            = latest.id,
            started_at    = latest.started_at,
            status        = ScanStatus.FAILED,
            files_scanned = latest.files_scanned,
            files_indexed = latest.files_indexed,
            errors        = latest.errors + 1,
            completed_at  = now,
        )
        await update_scan_record(db, stale)
        log.info('stale_scan_cleared', operation=operation, scan_id=latest.id)
    else:
        raise HTTPException(
            status_code=409,
            detail=(
                f'A {operation} is already running (started {int(age_seconds)}s ago). '
                f'Use force=true to cancel it, or wait for it to complete.'
            ),
        )
