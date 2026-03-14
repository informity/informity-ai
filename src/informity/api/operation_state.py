# ==============================================================================
# Informity AI — Operation State
# Module-level flags for long-running background operations (reset, etc.)
# shared across route modules. Used to prevent concurrent conflicting operations.
# Also contains shared helpers for scan/rebuild lifecycle management.
# ==============================================================================

import asyncio
from datetime import UTC, datetime

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
