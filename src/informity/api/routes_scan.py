# ==============================================================================
# Informity AI — Scan API Routes
# Endpoints for triggering scans, checking scan status, and browsing files.
# The scan background task wires together: crawler -> indexing pipeline.
# ==============================================================================

import asyncio
import contextlib
import gc
import os
import subprocess
import sys
import uuid
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import aiosqlite
import structlog
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from structlog.contextvars import bind_contextvars, clear_contextvars

import informity.api.operation_state as op_state
from informity.api.operation_state import resolve_running_scan
from informity.api.schemas import (
    FileListResponse,
    OpenFileRequest,
    ScanErrorItem,
    ScanRequest,
    ScanStatusResponse,
)
from informity.api.security import EndpointGuard
from informity.config import (
    get_effective_ignore_patterns,
    get_supported_extensions_for_scan,
    settings,
)
from informity.db.models import ScanErrorRecord, ScanRecord, ScanStatus
from informity.db.sqlite import (
    clear_file_failure,
    get_all_files_for_scan,
    get_chunk_count_for_file,
    get_db,
    get_file_by_id,
    get_file_by_path,
    get_files,
    get_index_integrity_issues,
    get_latest_scan,
    get_scan_error_records,
    get_scan_timeout_error_count,
    insert_scan_error_record,
    insert_scan_record,
    record_file_failure,
    should_skip_file_retry,
    update_scan_record,
)
from informity.indexer.pipeline import (
    IndexResult,
    index_file,
    reindex_file,
    remove_file,
)
from informity.scanner.crawler import (
    ScannedFile,
    compare_with_db,
    scan_directories,
    scanned_file_for_path,
)
from informity.scanner.extractors.base import register_extractors
from informity.utils.path_utils import normalize_path

# ==============================================================================
# Logger
# ==============================================================================

log = structlog.get_logger(__name__)
_SCAN_RUNTIME_EXCEPTIONS = (aiosqlite.Error, RuntimeError, ValueError, TypeError, OSError, TimeoutError)
SCAN_CANCEL_POLL_INTERVAL_SECONDS = 0.25
SCAN_PROGRESS_DB_BUSY_TIMEOUT_MS = 250
SCAN_PROGRESS_UPDATE_TIMEOUT_SECONDS = 1.0
SCAN_TERMINAL_UPDATE_RETRIES = 3
SCAN_TERMINAL_UPDATE_RETRY_DELAY_SECONDS = 0.2


class _ScanCancelledInFlight(Exception):
    """Raised when a scan cancellation request arrives during file processing."""

# ==============================================================================
# Router
# ==============================================================================

router = APIRouter(tags=['scan'])
SCAN_GUARD = EndpointGuard(
    name='scan',
    max_in_flight=1,
    max_requests_per_window=12,
    window_seconds=60,
)
MAX_SCAN_DIRECTORIES = 256
MAX_PATH_CHARS = 4096
SCAN_FILE_TIMEOUT_MIN_SECONDS = 0
SCAN_FILE_TIMEOUT_MAX_SECONDS = 600


# ==============================================================================
# POST /api/scan — trigger a scan
# ==============================================================================


def _clamp_scan_file_timeout_seconds(timeout_seconds: int) -> int:
    return max(
        SCAN_FILE_TIMEOUT_MIN_SECONDS,
        min(timeout_seconds, SCAN_FILE_TIMEOUT_MAX_SECONDS),
    )

@router.post('/api/scan')
async def trigger_scan(
    request: ScanRequest,
    background_tasks: BackgroundTasks,
    db: aiosqlite.Connection = Depends(get_db),
) -> dict:
    async with SCAN_GUARD.slot():
        if request.directories and len(request.directories) > MAX_SCAN_DIRECTORIES:
            raise HTTPException(
                status_code=413,
                detail=f'Too many directories (max {MAX_SCAN_DIRECTORIES}).',
            )
        if request.directories and any(len(path) > MAX_PATH_CHARS for path in request.directories):
            raise HTTPException(
                status_code=413,
                detail=f'Directory path too long (max {MAX_PATH_CHARS} characters).',
            )
        # Block if reset is in progress
        if await op_state.is_reset_in_progress():
            raise HTTPException(
                status_code=409,
                detail='Index reset is in progress. Please wait for it to complete.',
            )

        # Serialize scan/rebuild/reset transition checks + scan-record creation.
        async with op_state.get_scan_operation_lock():
            # Check if a scan is already running and resolve it (cancel / mark stale / block)
            await resolve_running_scan(db, force=request.force, operation='scan')

            # Determine directories to scan (validate BEFORE creating scan record)
            directories: list[Path] = []
            if request.directories:
                directories = [Path(d) for d in request.directories]
            elif settings.watched_directories:
                directories = list(settings.watched_directories)

            if not directories:
                raise HTTPException(
                    status_code=400,
                    detail='No directories to scan. Please configure watched directories in Settings, or pass directories in the request.',
                )

            # Create a scan record (only after validation passes)
            scan_record = ScanRecord(started_at=datetime.now(UTC))
            scan_record = await insert_scan_record(db, scan_record)

        # Launch background task
        background_tasks.add_task(
            _run_scan_task,
            scan_id=scan_record.id,
            directories=directories,
            force=request.force,
        )

        return {
            'scan_id': scan_record.id,
            'status':  'started',
            'directories': [str(d) for d in directories],
        }


# ==============================================================================
# GET /api/scan/status — current scan status
# ==============================================================================

@router.get('/api/scan/status', response_model=ScanStatusResponse)
async def get_scan_status(
    db: aiosqlite.Connection = Depends(get_db),
) -> ScanStatusResponse:
    latest = await get_latest_scan(db)
    if latest is None:
        raise HTTPException(status_code=404, detail='No scan has been run yet')

    now     = datetime.now(UTC)
    elapsed = (latest.completed_at or now) - latest.started_at

    recent_errors = await get_scan_error_records(db, latest.id or 0, limit=8)
    return ScanStatusResponse(
        status=latest.status.value,
        files_scanned=latest.files_scanned,
        files_indexed=latest.files_indexed,
        errors=latest.errors,
        timeout_errors=await get_scan_timeout_error_count(db, latest.id or 0),
        recent_errors=[
            ScanErrorItem(
                path=item.path,
                filename=item.filename,
                extension=item.extension,
                operation=item.operation,
                error_code=item.error_code,
                error_message=item.error_message,
                is_timeout=item.is_timeout,
                created_at=item.created_at,
            )
            for item in recent_errors
        ],
        started_at=latest.started_at,
        elapsed_seconds=elapsed.total_seconds(),
    )


@router.post('/api/scan/cancel')
async def cancel_scan(
    db: aiosqlite.Connection = Depends(get_db),
) -> dict:
    # Request cooperative cancellation for the latest running scan.
    # Idempotent: returns no_active_scan when nothing is running.
    async with op_state.get_scan_operation_lock():
        latest = await get_latest_scan(db)
        if latest is None or latest.status != ScanStatus.RUNNING:
            return {'status': 'no_active_scan', 'cancel_requested': False}
        await op_state.request_scan_cancel(latest.id)
        log.info('scan_cancel_requested', scan_id=latest.id)
        return {'status': 'cancelling', 'scan_id': latest.id, 'cancel_requested': True}


# ==============================================================================
# GET /api/files — list indexed files
# ==============================================================================

@router.get('/api/files', response_model=FileListResponse)
async def list_files(
    category:   str | None       = Query(default=None),
    extension:  str | list[str] | None = Query(default=None),
    search:     str | None       = Query(default=None),
    tag:        str | None       = Query(default=None),
    sort:       str              = Query(default='indexed_at'),
    order:      str              = Query(default='desc'),
    offset:     int              = Query(default=0, ge=0),
    limit:      int              = Query(default=50, ge=1, le=200),
    db: aiosqlite.Connection     = Depends(get_db),
) -> FileListResponse:
    # Normalise extension to list for DB layer (single value from query becomes list of one).
    extensions = [extension] if isinstance(extension, str) else (extension or None)
    files, total = await get_files(
        db,
        category=category,
        extensions=extensions,
        search=search,
        tag=tag,
        sort_by=sort,
        order=order,
        offset=offset,
        limit=limit,
    )

    return FileListResponse(
        files=[f.model_dump(mode='json') for f in files],
        total=total,
        offset=offset,
        limit=limit,
    )


# ==============================================================================
# GET /api/files/{file_id} — single file detail
# ==============================================================================

@router.get('/api/files/{file_id}')
async def get_file_detail(
    file_id: int,
    db: aiosqlite.Connection = Depends(get_db),
) -> dict:
    file = await get_file_by_id(db, file_id)
    if file is None:
        raise HTTPException(status_code=404, detail='File not found')
    chunk_count = await get_chunk_count_for_file(db, file_id)
    result = file.model_dump(mode='json')
    result['chunk_count'] = chunk_count
    return result


# ==============================================================================
# POST /api/files/{file_id}/reindex — re-index a single file
# ==============================================================================

@router.post('/api/files/{file_id}/reindex')
async def reindex_single_file(
    file_id: int,
    db: aiosqlite.Connection = Depends(get_db),
) -> dict:
    if await op_state.is_reset_in_progress():
        raise HTTPException(
            status_code=409,
            detail='Index reset is in progress. Please wait for it to complete.',
        )

    file = await get_file_by_id(db, file_id)
    if file is None:
        raise HTTPException(status_code=404, detail='File not found')

    file_path = Path(file.path)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail='File not found on disk')

    try:
        scanned = scanned_file_for_path(file_path)
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f'Cannot read file: {exc}') from exc
    if scanned is None:
        raise HTTPException(status_code=500, detail='Cannot compute file hash for re-index')

    result = await reindex_file(db, scanned)
    if not result.success:
        raise HTTPException(status_code=500, detail=result.error or 'Re-index failed')

    return {
        'file_id':        file_id,
        'success':        True,
        'chunks_created': result.chunks_created,
    }


# ==============================================================================
# DELETE /api/files/{file_id} — remove file from index
# ==============================================================================

@router.delete('/api/files/{file_id}')
async def remove_single_file(
    file_id: int,
    db: aiosqlite.Connection = Depends(get_db),
) -> dict:
    if await op_state.is_reset_in_progress():
        raise HTTPException(
            status_code=409,
            detail='Index reset is in progress. Please wait for it to complete.',
        )

    file = await get_file_by_id(db, file_id)
    if file is None:
        raise HTTPException(status_code=404, detail='File not found')

    removed = await remove_file(db, file)
    if not removed:
        raise HTTPException(status_code=500, detail='Failed to remove file')

    return {
        'file_id':  file_id,
        'deleted':  True,
    }


# ==============================================================================
# POST /api/files/open — open file in system default application
# ==============================================================================

@router.post('/api/files/open')
async def open_file(
    request: OpenFileRequest,
    db: aiosqlite.Connection = Depends(get_db),
) -> dict:
    # Opens the file as if double-clicked in Finder (macOS) or equivalent.
    raw_path = request.path.strip()
    if not raw_path:
        raise HTTPException(status_code=400, detail='Path cannot be empty')
    if len(raw_path) > MAX_PATH_CHARS:
        raise HTTPException(status_code=413, detail=f'Path too long (max {MAX_PATH_CHARS} characters)')

    path = Path(raw_path)
    if not path.is_absolute():
        raise HTTPException(status_code=400, detail='Path must be absolute')
    path = normalize_path(path, expand_user=False)
    if not path.exists():
        raise HTTPException(status_code=404, detail='File not found')
    if not path.is_file():
        raise HTTPException(status_code=400, detail='Path is not a file')

    indexed = await get_file_by_path(db, str(path))
    if indexed is None:
        raise HTTPException(
            status_code=403,
            detail='Opening files is only allowed for indexed files.',
        )

    try:
        if sys.platform == 'darwin':
            subprocess.run(['open', str(path)], check=True)
        elif sys.platform == 'win32':
            os.startfile(str(path))
        else:
            subprocess.run(['xdg-open', str(path)], check=True)
    except (subprocess.CalledProcessError, OSError) as exc:
        log.warning('open_file_failed', path=str(path), error=str(exc))
        raise HTTPException(status_code=500, detail='Failed to open file') from exc

    return {'opened': True, 'path': str(path)}


# ==============================================================================
# Background Scan Task
# ==============================================================================

async def _run_scan_task(
    scan_id: int,
    directories: list[Path],
    force: bool = False,
) -> None:
    # Background task that runs the full scan + index pipeline:
    # 1. Crawl directories (find all files, compute hashes)
    # 2. Compare with DB (new, changed, unchanged, deleted)
    # Only new and changed files are indexed below; unchanged are skipped
    # unless force=True.
    # 3. Index new files (extract → chunk → embed → store)
    # 4. Re-index changed files
    # 5. Remove deleted files from DB + SQLite vector storage
    # 6. Update scan record with final stats

    # We need our own DB connection since this runs in a background task
    from informity.db.sqlite import get_connection

    clear_contextvars()
    scan_operation_id = f'scan-{scan_id}-{uuid.uuid4().hex[:8]}'
    bind_contextvars(
        operation_type='scan',
        operation_id=scan_operation_id,
        scan_id=scan_id,
    )

    db = await get_connection()
    progress_db = await get_connection()
    await progress_db.execute(f'PRAGMA busy_timeout={SCAN_PROGRESS_DB_BUSY_TIMEOUT_MS}')

    scan_started_at  = datetime.now(UTC)
    files_scanned    = 0
    files_indexed    = 0
    errors           = 0
    processed        = 0
    total_to_process = 0
    chunks_total_created = 0
    success_by_extension: dict[str, int] = defaultdict(int)
    errors_by_extension: dict[str, int] = defaultdict(int)
    chunks_by_extension: dict[str, int] = defaultdict(int)
    extractor_success_counts: dict[str, int] = defaultdict(int)
    extractor_error_counts: dict[str, int] = defaultdict(int)
    ocr_used_count = 0
    timeout_seconds_configured = int(settings.scan_file_timeout_seconds)
    timeout_seconds_effective = _clamp_scan_file_timeout_seconds(timeout_seconds_configured)

    async def _update_scan_record_best_effort(
        record: ScanRecord,
        *,
        context: str,
        terminal: bool = False,
    ) -> None:
        attempts = SCAN_TERMINAL_UPDATE_RETRIES if terminal else 1
        for attempt in range(1, attempts + 1):
            try:
                await asyncio.wait_for(
                    update_scan_record(progress_db, record),
                    timeout=SCAN_PROGRESS_UPDATE_TIMEOUT_SECONDS,
                )
                return
            except TimeoutError:
                log.warning(
                    'scan_record_update_timeout',
                    scan_id=scan_id,
                    context=context,
                    attempt=attempt,
                    terminal=terminal,
                    timeout_seconds=SCAN_PROGRESS_UPDATE_TIMEOUT_SECONDS,
                )
            except _SCAN_RUNTIME_EXCEPTIONS as exc:
                log.warning(
                    'scan_record_update_failed',
                    scan_id=scan_id,
                    context=context,
                    attempt=attempt,
                    terminal=terminal,
                    error=str(exc),
                )

            if attempt < attempts:
                await asyncio.sleep(SCAN_TERMINAL_UPDATE_RETRY_DELAY_SECONDS)

    async def _process_file(
        sf: 'ScannedFile',
        action: str,
        handler,
    ) -> IndexResult:
        # Shared logic: run handler, update counters, persist progress after each file
        # to keep scan record in sync with database state.
        # Wrapped in try/except to ensure one file failure doesn't stop the entire scan.
        nonlocal processed, files_indexed, errors, chunks_total_created, ocr_used_count
        processed += 1
        log.info(
            'scan_file_processing',
            operation = action,
            progress = f'{processed}/{total_to_process}',
            file     = sf.filename,
            path     = str(sf.path),
        )

        async def _run_handler_with_cancel_polling() -> IndexResult:
            # Poll for scan cancellation while a single file is being processed so
            # cancel requests don't wait for full file timeout windows.
            handler_task = asyncio.create_task(handler(db, sf))
            deadline: float | None = None
            if timeout_seconds_effective > 0:
                deadline = asyncio.get_running_loop().time() + float(timeout_seconds_effective)

            try:
                while True:
                    if handler_task.done():
                        return await handler_task

                    if await op_state.is_scan_cancel_requested(scan_id):
                        log.info(
                            'scan_cancel_requested_inflight',
                            scan_id=scan_id,
                            operation=action,
                            path=str(sf.path),
                        )
                        handler_task.cancel()
                        with contextlib.suppress(asyncio.CancelledError, Exception):
                            await asyncio.wait_for(handler_task, timeout=0.1)
                        raise _ScanCancelledInFlight()

                    wait_timeout = SCAN_CANCEL_POLL_INTERVAL_SECONDS
                    if deadline is not None:
                        remaining = deadline - asyncio.get_running_loop().time()
                        if remaining <= 0:
                            handler_task.cancel()
                            with contextlib.suppress(asyncio.CancelledError, Exception):
                                await asyncio.wait_for(handler_task, timeout=0.1)
                            raise TimeoutError()
                        wait_timeout = min(wait_timeout, remaining)

                    try:
                        await asyncio.wait_for(
                            asyncio.shield(handler_task),
                            timeout=wait_timeout,
                        )
                    except TimeoutError:
                        # Expected: loop heartbeat for cancel polling.
                        continue
            finally:
                if not handler_task.done():
                    handler_task.cancel()

        try:
            result = await _run_handler_with_cancel_polling()
            if result.success:
                files_indexed += 1
                chunks_total_created += result.chunks_created
                success_by_extension[sf.extension] += 1
                chunks_by_extension[sf.extension] += result.chunks_created
                if result.extractor:
                    extractor_success_counts[result.extractor] += 1
                if result.ocr_used:
                    ocr_used_count += 1
            else:
                errors += 1
                errors_by_extension[sf.extension] += 1
                if result.extractor:
                    extractor_error_counts[result.extractor] += 1
                await insert_scan_error_record(
                    db,
                    ScanErrorRecord(
                        scan_id=scan_id,
                        path=str(sf.path),
                        filename=sf.filename,
                        extension=sf.extension,
                        operation=action,
                        error_code=result.error_code,
                        error_message=result.error or 'File processing failed',
                        is_timeout=(result.error_code == 'scan_file_timeout'),
                    ),
                )
                log.warning(
                    'scan_file_processing_failed',
                    operation=action,
                    path=str(sf.path),
                    error=result.error,
                )
        except _ScanCancelledInFlight:
            raise
        except TimeoutError:
            errors += 1
            errors_by_extension[sf.extension] += 1
            timeout_message = (
                f'File processing exceeded timeout '
                f'({timeout_seconds_effective}s)'
            )
            await insert_scan_error_record(
                db,
                ScanErrorRecord(
                    scan_id=scan_id,
                    path=str(sf.path),
                    filename=sf.filename,
                    extension=sf.extension,
                    operation=action,
                    error_code='scan_file_timeout',
                    error_message=timeout_message,
                    is_timeout=True,
                ),
            )
            log.warning(
                'scan_file_processing_timeout',
                operation=action,
                path=str(sf.path),
                timeout_seconds=timeout_seconds_effective,
            )
            result = IndexResult(
                success=False,
                chunks_created=0,
                error=timeout_message,
                error_code='scan_file_timeout',
                retryable=True,
            )
        except _SCAN_RUNTIME_EXCEPTIONS as exc:
            # Catch exceptions from handler to prevent one file from stopping the scan
            errors += 1
            errors_by_extension[sf.extension] += 1
            await insert_scan_error_record(
                db,
                ScanErrorRecord(
                    scan_id=scan_id,
                    path=str(sf.path),
                    filename=sf.filename,
                    extension=sf.extension,
                    operation=action,
                    error_code='scan_processing_exception',
                    error_message=str(exc),
                    is_timeout=False,
                ),
            )
            log.error(
                'scan_file_processing_exception',
                operation=action,
                path=str(sf.path),
                error=str(exc),
                exc_info=True,
            )
            result = IndexResult(
                success=False,
                chunks_created=0,
                error=str(exc),
                error_code='scan_processing_exception',
                retryable=True,
            )
        except Exception as exc:
            # Last-resort guard: never let unexpected exception classes kill
            # the background scan task silently.
            errors += 1
            errors_by_extension[sf.extension] += 1
            await insert_scan_error_record(
                db,
                ScanErrorRecord(
                    scan_id=scan_id,
                    path=str(sf.path),
                    filename=sf.filename,
                    extension=sf.extension,
                    operation=action,
                    error_code='scan_unhandled_exception',
                    error_message=str(exc),
                    is_timeout=False,
                ),
            )
            log.error(
                'scan_file_processing_unhandled_exception',
                operation=action,
                path=str(sf.path),
                error=str(exc),
                exception_type=type(exc).__name__,
                exc_info=True,
            )
            result = IndexResult(
                success=False,
                chunks_created=0,
                error=str(exc),
                error_code='scan_unhandled_exception',
                retryable=True,
            )

        # Update scan record after each file to keep it in sync with database state
        # (files are inserted immediately, so scan record should reflect current progress)
        await _update_scan_record_best_effort(
            ScanRecord(
                id=scan_id,
                started_at=scan_started_at,
                files_scanned=files_scanned,
                files_indexed=files_indexed,
                errors=errors,
                status=ScanStatus.RUNNING,
            ),
            context='per_file_progress',
        )

        # Explicit garbage collection to free memory back to OS after each file
        # This helps prevent memory accumulation during long scans, especially for large documents
        gc.collect()
        return result

    async def _finalize_cancelled() -> None:
        # Persist terminal cancelled state with current progress.
        await _update_scan_record_best_effort(
            ScanRecord(
                id=scan_id,
                started_at=scan_started_at,
                files_scanned=files_scanned,
                files_indexed=files_indexed,
                errors=errors,
                status=ScanStatus.CANCELLED,
                completed_at=datetime.now(UTC),
            ),
            context='cancelled_terminal',
            terminal=True,
        )

    async def _cancel_requested(stage: str) -> bool:
        requested = await op_state.is_scan_cancel_requested(scan_id)
        if not requested:
            return False
        log.info('scan_cancelled', scan_id=scan_id, stage=stage)
        await _finalize_cancelled()
        return True

    try:
        # Ensure extractors are registered
        register_extractors()
        if await _cancel_requested('pre_crawl'):
            return

        # 1. Crawl (run in thread to avoid blocking event loop)
        # Use persisted config for file types so the crawl respects the latest
        # saved Settings (e.g. PDF unchecked) even if the server started with defaults.
        supported_extensions = get_supported_extensions_for_scan()
        effective_ignores    = get_effective_ignore_patterns(settings)
        log.info(
            'scan_crawling',
            directories = [str(d) for d in directories],
            extensions  = supported_extensions,
            extension_count = len(supported_extensions),
        )
        scanned_files = await asyncio.to_thread(
            scan_directories,
            directories          = directories,
            ignore_patterns      = effective_ignores,
            supported_extensions = supported_extensions,
            follow_symlinks      = settings.follow_symlinks,
        )
        files_scanned = len(scanned_files)
        if await _cancel_requested('post_crawl'):
            return

        # 2. Compare with DB (load all indexed files so change detection is correct)
        async with op_state.get_ingestion_lock():
            if await _cancel_requested('pre_compare'):
                return
            db_files = await get_all_files_for_scan(db)
            changes  = compare_with_db(scanned_files, db_files)

            async def _filter_retry_suppressed(files: list[ScannedFile]) -> tuple[list[ScannedFile], int]:
                kept: list[ScannedFile] = []
                suppressed = 0
                for sf in files:
                    normalized_path = str(normalize_path(sf.path, expand_user=False))
                    skip, error_code = await should_skip_file_retry(
                        db,
                        normalized_path,
                        sf.content_hash,
                    )
                    if not skip:
                        kept.append(sf)
                        continue
                    suppressed += 1
                    log.info(
                        'scan_file_retry_suppressed',
                        path=normalized_path,
                        filename=sf.filename,
                        error_code=error_code,
                    )
                return kept, suppressed

            changes.new, suppressed_new = await _filter_retry_suppressed(changes.new)
            changes.changed, suppressed_changed = await _filter_retry_suppressed(changes.changed)
            if force:
                changes.unchanged, suppressed_unchanged = await _filter_retry_suppressed(changes.unchanged)
            else:
                suppressed_unchanged = 0

            # When force=True, also reindex unchanged files; otherwise skip them
            if force:
                total_to_process = (
                    len(changes.new) + len(changes.changed) + len(changes.unchanged)
                )
            else:
                total_to_process = len(changes.new) + len(changes.changed)

            log.info(
                'scan_indexing_start',
                scan_id   = scan_id,
                force     = force,
                new       = len(changes.new),
                changed   = len(changes.changed),
                unchanged = len(changes.unchanged),
                deleted   = len(changes.deleted),
                total     = total_to_process,
                retry_suppressed_new=suppressed_new,
                retry_suppressed_changed=suppressed_changed,
                retry_suppressed_unchanged=suppressed_unchanged,
            )

            # Persist "checked" count so UI shows we only index new/changed
            await _update_scan_record_best_effort(
                ScanRecord(
                    id=scan_id,
                    started_at=scan_started_at,
                    files_scanned=files_scanned,
                    files_indexed=0,
                    errors=0,
                    status=ScanStatus.RUNNING,
                ),
                context='post_crawl_baseline',
            )

            # 3. Index new files (sequential to preserve DB consistency)
            for sf in changes.new:
                if await _cancel_requested('index_new'):
                    return
                try:
                    result = await _process_file(sf, 'indexing_file', index_file)
                except _ScanCancelledInFlight:
                    if await _cancel_requested('index_new_inflight'):
                        return
                    raise
                normalized_path = str(normalize_path(sf.path, expand_user=False))
                # Persist failure/success state for retry suppression across scans.
                if result.success:
                    await clear_file_failure(db, normalized_path)
                else:
                    await record_file_failure(
                        db,
                        path=normalized_path,
                        content_hash=sf.content_hash,
                        error_code=result.error_code,
                        error_message=result.error,
                        retryable=result.retryable,
                    )
            log.info(
                'scan_loop_complete',
                loop='new_files',
                processed=len(changes.new),
                total=len(changes.new),
            )

            # 4. Re-index changed files (sequential to preserve DB consistency)
            for sf in changes.changed:
                if await _cancel_requested('index_changed'):
                    return
                try:
                    result = await _process_file(sf, 'reindexing_file', reindex_file)
                except _ScanCancelledInFlight:
                    if await _cancel_requested('index_changed_inflight'):
                        return
                    raise
                normalized_path = str(normalize_path(sf.path, expand_user=False))
                if result.success:
                    await clear_file_failure(db, normalized_path)
                else:
                    await record_file_failure(
                        db,
                        path=normalized_path,
                        content_hash=sf.content_hash,
                        error_code=result.error_code,
                        error_message=result.error,
                        retryable=result.retryable,
                    )
            log.info(
                'scan_loop_complete',
                loop='changed_files',
                processed=len(changes.changed),
                total=len(changes.changed),
            )

            # 4b. When force=True, re-index unchanged files as well
            if force:
                for sf in changes.unchanged:
                    if await _cancel_requested('index_unchanged'):
                        return
                    try:
                        result = await _process_file(sf, 'reindexing_unchanged', reindex_file)
                    except _ScanCancelledInFlight:
                        if await _cancel_requested('index_unchanged_inflight'):
                            return
                        raise
                    normalized_path = str(normalize_path(sf.path, expand_user=False))
                    if result.success:
                        await clear_file_failure(db, normalized_path)
                    else:
                        await record_file_failure(
                            db,
                            path=normalized_path,
                            content_hash=sf.content_hash,
                            error_code=result.error_code,
                            error_message=result.error,
                            retryable=result.retryable,
                        )
                log.info(
                    'scan_loop_complete',
                    loop='unchanged_files',
                    processed=len(changes.unchanged),
                    total=len(changes.unchanged),
                )

            # 5. Remove deleted files
            for df in changes.deleted:
                if await _cancel_requested('remove_deleted'):
                    return
                try:
                    removed = await remove_file(db, df)
                    if not removed:
                        log.warning('remove_file_failed', file_id=df.id, path=df.path)
                except _SCAN_RUNTIME_EXCEPTIONS as exc:
                    # Log but continue - deletion failure shouldn't stop the scan
                    log.error(
                        'remove_file_exception',
                        file_id=df.id,
                        path=df.path,
                        error=str(exc),
                        exc_info=True,
                    )

        # 6. Update scan record
        scan_record = ScanRecord(
            id            = scan_id,
            started_at    = scan_started_at,
            files_scanned = files_scanned,
            files_indexed = files_indexed,
            errors        = errors,
            status        = ScanStatus.COMPLETED,
            completed_at  = datetime.now(UTC),
        )
        await _update_scan_record_best_effort(
            scan_record,
            context='completed_terminal',
            terminal=True,
        )
        log.info(
            'scan_completed',
            scan_id  = scan_id,
            scanned  = files_scanned,
            indexed  = files_indexed,
            errors   = errors,
            deleted  = len(changes.deleted),
        )

        log.info(
            'scan_metrics_summary',
            scan_id=scan_id,
            chunks_total_created=chunks_total_created,
            success_by_extension=dict(sorted(success_by_extension.items())),
            errors_by_extension=dict(sorted(errors_by_extension.items())),
            chunks_by_extension=dict(sorted(chunks_by_extension.items())),
            extractor_success_counts=dict(sorted(extractor_success_counts.items())),
            extractor_error_counts=dict(sorted(extractor_error_counts.items())),
            ocr_used_count=ocr_used_count,
            coverage_fallback_mode='batched',
        )

        # Post-run integrity check to detect cross-store drift early.
        integrity_issues = await get_index_integrity_issues(db)
        non_zero_issues = {k: v for k, v in integrity_issues.items() if v > 0}
        if non_zero_issues:
            log.error(
                'scan_integrity_issues_detected',
                scan_id=scan_id,
                issues=non_zero_issues,
            )

        # Update adaptive top-k cache after corpus changed (force immediate recompute).
        try:
            from informity.indexer.adaptive_tuning import update_tuning_cache
            await update_tuning_cache(db, force_recompute=True)
        except (ImportError, _SCAN_RUNTIME_EXCEPTIONS) as exc:
            log.warning('adaptive_tuning_scan_update_failed', error=str(exc))

        # sqlite-vec path currently uses exact cosine distance search;
        # explicit log avoids implying ANN build behavior.
        log.debug('scan_vector_index_skipped', reason='exact_search_mode')

    except _ScanCancelledInFlight:
        if await op_state.is_scan_cancel_requested(scan_id):
            log.info('scan_cancelled', scan_id=scan_id, stage='inflight_fallback')
            await _finalize_cancelled()
            return
        raise
    except _SCAN_RUNTIME_EXCEPTIONS as exc:
        log.error('scan_failed', scan_id=scan_id, error=str(exc), exc_info=True)
        scan_record = ScanRecord(
            id           = scan_id,
            started_at   = scan_started_at,
            status       = ScanStatus.FAILED,
            errors       = errors + 1,
            completed_at = datetime.now(UTC),
        )
        await _update_scan_record_best_effort(
            scan_record,
            context='failed_terminal',
            terminal=True,
        )
    except Exception as exc:
        # Last-resort guard: ensure scan status does not remain "running"
        # when unexpected exceptions escape the scan loop.
        log.error(
            'scan_failed_unhandled_exception',
            scan_id=scan_id,
            error=str(exc),
            exception_type=type(exc).__name__,
            exc_info=True,
        )
        scan_record = ScanRecord(
            id=scan_id,
            started_at=scan_started_at,
            status=ScanStatus.FAILED,
            errors=errors + 1,
            completed_at=datetime.now(UTC),
        )
        await _update_scan_record_best_effort(
            scan_record,
            context='failed_terminal_unhandled',
            terminal=True,
        )

    finally:
        await op_state.clear_scan_cancel(scan_id)
        try:
            await db.close()
        except _SCAN_RUNTIME_EXCEPTIONS as exc:
            # Log but don't raise - connection closure failure shouldn't mask scan errors
            log.warning('db_close_failed', scan_id=scan_id, error=str(exc))
        try:
            await progress_db.close()
        except _SCAN_RUNTIME_EXCEPTIONS as exc:
            log.warning('progress_db_close_failed', scan_id=scan_id, error=str(exc))
        clear_contextvars()
