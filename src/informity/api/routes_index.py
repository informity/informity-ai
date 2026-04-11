# ==============================================================================
# Informity AI — Index API Routes
# Endpoints for managing the search index: rebuild and status.
# ==============================================================================

import asyncio
import shutil
import uuid
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import aiosqlite
import structlog
from fastapi import APIRouter, BackgroundTasks, Body, Depends, HTTPException
from structlog.contextvars import bind_contextvars, clear_contextvars

import informity.api.operation_state as op_state
from informity.api.chat_stream_registry import CHAT_STREAM_REGISTRY
from informity.api.schemas import IndexStatusResponse, RebuildRequest
from informity.api.security import EndpointGuard
from informity.config import DirNames, reset_to_factory_defaults, settings
from informity.db.models import ScanRecord, ScanStatus
from informity.db.sqlite import (
    get_chat_count,
    get_chunk_count,
    get_db,
    get_file_count,
    get_index_integrity_issues,
    get_index_scope_counts,
    get_indexed_content_size_bytes,
    get_latest_completed_scan,
    insert_scan_record,
    purge_term_dictionary,
    reset_all_data,
    reset_index_data_scope,
    update_scan_record,
)
from informity.db.vectors import vector_store
from informity.indexer.pipeline import reindex_file
from informity.indexer.term_dictionary_builder import (
    get_term_dictionary_build_status,
    rebuild_term_dictionary,
)
from informity.scanner.crawler import scanned_file_for_path
from informity.scanner.extractors.base import register_extractors
from informity.sources.base import FILESYSTEM_PROVIDER, SOURCE_ENTITY_FILE

# ==============================================================================
# Logger
# ==============================================================================

log = structlog.get_logger(__name__)
_INDEX_RUNTIME_EXCEPTIONS = (aiosqlite.Error, RuntimeError, ValueError, TypeError, OSError, TimeoutError)
_INDEX_CLEANUP_EXCEPTIONS = (OSError, PermissionError, RuntimeError)

# ==============================================================================
# Router
# ==============================================================================

router = APIRouter(tags=['index'])
REBUILD_GUARD = EndpointGuard(
    name='rebuild',
    max_in_flight=1,
    max_requests_per_window=12,
    window_seconds=60,
)


# ==============================================================================
# POST /api/index/rebuild — force full re-index
# ==============================================================================

@router.post('/api/index/rebuild')
async def rebuild_index(
    background_tasks: BackgroundTasks,
    db: aiosqlite.Connection = Depends(get_db),
    request: RebuildRequest | None = Body(None),
) -> dict:
    # Trigger a full re-index of all currently indexed files.
    # This re-extracts, re-chunks, re-embeds, and re-stores every file.
    req = request if request is not None else RebuildRequest()

    async with REBUILD_GUARD.slot():
        # Serialize scan/rebuild/reset transition checks + scan-record creation.
        async with op_state.get_scan_operation_lock():
            # Block if reset is in progress
            if await op_state.is_reset_in_progress():
                raise HTTPException(
                    status_code=409,
                    detail='Index reset is in progress. Please wait for it to complete.',
                )

            # Check if a scan/rebuild is already running and resolve it (cancel / mark stale / block)
            await op_state.resolve_running_scan(db, force=req.force, operation='rebuild')

            # Create a scan record for tracking
            scan_record = ScanRecord(started_at=datetime.now(UTC))
            scan_record = await insert_scan_record(db, scan_record)

        # Launch background task
        background_tasks.add_task(
            _run_rebuild_task,
            scan_id=scan_record.id,
        )

        return {
            'scan_id': scan_record.id,
            'status':  'rebuilding',
        }


# ==============================================================================
# GET /api/index/status — return index statistics
# ==============================================================================


def _compute_disk_sizes() -> tuple[int, int]:
    # Compute DB file size and model directory size (sync I/O).
    # Called via asyncio.to_thread from the async route handler.
    db_size_bytes = 0
    if settings.db_path is not None and settings.db_path.exists():
        db_size_bytes = settings.db_path.stat().st_size

    model_size_bytes = 0
    model_dirs = (
        settings.models_dir,
    )
    for models_dir in model_dirs:
        if models_dir is None or not models_dir.exists():
            continue
        for f in models_dir.rglob('*'):
            if f.is_file():
                model_size_bytes += f.stat().st_size

    return db_size_bytes, model_size_bytes


@router.get('/api/index/status', response_model=IndexStatusResponse)
async def get_index_status(
    db: aiosqlite.Connection = Depends(get_db),
) -> IndexStatusResponse:
    # Gather statistics from SQLite (including vector storage).
    reset_in_progress, last_reset_result = await op_state.get_reset_state_snapshot()

    # During reset, avoid DB reads from polling clients (Settings page polls every
    # 500ms) so reset compaction can obtain the exclusive lock for VACUUM.
    if reset_in_progress:
        db_size_bytes, model_size_bytes = await asyncio.to_thread(_compute_disk_sizes)
        return IndexStatusResponse(
            total_files                = 0,
            total_chunks               = 0,
            total_embeddings           = 0,
            chat_count                 = 0,
            last_scan_at               = None,
            db_size_bytes              = db_size_bytes,
            vectors_size_bytes         = 0,
            model_size_bytes           = model_size_bytes,
            indexed_content_size_bytes = 0,
            reset_in_progress          = True,
            last_reset_result          = last_reset_result,
            source_scope_stats         = [],
        )

    total_files  = await get_file_count(db)
    total_chunks = await get_chunk_count(db)

    # Get vector store stats
    try:
        vector_stats     = await asyncio.to_thread(vector_store.get_stats)
        total_embeddings = vector_stats['total_vectors']
        vectors_size     = vector_stats['storage_bytes']
    except _INDEX_RUNTIME_EXCEPTIONS:
        # Vector store may not be initialized yet
        total_embeddings = 0
        vectors_size     = 0

    # Last scan time: use most recent completed scan so we never show "Never" when a run is in progress
    latest_completed = await get_latest_completed_scan(db)
    last_scan_at     = latest_completed.completed_at if latest_completed else None

    indexed_content_size_bytes = await get_indexed_content_size_bytes(db)
    chat_count                 = await get_chat_count(db)
    source_scope_stats         = await get_index_scope_counts(db)

    # Compute DB and model directory sizes in a thread to avoid blocking the event loop
    db_size_bytes, model_size_bytes = await asyncio.to_thread(_compute_disk_sizes)

    return IndexStatusResponse(
        total_files                = total_files,
        total_chunks               = total_chunks,
        total_embeddings           = total_embeddings,
        chat_count                 = chat_count,
        last_scan_at               = last_scan_at,
        db_size_bytes              = db_size_bytes,
        vectors_size_bytes         = vectors_size,
        model_size_bytes          = model_size_bytes,
        indexed_content_size_bytes = indexed_content_size_bytes,
        reset_in_progress          = reset_in_progress,
        last_reset_result          = last_reset_result,
        source_scope_stats         = source_scope_stats,
    )


# ==============================================================================
# POST /api/index/reset — delete all indexed data
# ==============================================================================

@router.post('/api/index/reset')
async def reset_index(
    background_tasks: BackgroundTasks,
    force: bool = False,
    source_provider: str | None = None,
    entity_type: str | None = None,
    db: aiosqlite.Connection = Depends(get_db),
) -> dict:
    # Delete ALL user data: files, chunks, vectors, scan history,
    # chat messages, quality metrics, diagnostics (chat traces, evaluations, reports),
    # and logs (runtime application logs).
    # Downloaded models and cache assets (app_data_dir/models/, app_data_dir/cache/) are preserved.
    # Also resets all settings to factory defaults.
    # Runs in background so the request returns immediately.

    async with op_state.get_scan_operation_lock():
        source_provider = (source_provider or '').strip().lower() or None
        entity_type = (entity_type or '').strip().lower() or None
        if (source_provider is None) != (entity_type is None):
            raise HTTPException(
                status_code=400,
                detail='source_provider and entity_type must be provided together for scoped reset.',
            )

        if not await op_state.try_begin_reset():
            raise HTTPException(
                status_code=409,
                detail='A reset is already in progress. Please wait for it to complete.',
            )

        # Refuse if a scan/rebuild is currently running unless force=true.
        # Stale scans (>STALE_SCAN_THRESHOLD_SECONDS) are auto-cleared regardless.
        try:
            await op_state.resolve_running_scan(db, force=force, operation='reset')
        except HTTPException:
            await op_state.finish_reset(result=None)
            raise
        except _INDEX_RUNTIME_EXCEPTIONS:
            await op_state.finish_reset(result=None)
            raise

    background_tasks.add_task(
        _run_reset_task,
        source_provider=source_provider,
        entity_type=entity_type,
    )
    payload = {'status': 'reset_started'}
    if source_provider and entity_type:
        payload['scope'] = {
            'source_provider': source_provider,
            'entity_type': entity_type,
        }
    return payload


@router.get('/api/index/term-dictionary/status')
async def term_dictionary_status(
    db: aiosqlite.Connection = Depends(get_db),
) -> dict:
    return await get_term_dictionary_build_status(db)


@router.post('/api/index/term-dictionary/rebuild')
async def rebuild_term_dictionary_now(
    db: aiosqlite.Connection = Depends(get_db),
) -> dict:
    return await rebuild_term_dictionary(db)


@router.post('/api/index/term-dictionary/purge')
async def purge_term_dictionary_now(
    db: aiosqlite.Connection = Depends(get_db),
) -> dict:
    await purge_term_dictionary(db)
    return {'status': 'purged'}


# ==============================================================================
# Background Reset Task
# ==============================================================================

async def _run_reset_task(
    *,
    source_provider: str | None = None,
    entity_type: str | None = None,
) -> None:
    from informity.db.sqlite import get_connection

    clear_contextvars()
    reset_operation_id = f'reset-{uuid.uuid4().hex[:8]}'
    bind_contextvars(
        operation_type='reset',
        operation_id=reset_operation_id,
    )

    db = None
    reset_result: dict | None = None

    try:
        scoped_reset = bool(source_provider and entity_type)

        # Phase 1: Stop active operations first (chat streams) to release locks.
        if not scoped_reset:
            stopped_streams = await CHAT_STREAM_REGISTRY.stop_all()
            if stopped_streams > 0:
                await asyncio.sleep(0.25)
                log.info('reset_stopped_active_chat_streams', stream_count=stopped_streams)

        # Phase 2: Reset database state.
        db = await get_connection()
        db_counts: dict[str, object]
        if scoped_reset and source_provider and entity_type:
            db_counts = await reset_index_data_scope(
                db,
                source_provider=source_provider,
                entity_type=entity_type,
            )
        else:
            db_counts = await reset_all_data(db)

        # Phase 3: Delete user-generated directories.
        # Chat traces live under app_data_dir/chats/ when chat trace logging is enabled.
        # Diagnostics contains user traces/evaluations/reports.
        # Logs contain runtime application logs.
        chat_traces_deleted = False
        diagnostics_deleted = False
        logs_deleted = False

        if not scoped_reset:
            chat_traces_dir = settings.app_data_dir / DirNames.CHAT_LOGS
            if chat_traces_dir.exists():
                try:
                    await asyncio.to_thread(shutil.rmtree, chat_traces_dir)
                    chat_traces_deleted = True
                    log.info('chat_traces_deleted', path=str(chat_traces_dir))
                except _INDEX_CLEANUP_EXCEPTIONS as exc:
                    log.error('reset_chat_traces_failed', error=str(exc), exc_info=True)

            if settings.diagnostics_dir and settings.diagnostics_dir.exists():
                try:
                    await asyncio.to_thread(shutil.rmtree, settings.diagnostics_dir)
                    diagnostics_deleted = True
                    log.info('diagnostics_deleted', path=str(settings.diagnostics_dir))
                except _INDEX_CLEANUP_EXCEPTIONS as exc:
                    log.error('reset_diagnostics_failed', error=str(exc), exc_info=True)

            if settings.logs_dir and settings.logs_dir.exists():
                try:
                    await asyncio.to_thread(shutil.rmtree, settings.logs_dir)
                    logs_deleted = True
                    log.info('logs_deleted', path=str(settings.logs_dir))
                except _INDEX_CLEANUP_EXCEPTIONS as exc:
                    log.error('reset_logs_failed', error=str(exc), exc_info=True)

            # Phase 4: Reset settings/configuration to factory defaults.
            await asyncio.to_thread(reset_to_factory_defaults)
            log.info('settings_reset_to_factory_defaults_during_data_reset')

            # Keep runtime directories present after config reset.
            settings.ensure_directories()

        # Invalidate adaptive top-k cache (corpus is empty)
        from informity.indexer.adaptive_tuning import invalidate_tuning_cache
        invalidate_tuning_cache()

        reset_result = {
            'files_deleted':       db_counts.get('files', 0),
            'chunks_deleted':      db_counts.get('chunks', 0),
            'vectors_deleted':     db_counts.get('vec_chunks', 0),
            'chats_deleted':       db_counts.get('chat_messages', 0),
            'metrics_deleted':     db_counts.get('response_diagnostics_metrics', 0),
            'storage_compacted':   bool(db_counts.get('storage_compacted', False)),
            'compaction_error':    db_counts.get('compaction_error'),
            'chat_traces_deleted': chat_traces_deleted,
            'diagnostics_deleted': diagnostics_deleted,
            'logs_deleted':        logs_deleted,
            'scoped_reset':        scoped_reset,
        }
        if scoped_reset:
            reset_result.update(
                {
                    'source_provider': source_provider,
                    'entity_type': entity_type,
                    'files_deleted': db_counts.get('files_deleted', 0),
                    'chunks_deleted': db_counts.get('chunks_deleted', 0),
                    'vectors_deleted': db_counts.get('vectors_deleted', 0),
                    'file_failures_deleted': db_counts.get('file_failures_deleted', 0),
                }
            )

        log.info(
            'index_reset_completed',
            db_counts       = db_counts,
            vectors_deleted = db_counts.get('vec_chunks', 0),
        )

    except _INDEX_RUNTIME_EXCEPTIONS as exc:
        log.error('reset_failed', error=str(exc), exc_info=True)
        reset_result = {'error': str(exc)}

    finally:
        if db is not None:
            await db.close()
        await op_state.finish_reset(result=reset_result)
        clear_contextvars()


# ==============================================================================
# Background Rebuild Task
# ==============================================================================

async def _run_rebuild_task(scan_id: int) -> None:
    # Background task that re-indexes all files currently in the database.
    # For each file: re-crawl (to verify it still exists), re-extract,
    # re-chunk, re-embed, and re-store.

    from informity.db.sqlite import get_all_files_for_scan, get_connection

    clear_contextvars()
    rebuild_operation_id = f'rebuild-{scan_id}-{uuid.uuid4().hex[:8]}'
    bind_contextvars(
        operation_type='rebuild',
        operation_id=rebuild_operation_id,
        scan_id=scan_id,
    )

    scan_started_at = datetime.now(UTC)
    db              = await get_connection()

    files_scanned = 0
    files_indexed = 0
    errors        = 0
    chunks_total_created = 0
    success_by_extension: dict[str, int] = defaultdict(int)
    errors_by_extension: dict[str, int] = defaultdict(int)
    chunks_by_extension: dict[str, int] = defaultdict(int)
    extractor_success_counts: dict[str, int] = defaultdict(int)
    extractor_error_counts: dict[str, int] = defaultdict(int)
    ocr_used_count = 0

    try:
        register_extractors()

        # Get all currently indexed files (no arbitrary limit)
        all_files     = await get_all_files_for_scan(
            db,
            source_provider=FILESYSTEM_PROVIDER,
            entity_type=SOURCE_ENTITY_FILE,
        )
        files_scanned = len(all_files)

        # Drop and recreate the vector store before rebuilding so no stale vectors
        # from previous runs or schema mismatches accumulate. Per-file delete_by_file_id
        # (soft-delete) leaves orphaned Parquet fragments that confuse parent-chunk lookup.
        log.info('rebuild_clearing_vector_store')
        await asyncio.to_thread(vector_store.drop_all)

        log.info('rebuild_starting', total_files=files_scanned, provider=FILESYSTEM_PROVIDER)

        async with op_state.get_ingestion_lock():
            # Re-index each file
            for indexed_file in all_files:
                file_path = Path(indexed_file.path)

                # Skip files that no longer exist on disk
                if not file_path.exists():
                    log.warning('rebuild_file_missing', path=indexed_file.path)
                    errors += 1
                    continue

                # Recompute fresh hash/stat from disk (do not reuse stale DB hash).
                scanned = scanned_file_for_path(file_path)
                if scanned is None:
                    log.error('rebuild_scan_failed', path=indexed_file.path)
                    errors += 1
                    continue

                try:
                    result = await reindex_file(db, scanned)
                    if result.success:
                        files_indexed += 1
                        chunks_total_created += result.chunks_created
                        success_by_extension[scanned.extension] += 1
                        chunks_by_extension[scanned.extension] += result.chunks_created
                        if result.extractor:
                            extractor_success_counts[result.extractor] += 1
                        if result.ocr_used:
                            ocr_used_count += 1
                    else:
                        errors += 1
                        errors_by_extension[scanned.extension] += 1
                        if result.extractor:
                            extractor_error_counts[result.extractor] += 1
                        log.warning(
                            'rebuild_file_failed',
                            path  = indexed_file.path,
                            error = result.error,
                        )
                except _INDEX_RUNTIME_EXCEPTIONS as exc:
                    # Catch exceptions from reindex_file to prevent one file from stopping the rebuild
                    errors += 1
                    errors_by_extension[scanned.extension] += 1
                    log.error(
                        'rebuild_file_exception',
                        path=str(indexed_file.path),
                        error=str(exc),
                        exc_info=True,
                    )

        # Update scan record
        scan_record = ScanRecord(
            id            = scan_id,
            started_at    = scan_started_at,
            files_scanned = files_scanned,
            files_indexed = files_indexed,
            errors        = errors,
            status        = ScanStatus.COMPLETED,
            completed_at  = datetime.now(UTC),
        )
        await update_scan_record(db, scan_record)

        # Post-run integrity check to detect cross-store drift early.
        integrity_issues = await get_index_integrity_issues(db)
        non_zero_issues = {k: v for k, v in integrity_issues.items() if v > 0}
        if non_zero_issues:
            log.error(
                'rebuild_integrity_issues_detected',
                scan_id=scan_id,
                issues=non_zero_issues,
            )

        # Update adaptive top-k cache after corpus changed (force immediate recompute).
        try:
            from informity.indexer.adaptive_tuning import update_tuning_cache
            await update_tuning_cache(db, force_recompute=True)
        except (ImportError, _INDEX_RUNTIME_EXCEPTIONS) as exc:
            log.warning('adaptive_tuning_rebuild_update_failed', error=str(exc))

        # Post-rebuild term dictionary refresh (best-effort; does not fail rebuild).
        try:
            term_dictionary_result = await rebuild_term_dictionary(db, run_id=f'term-dict-rebuild-{scan_id}')
            log.info('term_dictionary_rebuild_update', scan_id=scan_id, result=term_dictionary_result)
        except _INDEX_RUNTIME_EXCEPTIONS as exc:
            log.warning('term_dictionary_rebuild_update_failed', scan_id=scan_id, error=str(exc))

        log.info(
            'rebuild_completed',
            scan_id = scan_id,
            provider = FILESYSTEM_PROVIDER,
            scanned = files_scanned,
            indexed = files_indexed,
            errors  = errors,
        )

        log.info(
            'rebuild_metrics_summary',
            scan_id=scan_id,
            chunks_total_created=chunks_total_created,
            success_by_extension=dict(sorted(success_by_extension.items())),
            errors_by_extension=dict(sorted(errors_by_extension.items())),
            chunks_by_extension=dict(sorted(chunks_by_extension.items())),
            extractor_success_counts=dict(sorted(extractor_success_counts.items())),
            extractor_error_counts=dict(sorted(extractor_error_counts.items())),
            ocr_used_count=ocr_used_count,
        )

        # sqlite-vec path currently uses exact cosine distance search;
        # explicit log avoids implying ANN build behavior.
        log.debug('rebuild_vector_index_skipped', reason='exact_search_mode')

    except _INDEX_RUNTIME_EXCEPTIONS as exc:
        log.error('rebuild_failed', scan_id=scan_id, error=str(exc), exc_info=True)
        scan_record = ScanRecord(
            id           = scan_id,
            started_at   = scan_started_at,
            status       = ScanStatus.FAILED,
            errors       = errors + 1,
            completed_at = datetime.now(UTC),
        )
        await update_scan_record(db, scan_record)

    finally:
        await db.close()
        clear_contextvars()
