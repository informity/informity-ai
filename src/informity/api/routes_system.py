# ==============================================================================
# Informity AI — System API Routes
# Endpoints for system operations: shutdown, diagnostics
# ==============================================================================

import asyncio
import math
import platform
from datetime import datetime
from typing import Literal

import psutil
import structlog
from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel

from informity.api.schemas import DiagnosticsMetricsSummaryResponse, DiagnosticsResponse
from informity.config import APP_DISPLAY_NAME, settings
from informity.db.sqlite import (
    CANONICAL_DIAGNOSTICS_QUERY_TYPES,
    CANONICAL_DIAGNOSTICS_TYPES,
    get_chunk_count,
    get_db,
    get_diagnostics_metrics_since,
    get_file_count,
    get_indexed_content_size_bytes,
)
from informity.db.vectors import vector_store
from informity.diagnostics.issue_types import IssueType
from informity.llm.engine import llm_engine
from informity.version import APP_VERSION

# ==============================================================================
# Logger
# ==============================================================================

log = structlog.get_logger(__name__)
_SYSTEM_DIAGNOSTICS_EXCEPTIONS = (OSError, RuntimeError, ValueError, TypeError)
_DIAGNOSTICS_SUMMARY_SCHEMA = 'informity.diagnostics.summary.v2'
_DIAGNOSTICS_SUMMARY_AGGREGATION_MODE = 'direct_window_scan'
_CANONICAL_DIAGNOSTICS_ISSUES = tuple(sorted(issue.value for issue in IssueType))

# ==============================================================================
# Router
# ==============================================================================

router = APIRouter(prefix='/api', tags=['system'])

# ==============================================================================
# Schemas
# ==============================================================================


class ShutdownResponse(BaseModel):
    """Shutdown confirmation."""
    message: str
    shutdown_initiated: bool = True


# ==============================================================================
# Endpoints
# ==============================================================================


@router.get('/diagnostics', response_model=DiagnosticsResponse)
async def get_diagnostics(request: Request) -> DiagnosticsResponse:
    """
    Returns system diagnostics: app version, Python version, OS, RAM, disk space,
    model info, DB stats, uptime. Useful for debugging issues in packaged builds.
    """
    # Get Python and platform info
    python_version = platform.python_version()
    platform_name = platform.system()
    platform_version = platform.version()
    architecture = platform.machine()

    # Get RAM info
    ram = psutil.virtual_memory()
    ram_total_gb = ram.total / (1024 ** 3)
    ram_available_gb = ram.available / (1024 ** 3)
    ram_used_gb = ram.used / (1024 ** 3)

    # Get disk info (for app data directory)
    disk = psutil.disk_usage(settings.app_data_dir)
    disk_total_gb = disk.total / (1024 ** 3)
    disk_available_gb = disk.free / (1024 ** 3)
    disk_used_gb = disk.used / (1024 ** 3)

    # Get model info
    model_loaded = llm_engine.model is not None
    model_filename = None
    model_size_gb = None
    if model_loaded:
        try:
            model_path = llm_engine._get_model_path()
            if model_path.exists():
                model_filename = model_path.name
                model_size_gb = model_path.stat().st_size / (1024 ** 3)
        except _SYSTEM_DIAGNOSTICS_EXCEPTIONS:
            pass

    # Get DB info
    db_path = str(settings.db_path)
    db_size_bytes = 0
    if settings.db_path and settings.db_path.exists():
        db_size_bytes = settings.db_path.stat().st_size
    db_size_mb = db_size_bytes / (1024 ** 2)

    # Get vectors info
    vectors_size_bytes = 0
    vectors_size_mb = 0.0
    try:
        stats = await asyncio.to_thread(vector_store.get_stats)
        vectors_size_bytes = stats.get('storage_bytes', 0)  # Changed from 'size_bytes' to 'storage_bytes'
        vectors_size_mb = vectors_size_bytes / (1024 ** 2)
    except _SYSTEM_DIAGNOSTICS_EXCEPTIONS:
        pass

    # Get index stats
    async with get_db() as db:
        total_files = await get_file_count(db)
        total_chunks = await get_chunk_count(db)
        indexed_content_size_bytes = await get_indexed_content_size_bytes(db)

    indexed_content_size_mb = indexed_content_size_bytes / (1024 ** 2)

    # Calculate uptime (if app started timestamp available)
    # For now, we don't track this, so return None
    uptime_seconds = None

    return DiagnosticsResponse(
        app_version=APP_VERSION,
        app_display_name=APP_DISPLAY_NAME,
        python_version=python_version,
        platform=platform_name,
        platform_version=platform_version,
        architecture=architecture,
        ram_total_gb=round(ram_total_gb, 2),
        ram_available_gb=round(ram_available_gb, 2),
        ram_used_gb=round(ram_used_gb, 2),
        disk_total_gb=round(disk_total_gb, 2),
        disk_available_gb=round(disk_available_gb, 2),
        disk_used_gb=round(disk_used_gb, 2),
        model_loaded=model_loaded,
        model_filename=model_filename,
        model_size_gb=round(model_size_gb, 2) if model_size_gb else None,
        db_path=db_path,
        db_size_bytes=db_size_bytes,
        db_size_mb=round(db_size_mb, 2),
        vectors_size_bytes=vectors_size_bytes,
        vectors_size_mb=round(vectors_size_mb, 2),
        total_files=total_files,
        total_chunks=total_chunks,
        indexed_content_size_bytes=indexed_content_size_bytes,
        indexed_content_size_mb=round(indexed_content_size_mb, 2),
        uptime_seconds=uptime_seconds,
    )


@router.post('/shutdown', response_model=ShutdownResponse)
async def shutdown(request: Request) -> ShutdownResponse:
    """
    Gracefully shuts down the application. Only callable from localhost.
    Tauri will call this before killing the sidecar process.
    """
    # Security: only allow shutdown from localhost
    client_host = request.client.host if request.client else None
    if client_host not in ('127.0.0.1', 'localhost', '::1'):
        raise HTTPException(
            status_code=403,
            detail='Shutdown endpoint is only accessible from localhost',
        )

    log.info('shutdown_requested', client_host=client_host)

    # Note: We can't actually shut down the FastAPI app from within a request handler.
    # The shutdown logic is handled by the lifespan context manager and signal handlers.
    # This endpoint just confirms that shutdown was requested and logs it.
    # Tauri will kill the process after calling this endpoint.

    return ShutdownResponse(
        message='Shutdown requested. Application will terminate.',
        shutdown_initiated=True,
    )


@router.get('/diagnostics/summary', response_model=DiagnosticsMetricsSummaryResponse)
async def get_diagnostics_summary(
    days: int = Query(default=30, ge=1, le=365),
    type_filter: Literal['user', 'evaluation'] | None = Query(default=None),
    run_id_filter: str | None = Query(default=None),
) -> DiagnosticsMetricsSummaryResponse:
    """
    Return aggregate runtime diagnostics metrics from response_diagnostics_metrics.
    Primarily used for operational trends and future stats dashboards.
    """
    async with get_db() as db:
        rows = await get_diagnostics_metrics_since(
            db=db,
            days=days,
            type_filter=type_filter,
            run_id_filter=run_id_filter,
        )

    total = len(rows)
    by_type: dict[str, int] = {}
    by_query_type: dict[str, int] = {}
    issue_counts: dict[str, int] = {}

    timeout_count = 0
    empty_answer_count = 0
    refusal_pattern_count = 0
    generation_seconds_values: list[float] = []
    sources_counts: list[int] = []
    raw_chunks_counts: list[int] = []
    created_at_values: list[datetime] = []

    for row in rows:
        metric_type = str(row.get('type') or '').strip().lower()
        if metric_type in CANONICAL_DIAGNOSTICS_TYPES:
            by_type[metric_type] = by_type.get(metric_type, 0) + 1
        else:
            log.warning('diagnostics_summary_unknown_type', raw_type=metric_type)

        query_type = str(row.get('query_type') or '').strip().lower()
        if query_type not in CANONICAL_DIAGNOSTICS_QUERY_TYPES:
            query_type = 'unknown'
            log.warning('diagnostics_summary_unknown_query_type')
        by_query_type[query_type] = by_query_type.get(query_type, 0) + 1

        if bool(row.get('timeout_occurred')):
            timeout_count += 1
        if bool(row.get('has_empty_answer')):
            empty_answer_count += 1
        if bool(row.get('has_refusal_pattern')):
            refusal_pattern_count += 1

        generation_seconds = row.get('generation_seconds')
        if isinstance(generation_seconds, int | float):
            generation_seconds_values.append(float(generation_seconds))

        sources_count = row.get('sources_count')
        if isinstance(sources_count, int):
            sources_counts.append(sources_count)

        raw_chunks_count = row.get('raw_chunks_count')
        if isinstance(raw_chunks_count, int):
            raw_chunks_counts.append(raw_chunks_count)

        detected_issues = row.get('detected_issues') or []
        if isinstance(detected_issues, list):
            for issue in detected_issues:
                issue_name = str(issue or '').strip().lower()
                if issue_name and issue_name in _CANONICAL_DIAGNOSTICS_ISSUES:
                    issue_counts[issue_name] = issue_counts.get(issue_name, 0) + 1

        created_at = row.get('created_at')
        if isinstance(created_at, datetime):
            created_at_values.append(created_at)

    def _avg(values: list[int] | list[float]) -> float:
        if not values:
            return 0.0
        return round(float(sum(values)) / len(values), 3)

    timeout_rate = round(timeout_count / total, 4) if total else 0.0
    empty_answer_rate = round(empty_answer_count / total, 4) if total else 0.0
    refusal_pattern_rate = round(refusal_pattern_count / total, 4) if total else 0.0

    p95_generation_seconds: float | None = None
    if generation_seconds_values:
        sorted_values = sorted(generation_seconds_values)
        idx = max(0, min(len(sorted_values) - 1, math.ceil(len(sorted_values) * 0.95) - 1))
        p95_generation_seconds = round(sorted_values[idx], 3)

    created_at_oldest = min(created_at_values) if created_at_values else None
    created_at_newest = max(created_at_values) if created_at_values else None

    return DiagnosticsMetricsSummaryResponse(
        summary_schema=_DIAGNOSTICS_SUMMARY_SCHEMA,
        aggregation_mode=_DIAGNOSTICS_SUMMARY_AGGREGATION_MODE,
        type_taxonomy=list(CANONICAL_DIAGNOSTICS_TYPES),
        query_type_taxonomy=list(CANONICAL_DIAGNOSTICS_QUERY_TYPES),
        issue_type_taxonomy=list(_CANONICAL_DIAGNOSTICS_ISSUES),
        window_days=days,
        type_filter=type_filter,
        run_id_filter=run_id_filter,
        total_responses=total,
        by_type=by_type,
        by_query_type=by_query_type,
        issue_counts=issue_counts,
        timeout_count=timeout_count,
        empty_answer_count=empty_answer_count,
        refusal_pattern_count=refusal_pattern_count,
        timeout_rate=timeout_rate,
        empty_answer_rate=empty_answer_rate,
        refusal_pattern_rate=refusal_pattern_rate,
        avg_generation_seconds=_avg(generation_seconds_values),
        p95_generation_seconds=p95_generation_seconds,
        avg_sources_count=_avg(sources_counts),
        avg_raw_chunks_count=_avg(raw_chunks_counts),
        created_at_oldest=created_at_oldest,
        created_at_newest=created_at_newest,
    )
