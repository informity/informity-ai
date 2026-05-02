# ==============================================================================
# Informity AI — File Watcher Module
# Uses watchdog to monitor watched directories. On create/modify: index the file.
# On delete: remove from DB and vectors. Debounces rapid changes (2s).
# ==============================================================================

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import Any

import structlog
from watchdog.events import FileMovedEvent, FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from informity.config import get_effective_ignore_patterns, get_supported_extensions_for_scan, settings
from informity.scanner.crawler import should_ignore
from informity.sources.base import FILESYSTEM_PROVIDER, SOURCE_ENTITY_FILE
from informity.utils.path_utils import normalize_path, resolve_and_check_path

# ==============================================================================
# Logger
# ==============================================================================

log = structlog.get_logger(__name__)
_WATCHER_STOP_EXCEPTIONS = (RuntimeError, OSError, ValueError)

# ==============================================================================
# Constants
# ==============================================================================

DEBOUNCE_SECONDS = 2.0

# ==============================================================================
# Path filtering (reuses crawler logic)
# ==============================================================================

# Cache for performance: patterns and extension set rebuilt only when settings change
_cached_ignore_patterns: list[str] | None = None
_cached_supported_extensions_set: frozenset[str] | None = None


def _get_cached_ignore_patterns() -> list[str]:
    """Get ignore patterns, using cache if available."""
    global _cached_ignore_patterns
    if _cached_ignore_patterns is None:
        _cached_ignore_patterns = get_effective_ignore_patterns(settings)
    return _cached_ignore_patterns


def _get_cached_supported_extensions_set() -> frozenset[str]:
    """Get supported extensions as a set for O(1) lookup."""
    global _cached_supported_extensions_set
    if _cached_supported_extensions_set is None:
        try:
            supported_extensions = get_supported_extensions_for_scan()
        except (OSError, ValueError, TypeError):
            # Fallback to in-memory singleton if persisted config cannot be read.
            supported_extensions = list(settings.supported_extensions)
        _cached_supported_extensions_set = frozenset(
            str(ext).strip().lower() for ext in supported_extensions if str(ext).strip()
        )
    return _cached_supported_extensions_set


def invalidate_watcher_cache() -> None:
    """Invalidate cached patterns/extensions (call when settings change)."""
    global _cached_ignore_patterns, _cached_supported_extensions_set
    _cached_ignore_patterns = None
    _cached_supported_extensions_set = None


def _is_watchable_file(path: Path) -> bool:
    """True if path is a supported file we should index (extension + not ignored)."""
    if not path.is_file():
        return False
    ext = path.suffix.lower()
    if ext not in _get_cached_supported_extensions_set():
        return False
    ignores = _get_cached_ignore_patterns()
    return not should_ignore(path, ignores)


# ==============================================================================
# Debouncer — coalesce events and flush after 2 seconds of inactivity
# ==============================================================================


class _Debouncer:
    """Thread-safe debouncer: collects path+action, flushes after DEBOUNCE_SECONDS."""

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        on_flush: Callable[[list[tuple[str, str]]], Coroutine[Any, Any, None]],
    ) -> None:
        self._lock       = threading.Lock()
        self._pending: dict[str, str] = {}  # path_str -> 'index' | 'delete'
        self._timer: threading.Timer | None = None
        self._loop       = loop
        self._on_flush   = on_flush  # async callback

    def enqueue(self, path: Path, action: str) -> None:
        # action is 'index' or 'delete'. For same path, delete wins (we coalesce).
        with self._lock:
            path_str = str(path.resolve())
            current = self._pending.get(path_str)
            if action == 'delete':
                self._pending[path_str] = 'delete'
            elif current != 'delete':
                self._pending[path_str] = 'index'

            self._schedule()

    def _schedule(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
        self._timer = threading.Timer(DEBOUNCE_SECONDS, self._flush)
        self._timer.daemon = True
        self._timer.start()

    def _flush(self) -> None:
        with self._lock:
            self._timer = None
            items = list(self._pending.items())
            self._pending.clear()
        if not items:
            return
        log.debug('watcher_flush', count=len(items), paths=[p for p, _ in items])
        asyncio.run_coroutine_threadsafe(self._on_flush(items), self._loop)


# ==============================================================================
# Watchdog event handler
# ==============================================================================


class _InformityEventHandler(FileSystemEventHandler):
    """Handles create/modify/delete; filters by extension and ignore patterns."""

    def __init__(self, debouncer: _Debouncer) -> None:
        super().__init__()
        self._debouncer = debouncer

    def _path(self, src_path: str) -> Path:
        return normalize_path(src_path, expand_user=False)

    def _enqueue_index(self, path: Path) -> None:
        if _is_watchable_file(path):
            self._debouncer.enqueue(path, 'index')

    def _enqueue_delete(self, path: Path) -> None:
        # Deleted path might not exist; we still want to remove from DB by path.
        ext = path.suffix.lower()
        if ext in _get_cached_supported_extensions_set():
            self._debouncer.enqueue(path, 'delete')

    def on_created(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        self._enqueue_index(self._path(event.src_path))

    def on_modified(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        self._enqueue_index(self._path(event.src_path))

    def on_deleted(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        self._enqueue_delete(self._path(event.src_path))

    def on_moved(self, event: FileMovedEvent) -> None:  # type: ignore[override]
        if event.is_directory:
            return
        # Treat move as delete (src) + create (dest)
        self._enqueue_delete(self._path(event.src_path))
        dest = self._path(event.dest_path)
        if dest.exists() and _is_watchable_file(dest):
            self._debouncer.enqueue(dest, 'index')


# ==============================================================================
# Watcher lifecycle — start/stop observer, process pending on main loop
# ==============================================================================

_observer: Observer | None = None
_debouncer: _Debouncer | None = None
# Serialize _process_pending so concurrent flushes don't race on the same paths.
_process_lock: asyncio.Lock | None = None


def _get_process_lock() -> asyncio.Lock:
    global _process_lock
    if _process_lock is None:
        _process_lock = asyncio.Lock()
    return _process_lock


async def _process_pending(items: list[tuple[str, str]]) -> None:
    """Run on the main event loop: index or remove each path."""
    import informity.api.operation_state as op_state
    from informity.db.sqlite import (
        clear_file_failure,
        get_connection,
        get_file_by_path,
        get_file_by_source_identity,
        record_file_failure,
        should_skip_file_retry,
    )
    from informity.indexer.pipeline import index_file, reindex_file, remove_file
    from informity.scanner.crawler import scanned_file_for_path
    from informity.scanner.extractors.base import register_extractors

    async with _get_process_lock(), op_state.get_ingestion_lock():
        register_extractors()
        db = await get_connection()
        try:
            for path_str, action in items:
                path = Path(path_str)
                source_item_id = str(normalize_path(path, expand_user=False))
                if action == 'delete':
                    existing = await get_file_by_source_identity(
                        db,
                        source_provider=FILESYSTEM_PROVIDER,
                        entity_type=SOURCE_ENTITY_FILE,
                        source_item_id=source_item_id,
                    )
                    if existing is None:
                        existing = await get_file_by_path(db, path_str)
                    if existing is not None:
                        await remove_file(db, existing)
                    else:
                        log.debug('watcher_delete_unknown_path', path=path_str)
                else:
                    scanned = scanned_file_for_path(path)
                    if scanned is None:
                        log.warning('watcher_scan_failed', path=path_str)
                        continue
                    skip_retry, error_code = await should_skip_file_retry(
                        db,
                        source_provider=FILESYSTEM_PROVIDER,
                        entity_type=SOURCE_ENTITY_FILE,
                        source_item_id=source_item_id,
                        content_hash=scanned.content_hash,
                    )
                    if skip_retry:
                        log.info(
                            'watcher_retry_suppressed',
                            path=path_str,
                            filename=scanned.filename,
                            error_code=error_code,
                        )
                        continue
                    existing = await get_file_by_source_identity(
                        db,
                        source_provider=FILESYSTEM_PROVIDER,
                        entity_type=SOURCE_ENTITY_FILE,
                        source_item_id=source_item_id,
                    )
                    if existing is None:
                        existing = await get_file_by_path(db, path_str)
                    if existing is not None:
                        result = await reindex_file(db, scanned)
                    else:
                        result = await index_file(db, scanned)
                    if result.success:
                        await clear_file_failure(
                            db,
                            source_provider=FILESYSTEM_PROVIDER,
                            entity_type=SOURCE_ENTITY_FILE,
                            source_item_id=source_item_id,
                        )
                        continue
                    await record_file_failure(
                        db,
                        source_provider=FILESYSTEM_PROVIDER,
                        entity_type=SOURCE_ENTITY_FILE,
                        source_item_id=source_item_id,
                        path=path_str,
                        content_hash=scanned.content_hash,
                        error_code=result.error_code,
                        error_message=result.error,
                        retryable=result.retryable,
                    )
                    log.warning('watcher_index_failed', path=path_str, error=result.error)
        finally:
            await db.close()


def start_watcher(loop: asyncio.AbstractEventLoop) -> None:
    """Start watching configured directories. Call from lifespan startup."""
    global _observer, _debouncer

    dirs = list(settings.watched_directories)
    if not dirs:
        log.info('watcher_skipped', reason='no_watched_directories')
        return

    # Resolve and filter existing directories
    watch_dirs: list[Path] = []
    for d in dirs:
        resolved, exists = resolve_and_check_path(d)
        if not exists:
            log.warning('watcher_directory_not_found', directory=str(resolved))
            continue
        if not resolved.is_dir():
            log.warning('watcher_path_not_directory', path=str(resolved))
            continue
        watch_dirs.append(resolved)

    if not watch_dirs:
        log.warning('watcher_skipped', reason='no_valid_directories')
        return

    _debouncer = _Debouncer(loop, _process_pending)
    handler    = _InformityEventHandler(_debouncer)
    _observer  = Observer()
    for directory in watch_dirs:
        _observer.schedule(handler, str(directory), recursive=True)
        log.debug('watcher_scheduled', directory=str(directory))

    _observer.start()
    log.info('watcher_started', directories=[str(d) for d in watch_dirs])


def stop_watcher() -> None:
    """Stop the file watcher. Call from lifespan shutdown."""
    global _observer, _debouncer

    if _observer is None:
        return
    try:
        _observer.stop()
        _observer.join(timeout=5.0)
    except _WATCHER_STOP_EXCEPTIONS as exc:
        log.warning('watcher_stop_error', error=str(exc))
    _observer = None
    _debouncer = None
    log.info('watcher_stopped')
