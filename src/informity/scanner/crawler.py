# ==============================================================================
# Informity AI — Filesystem Crawler
# Walks directories respecting ignore patterns and extension whitelist.
# Computes SHA-256 hashes, compares against DB to determine changes.
# ==============================================================================

from __future__ import annotations

import hashlib
import os
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import structlog
from pathspec import PathSpec

from informity.config import settings
from informity.db.models import IndexedFile
from informity.utils.path_utils import normalize_path, resolve_and_check_path

# ==============================================================================
# Logger
# ==============================================================================

log = structlog.get_logger(__name__)

# ==============================================================================
# Constants
# ==============================================================================

HASH_CHUNK_SIZE = 8192  # Read files in 8KB chunks for hashing
DEFAULT_SCAN_SOURCE_PROVIDER = 'filesystem'
DEFAULT_SCAN_ENTITY_TYPE = 'file'


# ==============================================================================
# Path Utilities
# ==============================================================================
# Note: resolve_and_check_path is now imported from utils.path_utils


# ==============================================================================
# ScannedFile — lightweight result from crawling (before extraction)
# ==============================================================================

@dataclass
class ScannedFile:
    # A file discovered during scanning, before text extraction.
    path:        Path
    filename:    str
    extension:   str
    size_bytes:  int
    content_hash: str        # SHA-256
    modified_at: datetime


# ==============================================================================
# Change Detection Result
# ==============================================================================

@dataclass
class ChangeSet:
    # Result of comparing scanned files against the database.
    new:       list[ScannedFile]    # Files not in DB
    changed:   list[ScannedFile]    # Files with different hash
    unchanged: list[ScannedFile]    # Files with same hash
    deleted:   list[IndexedFile]    # DB files no longer on disk


# ==============================================================================
# Scanning
# ==============================================================================

def scan_directories(
    directories: list[Path] | None = None,
    ignore_patterns: list[str] | None = None,
    supported_extensions: list[str] | None = None,
    follow_symlinks: bool | None = None,
) -> list[ScannedFile]:
    # Walk the given directories and return a list of ScannedFile objects.
    # Respects ignore patterns and extension whitelist from config.
    dirs       = directories if directories is not None else settings.watched_directories
    ignores    = ignore_patterns if ignore_patterns is not None else settings.ignore_patterns
    extensions = supported_extensions if supported_extensions is not None else settings.supported_extensions
    follow     = follow_symlinks if follow_symlinks is not None else settings.follow_symlinks

    if not dirs:
        log.warning('no_directories_to_scan')
        return []

    # Normalize extensions to lowercase set for fast lookup
    ext_set = {ext.lower() for ext in extensions}

    # Collect all candidate file paths
    candidate_paths: list[Path] = []

    for directory in dirs:
        directory, exists = resolve_and_check_path(directory)
        if not exists:
            log.warning('scan_directory_not_found', directory=str(directory))
            continue
        if not directory.is_dir():
            log.warning('scan_path_not_directory', path=str(directory))
            continue

        log.info('scanning_directory', directory=str(directory))
        allowed_roots = (directory.resolve(),)

        for file_path in _walk_directory(
            directory,
            ignores,
            ext_set,
            follow,
            _allowed_roots=allowed_roots,
        ):
            candidate_paths.append(file_path)

    if not candidate_paths:
        return []

    # Compute hashes (use adaptive executor for large batches)
    scanned = _build_scanned_files(candidate_paths)

    log.info('scan_complete', files=len(scanned))
    return scanned


# ==============================================================================
# Directory Walking
# ==============================================================================

def _walk_directory(
    directory: Path,
    ignore_patterns: list[str],
    extensions: set[str],
    follow_symlinks: bool,
    _visited: set[tuple[int, int]] | None = None,
    _ignore_spec: PathSpec | None = None,
    _allowed_roots: tuple[Path, ...] | None = None,
) -> list[Path]:
    # Recursively walk a directory, filtering by ignore patterns and extensions.
    # When follow_symlinks is True, tracks visited (dev, ino) pairs to prevent
    # infinite recursion from symlink cycles.
    results: list[Path] = []

    if _visited is None:
        _visited = set()
    if _ignore_spec is None:
        _ignore_spec = PathSpec.from_lines('gitignore', ignore_patterns)
    if _allowed_roots is None:
        _allowed_roots = (directory.resolve(),)

    # Protect against symlink cycles by tracking visited directory inodes
    if follow_symlinks:
        try:
            stat = directory.stat()
            key  = (stat.st_dev, stat.st_ino)
            if key in _visited:
                log.warning('symlink_cycle_detected', directory=str(directory))
                return results
            _visited.add(key)
        except OSError:
            pass  # stat failure; will be caught below

    try:
        entries = sorted(directory.iterdir())
    except PermissionError:
        log.warning('permission_denied', directory=str(directory))
        return results
    except OSError as exc:
        log.warning('directory_read_error', directory=str(directory), error=str(exc))
        return results

    for entry in entries:
        # Skip symlinks unless configured to follow
        if entry.is_symlink() and not follow_symlinks:
            continue

        if entry.is_symlink() and follow_symlinks:
            with_context = {'entry': str(entry)}
            try:
                target = entry.resolve()
            except OSError as exc:
                log.warning('symlink_resolve_failed', error=str(exc), **with_context)
                continue
            if not _is_within_allowed_roots(target, _allowed_roots):
                log.warning(
                    'symlink_target_outside_scan_root',
                    entry=str(entry),
                    target=str(target),
                    roots=[str(root) for root in _allowed_roots],
                )
                continue

        # Check ignore patterns against the entry name and relative parts
        if should_ignore(entry, ignore_patterns, ignore_spec=_ignore_spec):
            continue

        if entry.is_dir():
            results.extend(
                _walk_directory(
                    entry,
                    ignore_patterns,
                    extensions,
                    follow_symlinks,
                    _visited,
                    _ignore_spec,
                    _allowed_roots,
                )
            )
        elif entry.is_file() and entry.suffix.lower() in extensions:
            results.append(entry)

    return results


def _is_within_allowed_roots(path: Path, roots: tuple[Path, ...]) -> bool:
    try:
        resolved = path.resolve()
    except OSError:
        return False
    for root in roots:
        try:
            if resolved.is_relative_to(root):
                return True
        except ValueError:
            continue
    return False


def _normalize_match_path(path: Path) -> str:
    # Convert to normalized relative-style POSIX path for pathspec matching.
    return path.as_posix().lstrip('/')


def should_ignore(
    path: Path,
    ignore_patterns: list[str],
    *,
    ignore_spec: PathSpec | None = None,
) -> bool:
    # Check if a path matches any of the ignore patterns.
    # Matches against the filename and each component of the path.
    # Public so watcher.py can reuse the same logic.
    if not ignore_patterns:
        return False

    spec = ignore_spec or PathSpec.from_lines('gitignore', ignore_patterns)
    return bool(spec.match_file(_normalize_match_path(path)))


# ==============================================================================
# Hashing
# ==============================================================================


def _compute_file_hash_and_stat(file_path: str) -> tuple[str, int, float] | None:
    # Compute SHA-256 hash, size, and mtime of a file in one pass.
    # Takes a string path for parallel executors.
    # Returns (hash, size_bytes, mtime) or None on error.
    sha256 = hashlib.sha256()
    try:
        stat_result = os.stat(file_path)
        size_bytes = stat_result.st_size
        mtime = stat_result.st_mtime
        max_hash_bytes = int(getattr(settings, 'scan_hash_max_file_size_bytes', 0) or 0)
        if max_hash_bytes > 0 and size_bytes > max_hash_bytes:
            pseudo_hash = hashlib.sha256(f'oversized:{size_bytes}:{mtime}'.encode()).hexdigest()
            log.warning(
                'scan_hash_skipped_oversized_file',
                path=file_path,
                size_bytes=size_bytes,
                max_hash_bytes=max_hash_bytes,
            )
            return pseudo_hash, size_bytes, mtime

        with open(file_path, 'rb') as f:  # noqa: ASYNC230
            while True:
                chunk = f.read(HASH_CHUNK_SIZE)
                if not chunk:
                    break
                sha256.update(chunk)
        return sha256.hexdigest(), size_bytes, mtime
    except OSError:
        return None


def _build_scanned_files(paths: list[Path]) -> list[ScannedFile]:
    # Build ScannedFile objects with hashes for all candidate paths.
    # Uses parallel hashing for batches larger than a threshold.
    parallel_threshold = 50
    hash_stage_start = time.perf_counter()

    if len(paths) > parallel_threshold:
        cpu_count = os.cpu_count() or 4
        max_workers = settings.scan_hash_workers or min(4, max(2, cpu_count // 3))
        pool_kind = settings.scan_hash_pool
        if pool_kind not in {'thread', 'process'}:
            pool_kind = 'thread'
        results = _build_scanned_files_parallel(paths, max_workers=max_workers, pool_kind=pool_kind)
        duration = time.perf_counter() - hash_stage_start
        files_per_second = len(paths) / max(duration, 1e-9)
        log.info(
            'scan_hash_stage_complete',
            mode=pool_kind,
            max_workers=max_workers,
            files=len(paths),
            duration_seconds=round(duration, 3),
            files_per_second=round(files_per_second, 1),
        )
        return results

    results = _build_scanned_files_sequential(paths)
    duration = time.perf_counter() - hash_stage_start
    files_per_second = len(paths) / max(duration, 1e-9)
    log.info(
        'scan_hash_stage_complete',
        mode='sequential',
        max_workers=1,
        files=len(paths),
        duration_seconds=round(duration, 3),
        files_per_second=round(files_per_second, 1),
    )
    return results


def _build_scanned_files_sequential(paths: list[Path]) -> list[ScannedFile]:
    # Build ScannedFile objects sequentially.
    results: list[ScannedFile] = []
    for path in paths:
        scanned = _path_to_scanned_file(path)
        if scanned is not None:
            results.append(scanned)
    return results


def _build_scanned_files_parallel(
    paths: list[Path],
    *,
    max_workers: int,
    pool_kind: str,
) -> list[ScannedFile]:
    # Build ScannedFile objects using parallel hash computation.
    # Compute hash, size, and mtime in one pass.
    path_strings = [str(p) for p in paths]

    executor_cls = ThreadPoolExecutor if pool_kind == 'thread' else ProcessPoolExecutor
    with executor_cls(max_workers=max_workers) as executor:
        hash_stat_results = list(executor.map(_compute_file_hash_and_stat, path_strings))

    results: list[ScannedFile] = []
    for path, result in zip(paths, hash_stat_results, strict=True):
        if result is None:
            log.warning('hash_failed', path=str(path))
            continue
        content_hash, size_bytes, mtime = result
        try:
            results.append(ScannedFile(
                path=path.resolve(),
                filename=path.name,
                extension=path.suffix.lower(),
                size_bytes=size_bytes,
                content_hash=content_hash,
                modified_at=datetime.fromtimestamp(mtime, tz=UTC),
            ))
        except OSError as exc:
            log.warning('file_processing_failed', path=str(path), error=str(exc))
    return results


def _path_to_scanned_file(path: Path) -> ScannedFile | None:
    # Convert a single Path to a ScannedFile, or None on error.
    result = _compute_file_hash_and_stat(str(path))
    if result is None:
        log.warning('scan_file_error', path=str(path), error='hash_or_stat_failed')
        return None
    content_hash, size_bytes, mtime = result
    try:
        return ScannedFile(
            path=path.resolve(),
            filename=path.name,
            extension=path.suffix.lower(),
            size_bytes=size_bytes,
            content_hash=content_hash,
            modified_at=datetime.fromtimestamp(mtime, tz=UTC),
        )
    except OSError as exc:
        log.warning('scan_file_error', path=str(path), error=str(exc))
        return None


def scanned_file_for_path(path: Path) -> ScannedFile | None:
    """
    Build a ScannedFile for a single path (hash + metadata).
    Used by the watcher for incremental indexing. Returns None on error.
    """
    return _path_to_scanned_file(path)


# ==============================================================================
# Change Detection
# ==============================================================================

def compare_with_db(
    scanned: list[ScannedFile],
    db_files: list[IndexedFile],
    *,
    source_provider: str = DEFAULT_SCAN_SOURCE_PROVIDER,
    entity_type: str = DEFAULT_SCAN_ENTITY_TYPE,
) -> ChangeSet:
    # Compare scanned files against database records.
    # Returns a ChangeSet with new, changed, unchanged, and deleted lists.
    #
    # Change detection is content-based only: we use SHA-256 content_hash.
    # We do not use size_bytes or modified_at to decide if a file changed.
    # That way we only re-index when file contents actually differ.
    # Uses normalized paths so symlinks / slight path differences don't cause
    # false "new" or "changed" classification.

    scoped_db_files = [
        f
        for f in db_files
        if f.source_provider == source_provider and f.entity_type == entity_type
    ]

    # Build a lookup from normalized path -> IndexedFile
    db_by_path: dict[str, IndexedFile] = {
        str(normalize_path(f.path)): f for f in scoped_db_files
    }

    # Set of normalized scanned paths for deletion detection
    scanned_paths_norm: set[str] = set()

    new_files:       list[ScannedFile] = []
    changed_files:   list[ScannedFile] = []
    unchanged_files: list[ScannedFile] = []

    for sf in scanned:
        path_norm = str(normalize_path(sf.path))
        scanned_paths_norm.add(path_norm)

        db_file = db_by_path.get(path_norm)
        if db_file is None:
            new_files.append(sf)
        elif db_file.content_hash != sf.content_hash:
            changed_files.append(sf)
        else:
            unchanged_files.append(sf)

    # Files in DB but not on disk = deleted (compare using normalized paths)
    deleted_files = [
        f for f in scoped_db_files
        if str(normalize_path(f.path)) not in scanned_paths_norm
    ]

    log.info(
        'change_detection_complete',
        source_provider=source_provider,
        entity_type=entity_type,
        new=len(new_files),
        changed=len(changed_files),
        unchanged=len(unchanged_files),
        deleted=len(deleted_files),
    )

    return ChangeSet(
        new=new_files,
        changed=changed_files,
        unchanged=unchanged_files,
        deleted=deleted_files,
    )
