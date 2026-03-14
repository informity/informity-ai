# ==============================================================================
# Informity AI — Adaptive RAG Tuning
# Corpus-aware top-k tuning. See .internal/features/adaptive-tuning.md.
# ==============================================================================

import math
from datetime import UTC, datetime
from typing import Any

import aiosqlite
import structlog

from informity.config import settings
from informity.db.sqlite import get_corpus_stats
from informity.llm.model_adapter import get_profile

log = structlog.get_logger(__name__)
_ADAPTIVE_TUNING_EXCEPTIONS = (aiosqlite.Error, RuntimeError, ValueError, TypeError, OSError)

# ==============================================================================
# In-Memory Cache (sync-accessible by get_retrieval_top_k)
# ==============================================================================

_cached_focused_top_k: int | None   = None
_cached_coverage_top_k: int | None  = None
_cache_valid: bool                  = False
_parent_chunks_at_compute: int      = 0
_files_at_compute: int             = 0
_last_computed_at: datetime | None  = None
_profile_name_at_compute: str | None = None


# ==============================================================================
# Formula
# ==============================================================================

def calculate_adaptive_top_k(
    total_files:         int,
    total_parent_chunks: int,
    query_type:          str,
    profile_base:        int,
) -> int:
    """
    Compute adaptive top-k from corpus stats and profile base.

    Uses config constants (no magic numbers). Falls back to profile_base
    when corpus is empty or stats unavailable.
    """
    s = settings
    if query_type == 'coverage':
        if total_files <= 0:
            return profile_base
        adaptive = min(
            int(total_files * s.adaptive_top_k_coverage_ratio),
            s.adaptive_top_k_coverage_max,
        )
        return max(adaptive, profile_base)

    # Focused
    if total_parent_chunks <= 0:
        return profile_base
    threshold = s.adaptive_top_k_focused_small_threshold
    small_cap  = s.adaptive_top_k_focused_small_cap
    base       = s.adaptive_top_k_focused_base
    scale      = s.adaptive_top_k_focused_scale
    max_k      = s.adaptive_top_k_focused_max

    if total_parent_chunks < threshold:
        return max(10, min(profile_base, small_cap))

    adaptive = base + int(math.log2(max(total_parent_chunks / 100, 1)) * scale)
    return max(profile_base, min(adaptive, max_k))


def _is_cache_stale(stats: dict[str, Any]) -> bool:
    """Return True if cache should be recomputed."""
    if not _cache_valid or _last_computed_at is None:
        return True
    current_profile_name = get_profile().name
    if _profile_name_at_compute != current_profile_name:
        return True
    delta_hours = (datetime.now(UTC) - _last_computed_at).total_seconds() / 3600
    if delta_hours >= settings.adaptive_top_k_staleness_hours:
        return True
    cached_parents = _parent_chunks_at_compute
    current_parents = stats.get('total_parent_chunks', 0) or 0
    if cached_parents <= 0:
        return current_parents != cached_parents
    parent_delta_ratio = abs(current_parents - cached_parents) / cached_parents

    cached_files = _files_at_compute
    current_files = stats.get('total_files', 0) or 0
    if cached_files <= 0:
        files_stale = current_files != cached_files
    else:
        files_delta_ratio = abs(current_files - cached_files) / cached_files
        files_stale = files_delta_ratio > settings.adaptive_top_k_staleness_delta

    return parent_delta_ratio > settings.adaptive_top_k_staleness_delta or files_stale


# ==============================================================================
# Cache Update (async)
# ==============================================================================

async def update_tuning_cache(
    db: aiosqlite.Connection,
    force_recompute: bool = False,
) -> None:
    """
    Recompute adaptive top-k from corpus stats and update in-memory cache.
    Called at startup, scan completion, rebuild completion. Invalidates on reset.

    Args:
        db: Active database connection.
        force_recompute: When True, bypass staleness checks and recompute now.
    """
    global _cached_focused_top_k, _cached_coverage_top_k, _cache_valid
    global _parent_chunks_at_compute, _files_at_compute, _last_computed_at, _profile_name_at_compute

    if not settings.adaptive_rag_tuning:
        _cache_valid = False
        return

    try:
        stats = await get_corpus_stats(db)
        total_files        = stats.get('total_files', 0) or 0
        total_parent_chunks = stats.get('total_parent_chunks', 0) or 0

        if not force_recompute and not _is_cache_stale(stats):
            return

        profile = get_profile()
        focused  = calculate_adaptive_top_k(
            total_files, total_parent_chunks, 'focused', profile.rag_top_k
        )
        coverage = calculate_adaptive_top_k(
            total_files, total_parent_chunks, 'coverage', profile.coverage_top_k
        )

        _cached_focused_top_k   = focused
        _cached_coverage_top_k  = coverage
        _parent_chunks_at_compute = total_parent_chunks
        _files_at_compute       = total_files
        _last_computed_at       = datetime.now(UTC)
        _profile_name_at_compute = profile.name
        _cache_valid            = True

        log.debug(
            'adaptive_tuning_updated',
            force_recompute=force_recompute,
            focused_top_k=focused,
            coverage_top_k=coverage,
            profile_name=profile.name,
            total_files=total_files,
            total_parent_chunks=total_parent_chunks,
        )
    except _ADAPTIVE_TUNING_EXCEPTIONS as exc:
        log.warning('adaptive_tuning_update_failed', error=str(exc))
        _cache_valid = False


def invalidate_tuning_cache() -> None:
    """Clear cache. Call on index reset."""
    global _cached_focused_top_k, _cached_coverage_top_k, _cache_valid
    global _parent_chunks_at_compute, _files_at_compute, _last_computed_at, _profile_name_at_compute
    _cached_focused_top_k   = None
    _cached_coverage_top_k  = None
    _cache_valid            = False
    _parent_chunks_at_compute = 0
    _files_at_compute       = 0
    _last_computed_at       = None
    _profile_name_at_compute = None
    log.debug('adaptive_tuning_cache_invalidated')


def get_effective_top_k(query_type: str) -> int | None:
    """
    Return cached adaptive top-k if valid and enabled; else None (caller uses profile).
    Sync, for use by get_retrieval_top_k.
    """
    if not settings.adaptive_rag_tuning or not _cache_valid:
        return None
    if query_type == 'coverage' and _cached_coverage_top_k is not None:
        return _cached_coverage_top_k
    if query_type == 'focused' and _cached_focused_top_k is not None:
        return _cached_focused_top_k
    return None
