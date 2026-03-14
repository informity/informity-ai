# ==============================================================================
# Informity AI — Runtime Resource Snapshot
# Lightweight CPU/RAM snapshots for per-query diagnostics and troubleshooting.
# ==============================================================================

from __future__ import annotations

import time
from typing import Any

import psutil

_RESOURCE_SNAPSHOT_EXCEPTIONS = (psutil.Error, OSError)


def _round_metric(value: float | int | None, digits: int = 2) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def capture_resource_snapshot() -> dict[str, Any]:
    """
    Capture a compact CPU/RAM snapshot for diagnostics.

    Returns a resilient payload; failures are represented as capture_error.
    """
    try:
        process = psutil.Process()
        process_mem = process.memory_info()
        virtual_mem = psutil.virtual_memory()
        return {
            'captured_at_epoch_ms': int(time.time() * 1000),
            'system_cpu_percent': _round_metric(psutil.cpu_percent(interval=None)),
            'process_cpu_percent': _round_metric(process.cpu_percent(interval=None)),
            'process_rss_mb': _round_metric(process_mem.rss / (1024 * 1024)),
            'process_vms_mb': _round_metric(process_mem.vms / (1024 * 1024)),
            'system_memory_used_percent': _round_metric(virtual_mem.percent),
            'system_memory_available_mb': _round_metric(virtual_mem.available / (1024 * 1024)),
            'system_memory_used_mb': _round_metric(virtual_mem.used / (1024 * 1024)),
            'logical_cpu_count': psutil.cpu_count(logical=True),
        }
    except _RESOURCE_SNAPSHOT_EXCEPTIONS as exc:
        return {
            'captured_at_epoch_ms': int(time.time() * 1000),
            'capture_error': str(exc),
        }


def build_resource_delta(
    *,
    before: dict[str, Any] | None,
    after: dict[str, Any] | None,
) -> dict[str, float]:
    """Compute useful before/after deltas for key process/system metrics."""
    if not isinstance(before, dict) or not isinstance(after, dict):
        return {}

    delta_map: dict[str, float] = {}
    tracked_keys = (
        'system_cpu_percent',
        'process_cpu_percent',
        'process_rss_mb',
        'process_vms_mb',
        'system_memory_used_percent',
        'system_memory_available_mb',
        'system_memory_used_mb',
    )
    for key in tracked_keys:
        before_value = before.get(key)
        after_value = after.get(key)
        if isinstance(before_value, (int, float)) and isinstance(after_value, (int, float)):
            delta_map[f'{key}_delta'] = round(float(after_value) - float(before_value), 2)
    return delta_map
