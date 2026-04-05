# ==============================================================================
# Informity AI — Metrics Payload Builder
# Shared helper for runtime metrics payload shape consistency.
# ==============================================================================

from __future__ import annotations

from typing import Any


def build_metrics_payload(
    *,
    query_type: str,
    raw_chunks_count: int,
    **optional_fields: Any,
) -> dict[str, object]:
    payload: dict[str, object] = {
        'query_type': query_type,
        'raw_chunks_count': int(raw_chunks_count),
    }
    for key, value in optional_fields.items():
        if value is not None:
            payload[key] = value
    return payload
