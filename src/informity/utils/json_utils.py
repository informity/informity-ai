# ==============================================================================
# Informity AI — JSON Utilities (v2)
# Standardized JSON serialization patterns
# ==============================================================================

import json
from typing import Any

_JSON_INDENT = 2


def _serialize_with_default_str(data: dict[str, Any], *, ensure_ascii: bool = True) -> str:
    # Use only for non-API persisted artifacts where string coercion is acceptable.
    return json.dumps(data, indent=_JSON_INDENT, default=str, ensure_ascii=ensure_ascii)


def serialize_config(data: dict[str, Any]) -> str:
    """
    Serialize configuration data to JSON string.

    Used for config.json files with human-readable formatting.

    Args:
        data: Dictionary to serialize

    Returns:
        JSON string with 2-space indent, default=str for non-serializable types
    """
    return _serialize_with_default_str(data) + '\n'


def serialize_trace(data: dict[str, Any]) -> str:
    """
    Serialize trace data to JSON string.

    Used for diagnostic trace files with full Unicode support.

    Args:
        data: Dictionary to serialize

    Returns:
        JSON string with 2-space indent, ensure_ascii=False, default=str
    """
    return _serialize_with_default_str(data, ensure_ascii=False)


def serialize_api_response(data: dict[str, Any]) -> str:
    """
    Serialize API response data to JSON string.

    Used for SSE events and API responses (compact format).

    Args:
        data: Dictionary to serialize

    Returns:
        Compact JSON string (no indent)
    """
    try:
        return json.dumps(data)
    except TypeError as exc:
        raise TypeError(
            'serialize_api_response requires JSON-serializable payload values '
            '(convert non-JSON types before serialization).',
        ) from exc
