# ==============================================================================
# Informity AI — JSON Utilities (v2)
# Standardized JSON serialization patterns
# ==============================================================================

import json
from typing import Any

import structlog

log = structlog.get_logger(__name__)
_JSON_FALLBACK_PREVIEW_CHARS = 160
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


def parse_json_safe(value: str | None, default: Any = None) -> Any:
    """
    Parse JSON string with safe fallback.

    Args:
        value: JSON string to parse
        default: Default value to return if parsing fails (default: None)

    Returns:
        Parsed JSON value or default if parsing fails
    """
    if not value:
        return default
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError) as exc:
        log.warning(
            'json_parse_safe_fallback',
            error=str(exc),
            value_preview=str(value)[:_JSON_FALLBACK_PREVIEW_CHARS],
            default_type=type(default).__name__,
        )
        return default
