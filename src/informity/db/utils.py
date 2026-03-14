# ==============================================================================
# Informity AI — Database Utilities (v2)
# Shared utilities for database operations: timestamp parsing, row conversion helpers
# ==============================================================================

import json
from datetime import UTC, datetime

import structlog

from informity.db.models import FileCategory

log = structlog.get_logger(__name__)
_JSON_FALLBACK_PREVIEW_CHARS = 160


# ==============================================================================
# Timestamp Parsing
# ==============================================================================

def parse_timestamp(value: str | datetime | None) -> datetime | None:
    """
    Parse a SQLite timestamp string into a datetime object.

    Handles multiple timestamp formats:
    - '%Y-%m-%d %H:%M:%S'
    - '%Y-%m-%dT%H:%M:%S'
    - '%Y-%m-%dT%H:%M:%S.%f'
    - ISO format (fromisoformat)

    Args:
        value: Timestamp string, datetime object, or None

    Returns:
        datetime object with UTC timezone, or None if parsing fails
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    for fmt in ('%Y-%m-%d %H:%M:%S', '%Y-%m-%dT%H:%M:%S', '%Y-%m-%dT%H:%M:%S.%f'):
        try:
            return datetime.strptime(value, fmt).replace(tzinfo=UTC)
        except ValueError:
            continue
    try:
        dt = datetime.fromisoformat(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return dt
    except ValueError:
        log.warning('unparseable_timestamp', value=value)
        return None


# ==============================================================================
# Row Conversion Helpers
# ==============================================================================

def parse_file_category(value: str | None) -> FileCategory:
    """
    Parse FileCategory from database value with safe fallback.

    Args:
        value: Category string from database (e.g., 'document', 'plaintext')

    Returns:
        FileCategory enum value, defaults to FileCategory.OTHER if invalid
    """
    if not value:
        return FileCategory.OTHER
    try:
        return FileCategory(value)
    except (ValueError, KeyError):
        return FileCategory.OTHER


def parse_json_tags(value: str | None) -> list[str]:
    """
    Parse JSON tags from database with safe fallback.

    Args:
        value: JSON string from database (e.g., '["tag1", "tag2"]')

    Returns:
        List of tag strings, empty list if parsing fails or invalid
    """
    if not value:
        return []
    try:
        tags = json.loads(value)
        if not isinstance(tags, list):
            log.warning(
                'db_json_parse_fallback',
                field='tags',
                reason='not_list',
                value_preview=value[:_JSON_FALLBACK_PREVIEW_CHARS],
            )
            return []
        return tags
    except (json.JSONDecodeError, TypeError) as exc:
        log.warning(
            'db_json_parse_fallback',
            field='tags',
            reason='invalid_json',
            error=str(exc),
            value_preview=str(value)[:_JSON_FALLBACK_PREVIEW_CHARS],
        )
        return []


def parse_json_sources(value: str | None) -> list[dict]:
    """
    Parse JSON sources from database with safe fallback.

    Args:
        value: JSON string from database (e.g., '[{"filename": "foo.pdf", ...}]')

    Returns:
        List of source dictionaries, empty list if parsing fails or invalid
    """
    if not value:
        return []
    try:
        sources = json.loads(value)
        if not isinstance(sources, list):
            log.warning(
                'db_json_parse_fallback',
                field='sources',
                reason='not_list',
                value_preview=value[:_JSON_FALLBACK_PREVIEW_CHARS],
            )
            return []
        return sources
    except (json.JSONDecodeError, TypeError) as exc:
        log.warning(
            'db_json_parse_fallback',
            field='sources',
            reason='invalid_json',
            error=str(exc),
            value_preview=str(value)[:_JSON_FALLBACK_PREVIEW_CHARS],
        )
        return []
