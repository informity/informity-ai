# ==============================================================================
# Informity AI — File Classifier (v2)
# Simple extension-based classification and year extraction
# ==============================================================================

from pathlib import Path

from informity.category_patterns import get_category_for_extension
from informity.db.models import FileCategory
from informity.file_patterns import YEAR_PATTERN

_YEAR_EXTRACTION_TEXT_LIMIT = 1000
_SKIP_DIRS: frozenset[str] = frozenset({
    'library', 'applications', 'system', 'users', 'home',
    '.git', '.venv', '.env', 'node_modules', '__pycache__',
    'desktop', 'documents', 'downloads', 'movies', 'music', 'pictures',
})


def classify_file(path: Path, extension: str) -> FileCategory:
    """
    Classify file by extension using centralized mapping.

    Uses category_patterns.get_category_for_extension() as single source of truth.
    """
    return get_category_for_extension(extension)


def extract_year(path: Path, text: str) -> int | None:
    """
    Extract year from filename/path/text using standardized pattern.

    Tries filename first, then path, then text (first 1000 chars).

    Args:
        path: File path
        text: Extracted text content

    Returns:
        Year as integer if found, None otherwise
    """
    # Try filename first
    year_match = YEAR_PATTERN.search(path.name)
    if year_match:
        return int(year_match.group(0))

    # Try path
    year_match = YEAR_PATTERN.search(str(path))
    if year_match:
        return int(year_match.group(0))

    # Try text (first 1000 chars)
    year_match = YEAR_PATTERN.search(text[:_YEAR_EXTRACTION_TEXT_LIMIT])
    if year_match:
        return int(year_match.group(0))

    return None


def generate_tags(path: Path) -> list[str]:
    # Generate tags from directory path components.
    # Extracts meaningful directory names, filters system dirs, normalizes.
    tags: list[str] = []

    # Extract directory components
    parts = path.parent.parts

    for part in parts:
        # Skip empty, single-char, or system dirs
        if len(part) <= 1 or part.lower() in _SKIP_DIRS:
            continue

        # Normalize: lowercase, replace spaces/hyphens with underscores
        tag = part.lower().replace(' ', '_').replace('-', '_')

        # Skip if already added or invalid
        if tag and tag not in tags:
            tags.append(tag)

        # Limit to 5 tags, prefer deeper directories
        if len(tags) >= 5:
            break

    return tags
