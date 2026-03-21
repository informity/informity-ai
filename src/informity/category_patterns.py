# ==============================================================================
# Informity AI — Category Patterns (v2)
# Single source of truth for extension→category mapping
# Used by indexer (file storage) and query classifier (metadata queries only)
# ==============================================================================

from informity.db.models import FileCategory

# ==============================================================================
# Extension→Category Mapping (Canonical)
# ==============================================================================

# Extension→Category mapping (single source of truth, immutable)
# Categories reflect how files are processed, not philosophical classification:
# - DOCUMENT: Rich format documents (PDF, Word, PowerPoint)
# - PLAINTEXT: Text files read as-is for semantic search (including config formats)
# - DATA: Tabular/structured data (CSV, Excel)
# - WEB: Web pages (HTML)
# - CODE: Source code (future)
# - OTHER: Unsupported extensions
EXTENSION_CATEGORY_MAP: dict[str, FileCategory] = {
    # Document files (rich formats)
    '.pdf':   FileCategory.DOCUMENT,
    '.docx':  FileCategory.DOCUMENT,
    '.pptx':  FileCategory.DOCUMENT,

    # Plaintext files (including config formats read as text for RAG)
    # Note: .json, .yaml, .yml, .toml are extracted as plain text (TextExtractor)
    # and searched semantically, so they're classified as PLAINTEXT not DATA
    '.txt':   FileCategory.PLAINTEXT,
    '.md':    FileCategory.PLAINTEXT,
    '.rst':   FileCategory.PLAINTEXT,
    '.log':   FileCategory.PLAINTEXT,
    '.json':  FileCategory.PLAINTEXT,  # Config files, read as text for RAG
    '.yaml':  FileCategory.PLAINTEXT,  # Config files, read as text for RAG
    '.yml':   FileCategory.PLAINTEXT,  # Config files, read as text for RAG
    '.toml':  FileCategory.PLAINTEXT,  # Config files, read as text for RAG

    # Data files (tabular/structured formats)
    '.csv':   FileCategory.DATA,
    '.xlsx':  FileCategory.DATA,

    # Web files
    '.html':  FileCategory.WEB,
    '.htm':   FileCategory.WEB,
}


def get_category_for_extension(extension: str) -> FileCategory:
    """
    Get category for a file extension using centralized mapping.

    This is the single source of truth for extension→category classification.
    Used by:
    - indexer/classifier.py (at index time)
    - llm/query_classifier.py (for deterministic slot extraction)

    Args:
        extension: File extension (e.g., '.pdf', '.md', 'txt')

    Returns:
        FileCategory enum value, defaults to OTHER if extension not recognized
    """
    # Normalize extension (ensure lowercase, starts with dot)
    ext = extension.lower()
    if not ext.startswith('.'):
        ext = f'.{ext}'

    return EXTENSION_CATEGORY_MAP.get(ext, FileCategory.OTHER)
