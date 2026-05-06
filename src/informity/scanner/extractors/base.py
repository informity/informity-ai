# ==============================================================================
# Informity AI — Base Extractor (v2)
# Defines the ExtractedDocument dataclass and BaseExtractor protocol.
# Simplified v2 version.
# ==============================================================================

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

import structlog

log = structlog.get_logger(__name__)

# Extracted text preview constants
MAX_EXTRACTED_TEXT_PREVIEW = 500  # Maximum length of extracted text preview (first N chars)

# ==============================================================================
# ExtractedDocument — output of any extractor
# ==============================================================================

@dataclass(frozen=True)
class ExtractedDocument:
    # Output of any extractor. Immutable value object.
    text:                 str                         # Full extracted text
    source_path:          Path                        # Absolute path to source file
    metadata:             dict[str, str] = field(default_factory=dict)  # Format-specific metadata
    page_count:           int | None     = None       # For PDFs, PPTX
    word_count:           int            = 0          # Computed from text
    extraction_time_ms:   float          = 0.0        # How long extraction took
    error:                str | None     = None       # Non-fatal extraction warnings
    preview_text:         str            = ''         # Clean preview text (first 500 chars, no markdown noise)
    # Per-chunk metadata mappings (for docling formats with provenance, primarily PDFs/PPTX)
    # Range-based storage: list of (start, end, value) tuples for memory efficiency
    # O(N items) instead of O(N characters) - typically 100-1000x fewer entries
    char_to_page_ranges:     list[tuple[int, int, int]] | None = None  # (start, end, page_no) ranges
    char_to_block_type_ranges: list[tuple[int, int, str]] | None = None  # (start, end, block_type) ranges
    char_to_header_level_ranges: list[tuple[int, int, int]] | None = None  # (start, end, header_level) ranges


# ==============================================================================
# BaseExtractor — protocol that all extractors must implement
# ==============================================================================

@runtime_checkable
class BaseExtractor(Protocol):
    # Protocol that all extractors must implement.
    # Extractors must never raise exceptions — return errors in
    # ExtractedDocument.error instead.
    supported_extensions: list[str]

    def extract(self, path: Path) -> ExtractedDocument:
        # Extract text content from a file.
        # Must not raise — return error in ExtractedDocument.error.
        ...

    def can_handle(self, path: Path) -> bool:
        # Check if this extractor supports the given file.
        ...


# ==============================================================================
# Extractor Registry
# ==============================================================================

EXTRACTOR_REGISTRY: dict[str, BaseExtractor] = {}
_registry_initialized = False


def register_extractors() -> None:
    # Instantiate all extractors and register them by supported extension.
    global _registry_initialized

    if _registry_initialized:
        return

    from informity.scanner.extractors.docling import DoclingExtractor
    from informity.scanner.extractors.epub import EpubExtractor
    from informity.scanner.extractors.text import TextExtractor

    extractor_classes = [
        DoclingExtractor,  # Unified extractor for PDF, DOCX, PPTX, XLSX, HTML, CSV
        EpubExtractor,     # EPUB ebooks
        TextExtractor,      # Plain text files (.txt, .md, .rst, .log)
    ]

    for extractor_class in extractor_classes:
        extractor = extractor_class()
        for ext in extractor.supported_extensions:
            EXTRACTOR_REGISTRY[ext] = extractor

    _registry_initialized = True
    log.info('extractors_registered', count=len(extractor_classes), extractors=[cls.__name__ for cls in extractor_classes])


def get_extractor(path: Path) -> BaseExtractor | None:
    # Look up the appropriate extractor for a file by its extension.
    if not _registry_initialized:
        register_extractors()
    return EXTRACTOR_REGISTRY.get(path.suffix.lower())


def get_all_extractable_extensions() -> list[str]:
    # Get all file extensions that have extractors available.
    register_extractors()
    return sorted(EXTRACTOR_REGISTRY.keys())
