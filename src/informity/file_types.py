# ==============================================================================
# Informity AI — File Type Options (canonical)
# Single source of truth for file categories and extensions used by:
# - Settings (file types to index), Files filtering/display, dashboard.
# Extensions must match backend extractors and classifier where applicable.
# ==============================================================================

"""Canonical file type option lists used throughout the app."""

from __future__ import annotations

# ==============================================================================
# File type options: id, label, extensions
# ==============================================================================

FILE_TYPE_OPTIONS: list[dict[str, str | list[str]]] = [
    {"id": "pdf", "label": "PDF Documents", "extensions": [".pdf"]},
    {
        "id": "image",
        "label": "Images",
        "extensions": [".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"],
    },
    {
        "id": "docx",
        "label": "Word Documents",
        "extensions": [".docx"],
    },
    {
        "id": "spreadsheet",
        "label": "Spreadsheets",
        "extensions": [".xlsx", ".csv"],
    },
    {
        "id": "pptx",
        "label": "PowerPoint Presentations",
        "extensions": [".pptx"],
    },
    {
        "id": "epub",
        "label": "EPUB E-books",
        "extensions": [".epub"],
    },
    {
        "id": "web",
        "label": "Web Pages",
        "extensions": [".html", ".htm"],
    },
    {
        "id": "text",
        "label": "Text and Markdown Files",
        "extensions": [".txt", ".md", ".rst", ".log"],
    },
    {
        "id": "data",
        "label": "Data and Configuration Files",
        "extensions": [".json", ".yaml", ".yml", ".toml"],
    },
]

SUPPORTED_EXTENSIONS_CANONICAL_ORDER: tuple[str, ...] = tuple(
    ext for option in FILE_TYPE_OPTIONS for ext in option["extensions"]
)

PLAINTEXT_EXTENSIONS: tuple[str, ...] = (
    ".txt",
    ".md",
    ".rst",
    ".log",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
)


def get_file_type_options() -> list[dict[str, str | list[str]]]:
    """Return the canonical list of file type options (id, label, extensions)."""
    # Deep-copy extensions lists so callers cannot mutate the module constant.
    return [{**opt, "extensions": list(opt["extensions"])} for opt in FILE_TYPE_OPTIONS]
