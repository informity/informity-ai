"""Upload policy constants and validation helpers."""

from __future__ import annotations

from pathlib import Path

from informity.config import settings

UPLOAD_PROVIDER = "upload.local"
UPLOAD_ENTITY_TYPE = "file"
UPLOAD_STORAGE_DIRNAME = "storage/uploads"

MAX_UPLOAD_FILE_SIZE_MB = 50
MAX_UPLOAD_FILES_PER_CHAT = 10
MAX_UPLOAD_TOTAL_SIZE_MB = 200


ALLOWED_MIME_PREFIXES: tuple[str, ...] = (
    "text/",
    "application/pdf",
    "application/epub+zip",
    "application/json",
    "application/xml",
    "application/vnd.openxmlformats-officedocument",
    "application/vnd.ms-excel",
    "application/vnd.ms-powerpoint",
    "application/msword",
)


def upload_root_dir() -> Path:
    """Return the root directory used for uploaded files."""
    return settings.app_data_dir / UPLOAD_STORAGE_DIRNAME


def max_upload_file_size_bytes() -> int:
    """Return the per-file upload size limit in bytes."""
    return MAX_UPLOAD_FILE_SIZE_MB * 1024 * 1024


def max_upload_total_size_bytes() -> int:
    """Return the total upload size limit in bytes."""
    return MAX_UPLOAD_TOTAL_SIZE_MB * 1024 * 1024


def _extract_extension(filename: str) -> str:
    """Extract and normalize a file extension."""
    ext = Path(str(filename or "")).suffix.lower().strip()
    if not ext:
        return ""
    if not ext.startswith("."):
        return f".{ext}"
    return ext


def allowed_extensions() -> set[str]:
    """Return the allowed file extensions for uploads."""
    configured = {
        normalized
        for ext in (settings.supported_extensions or [])
        if (normalized := str(ext).strip().lower())
    }
    # Keep text as safe fallback even if settings are misconfigured.
    configured.update({".txt", ".md"})
    return configured


def is_allowed_extension(filename: str) -> bool:
    """Return whether the uploaded filename has an allowed extension."""
    ext = _extract_extension(filename)
    if not ext:
        return False
    return ext in allowed_extensions()


def is_allowed_mime(content_type: str | None) -> bool:
    """Return whether the upload MIME type is allowed."""
    value = str(content_type or "").strip().lower()
    if not value:
        return True
    base_mime = value.split(";", 1)[0].strip()
    if not base_mime:
        return True
    for allowed in ALLOWED_MIME_PREFIXES:
        normalized_allowed = str(allowed).strip().lower().rstrip("/")
        if not normalized_allowed:
            continue
        if base_mime.startswith(normalized_allowed):
            return True
    return False
