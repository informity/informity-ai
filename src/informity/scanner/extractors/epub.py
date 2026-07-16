# ==============================================================================
# Informity AI — EPUB Extractor (v2)
# Handles EPUB ebook extraction using EbookLib.
# ==============================================================================

"""Module for scanner extractors epub."""

from __future__ import annotations

import re
import time
import zipfile
from html.parser import HTMLParser
from pathlib import Path

import structlog

from informity.scanner.extractors.base import MAX_EXTRACTED_TEXT_PREVIEW, ExtractedDocument
from informity.scanner.extractors.boundary_rules import join_structured_blocks
from informity.scanner.extractors.text_utils import elapsed_ms, get_max_file_size_bytes

log = structlog.get_logger(__name__)

_EPUB_SUPPORTED_EXTENSIONS = [".epub"]


class _HTMLTextExtractor(HTMLParser):
    """Small HTML-to-text extractor with conservative block/newline handling."""

    _BLOCK_TAGS = {
        "p",
        "div",
        "section",
        "article",
        "header",
        "footer",
        "nav",
        "aside",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "li",
        "ul",
        "ol",
        "blockquote",
        "table",
        "tr",
        "th",
        "td",
        "br",
    }
    _SKIP_TAGS = {"script", "style"}

    def __init__(self) -> None:
        """Initialize the instance."""
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """Handle starttag."""
        del attrs
        tag_lower = tag.lower()
        if tag_lower in self._SKIP_TAGS:
            self._skip_depth += 1
            return
        if tag_lower in self._BLOCK_TAGS:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        """Handle endtag."""
        tag_lower = tag.lower()
        if tag_lower in self._SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1
            return
        if tag_lower in self._BLOCK_TAGS:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        """Handle data."""
        if self._skip_depth > 0:
            return
        if data and data.strip():
            self._parts.append(data.strip())

    def get_text(self) -> str:
        """Get text."""
        text = " ".join(self._parts)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r" *\n+ *", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()


def _classify_epub_exception(exc: Exception) -> tuple[str, bool]:
    """Internal helper for classify epub exception."""
    error_str = str(exc).lower()
    if "encrypted" in error_str or "drm" in error_str:
        return "epub_encrypted_or_drm", False
    if "zip" in error_str or "archive" in error_str or "container" in error_str:
        return "epub_invalid_or_corrupt", False
    return "epub_extraction_error", True


class EpubExtractor:
    """Class docstring."""
    supported_extensions: list[str] = _EPUB_SUPPORTED_EXTENSIONS

    def can_handle(self, path: Path) -> bool:
        """Can handle."""
        return path.suffix.lower() in self.supported_extensions

    def extract(self, path: Path) -> ExtractedDocument:
        """Extract."""
        start_time = time.perf_counter()
        path_str = str(path)
        file_size = 0
        try:
            file_size = path.stat().st_size
            if file_size == 0:
                doc = ExtractedDocument(
                    text="",
                    source_path=path,
                    extraction_time_ms=elapsed_ms(start_time),
                    preview_text="",
                    error="File is empty",
                )
                log.info(
                    "epub_extraction_result",
                    path=path_str,
                    status="error",
                    error_code="file_empty",
                    retryable=False,
                    extraction_time_ms=doc.extraction_time_ms,
                    file_size_bytes=file_size,
                )
                return doc
            if file_size > get_max_file_size_bytes():
                doc = ExtractedDocument(
                    text="",
                    source_path=path,
                    extraction_time_ms=elapsed_ms(start_time),
                    preview_text="",
                    error=f"File too large: {file_size} bytes",
                )
                log.info(
                    "epub_extraction_result",
                    path=path_str,
                    status="error",
                    error_code="file_too_large",
                    retryable=False,
                    extraction_time_ms=doc.extraction_time_ms,
                    file_size_bytes=file_size,
                )
                return doc

            try:
                import ebooklib
                from ebooklib import epub
            except ImportError as exc:
                doc = ExtractedDocument(
                    text="",
                    source_path=path,
                    metadata={
                        "error_code": "epub_dependency_missing",
                        "retryable": "false",
                    },
                    extraction_time_ms=elapsed_ms(start_time),
                    preview_text="",
                    error=f"EPUB dependency not available: {exc}",
                )
                log.info(
                    "epub_extraction_result",
                    path=path_str,
                    status="error",
                    error_code="epub_dependency_missing",
                    retryable=False,
                    extraction_time_ms=doc.extraction_time_ms,
                    file_size_bytes=file_size,
                )
                return doc

            epub_exception_types: tuple[type[BaseException], ...] = (
                AttributeError,
                KeyError,
                OSError,
                RuntimeError,
                ValueError,
                zipfile.BadZipFile,
            )
            epub_exception = getattr(epub, "EpubException", None)
            if isinstance(epub_exception, type) and issubclass(epub_exception, BaseException):
                epub_exception_types += (epub_exception,)

            try:
                book = epub.read_epub(str(path))
            except epub_exception_types as exc:
                error_code, retryable = _classify_epub_exception(exc)
                doc = ExtractedDocument(
                    text="",
                    source_path=path,
                    metadata={
                        "error_code": error_code,
                        "retryable": "false" if not retryable else "true",
                    },
                    extraction_time_ms=elapsed_ms(start_time),
                    preview_text="",
                    error=f"EPUB extraction failed: {exc}",
                )
                log.info(
                    "epub_extraction_result",
                    path=path_str,
                    status="error",
                    error_code=error_code,
                    retryable=retryable,
                    extraction_time_ms=doc.extraction_time_ms,
                    file_size_bytes=file_size,
                )
                return doc

            text_parts: list[str] = []
            chapter_count = 0

            for item in book.get_items():
                if item.get_type() not in {ebooklib.ITEM_DOCUMENT}:
                    continue
                chapter_count += 1
                parser = _HTMLTextExtractor()
                content = item.get_content()
                html = (
                    content.decode("utf-8", errors="replace")
                    if isinstance(content, bytes)
                    else str(content)
                )
                parser.feed(html)
                parsed = parser.get_text()
                if parsed:
                    text_parts.append(parsed)

            text = join_structured_blocks(text_parts)
            word_count = len(text.split()) if text else 0

            # Metadata keys are library-defined tuples in EbookLib.
            title_value = ""
            language_value = ""
            titles = book.get_metadata("DC", "title")
            if titles and titles[0]:
                title_value = str(titles[0][0] or "")
            languages = book.get_metadata("DC", "language")
            if languages and languages[0]:
                language_value = str(languages[0][0] or "")

            metadata = {
                "converter": "ebooklib",
                "mime_type": "application/epub+zip",
                "chapter_count": str(chapter_count),
            }
            if title_value:
                metadata["title"] = title_value
            if language_value:
                metadata["language"] = language_value

            if not text:
                doc = ExtractedDocument(
                    text="",
                    source_path=path,
                    metadata={
                        **metadata,
                        "error_code": "epub_no_text_extracted",
                        "retryable": "false",
                    },
                    word_count=0,
                    extraction_time_ms=elapsed_ms(start_time),
                    preview_text="",
                    error="EPUB extracted no readable text",
                )
                log.info(
                    "epub_extraction_result",
                    path=path_str,
                    status="error",
                    error_code="epub_no_text_extracted",
                    retryable=False,
                    extraction_time_ms=doc.extraction_time_ms,
                    file_size_bytes=file_size,
                    chapter_count=chapter_count,
                )
                return doc

            doc = ExtractedDocument(
                text=text,
                source_path=path,
                metadata=metadata,
                word_count=word_count,
                extraction_time_ms=elapsed_ms(start_time),
                preview_text=text[:MAX_EXTRACTED_TEXT_PREVIEW],
                error=None,
            )
            log.info(
                "epub_extraction_result",
                path=path_str,
                status="success",
                extraction_time_ms=doc.extraction_time_ms,
                file_size_bytes=file_size,
                word_count=word_count,
                chapter_count=chapter_count,
            )
            return doc
        except OSError as exc:
            doc = ExtractedDocument(
                text="",
                source_path=path,
                extraction_time_ms=elapsed_ms(start_time),
                preview_text="",
                error=f"Failed to read file: {exc}",
            )
            log.info(
                "epub_extraction_result",
                path=path_str,
                status="error",
                error_code="file_io_error",
                retryable=True,
                extraction_time_ms=doc.extraction_time_ms,
                file_size_bytes=file_size,
            )
            return doc
