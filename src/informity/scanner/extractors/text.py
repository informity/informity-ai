# ==============================================================================
# Informity AI — Text Extractor (v2)
# Handles plain text files not supported by docling: .txt, .md, .rst, .log
# Optimized for simple, fast text file reading with encoding detection
# ==============================================================================

import time
from pathlib import Path

import structlog

from informity.scanner.extractors.base import MAX_EXTRACTED_TEXT_PREVIEW, ExtractedDocument
from informity.scanner.extractors.text_utils import MAX_FILE_SIZE_BYTES, decode_bytes, elapsed_ms

log = structlog.get_logger(__name__)

# Plain text formats not supported by docling
# Includes structured data formats (JSON/YAML/TOML) - read as plain text for RAG
_PLAINTEXT_EXTENSIONS = ['.txt', '.md', '.rst', '.log', '.json', '.yaml', '.yml', '.toml']


class TextExtractor:
    """
    Extractor for plain text files not supported by docling.
    Handles encoding detection and simple text extraction.
    Also handles structured data formats (JSON/YAML/TOML) - reads as plain text for RAG.
    """
    supported_extensions: list[str] = _PLAINTEXT_EXTENSIONS

    def can_handle(self, path: Path) -> bool:
        return path.suffix.lower() in self.supported_extensions

    def extract(self, path: Path) -> ExtractedDocument:
        start_time = time.perf_counter()
        try:
            file_size = path.stat().st_size
            if file_size == 0:
                return ExtractedDocument(
                    text='',
                    source_path=path,
                    word_count=0,
                    extraction_time_ms=elapsed_ms(start_time),
                    preview_text='',
                    error='File is empty',
                )

            if file_size > MAX_FILE_SIZE_BYTES:
                return ExtractedDocument(
                    text='',
                    source_path=path,
                    word_count=0,
                    extraction_time_ms=elapsed_ms(start_time),
                    preview_text='',
                    error=f'File too large: {file_size} bytes',
                )

            # Read and decode text with encoding detection
            raw_bytes = path.read_bytes()
            text, encoding, error = decode_bytes(raw_bytes)
            word_count = len(text.split()) if text else 0

            return ExtractedDocument(
                text=text,
                source_path=path,
                metadata={'encoding': encoding, 'converter': 'plain_text'},
                word_count=word_count,
                extraction_time_ms=elapsed_ms(start_time),
                preview_text=text[:MAX_EXTRACTED_TEXT_PREVIEW],
                error=error,
            )
        except OSError as exc:
            return ExtractedDocument(
                text='',
                source_path=path,
                extraction_time_ms=elapsed_ms(start_time),
                preview_text='',
                error=f'Failed to read file: {exc}',
            )
