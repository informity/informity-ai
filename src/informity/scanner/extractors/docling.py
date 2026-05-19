# ==============================================================================
# Informity AI — Docling Extractor (v2)
# Unified extractor for docling-supported formats: PDF, DOCX, PPTX, XLSX, HTML, CSV
# Uses docling to convert documents to markdown with better structure preservation
# ==============================================================================

from __future__ import annotations

import gc
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING

import structlog

from informity.config import (
    DirNames,
    configure_hf_environment,
    ensure_docling_rapidocr_cache_compat,
    settings,
)
from informity.scanner.extractors.base import MAX_EXTRACTED_TEXT_PREVIEW, ExtractedDocument
from informity.scanner.extractors.text_utils import elapsed_ms, get_max_file_size_bytes
from informity.utils.directory_utils import ensure_directory

log = structlog.get_logger(__name__)
_DOCLING_RUNTIME_EXCEPTIONS = (
    RuntimeError,
    ValueError,
    TypeError,
    AttributeError,
    KeyError,
    IndexError,
    OSError,
    ImportError,
    MemoryError,
    AssertionError,
)

if TYPE_CHECKING:
    from docling.document_converter import DocumentConverter

# ==============================================================================
# Converter Singleton with Periodic Reset
# ==============================================================================

# Reuse DocumentConverter instance for performance, but recreate periodically
# to avoid memory leaks (docling has known memory leak when reusing instances)
# Increased from 10 to 25 to reduce reset frequency while maintaining memory safety
# With range-based metadata storage (#6) and explicit GC (#8), memory pressure is lower
_MAX_CONVERSIONS_BEFORE_RESET = 25  # Recreate converter every N conversions

# Docling-supported formats (excluding images for now)
_DOCLING_SUPPORTED_EXTENSIONS = ['.pdf', '.docx', '.pptx', '.xlsx', '.html', '.htm', '.csv']


def _classify_docling_exception(exc: Exception) -> tuple[str, bool]:
    # Map docling/pdfium failures to stable, generic error codes and retryability.
    error_str = str(exc).lower()

    if 'pdf_resources_dir' in error_str:
        # Packaged runtime is missing docling parse resources; this is an app setup issue.
        return 'docling_runtime_resource_missing', True
    if 'incorrect password' in error_str or ('password' in error_str and 'pdfium' in error_str):
        return 'pdf_password_protected', False
    if 'data format error' in error_str or 'is not valid' in error_str:
        return 'pdf_invalid_or_corrupt', False
    if 'font_name' in error_str and 'is not known' in error_str:
        return 'pdf_unsupported_font_map', False
    return 'docling_extraction_error', True


class DoclingExtractor:
    """
    Unified extractor for docling-supported document formats.
    Converts documents to markdown with structure preservation and metadata extraction.
    """
    supported_extensions: list[str] = _DOCLING_SUPPORTED_EXTENSIONS

    # Class-level singleton converter with reset counter
    _converter: DocumentConverter | None = None
    _conversion_count: int = 0

    def can_handle(self, path: Path) -> bool:
        return path.suffix.lower() in self.supported_extensions

    @classmethod
    def reset_converter(cls) -> None:
        # Force-reset the converter singleton. Called after a timeout or cancellation
        # so the next file gets a fresh converter rather than one in a corrupted mid-run state.
        if cls._converter is not None:
            del cls._converter
            cls._converter = None
            gc.collect()
        cls._conversion_count = 0

    @classmethod
    def _get_converter(cls) -> DocumentConverter:
        # Lazy initialization: create converter on first use
        # Reset periodically to avoid memory leaks
        if cls._converter is None or cls._conversion_count >= _MAX_CONVERSIONS_BEFORE_RESET:
            if cls._converter is not None:
                del cls._converter
                gc.collect()

            # Configure docling to use our cache directory for models
            # Docling uses DOCLING_ARTIFACTS_PATH env var or downloads to default cache
            # We set it to our unified cache directory so models are cached there
            # Set docling artifacts path (flat structure: cache/docling/)
            # Docling will create its own subdirectories inside as needed
            docling_cache = settings.cache_dir / DirNames.DOCLING
            ensure_directory(docling_cache)
            os.environ['DOCLING_ARTIFACTS_PATH'] = str(docling_cache)

            # Ensure HF environment is configured (for docling's HuggingFace dependencies)
            # This will raise ConfigurationError if Full Privacy is enabled but models aren't cached
            configure_hf_environment()
            ensure_docling_rapidocr_cache_compat(settings.cache_dir)

            try:
                # Import only after DOCLING_ARTIFACTS_PATH is set so docling's settings see it
                from docling.datamodel.base_models import InputFormat
                from docling.datamodel.pipeline_options import (
                    AcceleratorOptions,
                    PdfPipelineOptions,
                )
                from docling.document_converter import DocumentConverter, PdfFormatOption

                # Configure accelerator options with thread alignment
                accelerator_options = AcceleratorOptions(
                    num_threads=settings.embedding_max_threads or 4
                )

                # Configure PDF pipeline options
                pdf_pipeline_options = PdfPipelineOptions(
                    accelerator_options=accelerator_options
                )

                # Create format_options dict mapping formats to their options
                # PDF gets custom pipeline options; other formats use defaults
                format_options = {
                    InputFormat.PDF: PdfFormatOption(
                        pipeline_options=pdf_pipeline_options
                    )
                }

                cls._converter = DocumentConverter(format_options=format_options)
                cls._conversion_count = 0
            except _DOCLING_RUNTIME_EXCEPTIONS as exc:
                # If converter creation fails, log error with helpful context
                log.error(
                    'docling_converter_failed',
                    error=str(exc),
                    cache_path=str(docling_cache),
                    suggestion='If Full Privacy is enabled, ensure install script completed successfully',
                    exc_info=True,
                )
                raise

        cls._conversion_count += 1
        return cls._converter

    @classmethod
    def _create_ocr_converter(cls) -> DocumentConverter:
        """
        Create a DocumentConverter with OCR enabled for image-only PDFs.
        Used as fallback when regular extraction returns empty text.
        """
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import (
            AcceleratorOptions,
            PdfPipelineOptions,
            RapidOcrOptions,
        )
        from docling.document_converter import DocumentConverter, PdfFormatOption

        # Configure accelerator options
        accelerator_options = AcceleratorOptions(
            num_threads=settings.embedding_max_threads or 4
        )

        # Configure OCR options: force full-page OCR for image-only PDFs
        # Use RapidOcrOptions since we're using RapidOCR (downloaded via bootstrap)
        ocr_options = RapidOcrOptions(lang=[])  # Empty lang list = auto-detect language
        ocr_options.force_full_page_ocr = True

        # Configure PDF pipeline options with OCR enabled
        pdf_pipeline_options = PdfPipelineOptions(
            accelerator_options=accelerator_options,
            do_ocr=True,  # Enable OCR
            ocr_options=ocr_options,
        )

        # Create format_options dict with OCR-enabled PDF options
        format_options = {
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pdf_pipeline_options
            )
        }

        return DocumentConverter(format_options=format_options)

    def extract(self, path: Path) -> ExtractedDocument:
        start_time = time.perf_counter()
        try:
            file_size = path.stat().st_size
            if file_size > get_max_file_size_bytes():
                return ExtractedDocument(
                    text='',
                    source_path=path,
                    extraction_time_ms=elapsed_ms(start_time),
                    preview_text='',
                    error=f'File too large: {file_size} bytes',
                )

            # PDF preflight: fail fast on password-protected/corrupt PDFs before full conversion.
            if path.suffix.lower() == '.pdf':
                try:
                    import pypdfium2 as pdfium

                    preflight_doc = pdfium.PdfDocument(str(path))
                    # Explicit close to avoid leaking handles in long scans.
                    if hasattr(preflight_doc, 'close'):
                        preflight_doc.close()
                except _DOCLING_RUNTIME_EXCEPTIONS as preflight_exc:
                    error_code, retryable = _classify_docling_exception(preflight_exc)
                    return ExtractedDocument(
                        text='',
                        source_path=path,
                        metadata={
                            'error_code': error_code,
                            'retryable': 'false' if not retryable else 'true',
                        },
                        extraction_time_ms=elapsed_ms(start_time),
                        preview_text='',
                        error=f'Docling extraction failed: {preflight_exc}',
                    )

            # Get reusable converter instance (lazy-loaded, periodically reset)
            converter = self._get_converter()

            # Attempt conversion
            # If models are cached (from install script), we respect Full Privacy completely
            # If models aren't cached and offline mode is set, conversion will fail with clear error
            result = converter.convert(str(path))

            # Extract metadata from docling result
            doc = result.document

            # Build markdown text ourselves using iterate_items() to get accurate char positions
            # This enables correct page number and block type assignment (charspan from provenance
            # maps to original document positions, not exported markdown positions)
            # Use range-based storage (start, end, value) instead of per-character dicts for memory efficiency
            text_parts: list[str] = []
            char_to_page_ranges: list[tuple[int, int, int]] = []  # (start, end, page_no)
            char_to_block_type_ranges: list[tuple[int, int, str]] = []  # (start, end, block_type)
            char_to_header_level_ranges: list[tuple[int, int, int]] = []  # (start, end, header_level)
            char_pos = 0

            try:
                # Import docling types for isinstance checks
                from docling_core.types.doc.document import (
                    KeyValueItem,
                    SectionHeaderItem,
                    TableItem,
                    TextItem,
                )

                # Iterate all document items in order
                for item, _level in doc.iterate_items(with_groups=True):
                    item_text = ''
                    page_no: int | None = None
                    block_type: str | None = None
                    header_level: int | None = None

                    # Extract page number from provenance (first prov item)
                    if hasattr(item, 'prov') and item.prov:
                        prov = item.prov[0]
                        if hasattr(prov, 'page_no'):
                            page_no = prov.page_no

                    # Determine block type and extract text based on item type
                    if isinstance(item, SectionHeaderItem):
                        # Section headers: use text (formatted) or orig (raw), track level
                        header_level = item.level if hasattr(item, 'level') else 1
                        header_text = item.text if hasattr(item, 'text') else (item.orig if hasattr(item, 'orig') else '')
                        block_type = 'narrative'
                        # Format as markdown header (ensure proper formatting)
                        if header_text.strip().startswith('#'):
                            # Already formatted, use as-is
                            item_text = header_text if header_text.endswith('\n') else header_text + '\n'
                        else:
                            # Format manually
                            item_text = f'{"#" * header_level} {header_text.strip()}\n'

                    elif isinstance(item, TableItem):
                        # Tables: use export_to_markdown with doc context for proper rendering
                        try:
                            item_text = item.export_to_markdown(doc=doc)
                            if not item_text.endswith('\n'):
                                item_text += '\n'
                        except _DOCLING_RUNTIME_EXCEPTIONS:
                            # Fallback: try text attribute
                            item_text = item.text if hasattr(item, 'text') else ''
                        block_type = 'table'

                    elif isinstance(item, KeyValueItem):
                        # Key-value items (form fields): format as markdown
                        key = item.key if hasattr(item, 'key') else ''
                        value = item.value if hasattr(item, 'value') else ''
                        if key and value:
                            item_text = f'**{key}:** {value}\n'
                        block_type = 'form'

                    elif isinstance(item, TextItem):
                        # Regular text items: use text attribute
                        item_text = item.text if hasattr(item, 'text') else ''
                        block_type = 'narrative'

                    else:
                        # Unknown item type: try to get text
                        if hasattr(item, 'text'):
                            item_text = item.text
                            block_type = 'narrative'
                        elif hasattr(item, 'orig'):
                            item_text = item.orig
                            block_type = 'narrative'

                    # Map character positions for this item's text using ranges (memory-efficient)
                    # Store (start, end, value) tuples instead of per-character dict entries
                    if item_text:
                        item_end = char_pos + len(item_text)
                        if page_no is not None:
                            char_to_page_ranges.append((char_pos, item_end, page_no))
                        if block_type:
                            char_to_block_type_ranges.append((char_pos, item_end, block_type))
                        if header_level is not None:
                            char_to_header_level_ranges.append((char_pos, item_end, header_level))

                    # Append to text
                    text_parts.append(item_text)
                    char_pos += len(item_text)

                # Join all parts into final markdown text
                text = ''.join(text_parts)

            except _DOCLING_RUNTIME_EXCEPTIONS as exc:
                # Fallback: use export_to_markdown if iterate_items fails
                log.warning('iterate_items_failed_fallback', path=str(path), error=str(exc))
                text = doc.export_to_markdown()
                # Clear mappings since they're unreliable
                char_to_page_ranges = []
                char_to_block_type_ranges = []
                char_to_header_level_ranges = []

            # High-value metadata: Page count (for PDF, PPTX - pages/slides)
            page_count: int | None = None
            try:
                # Prefer result.input.page_count (most reliable for PDFs)
                if hasattr(result, 'input') and hasattr(result.input, 'page_count'):
                    page_count = result.input.page_count
                # Fallback to counting pages dict
                elif hasattr(doc, 'pages') and doc.pages:
                    page_count = len(doc.pages)
                # Last resort: num_pages() method
                elif hasattr(doc, 'num_pages'):
                    page_count = doc.num_pages()
            except _DOCLING_RUNTIME_EXCEPTIONS:
                log.debug('page_count_extraction_failed', path=str(path))

            # High-value metadata: Content statistics
            tables_count = len(doc.tables) if hasattr(doc, 'tables') else 0
            form_items_count = len(doc.form_items) if hasattr(doc, 'form_items') else 0
            key_value_items_count = len(doc.key_value_items) if hasattr(doc, 'key_value_items') else 0

            # Medium-value metadata
            pictures_count = len(doc.pictures) if hasattr(doc, 'pictures') else 0
            document_hash: str | None = None
            try:
                if hasattr(result, 'input') and hasattr(result.input, 'document_hash'):
                    document_hash = result.input.document_hash
            except _DOCLING_RUNTIME_EXCEPTIONS:
                pass

            # Log mapping statistics for debugging (only if mappings exist)
            if char_to_page_ranges:
                log.debug(
                    'per_chunk_metadata_ranges_created',
                    path=str(path),
                    page_ranges=len(char_to_page_ranges),
                    block_type_ranges=len(char_to_block_type_ranges),
                    header_level_ranges=len(char_to_header_level_ranges),
                    markdown_length=len(text),
                )

            # Build metadata dict
            metadata: dict[str, str] = {
                'converter': 'docling',
                'format': path.suffix.lower(),
                'page_count': str(page_count) if page_count else 'unknown',
                'tables_count': str(tables_count),
                'form_items_count': str(form_items_count),
                'key_value_items_count': str(key_value_items_count),
                'pictures_count': str(pictures_count),
            }
            if document_hash:
                metadata['document_hash'] = document_hash

            word_count = len(text.split()) if text else 0

            # If extraction returned empty text and OCR is enabled, try OCR as fallback
            if not text.strip() and settings.enable_ocr_for_images and path.suffix.lower() == '.pdf':
                log.info(
                    'trying_ocr_fallback',
                    path=str(path),
                    reason='regular_extraction_returned_empty_text'
                )
                try:
                    # Create OCR-enabled converter for retry
                    ocr_converter = self._create_ocr_converter()
                    ocr_result = ocr_converter.convert(str(path))
                    ocr_doc = ocr_result.document

                    # Extract text from OCR result
                    ocr_text = ocr_doc.export_to_markdown() if hasattr(ocr_doc, 'export_to_markdown') else ''

                    if ocr_text.strip():
                        log.info(
                            'ocr_extraction_succeeded',
                            path=str(path),
                            text_length=len(ocr_text),
                            word_count=len(ocr_text.split())
                        )
                        # Use OCR-extracted text
                        text = ocr_text
                        word_count = len(text.split())

                        # Update metadata to indicate OCR was used
                        metadata['ocr_used'] = 'true'
                        metadata['converter'] = 'docling+ocr'

                        # Re-extract page count from OCR result
                        try:
                            if hasattr(ocr_result, 'input') and hasattr(ocr_result.input, 'page_count'):
                                page_count = ocr_result.input.page_count
                            elif hasattr(ocr_doc, 'pages') and ocr_doc.pages:
                                page_count = len(ocr_doc.pages)
                            elif hasattr(ocr_doc, 'num_pages'):
                                page_count = ocr_doc.num_pages()
                        except _DOCLING_RUNTIME_EXCEPTIONS:
                            pass

                        # Use OCR document for preview
                        try:
                            preview_text = ocr_doc.export_to_text()[:MAX_EXTRACTED_TEXT_PREVIEW]
                        except _DOCLING_RUNTIME_EXCEPTIONS:
                            preview_text = text[:MAX_EXTRACTED_TEXT_PREVIEW]
                    else:
                        log.debug(
                            'ocr_extraction_empty',
                            path=str(path),
                            reason='ocr_also_returned_empty_text'
                        )
                        # OCR also returned empty - use original empty result
                        preview_text = text[:MAX_EXTRACTED_TEXT_PREVIEW]
                except _DOCLING_RUNTIME_EXCEPTIONS as ocr_exc:
                    log.warning(
                        'ocr_fallback_failed',
                        path=str(path),
                        error=str(ocr_exc),
                        error_type=type(ocr_exc).__name__,
                        action='indexing_file_with_zero_chunks'
                    )
                    # OCR failed - use original empty result (file will be indexed with 0 chunks)
                    preview_text = text[:MAX_EXTRACTED_TEXT_PREVIEW]
            else:
                # Get clean preview text (without markdown noise)
                try:
                    preview_text = doc.export_to_text()[:MAX_EXTRACTED_TEXT_PREVIEW]
                except _DOCLING_RUNTIME_EXCEPTIONS:
                    preview_text = text[:MAX_EXTRACTED_TEXT_PREVIEW]

            return ExtractedDocument(
                text=text,
                source_path=path,
                metadata=metadata,
                page_count=page_count,
                word_count=word_count,
                extraction_time_ms=elapsed_ms(start_time),
                preview_text=preview_text,
                char_to_page_ranges=char_to_page_ranges if char_to_page_ranges else None,
                char_to_block_type_ranges=char_to_block_type_ranges if char_to_block_type_ranges else None,
                char_to_header_level_ranges=char_to_header_level_ranges if char_to_header_level_ranges else None,
            )
        except _DOCLING_RUNTIME_EXCEPTIONS as exc:
            error_str = str(exc).lower()
            # Provide helpful error message for offline mode issues
            if 'offline' in error_str or 'cached snapshot' in error_str:
                return ExtractedDocument(
                    text='',
                    source_path=path,
                    metadata={
                        'error_code': 'docling_models_unavailable_offline',
                        'retryable': 'true',
                    },
                    extraction_time_ms=elapsed_ms(start_time),
                    preview_text='',
                    error='Docling extraction failed: Models not cached and offline mode enabled. Run install script to download models, or disable full_privacy temporarily.',
                )

            error_code, retryable = _classify_docling_exception(exc)

            # If OCR is enabled and this is a PDF, try OCR as fallback for extraction failures
            if settings.enable_ocr_for_images and path.suffix.lower() == '.pdf':
                log.info(
                    'trying_ocr_fallback_after_exception',
                    path=str(path),
                    error=str(exc),
                    error_type=type(exc).__name__,
                    reason='regular_extraction_raised_exception'
                )
                try:
                    # Create OCR-enabled converter for retry
                    ocr_converter = self._create_ocr_converter()
                    ocr_result = ocr_converter.convert(str(path))
                    ocr_doc = ocr_result.document

                    # Extract text from OCR result
                    ocr_text = ocr_doc.export_to_markdown() if hasattr(ocr_doc, 'export_to_markdown') else ''

                    if ocr_text.strip():
                        log.info(
                            'ocr_extraction_succeeded_after_exception',
                            path=str(path),
                            text_length=len(ocr_text),
                            word_count=len(ocr_text.split())
                        )
                        # Use OCR-extracted text
                        word_count = len(ocr_text.split())

                        # Get page count from OCR result
                        page_count: int | None = None
                        try:
                            if hasattr(ocr_result, 'input') and hasattr(ocr_result.input, 'page_count'):
                                page_count = ocr_result.input.page_count
                            elif hasattr(ocr_doc, 'pages') and ocr_doc.pages:
                                page_count = len(ocr_doc.pages)
                            elif hasattr(ocr_doc, 'num_pages'):
                                page_count = ocr_doc.num_pages()
                        except _DOCLING_RUNTIME_EXCEPTIONS:
                            pass

                        # Build metadata indicating OCR was used
                        metadata: dict[str, str] = {
                            'converter': 'docling+ocr',
                            'format': path.suffix.lower(),
                            'page_count': str(page_count) if page_count else 'unknown',
                            'ocr_used': 'true',
                            'original_error': str(exc)[:200],  # Truncate long error messages
                        }

                        # Get preview text
                        try:
                            preview_text = ocr_doc.export_to_text()[:MAX_EXTRACTED_TEXT_PREVIEW]
                        except _DOCLING_RUNTIME_EXCEPTIONS:
                            preview_text = ocr_text[:MAX_EXTRACTED_TEXT_PREVIEW]

                        return ExtractedDocument(
                            text=ocr_text,
                            source_path=path,
                            metadata=metadata,
                            page_count=page_count,
                            word_count=word_count,
                            extraction_time_ms=elapsed_ms(start_time),
                            preview_text=preview_text,
                        )
                    else:
                        log.debug(
                            'ocr_extraction_empty_after_exception',
                            path=str(path),
                            reason='ocr_also_returned_empty_text'
                        )
                        # OCR also returned empty - return error result
                        return ExtractedDocument(
                            text='',
                            source_path=path,
                            metadata={
                                'error_code': error_code,
                                'retryable': 'false' if not retryable else 'true',
                            },
                            extraction_time_ms=elapsed_ms(start_time),
                            preview_text='',
                            error=f'Docling extraction failed: {exc}. OCR fallback also returned empty text.',
                        )
                except _DOCLING_RUNTIME_EXCEPTIONS as ocr_exc:
                    log.warning(
                        'ocr_fallback_failed_after_exception',
                        path=str(path),
                        original_error=str(exc),
                        ocr_error=str(ocr_exc),
                        error_type=type(ocr_exc).__name__,
                        action='indexing_file_with_zero_chunks'
                    )
                    # OCR failed - return error result (file will be indexed with 0 chunks)
                    return ExtractedDocument(
                        text='',
                        source_path=path,
                        metadata={
                            'error_code': error_code,
                            'retryable': 'false' if not retryable else 'true',
                        },
                        extraction_time_ms=elapsed_ms(start_time),
                        preview_text='',
                        error=f'Docling extraction failed: {exc}. OCR fallback also failed: {ocr_exc}',
                    )

            # OCR not enabled or not a PDF - return error result
            return ExtractedDocument(
                text='',
                source_path=path,
                metadata={
                    'error_code': error_code,
                    'retryable': 'false' if not retryable else 'true',
                },
                extraction_time_ms=elapsed_ms(start_time),
                preview_text='',
                error=f'Docling extraction failed: {exc}',
            )
