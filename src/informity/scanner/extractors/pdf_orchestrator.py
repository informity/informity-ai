# ==============================================================================
# Informity AI — PDF Extraction Orchestrator
# Centralized, strategy-based extraction for PDFs with bounded time budgets.
# ==============================================================================

from __future__ import annotations

import multiprocessing as mp
import queue
import time
import tempfile
from pathlib import Path
from typing import Literal

import structlog

from informity.config import (
    DEFAULT_PDF_EXTRACTION_STRATEGY_ORDER,
    PDF_EXTRACTION_STRATEGIES,
    settings,
)
from informity.scanner.extractors.base import MAX_EXTRACTED_TEXT_PREVIEW, ExtractedDocument
from informity.scanner.extractors.boundary_rules import join_structured_blocks
from informity.scanner.extractors.docling import DoclingExtractor
from informity.scanner.extractors.text_utils import elapsed_ms

PdfStrategy = Literal['docling_full', 'docling_fast', 'pdf_text_layer']
_ALLOWED_STRATEGIES: set[str] = set(PDF_EXTRACTION_STRATEGIES)
OCR_ESCALATION_MAX_PAGES = 10
_LIVE_TRACE_FILENAME = '2023 Taxes - Completed and Signed.pdf'
log = structlog.get_logger(__name__)

_DOCLING_WORKER_EXCEPTIONS = (
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


def _serialize_doc(doc: ExtractedDocument) -> dict[str, object]:
    return {
        'text': doc.text,
        'source_path': str(doc.source_path),
        'metadata': dict(doc.metadata),
        'status': doc.status,
        'skip_reason': doc.skip_reason,
        'page_count': doc.page_count,
        'word_count': doc.word_count,
        'extraction_time_ms': doc.extraction_time_ms,
        'error': doc.error,
        'preview_text': doc.preview_text,
        'char_to_page_ranges': doc.char_to_page_ranges,
        'char_to_block_type_ranges': doc.char_to_block_type_ranges,
        'char_to_header_level_ranges': doc.char_to_header_level_ranges,
    }


def _deserialize_doc(payload: dict[str, object], *, source_path: Path, elapsed_ms_value: float) -> ExtractedDocument:
    return ExtractedDocument(
        text=str(payload.get('text') or ''),
        source_path=Path(str(payload.get('source_path') or str(source_path))),
        metadata=dict(payload.get('metadata') or {}),
        status=str(payload.get('status') or 'ok'),
        skip_reason=str(payload.get('skip_reason')) if payload.get('skip_reason') is not None else None,
        page_count=payload.get('page_count') if isinstance(payload.get('page_count'), int) else None,
        word_count=int(payload.get('word_count') or 0),
        extraction_time_ms=elapsed_ms_value,
        error=(str(payload.get('error')) if payload.get('error') is not None else None),
        preview_text=str(payload.get('preview_text') or ''),
        char_to_page_ranges=payload.get('char_to_page_ranges') if isinstance(payload.get('char_to_page_ranges'), list) else None,
        char_to_block_type_ranges=payload.get('char_to_block_type_ranges') if isinstance(payload.get('char_to_block_type_ranges'), list) else None,
        char_to_header_level_ranges=payload.get('char_to_header_level_ranges') if isinstance(payload.get('char_to_header_level_ranges'), list) else None,
    )


def _docling_extract_worker(
    path_str: str,
    mode: str,
    use_ocr: bool,
    max_pages: int | None,
    result_queue: mp.Queue,
) -> None:
    try:
        from informity.scanner.extractors.docling_runtime import (
            build_pdf_converter,
            prepare_docling_runtime,
        )

        path = Path(path_str)
        prepare_docling_runtime()
        convert_path = path
        temp_pdf_path: Path | None = None
        if max_pages is not None and max_pages > 0:
            import fitz

            source_pdf = fitz.open(str(path))
            try:
                page_count = min(int(max_pages), source_pdf.page_count)
                if page_count > 0:
                    limited_pdf = fitz.open()
                    limited_pdf.insert_pdf(source_pdf, from_page=0, to_page=page_count - 1)
                    with tempfile.NamedTemporaryFile(prefix=f'{path.stem}_ocr_', suffix='.pdf', delete=False) as temp_handle:
                        temp_pdf_path = Path(temp_handle.name)
                    limited_pdf.save(str(temp_pdf_path))
                    limited_pdf.close()
                    convert_path = temp_pdf_path
            finally:
                source_pdf.close()

        converter = build_pdf_converter(do_ocr=(mode != 'docling_fast' and use_ocr), force_full_page_ocr=True)
        result = converter.convert(str(convert_path))
        doc = result.document
        markdown = (doc.export_to_markdown() or '').strip()
        if not markdown:
            result_queue.put({'ok': False, 'error': 'docling_empty_text'})
            return
        extracted = ExtractedDocument(
            text=markdown,
            source_path=path,
            metadata={
                'converter': 'docling+ocr' if use_ocr else 'docling',
                'format': '.pdf',
                'extractor_strategy': mode,
                'ocr_used': 'true' if use_ocr else 'false',
            },
            page_count=(result.input.page_count if hasattr(result, 'input') and hasattr(result.input, 'page_count') else None),
            word_count=len(markdown.split()),
            preview_text=markdown[:MAX_EXTRACTED_TEXT_PREVIEW],
        )
        result_queue.put({'ok': True, 'doc': _serialize_doc(extracted)})
    except _DOCLING_WORKER_EXCEPTIONS as exc:
        result_queue.put({'ok': False, 'error': str(exc)})
    finally:
        try:
            if 'temp_pdf_path' in locals() and temp_pdf_path is not None and temp_pdf_path.exists():
                temp_pdf_path.unlink()
        except Exception:
            pass


def _pdf_is_image_only(file_path: Path) -> bool:
    try:
        import pypdfium2 as pdfium

        document = pdfium.PdfDocument(str(file_path))
    except Exception:
        return False

    try:
        page_count = len(document)
        if page_count <= 0:
            return False

        for page_index in range(page_count):
            page = document.get_page(page_index)
            text_page = None
            try:
                text_page = page.get_textpage()
                page_text = text_page.get_text_range()
                if str(page_text or '').strip():
                    return False
            finally:
                try:
                    if text_page is not None:
                        text_page.close()
                except Exception:
                    pass
                try:
                    page.close()
                except Exception:
                    pass
        return True
    finally:
        try:
            document.close()
        except Exception:
            pass


def _get_page_count(file_path: Path) -> int:
    """Returns PDF page count using pypdfium2. Returns 0 on any error."""
    try:
        import pypdfium2 as pdfium

        document = pdfium.PdfDocument(str(file_path))
        try:
            return len(document)
        finally:
            try:
                document.close()
            except Exception:
                pass
    except Exception:
        return 0


def _run_docling_strategy(
    path: Path,
    *,
    mode: str,
    timeout_seconds: int,
    use_ocr: bool,
    max_pages: int | None = None,
) -> ExtractedDocument:
    start_time = time.perf_counter()
    ctx = mp.get_context('spawn')
    result_queue: mp.Queue = ctx.Queue(maxsize=1)
    process = ctx.Process(
        target=_docling_extract_worker,
        args=(str(path), mode, use_ocr, max_pages, result_queue),
        daemon=True,
    )
    process.start()
    process.join(timeout=float(max(1, timeout_seconds)))
    if process.is_alive():
        process.terminate()
        process.join(timeout=2.0)
        return ExtractedDocument(
            text='',
            source_path=path,
            metadata={'error_code': 'scan_file_timeout', 'retryable': 'true', 'extractor_strategy': mode},
            status='failed_unknown',
            extraction_time_ms=elapsed_ms(start_time),
            preview_text='',
            error=f'{mode} timed out ({timeout_seconds}s)',
        )
    try:
        payload = result_queue.get_nowait()
    except queue.Empty:
        payload = None
    if not isinstance(payload, dict):
        return ExtractedDocument(
            text='',
            source_path=path,
            metadata={'error_code': 'docling_worker_no_result', 'retryable': 'true', 'extractor_strategy': mode},
            status='failed_unknown',
            extraction_time_ms=elapsed_ms(start_time),
            preview_text='',
            error=f'{mode} worker exited without payload',
        )
    if bool(payload.get('ok')) and isinstance(payload.get('doc'), dict):
        return _deserialize_doc(payload['doc'], source_path=path, elapsed_ms_value=elapsed_ms(start_time))
    return ExtractedDocument(
        text='',
        source_path=path,
        metadata={'error_code': 'docling_extraction_error', 'retryable': 'true', 'extractor_strategy': mode},
        status='failed_unknown',
        extraction_time_ms=elapsed_ms(start_time),
        preview_text='',
        error=f"{mode} failed: {payload.get('error') or 'unknown error'}",
    )


def _extract_pdf_text_layer(path: Path) -> ExtractedDocument:
    start_time = time.perf_counter()
    try:
        import pypdfium2 as pdfium

        doc = pdfium.PdfDocument(str(path))
        chunks: list[str] = []
        page_count = len(doc)
        for page_index in range(page_count):
            text = (doc[page_index].get_textpage().get_text_range() or '').strip()
            if text:
                chunks.append(text)
        if hasattr(doc, 'close'):
            doc.close()
        merged = join_structured_blocks(chunks)
        if not merged:
            return ExtractedDocument(
                text='',
                source_path=path,
                metadata={'error_code': 'pdf_text_layer_empty', 'retryable': 'true', 'extractor_strategy': 'pdf_text_layer'},
                status='skipped_empty',
                skip_reason='file skipped — no extractable text found',
                extraction_time_ms=elapsed_ms(start_time),
                preview_text='',
                error='file skipped — no extractable text found',
            )
        return ExtractedDocument(
            text=merged,
            source_path=path,
            metadata={'converter': 'pdf_text_layer', 'format': '.pdf', 'extractor_strategy': 'pdf_text_layer'},
            status='ok',
            page_count=page_count,
            word_count=len(merged.split()),
            extraction_time_ms=elapsed_ms(start_time),
            preview_text=merged[:MAX_EXTRACTED_TEXT_PREVIEW],
        )
    except _DOCLING_WORKER_EXCEPTIONS as exc:
        lowered = str(exc).casefold()
        status = 'failed_unknown'
        skip_reason = None
        if 'password' in lowered or 'encrypted' in lowered:
            status = 'skipped_encrypted'
            skip_reason = 'file skipped — password protected'
        elif 'corrupt' in lowered or 'invalid' in lowered:
            status = 'skipped_corrupted'
            skip_reason = 'file skipped — file is corrupted'
        elif 'unsupported' in lowered or 'not known' in lowered:
            status = 'skipped_unsupported'
            skip_reason = 'file skipped — unsupported format'
        return ExtractedDocument(
            text='',
            source_path=path,
            metadata={
                'error_code': status if status.startswith('skipped_') else 'pdf_text_layer_failed',
                'retryable': 'false' if status.startswith('skipped_') else 'true',
                'extractor_strategy': 'pdf_text_layer',
            },
            status=status,
            skip_reason=skip_reason,
            extraction_time_ms=elapsed_ms(start_time),
            preview_text='',
            error=skip_reason or f'PDF text layer extraction failed: {exc}',
        )


def extract_pdf_with_orchestrator(path: Path, *, timeout_seconds: int) -> ExtractedDocument:
    start_time = time.perf_counter()
    strategy_order = [s for s in settings.pdf_extraction_strategy_order if s in _ALLOWED_STRATEGIES]
    if not strategy_order:
        strategy_order = list(DEFAULT_PDF_EXTRACTION_STRATEGY_ORDER)
    trace_live_path = path.name == _LIVE_TRACE_FILENAME
    if trace_live_path:
        log.debug(
            'pdf_orchestrator_trace_start',
            path=str(path),
            filename=path.name,
            timeout_seconds=timeout_seconds,
            strategy_order=strategy_order,
            scan_file_timeout_seconds=getattr(settings, 'scan_file_timeout_seconds', None),
            enable_ocr_for_images=getattr(settings, 'enable_ocr_for_images', None),
        )

    log.debug(
        'pdf_orchestrator_start',
        path=str(path),
        timeout_seconds=timeout_seconds,
        scan_file_timeout_seconds=getattr(settings, 'scan_file_timeout_seconds', None),
        enable_ocr_for_images=getattr(settings, 'enable_ocr_for_images', None),
        strategy_order=strategy_order,
    )

    total = max(1, int(timeout_seconds))
    weights: dict[PdfStrategy, int] = {
        'docling_full': 55,
        'docling_fast': 30,
        'pdf_text_layer': 15,
    }
    total_weight = sum(weights.get(strategy, 0) for strategy in strategy_order) or 100

    failures: list[str] = []
    failure_codes: list[str] = []
    deadline = time.perf_counter() + float(total)
    ocr_attempted = False
    all_strategies_zero_chars = True
    for strategy in strategy_order:
        remaining = int(max(1.0, deadline - time.perf_counter()))
        if remaining <= 0:
            break
        strategy_budget = max(1, int(total * (weights.get(strategy, 0) / total_weight)))
        strategy_budget = min(strategy_budget, remaining)
        log.debug(
            'pdf_orchestrator_strategy_start',
            path=str(path),
            strategy=strategy,
            strategy_budget_seconds=strategy_budget,
            remaining_seconds=remaining,
        )
        if strategy == 'docling_full':
            doc = _run_docling_strategy(
                path,
                mode='docling_full',
                timeout_seconds=strategy_budget,
                use_ocr=False,
            )
        elif strategy == 'docling_fast':
            doc = _run_docling_strategy(
                path,
                mode='docling_fast',
                timeout_seconds=strategy_budget,
                use_ocr=False,
            )
        else:
            doc = _extract_pdf_text_layer(path)

        log.debug(
            'pdf_orchestrator_strategy_result',
            path=str(path),
            strategy=strategy,
            extracted_chars=len(doc.text or ''),
            status=doc.status,
            method=doc.metadata.get('converter') or doc.metadata.get('extractor_strategy'),
            error=doc.error,
        )

        if len((doc.text or '').strip()) > 0:
            all_strategies_zero_chars = False

        if doc.text and not DoclingExtractor._looks_effectively_empty(doc.text):
            merged = dict(doc.metadata)
            merged['fallback_used'] = 'true' if strategy != strategy_order[0] else 'false'
            return ExtractedDocument(
                text=doc.text,
                source_path=doc.source_path,
                metadata=merged,
                page_count=doc.page_count,
                word_count=doc.word_count,
                extraction_time_ms=elapsed_ms(start_time),
                error=doc.error,
                preview_text=doc.preview_text,
                char_to_page_ranges=doc.char_to_page_ranges,
                char_to_block_type_ranges=doc.char_to_block_type_ranges,
                char_to_header_level_ranges=doc.char_to_header_level_ranges,
            )
        code = str(doc.metadata.get('error_code') or '').strip()
        if code:
            failure_codes.append(code)
        failures.append(f'{strategy}:{doc.error or "empty"}')

    image_only = _pdf_is_image_only(path)
    page_count = _get_page_count(path)
    log.debug(
        'pdf_orchestrator_ocr_evaluation',
        path=str(path),
        image_only=image_only,
        page_count=page_count,
        enable_ocr_for_images=getattr(settings, 'enable_ocr_for_images', None),
        all_strategies_zero_chars=all_strategies_zero_chars,
        ocr_attempted=ocr_attempted,
    )
    if image_only and settings.enable_ocr_for_images:
        ocr_attempted = True
        remaining = int(max(1.0, deadline - time.perf_counter()))
        if remaining > 0:
            log.debug(
                'pdf_orchestrator_ocr_attempt_start',
                path=str(path),
                timeout_seconds=remaining,
                max_pages=OCR_ESCALATION_MAX_PAGES,
            )
            ocr_doc = _run_docling_strategy(
                path,
                mode='docling_full',
                timeout_seconds=remaining,
                use_ocr=True,
                max_pages=OCR_ESCALATION_MAX_PAGES,
            )
            log.debug(
                'pdf_orchestrator_ocr_attempt_result',
                path=str(path),
                extracted_chars=len(ocr_doc.text or ''),
                status=ocr_doc.status,
                method=ocr_doc.metadata.get('converter') or ocr_doc.metadata.get('extractor_strategy'),
                error=ocr_doc.error,
            )
            if ocr_doc.text and not DoclingExtractor._looks_effectively_empty(ocr_doc.text):
                merged = dict(ocr_doc.metadata)
                merged['fallback_used'] = 'true'
                merged['converter'] = 'docling+ocr'
                merged['ocr_used'] = 'true'
                return ExtractedDocument(
                    text=ocr_doc.text,
                    source_path=ocr_doc.source_path,
                    metadata=merged,
                    status='ok',
                    page_count=ocr_doc.page_count,
                    word_count=ocr_doc.word_count,
                    extraction_time_ms=elapsed_ms(start_time),
                    error=ocr_doc.error,
                    preview_text=ocr_doc.preview_text,
                    char_to_page_ranges=ocr_doc.char_to_page_ranges,
                    char_to_block_type_ranges=ocr_doc.char_to_block_type_ranges,
                    char_to_header_level_ranges=ocr_doc.char_to_header_level_ranges,
                )
            if ocr_doc.error and 'worker exited without payload' in ocr_doc.error:
                return ocr_doc
            if ocr_doc.status.startswith('skipped_') or not ocr_doc.text or DoclingExtractor._looks_effectively_empty(ocr_doc.text):
                return ExtractedDocument(
                    text='',
                    source_path=path,
                    metadata={
                        'error_code': 'skipped_empty',
                        'retryable': 'false',
                        'extractor_strategy': 'docling+ocr',
                    },
                    status='skipped_empty',
                    skip_reason='file skipped — no extractable text found',
                    extraction_time_ms=elapsed_ms(start_time),
                    preview_text='',
                    error='file skipped — no extractable text found',
                )

    if (
        not ocr_attempted
        and settings.enable_ocr_for_images
        and all_strategies_zero_chars
        and page_count > 0
    ):
        ocr_attempted = True
        remaining = int(max(1.0, deadline - time.perf_counter()))
        if remaining > 0:
            log.debug(
                'pdf_orchestrator_ocr_escalation_start',
                path=str(path),
                timeout_seconds=remaining,
                max_pages=OCR_ESCALATION_MAX_PAGES,
            )
            ocr_doc = _run_docling_strategy(
                path,
                mode='docling_full',
                timeout_seconds=remaining,
                use_ocr=True,
                max_pages=OCR_ESCALATION_MAX_PAGES,
            )
            log.debug(
                'pdf_orchestrator_ocr_escalation_result',
                path=str(path),
                extracted_chars=len(ocr_doc.text or ''),
                status=ocr_doc.status,
                method=ocr_doc.metadata.get('converter') or ocr_doc.metadata.get('extractor_strategy'),
                error=ocr_doc.error,
            )
            if ocr_doc.text and not DoclingExtractor._looks_effectively_empty(ocr_doc.text):
                merged = dict(ocr_doc.metadata)
                merged['fallback_used'] = 'true'
                merged['converter'] = 'docling+ocr'
                merged['ocr_used'] = 'true'
                return ExtractedDocument(
                    text=ocr_doc.text,
                    source_path=ocr_doc.source_path,
                    metadata=merged,
                    status='ok',
                    page_count=ocr_doc.page_count,
                    word_count=ocr_doc.word_count,
                    extraction_time_ms=elapsed_ms(start_time),
                    error=ocr_doc.error,
                    preview_text=ocr_doc.preview_text,
                    char_to_page_ranges=ocr_doc.char_to_page_ranges,
                    char_to_block_type_ranges=ocr_doc.char_to_block_type_ranges,
                    char_to_header_level_ranges=ocr_doc.char_to_header_level_ranges,
                )

    if failure_codes and all(code == 'scan_file_timeout' for code in failure_codes):
        final_error_code = 'scan_file_timeout'
    else:
        final_error_code = 'pdf_extraction_failed'
    combined_failure_text = ' | '.join(failures[:4]).casefold()
    status = 'failed_unknown'
    skip_reason = None
    if 'password' in combined_failure_text or 'encrypted' in combined_failure_text:
        status = 'skipped_encrypted'
        skip_reason = 'file skipped — password protected'
        final_error_code = 'skipped_encrypted'
    elif 'corrupt' in combined_failure_text or 'invalid' in combined_failure_text:
        status = 'skipped_corrupted'
        skip_reason = 'file skipped — file is corrupted'
        final_error_code = 'skipped_corrupted'
    elif 'unsupported' in combined_failure_text or 'not known' in combined_failure_text:
        status = 'skipped_unsupported'
        skip_reason = 'file skipped — unsupported format'
        final_error_code = 'skipped_unsupported'
    elif 'empty' in combined_failure_text:
        status = 'skipped_empty'
        skip_reason = 'file skipped — no extractable text found'
        final_error_code = 'skipped_empty'
    if image_only and not status.startswith('skipped_') and status != 'ok':
        status = 'skipped_empty'
        skip_reason = 'file skipped — no extractable text found'
        final_error_code = 'skipped_empty'
    log.debug(
        'pdf_orchestrator_returning',
        path=str(path),
        status=status,
        method=strategy_order[-1],
        final_error_code=final_error_code,
        skip_reason=skip_reason,
        ocr_attempted=ocr_attempted,
        image_only=image_only,
        page_count=page_count,
        all_strategies_zero_chars=all_strategies_zero_chars,
        elapsed_ms=elapsed_ms(start_time),
    )
    return ExtractedDocument(
        text='',
        source_path=path,
        metadata={
            'error_code': final_error_code,
            'retryable': 'true',
            'extractor_strategy': strategy_order[-1],
        },
        status=status,
        skip_reason=skip_reason,
        extraction_time_ms=elapsed_ms(start_time),
        preview_text='',
        error=skip_reason or ('PDF extraction failed across strategies: ' + ' | '.join(failures[:4])),
    )
