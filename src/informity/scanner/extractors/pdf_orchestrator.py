# ==============================================================================
# Informity AI — PDF Extraction Orchestrator
# Centralized, strategy-based extraction for PDFs with bounded time budgets.
# ==============================================================================

from __future__ import annotations

import multiprocessing as mp
import os
import queue
import time
from pathlib import Path
from typing import Literal

from informity.config import (
    DEFAULT_PDF_EXTRACTION_STRATEGY_ORDER,
    PDF_EXTRACTION_STRATEGIES,
    DirNames,
    configure_hf_environment,
    ensure_docling_rapidocr_cache_compat,
    settings,
)
from informity.scanner.extractors.base import ExtractedDocument
from informity.scanner.extractors.text_utils import elapsed_ms
from informity.utils.directory_utils import ensure_directory

PdfStrategy = Literal['docling_full', 'docling_fast', 'pdf_text_layer']
_ALLOWED_STRATEGIES: set[str] = set(PDF_EXTRACTION_STRATEGIES)

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
        page_count=payload.get('page_count') if isinstance(payload.get('page_count'), int) else None,
        word_count=int(payload.get('word_count') or 0),
        extraction_time_ms=elapsed_ms_value,
        error=(str(payload.get('error')) if payload.get('error') is not None else None),
        preview_text=str(payload.get('preview_text') or ''),
        char_to_page_ranges=payload.get('char_to_page_ranges') if isinstance(payload.get('char_to_page_ranges'), list) else None,
        char_to_block_type_ranges=payload.get('char_to_block_type_ranges') if isinstance(payload.get('char_to_block_type_ranges'), list) else None,
        char_to_header_level_ranges=payload.get('char_to_header_level_ranges') if isinstance(payload.get('char_to_header_level_ranges'), list) else None,
    )


def _prepare_docling_runtime() -> Path:
    docling_cache = settings.cache_dir / DirNames.DOCLING
    ensure_directory(docling_cache)
    configure_hf_environment()
    ensure_docling_rapidocr_cache_compat(settings.cache_dir)
    return docling_cache


def _docling_extract_worker(path_str: str, mode: str, use_ocr: bool, result_queue: mp.Queue) -> None:
    try:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import (
            AcceleratorOptions,
            PdfPipelineOptions,
            RapidOcrOptions,
        )
        from docling.document_converter import DocumentConverter, PdfFormatOption

        path = Path(path_str)
        docling_cache = _prepare_docling_runtime()
        os.environ['DOCLING_ARTIFACTS_PATH'] = str(docling_cache)

        accelerator_options = AcceleratorOptions(num_threads=settings.embedding_max_threads or 4)
        if mode == 'docling_fast':
            pipeline_options = PdfPipelineOptions(
                accelerator_options=accelerator_options,
                do_ocr=False,
            )
        else:
            if use_ocr:
                ocr_options = RapidOcrOptions(lang=[])
                ocr_options.force_full_page_ocr = True
                pipeline_options = PdfPipelineOptions(
                    accelerator_options=accelerator_options,
                    do_ocr=True,
                    ocr_options=ocr_options,
                )
            else:
                pipeline_options = PdfPipelineOptions(accelerator_options=accelerator_options)

        converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
        )
        result = converter.convert(str(path))
        doc = result.document
        markdown = (doc.export_to_markdown() or '').strip()
        if not markdown:
            result_queue.put({'ok': False, 'error': 'docling_empty_text'})
            return
        extracted = ExtractedDocument(
            text=markdown,
            source_path=path,
            metadata={
                'converter': 'docling',
                'format': '.pdf',
                'extractor_strategy': mode,
                'ocr_used': 'true' if use_ocr else 'false',
            },
            page_count=(result.input.page_count if hasattr(result, 'input') and hasattr(result.input, 'page_count') else None),
            word_count=len(markdown.split()),
            preview_text=markdown[:500],
        )
        result_queue.put({'ok': True, 'doc': _serialize_doc(extracted)})
    except _DOCLING_WORKER_EXCEPTIONS as exc:
        result_queue.put({'ok': False, 'error': str(exc)})


def _run_docling_strategy(path: Path, *, mode: str, timeout_seconds: int, use_ocr: bool) -> ExtractedDocument:
    start_time = time.perf_counter()
    ctx = mp.get_context('spawn')
    result_queue: mp.Queue = ctx.Queue(maxsize=1)
    process = ctx.Process(
        target=_docling_extract_worker,
        args=(str(path), mode, use_ocr, result_queue),
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
        merged = '\n\n'.join(chunks).strip()
        if not merged:
            return ExtractedDocument(
                text='',
                source_path=path,
                metadata={'error_code': 'pdf_text_layer_empty', 'retryable': 'true', 'extractor_strategy': 'pdf_text_layer'},
                extraction_time_ms=elapsed_ms(start_time),
                preview_text='',
                error='PDF text layer extraction returned empty text',
            )
        return ExtractedDocument(
            text=merged,
            source_path=path,
            metadata={'converter': 'pdf_text_layer', 'format': '.pdf', 'extractor_strategy': 'pdf_text_layer'},
            page_count=page_count,
            word_count=len(merged.split()),
            extraction_time_ms=elapsed_ms(start_time),
            preview_text=merged[:500],
        )
    except _DOCLING_WORKER_EXCEPTIONS as exc:
        return ExtractedDocument(
            text='',
            source_path=path,
            metadata={'error_code': 'pdf_text_layer_failed', 'retryable': 'true', 'extractor_strategy': 'pdf_text_layer'},
            extraction_time_ms=elapsed_ms(start_time),
            preview_text='',
            error=f'PDF text layer extraction failed: {exc}',
        )


def extract_pdf_with_orchestrator(path: Path, *, timeout_seconds: int) -> ExtractedDocument:
    start_time = time.perf_counter()
    strategy_order = [s for s in settings.pdf_extraction_strategy_order if s in _ALLOWED_STRATEGIES]
    if not strategy_order:
        strategy_order = list(DEFAULT_PDF_EXTRACTION_STRATEGY_ORDER)

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
    for strategy in strategy_order:
        remaining = int(max(1.0, deadline - time.perf_counter()))
        if remaining <= 0:
            break
        strategy_budget = max(1, int(total * (weights.get(strategy, 0) / total_weight)))
        strategy_budget = min(strategy_budget, remaining)
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

        if doc.text.strip():
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

    if failure_codes and all(code == 'scan_file_timeout' for code in failure_codes):
        final_error_code = 'scan_file_timeout'
    else:
        final_error_code = 'pdf_extraction_failed'
    return ExtractedDocument(
        text='',
        source_path=path,
        metadata={
            'error_code': final_error_code,
            'retryable': 'true',
            'extractor_strategy': strategy_order[-1],
        },
        extraction_time_ms=elapsed_ms(start_time),
        preview_text='',
        error='PDF extraction failed across strategies: ' + ' | '.join(failures[:4]),
    )
