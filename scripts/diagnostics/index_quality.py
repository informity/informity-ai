#!/usr/bin/env python3
"""Read-only indexing quality diagnostic for a single file."""

# ruff: noqa: I001

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import signal
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import Any

import pypdfium2 as pdfium

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / 'src'))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from informity.config import settings  # noqa: E402
from informity.indexer.chunker import ChunkData, chunk_text, create_child_chunks  # noqa: E402
from informity.indexer.embedder import embedder  # noqa: E402
from informity.indexer.post_process import post_process_extracted_text  # noqa: E402
from informity.indexer.pipeline import filter_noise_chunks  # noqa: E402
from informity.scanner.extractors.base import ExtractedDocument, get_extractor  # noqa: E402
from informity.scanner.extractors.docling import DoclingExtractor  # noqa: E402
from informity.scanner.extractors.pdf_orchestrator import extract_pdf_with_orchestrator  # noqa: E402

_MIN_VIABLE_EXTRACTED_CHARS = 100
_MIN_USEFUL_CHUNK_CHARS = 50
_MAX_USEFUL_CHUNK_CHARS = 2000
_EMBEDDING_MODEL_MAX_TOKENS = 8192
_EXTRACTION_PREVIEW_CHARS = 500
_RUN_DIRECT_ENV_VAR = 'INFORMITY_INDEX_QUALITY_RUN_DIRECT'


@dataclass(frozen=True)
class StageResult:
    name: str
    status: str
    summary: str
    data: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Inspect indexing quality without modifying the index.')
    parser.add_argument('--file', type=Path, help='Path to the document to inspect.')
    parser.add_argument('--files', nargs='+', type=Path, help='Specific files to inspect instead of a single file.')
    parser.add_argument('--verbose', action='store_true', help='Print full extracted text and chunk contents.')
    parser.add_argument('--diagnose-unknowns', action='store_true', help='Add direct PDF inspection details for failed_unknown files.')
    parser.add_argument('--timeout-seconds', type=int, default=120, help='Per-file timeout when using --files.')
    return parser.parse_args()


def _sanitize_report_name(file_path: Path) -> str:
    sanitized = re.sub(r'[^A-Za-z0-9._-]+', '_', file_path.name).strip('._-')
    return sanitized or 'document'


def _count_stripped_chars(text: str) -> int:
    return len((text or '').strip())


def _stage_status(*, has_error: bool, has_warn: bool) -> str:
    if has_error:
        return 'ERROR'
    if has_warn:
        return 'WARN'
    return 'OK'


def _looks_like_page_noise(content: str) -> bool:
    stripped = content.strip()
    if not stripped:
        return True
    if re.fullmatch(r'(?:page\s*)?\d+(?:\s*/\s*\d+)?', stripped, flags=re.IGNORECASE):
        return True
    return bool(re.fullmatch(r'#+\s+\S+(?:\s+#+)?', stripped))


def _is_structural_noise(chunk: ChunkData) -> tuple[bool, str | None]:
    stripped = chunk.content.strip()
    if not stripped:
        return True, 'whitespace_only'

    try:
        from informity.indexer import chunker as chunker_module

        if chunker_module._is_header_only_chunk(chunk.content):  # noqa: SLF001
            return True, 'header_only'
    except Exception:
        pass

    if _looks_like_page_noise(stripped):
        return True, 'page_or_header_noise'

    return False, None


def _extract_document(file_path: Path) -> ExtractedDocument:
    extractor = get_extractor(file_path)
    if extractor is None:
        return ExtractedDocument(
            text='',
            source_path=file_path,
            metadata={'error_code': 'no_extractor_for_extension', 'retryable': 'false'},
            preview_text='',
            error=f'No extractor for extension: {file_path.suffix.lower()}',
        )

    if (
        isinstance(extractor, DoclingExtractor)
        and file_path.suffix.lower() == '.pdf'
        and int(getattr(settings, 'scan_file_timeout_seconds', 0) or 0) > 0
    ):
        return extract_pdf_with_orchestrator(
            file_path,
            timeout_seconds=int(settings.scan_file_timeout_seconds),
        )

    return extractor.extract(file_path)




def _inspect_pdf_directly(file_path: Path) -> dict[str, Any]:
    file_size_bytes = file_path.stat().st_size
    direct_error = None
    opens_successfully = False
    page_count = None
    has_text = None
    appears_image_only = None
    pdf_version = None
    page_text_flags: list[bool] = []

    try:
        document = pdfium.PdfDocument(str(file_path))
        opens_successfully = True
        try:
            page_count = len(document)
        except Exception:
            page_count = None
        try:
            pdf_version = document.get_version()
        except Exception:
            pdf_version = None

        text_pages = 0
        total_pages = page_count or 0
        for page_index in range(total_pages):
            page = document.get_page(page_index)
            text_page = None
            try:
                text_page = page.get_textpage()
                try:
                    page_text = text_page.get_text_range()
                except TypeError:
                    page_text = text_page.get_text_range(0, text_page.count_chars())
                has_page_text = bool((page_text or '').strip())
                page_text_flags.append(has_page_text)
                if has_page_text:
                    text_pages += 1
            finally:
                try:
                    text_page.close()
                except Exception:
                    pass
                try:
                    page.close()
                except Exception:
                    pass

        try:
            document.close()
        except Exception:
            pass

        has_text = any(page_text_flags)
        appears_image_only = bool(total_pages > 0 and not has_text)
    except Exception as exc:
        direct_error = str(exc)

    return {
        'file_size_bytes': file_size_bytes,
        'pdf_version': pdf_version,
        'opens_successfully': opens_successfully,
        'page_count': page_count,
        'has_text': has_text,
        'appears_image_only': appears_image_only,
        'direct_open_error': direct_error,
        'under_1kb': file_size_bytes < 1024,
        'page_text_flags': page_text_flags,
    }

def _summarize_extraction(document: ExtractedDocument) -> StageResult:
    extracted_text = document.text or ''
    extracted_chars = len(extracted_text)
    stripped_chars = _count_stripped_chars(extracted_text)
    method = str(document.metadata.get('converter') or document.metadata.get('extractor_strategy') or 'unknown')
    ocr_triggered = method.endswith('+ocr') and document.source_path.suffix.lower() == '.pdf'
    below_threshold = stripped_chars < _MIN_VIABLE_EXTRACTED_CHARS
    failure_reason = _classify_extraction_failure_reason(document, stripped_chars)
    extraction_status = str(getattr(document, 'status', 'ok') or 'ok')

    has_error = bool(document.error) or extraction_status.startswith('failed_') or extraction_status.startswith('skipped_')
    has_warn = False
    if stripped_chars == 0:
        has_error = True
    elif below_threshold:
        has_warn = True
    if method == 'unknown':
        has_warn = True

    summary = (
        f'{method}; chars={extracted_chars}; stripped={stripped_chars}; '
        f'ocr_fallback={ocr_triggered}; below_threshold={below_threshold}'
    )
    return StageResult(
        name='Extraction',
        status=_stage_status(has_error=has_error, has_warn=has_warn),
        summary=summary,
        data={
            'extracted_text_length_chars': extracted_chars,
            'stripped_text_length_chars': stripped_chars,
            'extraction_method': method,
            'ocr_fallback_triggered': ocr_triggered,
            'below_minimum_viable_threshold': below_threshold,
            'threshold_chars': _MIN_VIABLE_EXTRACTED_CHARS,
            'extraction_status': extraction_status,
            'skip_reason': document.skip_reason,
            'extraction_failure_reason': failure_reason,
            'first_500_characters': extracted_text[:_EXTRACTION_PREVIEW_CHARS],
            'error': document.error,
            'error_code': document.metadata.get('error_code'),
            'extractor_metadata': dict(document.metadata),
        },
    )


def _summarize_post_processing(raw_text: str, cleaned_text: str) -> StageResult:
    before_chars = len(raw_text)
    after_chars = len(cleaned_text)
    reduction = 0.0 if before_chars == 0 else max(0.0, (before_chars - after_chars) / before_chars)
    significant_reduction = reduction > 0.20

    has_error = before_chars > 0 and after_chars == 0
    has_warn = significant_reduction

    summary = (
        f'before={before_chars}; after={after_chars}; reduction={reduction:.1%}; '
        f'significant_reduction={significant_reduction}'
    )
    return StageResult(
        name='Post-processing',
        status=_stage_status(has_error=has_error, has_warn=has_warn),
        summary=summary,
        data={
            'text_length_before_chars': before_chars,
            'text_length_after_chars': after_chars,
            'content_reduction_ratio': reduction,
            'significant_content_removed': significant_reduction,
        },
    )


def _classify_extraction_failure_reason(document: ExtractedDocument, stripped_chars: int) -> str | None:
    if stripped_chars > 0:
        return None

    status = str(getattr(document, 'status', '') or '')
    if status.startswith('skipped_'):
        return {
            'skipped_encrypted': 'password_protected',
            'skipped_corrupted': 'corrupted',
            'skipped_unsupported': 'unsupported_format',
            'skipped_empty': 'empty_document',
        }.get(status, 'unknown')

    if not document.error:
        return 'empty_document'

    error_text = ' '.join(
        part
        for part in (
            document.error,
            str(document.metadata.get('error_code') or ''),
            str(document.metadata.get('original_error') or ''),
        )
        if part
    ).casefold()

    if 'password' in error_text or 'encrypted' in error_text:
        return 'password_protected'
    if 'corrupt' in error_text or 'invalid' in error_text:
        return 'corrupted'
    if 'unsupported' in error_text or 'no_extractor_for_extension' in error_text:
        return 'unsupported_format'

    return 'unknown'


def _summarize_chunking(
    parent_chunks: list[ChunkData],
    raw_child_chunks: list[ChunkData],
    child_chunks: list[ChunkData],
    noise_chunks_filtered: int,
    *,
    source_text_empty: bool,
) -> StageResult:
    child_lengths = [len(chunk.content.strip()) for chunk in child_chunks]
    clean_chunk_indexes = {chunk.chunk_index for chunk in child_chunks}
    noise_chunks: list[dict[str, Any]] = []
    below_minimum: list[dict[str, Any]] = []
    above_maximum: list[dict[str, Any]] = []

    for chunk in raw_child_chunks:
        stripped = chunk.content.strip()
        stripped_len = len(stripped)
        is_noise, reason = _is_structural_noise(chunk)
        if is_noise:
            noise_chunks.append({
                'chunk_index': chunk.chunk_index,
                'reason': reason,
                'length_chars': stripped_len,
                'content': chunk.content,
            })
        if chunk.chunk_index in clean_chunk_indexes and stripped_len < _MIN_USEFUL_CHUNK_CHARS:
            below_minimum.append({'chunk_index': chunk.chunk_index, 'length_chars': stripped_len, 'content': chunk.content})
        if chunk.chunk_index in clean_chunk_indexes and stripped_len > _MAX_USEFUL_CHUNK_CHARS:
            above_maximum.append({'chunk_index': chunk.chunk_index, 'length_chars': stripped_len, 'content': chunk.content})

    has_error = False
    if not child_chunks and not source_text_empty and not raw_child_chunks:
        has_error = True
    has_warn = bool(below_minimum or above_maximum or noise_chunks_filtered)
    if raw_child_chunks and not child_chunks and noise_chunks_filtered == len(raw_child_chunks):
        has_warn = True

    summary = (
        f'parents={len(parent_chunks)}; raw_children={len(raw_child_chunks)}; '
        f'clean_children={len(child_chunks)}; min={min(child_lengths) if child_lengths else 0}; '
        f'max={max(child_lengths) if child_lengths else 0}; noise={noise_chunks_filtered}'
    )
    return StageResult(
        name='Chunking',
        status=_stage_status(has_error=has_error, has_warn=has_warn),
        summary=summary,
        data={
            'total_parent_chunks': len(parent_chunks),
            'total_child_chunks': len(child_chunks),
            'raw_child_chunks': len(raw_child_chunks),
            'total_clean_child_chunks': len(child_chunks),
            'noise_chunks_filtered': noise_chunks_filtered,
            'child_chunk_length_chars': {
                'average': mean(child_lengths) if child_lengths else 0.0,
                'minimum': min(child_lengths) if child_lengths else 0,
                'maximum': max(child_lengths) if child_lengths else 0,
            },
            'chunks_below_minimum_useful_length': len(below_minimum),
            'chunks_above_maximum_useful_length': len(above_maximum),
            'structural_noise_chunks': noise_chunks,
            'small_chunk_examples': below_minimum[:10],
            'large_chunk_examples': above_maximum[:10],
        },
    )


def _embed_child_chunks(child_chunks: list[ChunkData]) -> StageResult:
    total_child_chunks = len(child_chunks)
    if total_child_chunks == 0:
        return StageResult(
            name='Embedding',
            status='WARN',
            summary='no child chunks to embed',
            data={
                'embedding_succeeded': False,
                'embedding_vector_dimension': None,
                'expected_embedding_dimension': embedder.get_embedding_dimension(),
                'chunks_failed_to_embed': 0,
                'chunks_skipped_due_to_context': 0,
                'embeddable_chunks': 0,
                'embedded_chunks': 0,
                'failed_chunk_indexes': [],
                'skipped_chunk_indexes': [],
            },
        )

    batch_size = embedder.get_effective_batch_size()
    skipped_chunk_indexes: list[int] = []
    failed_chunk_indexes: list[int] = []
    embedded_vectors: list[list[float]] = []
    embeddable_chunks = 0

    for batch_start in range(0, total_child_chunks, batch_size):
        batch = child_chunks[batch_start:batch_start + batch_size]
        valid_chunks = []
        for chunk in batch:
            if chunk.token_count > _EMBEDDING_MODEL_MAX_TOKENS:
                skipped_chunk_indexes.append(chunk.chunk_index)
                continue
            valid_chunks.append(chunk)

        if not valid_chunks:
            continue

        embeddable_chunks += len(valid_chunks)
        try:
            batch_vectors = embedder.embed_texts([chunk.content for chunk in valid_chunks])
            if len(batch_vectors) != len(valid_chunks):
                raise ValueError(f'Embedding count mismatch: expected {len(valid_chunks)}, got {len(batch_vectors)}')
            embedded_vectors.extend(batch_vectors)
        except Exception:
            failed_chunk_indexes.extend(chunk.chunk_index for chunk in valid_chunks)

    embedded_count = len(embedded_vectors)
    if embeddable_chunks > 0 and embedded_count == 0:
        return StageResult(
            name='Embedding',
            status='ERROR',
            summary='embedding failed for all embeddable chunks',
            data={
                'embedding_succeeded': False,
                'embedding_vector_dimension': None,
                'expected_embedding_dimension': embedder.get_embedding_dimension(),
                'chunks_failed_to_embed': len(failed_chunk_indexes),
                'chunks_skipped_due_to_context': len(skipped_chunk_indexes),
                'embeddable_chunks': embeddable_chunks,
                'embedded_chunks': embedded_count,
                'failed_chunk_indexes': failed_chunk_indexes,
                'skipped_chunk_indexes': skipped_chunk_indexes,
            },
        )

    expected_dimension = embedder.get_embedding_dimension()
    actual_dimension = len(embedded_vectors[0]) if embedded_vectors else None
    dimension_mismatch = (
        actual_dimension is not None
        and expected_dimension > 0
        and actual_dimension != expected_dimension
    )
    has_error = bool(failed_chunk_indexes) or dimension_mismatch
    has_warn = bool(skipped_chunk_indexes) and not has_error
    if embedded_count < embeddable_chunks and not has_error:
        has_warn = True
    if failed_chunk_indexes:
        has_error = True

    summary = (
        f'success={embedded_count > 0}; dimension={actual_dimension or "n/a"}; '
        f'embedded={embedded_count}/{embeddable_chunks}; skipped={len(skipped_chunk_indexes)}'
    )
    return StageResult(
        name='Embedding',
        status=_stage_status(has_error=has_error, has_warn=has_warn),
        summary=summary,
        data={
            'embedding_succeeded': embedded_count > 0 and not has_error,
            'embedding_vector_dimension': actual_dimension,
            'expected_embedding_dimension': expected_dimension,
            'chunks_failed_to_embed': len(failed_chunk_indexes),
            'chunks_skipped_due_to_context': len(skipped_chunk_indexes),
            'embeddable_chunks': embeddable_chunks,
            'embedded_chunks': embedded_count,
            'dimension_mismatch': dimension_mismatch,
            'failed_chunk_indexes': failed_chunk_indexes,
            'skipped_chunk_indexes': skipped_chunk_indexes,
        },
    )


def _print_stage(stage_number: int, stage: StageResult) -> None:
    print(f'Stage {stage_number} — {stage.name} [{stage.status}] {stage.summary}')


def _write_timeout_report(
    file_path: Path,
    *,
    verbose: bool,
    diagnose_unknowns: bool,
    output_dir: Path | None = None,
    timeout_seconds: int = 120,
) -> dict[str, Any]:
    resolved_path = file_path.expanduser().resolve()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_root = output_dir or (Path.home() / '.informity' / 'diagnostics')
    report_root.mkdir(parents=True, exist_ok=True)
    report_path = report_root / f'index_quality_{_sanitize_report_name(resolved_path)}_{timestamp}.json'

    extraction_stage = StageResult(
        name='Extraction',
        status='ERROR',
        summary='timed out before extraction completed',
        data={
            'extracted_text_length_chars': 0,
            'stripped_text_length_chars': 0,
            'extraction_method': 'unknown',
            'ocr_fallback_triggered': False,
            'below_minimum_viable_threshold': True,
            'threshold_chars': _MIN_VIABLE_EXTRACTED_CHARS,
            'extraction_status': 'timeout',
            'skip_reason': f'diagnostic timed out after {timeout_seconds}s',
            'extraction_failure_reason': 'timeout',
            'first_500_characters': '',
            'error': f'diagnostic timed out after {timeout_seconds}s',
            'error_code': 'timeout',
            'extractor_metadata': {},
        },
    )
    post_processing_stage = StageResult(
        name='Post-processing',
        status='ERROR',
        summary='not run due to timeout',
        data={
            'text_length_before_chars': 0,
            'text_length_after_chars': 0,
            'content_reduction_ratio': 0.0,
            'significant_content_removed': False,
        },
    )
    chunking_stage = StageResult(
        name='Chunking',
        status='ERROR',
        summary='not run due to timeout',
        data={
            'total_parent_chunks': 0,
            'total_child_chunks': 0,
            'raw_child_chunks': 0,
            'total_clean_child_chunks': 0,
            'noise_chunks_filtered': 0,
            'child_chunk_length_chars': {'average': 0.0, 'minimum': 0, 'maximum': 0},
            'chunks_below_minimum_useful_length': 0,
            'chunks_above_maximum_useful_length': 0,
            'structural_noise_chunks': [],
            'small_chunk_examples': [],
            'large_chunk_examples': [],
        },
    )
    embedding_stage = StageResult(
        name='Embedding',
        status='ERROR',
        summary='not run due to timeout',
        data={
            'embedding_succeeded': False,
            'embedding_vector_dimension': None,
            'expected_embedding_dimension': embedder.get_embedding_dimension(),
            'chunks_failed_to_embed': 0,
            'chunks_skipped_due_to_context': 0,
            'embeddable_chunks': 0,
            'embedded_chunks': 0,
            'failed_chunk_indexes': [],
            'skipped_chunk_indexes': [],
        },
    )

    report: dict[str, Any] = {
        'input_file': str(resolved_path),
        'timestamp': timestamp,
        'overall_status': 'ERROR',
        'stages': {
            extraction_stage.name: asdict(extraction_stage),
            post_processing_stage.name: asdict(post_processing_stage),
            chunking_stage.name: asdict(chunking_stage),
            embedding_stage.name: asdict(embedding_stage),
        },
        'artifacts': {
            'full_extracted_text': '' if verbose else None,
            'cleaned_text': '' if verbose else None,
            'parent_chunks': [] if verbose else None,
            'child_chunks': [] if verbose else None,
        },
        'report_path': str(report_path),
    }
    if diagnose_unknowns:
        report['unknown_diagnostics'] = {
            'raw_extraction_error': f'diagnostic timed out after {timeout_seconds}s',
            'pdf_direct_inspection': None,
        }

    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    return report


def _run_single_file_via_subprocess(
    file_path: Path,
    *,
    verbose: bool,
    diagnose_unknowns: bool,
    timeout_seconds: int,
) -> int:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        '--file',
        str(file_path),
        '--timeout-seconds',
        str(timeout_seconds),
    ]
    if verbose:
        command.append('--verbose')
    if diagnose_unknowns:
        command.append('--diagnose-unknowns')

    env = os.environ.copy()
    env[_RUN_DIRECT_ENV_VAR] = '1'
    process = subprocess.Popen(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=env,
        start_new_session=True,
    )
    try:
        return_code = process.wait(timeout=timeout_seconds)
        return return_code
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except Exception:
            pass
        try:
            process.wait(timeout=10)
        except Exception:
            pass
        _write_timeout_report(
            file_path,
            verbose=verbose,
            diagnose_unknowns=diagnose_unknowns,
            timeout_seconds=timeout_seconds,
        )
        return 124


def _run_file_list(
    file_paths: list[Path],
    *,
    verbose: bool,
    diagnose_unknowns: bool,
    timeout_seconds: int,
) -> int:
    failures = 0
    for index, file_path in enumerate(file_paths, 1):
        print(f'[{index}/{len(file_paths)}] {file_path}')
        return_code = _run_single_file_via_subprocess(
            file_path,
            verbose=verbose,
            diagnose_unknowns=diagnose_unknowns,
            timeout_seconds=timeout_seconds,
        )
        if return_code != 0:
            failures += 1
    return failures


def run_quality_diagnostic(
    file_path: Path,
    *,
    verbose: bool = False,
    diagnose_unknowns: bool = False,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    resolved_path = file_path.expanduser().resolve()
    if not resolved_path.exists():
        raise FileNotFoundError(f'File not found: {resolved_path}')

    extraction = _extract_document(resolved_path)
    extraction_stage = _summarize_extraction(extraction)

    cleaned_text = post_process_extracted_text(extraction.text or '')
    post_processing_stage = _summarize_post_processing(extraction.text or '', cleaned_text)

    parent_chunks = chunk_text(
        cleaned_text,
        char_to_page_ranges=extraction.char_to_page_ranges,
        char_to_block_type_ranges=extraction.char_to_block_type_ranges,
        char_to_header_level_ranges=extraction.char_to_header_level_ranges,
    )
    raw_child_chunks = create_child_chunks(parent_chunks)
    child_chunks, noise_chunks_filtered = filter_noise_chunks(raw_child_chunks)
    chunking_stage = _summarize_chunking(
        parent_chunks,
        raw_child_chunks,
        child_chunks,
        noise_chunks_filtered,
        source_text_empty=not cleaned_text.strip(),
    )
    embedding_stage = _embed_child_chunks(child_chunks)

    overall_status = 'OK'
    if any(stage.status == 'ERROR' for stage in (extraction_stage, post_processing_stage, chunking_stage, embedding_stage)):
        overall_status = 'ERROR'
    elif any(stage.status == 'WARN' for stage in (extraction_stage, post_processing_stage, chunking_stage, embedding_stage)):
        overall_status = 'WARN'

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_root = output_dir or (Path.home() / '.informity' / 'diagnostics')
    report_root.mkdir(parents=True, exist_ok=True)
    report_path = report_root / f'index_quality_{_sanitize_report_name(resolved_path)}_{timestamp}.json'

    report: dict[str, Any] = {
        'input_file': str(resolved_path),
        'timestamp': timestamp,
        'overall_status': overall_status,
        'stages': {
            extraction_stage.name: asdict(extraction_stage),
            post_processing_stage.name: asdict(post_processing_stage),
            chunking_stage.name: asdict(chunking_stage),
            embedding_stage.name: asdict(embedding_stage),
        },
        'artifacts': {
            'full_extracted_text': extraction.text if verbose else None,
            'cleaned_text': cleaned_text if verbose else None,
            'parent_chunks': [
                {
                    'chunk_index': chunk.chunk_index,
                    'length_chars': len(chunk.content),
                    'content': chunk.content,
                }
                for chunk in parent_chunks
            ] if verbose else None,
            'child_chunks': [
                {
                    'chunk_index': chunk.chunk_index,
                    'parent_chunk_index': chunk.parent_chunk_index,
                    'length_chars': len(chunk.content),
                    'content': chunk.content,
                }
                for chunk in child_chunks
            ] if verbose else None,
        },
    }
    if diagnose_unknowns and extraction_stage.data.get('extraction_failure_reason') == 'unknown':
        report['unknown_diagnostics'] = {
            'raw_extraction_error': extraction.error or extraction_stage.data.get('error') or None,
            'pdf_direct_inspection': _inspect_pdf_directly(resolved_path),
        }

    report['report_path'] = str(report_path)

    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')

    _print_stage(1, extraction_stage)
    _print_stage(2, post_processing_stage)
    _print_stage(3, chunking_stage)
    _print_stage(4, embedding_stage)
    print(f'Overall [{overall_status}] report written to {report_path}')

    if verbose:
        print('\n=== Extracted Text ===')
        print(extraction.text or '')
        print('\n=== Parent Chunks ===')
        for chunk in parent_chunks:
            print(f'[{chunk.chunk_index}] parent len={len(chunk.content)}')
            print(chunk.content)
            print('---')
        print('\n=== Child Chunks ===')
        for chunk in child_chunks:
            print(f'[{chunk.chunk_index}] child len={len(chunk.content)} parent={chunk.parent_chunk_index}')
            print(chunk.content)
            print('---')

    return report


def main() -> int:
    args = parse_args()
    if os.environ.get(_RUN_DIRECT_ENV_VAR) == '1':
        report = run_quality_diagnostic(args.file, verbose=args.verbose, diagnose_unknowns=args.diagnose_unknowns)
        return 1 if report['overall_status'] == 'ERROR' else 0
    if args.files:
        return _run_file_list(
            args.files,
            verbose=args.verbose,
            diagnose_unknowns=args.diagnose_unknowns,
            timeout_seconds=args.timeout_seconds,
        )
    if args.file is None:
        raise SystemExit('Provide either --file or --files')

    return _run_single_file_via_subprocess(
        args.file,
        verbose=args.verbose,
        diagnose_unknowns=args.diagnose_unknowns,
        timeout_seconds=args.timeout_seconds,
    )


if __name__ == '__main__':
    raise SystemExit(main())
