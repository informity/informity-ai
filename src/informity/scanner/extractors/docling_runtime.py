from __future__ import annotations

import os
from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    AcceleratorOptions,
    PdfPipelineOptions,
    RapidOcrOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption

from informity.config import (
    DirNames,
    configure_hf_environment,
    ensure_docling_rapidocr_cache_compat,
    settings,
)
from informity.utils.directory_utils import ensure_directory

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


def prepare_docling_runtime() -> Path:
    docling_cache = settings.cache_dir / DirNames.DOCLING
    ensure_directory(docling_cache)
    os.environ['DOCLING_ARTIFACTS_PATH'] = str(docling_cache)
    configure_hf_environment()
    ensure_docling_rapidocr_cache_compat(settings.cache_dir)
    return docling_cache


def build_pdf_converter(*, do_ocr: bool, force_full_page_ocr: bool = False) -> DocumentConverter:
    accelerator_options = AcceleratorOptions(num_threads=settings.embedding_max_threads or 4)
    if do_ocr:
        ocr_options = RapidOcrOptions(lang=[])
        ocr_options.force_full_page_ocr = force_full_page_ocr
        pipeline_options = PdfPipelineOptions(
            accelerator_options=accelerator_options,
            do_ocr=True,
            ocr_options=ocr_options,
        )
    else:
        pipeline_options = PdfPipelineOptions(accelerator_options=accelerator_options)

    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
        }
    )
