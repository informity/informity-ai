"""Test module for tests test docling runtime."""

from __future__ import annotations

from docling.datamodel.base_models import InputFormat

from informity.scanner.extractors.docling_runtime import build_docling_converter


def test_build_docling_converter_includes_image_format() -> None:
    """Test build docling converter includes image format."""
    converter = build_docling_converter(do_ocr=False, include_image_formats=True)

    assert InputFormat.PDF in converter.format_to_options
    assert InputFormat.IMAGE in converter.format_to_options
    assert converter.format_to_options[InputFormat.IMAGE].__class__.__name__ == "ImageFormatOption"
