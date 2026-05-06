# ==============================================================================
# Informity AI — Extractor Tests (v2)
# Tests DoclingExtractor (unified) and TextExtractor with sample files and edge cases.
# ==============================================================================

from pathlib import Path

import pytest

from informity.scanner.extractors.base import (
    BaseExtractor,
    get_extractor,
    register_extractors,
)
from informity.scanner.extractors.docling import DoclingExtractor
from informity.scanner.extractors.epub import EpubExtractor
from informity.scanner.extractors.text import TextExtractor

# ==============================================================================
# Registry Tests
# ==============================================================================


class TestExtractorRegistry:
    def test_register_extractors(self) -> None:
        register_extractors()
        # DoclingExtractor handles: .pdf, .docx, .pptx, .xlsx, .html, .htm, .csv
        assert get_extractor(Path("test.pdf")) is not None
        assert get_extractor(Path("test.docx")) is not None
        assert get_extractor(Path("test.pptx")) is not None
        assert get_extractor(Path("test.xlsx")) is not None
        assert get_extractor(Path("test.csv")) is not None
        assert get_extractor(Path("test.html")) is not None
        assert get_extractor(Path("test.htm")) is not None
        assert get_extractor(Path("test.epub")) is not None
        # TextExtractor handles: .txt, .md, .rst, .log, .json, .yaml, .yml, .toml
        assert get_extractor(Path("test.txt")) is not None
        assert get_extractor(Path("test.md")) is not None
        assert get_extractor(Path("test.rst")) is not None
        assert get_extractor(Path("test.log")) is not None

    def test_unknown_extension_returns_none(self) -> None:
        register_extractors()
        assert get_extractor(Path("test.xyz")) is None
        assert get_extractor(Path("test.mp4")) is None

    def test_extractors_implement_protocol(self) -> None:
        extractors = [
            TextExtractor(),
            DoclingExtractor(),
            EpubExtractor(),
        ]
        for ext in extractors:
            assert isinstance(ext, BaseExtractor)


# ==============================================================================
# TextExtractor Tests
# ==============================================================================


class TestTextExtractor:
    def setup_method(self) -> None:
        self.extractor = TextExtractor()

    def test_can_handle(self) -> None:
        assert self.extractor.can_handle(Path("readme.txt"))
        assert self.extractor.can_handle(Path("notes.md"))
        assert self.extractor.can_handle(Path("doc.rst"))
        assert self.extractor.can_handle(Path("server.log"))
        assert self.extractor.can_handle(Path("config.json"))
        assert self.extractor.can_handle(Path("config.yaml"))
        assert not self.extractor.can_handle(Path("file.pdf"))

    def test_extract_txt(self, sample_txt: Path) -> None:
        doc = self.extractor.extract(sample_txt)
        assert doc.text.startswith("Hello, Informity AI!")
        assert doc.word_count > 0
        assert doc.error is None
        assert doc.metadata["encoding"] == "utf-8"
        assert doc.source_path == sample_txt

    def test_extract_md(self, sample_md: Path) -> None:
        doc = self.extractor.extract(sample_md)
        assert "# Heading" in doc.text
        assert doc.word_count > 0
        assert doc.error is None

    def test_empty_file(self, tmp_path: Path) -> None:
        f = tmp_path / "empty.txt"
        f.write_text("")
        doc = self.extractor.extract(f)
        assert doc.text == ""
        assert doc.word_count == 0
        assert doc.error is not None
        assert "empty" in doc.error.lower()

    def test_missing_file(self, tmp_path: Path) -> None:
        doc = self.extractor.extract(tmp_path / "missing.txt")
        assert doc.text == ""
        assert doc.error is not None
        assert "Failed to read" in doc.error

    def test_latin1_encoding(self, tmp_path: Path) -> None:
        f = tmp_path / "latin.txt"
        f.write_bytes("Café crème résumé".encode("latin-1"))
        doc = self.extractor.extract(f)
        assert "Café" in doc.text or "Caf" in doc.text
        assert doc.word_count > 0

    def test_extraction_timing(self, sample_txt: Path) -> None:
        doc = self.extractor.extract(sample_txt)
        assert doc.extraction_time_ms >= 0

    def test_immutable_result(self, sample_txt: Path) -> None:
        doc = self.extractor.extract(sample_txt)
        with pytest.raises(AttributeError):
            doc.text = "modified"  # type: ignore[misc]


# ==============================================================================
# DoclingExtractor Tests
# ==============================================================================


class TestDoclingExtractor:
    pytestmark = pytest.mark.integration

    @staticmethod
    def _skip_if_models_unavailable(doc) -> None:
        error_text = str(getattr(doc, 'error', '') or '')
        if (
            'Full Privacy' in error_text
            or 'required models are not cached' in error_text
        ):
            pytest.skip('Docling models are not cached in this environment')

    def setup_method(self) -> None:
        self.extractor = DoclingExtractor()

    def test_can_handle(self) -> None:
        assert self.extractor.can_handle(Path("doc.pdf"))
        assert self.extractor.can_handle(Path("doc.docx"))
        assert self.extractor.can_handle(Path("slides.pptx"))
        assert self.extractor.can_handle(Path("data.xlsx"))
        assert self.extractor.can_handle(Path("data.csv"))
        assert self.extractor.can_handle(Path("page.html"))
        assert self.extractor.can_handle(Path("page.htm"))
        assert not self.extractor.can_handle(Path("doc.txt"))

    def test_extract_pdf(self, sample_pdf: Path) -> None:
        doc = self.extractor.extract(sample_pdf)
        self._skip_if_models_unavailable(doc)
        # Docling extracts text from PDFs
        assert len(doc.text) > 0
        assert doc.page_count == 2
        assert doc.word_count > 0
        assert doc.error is None
        assert doc.metadata.get("page_count") == "2"

    def test_extract_docx(self, sample_docx: Path) -> None:
        doc = self.extractor.extract(sample_docx)
        self._skip_if_models_unavailable(doc)
        assert "Document Title" in doc.text or "First paragraph" in doc.text
        assert doc.word_count > 0
        assert doc.error is None

    def test_extract_pptx(self, sample_pptx: Path) -> None:
        doc = self.extractor.extract(sample_pptx)
        self._skip_if_models_unavailable(doc)
        assert len(doc.text) > 0
        assert doc.page_count == 2
        assert doc.word_count > 0
        assert doc.error is None

    def test_extract_xlsx(self, sample_xlsx: Path) -> None:
        doc = self.extractor.extract(sample_xlsx)
        self._skip_if_models_unavailable(doc)
        assert len(doc.text) > 0
        assert doc.word_count > 0
        assert doc.error is None

    def test_extract_csv(self, sample_csv: Path) -> None:
        doc = self.extractor.extract(sample_csv)
        self._skip_if_models_unavailable(doc)
        assert "name" in doc.text.lower() or "alice" in doc.text.lower()
        assert doc.word_count > 0
        assert doc.error is None

    def test_extract_html(self, sample_html: Path) -> None:
        doc = self.extractor.extract(sample_html)
        self._skip_if_models_unavailable(doc)
        assert len(doc.text) > 0
        assert doc.word_count > 0
        assert doc.error is None

    def test_missing_file(self, tmp_path: Path) -> None:
        doc = self.extractor.extract(tmp_path / "missing.pdf")
        assert doc.text == ""
        assert doc.error is not None

    def test_corrupt_file(self, tmp_path: Path) -> None:
        f = tmp_path / "corrupt.pdf"
        f.write_bytes(b"this is not a pdf file at all")
        doc = self.extractor.extract(f)
        # Docling may return empty text or error for corrupt files
        assert doc.error is not None or doc.text == ""

    def test_extraction_timing(self, sample_pdf: Path) -> None:
        doc = self.extractor.extract(sample_pdf)
        self._skip_if_models_unavailable(doc)
        assert doc.extraction_time_ms >= 0

    def test_extraction_metadata(self, sample_pdf: Path) -> None:
        doc = self.extractor.extract(sample_pdf)
        self._skip_if_models_unavailable(doc)
        # Docling provides metadata
        assert isinstance(doc.metadata, dict)


class TestEpubExtractor:
    def setup_method(self) -> None:
        self.extractor = EpubExtractor()

    def test_can_handle(self) -> None:
        assert self.extractor.can_handle(Path("book.epub"))
        assert not self.extractor.can_handle(Path("book.pdf"))

    def test_extract_epub(self, sample_epub: Path) -> None:
        doc = self.extractor.extract(sample_epub)
        if doc.error and 'dependency not available' in doc.error.lower():
            pytest.skip('ebooklib is not installed in this environment')
        assert doc.error is None
        assert "Hello from EPUB chapter one." in doc.text
        assert doc.word_count > 0
        assert doc.metadata.get("converter") == "ebooklib"
        assert doc.metadata.get("mime_type") == "application/epub+zip"
        assert doc.metadata.get("title") == "Test EPUB"

    def test_missing_file(self, tmp_path: Path) -> None:
        doc = self.extractor.extract(tmp_path / "missing.epub")
        assert doc.text == ""
        assert doc.error is not None

    def test_corrupt_epub(self, tmp_path: Path) -> None:
        f = tmp_path / "corrupt.epub"
        f.write_bytes(b"not-a-valid-epub")
        doc = self.extractor.extract(f)
        if doc.error and 'dependency not available' in doc.error.lower():
            pytest.skip('ebooklib is not installed in this environment')
        assert doc.error is not None
