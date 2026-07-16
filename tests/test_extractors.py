# ==============================================================================
# Informity AI — Extractor Tests (v2)
# Tests DoclingExtractor (unified) and TextExtractor with sample files and edge cases.
# ==============================================================================

"""Test module for tests test extractors."""

from pathlib import Path

# pylint: disable=redefined-outer-name
# pylint: disable=unused-argument
import pytest

from informity.api import routes_translate
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
    """Class docstring."""
    def test_register_extractors(self) -> None:
        """Test register extractors."""
        register_extractors()
        # DoclingExtractor handles: .pdf, image, .docx, .pptx, .xlsx, .html, .htm, .csv
        assert get_extractor(Path("test.pdf")) is not None
        assert get_extractor(Path("test.jpg")) is not None
        assert get_extractor(Path("test.png")) is not None
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
        """Test unknown extension returns none."""
        register_extractors()
        assert get_extractor(Path("test.xyz")) is None
        assert get_extractor(Path("test.mp4")) is None

    def test_extractors_implement_protocol(self) -> None:
        """Test extractors implement protocol."""
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
    """Class docstring."""
    extractor: TextExtractor

    def setup_method(self) -> None:
        """Setup method."""
        self.extractor = TextExtractor()

    def test_can_handle(self) -> None:
        """Test can handle."""
        assert self.extractor.can_handle(Path("readme.txt"))
        assert self.extractor.can_handle(Path("notes.md"))
        assert self.extractor.can_handle(Path("doc.rst"))
        assert self.extractor.can_handle(Path("server.log"))
        assert self.extractor.can_handle(Path("config.json"))
        assert self.extractor.can_handle(Path("config.yaml"))
        assert not self.extractor.can_handle(Path("file.pdf"))

    def test_extract_txt(self, sample_txt: Path) -> None:
        """Test extract txt."""
        doc = self.extractor.extract(sample_txt)
        assert doc.text.startswith("Hello, Informity AI!")
        assert doc.word_count > 0
        assert doc.error is None
        assert doc.metadata["encoding"] == "utf-8"
        assert doc.source_path == sample_txt

    def test_extract_md(self, sample_md: Path) -> None:
        """Test extract md."""
        doc = self.extractor.extract(sample_md)
        assert "# Heading" in doc.text
        assert doc.word_count > 0
        assert doc.error is None

    def test_empty_file(self, tmp_path: Path) -> None:
        """Test empty file."""
        f = tmp_path / "empty.txt"
        f.write_text("")
        doc = self.extractor.extract(f)
        assert doc.text == ""
        assert doc.word_count == 0
        assert doc.error is not None
        assert "empty" in doc.error.lower()

    def test_missing_file(self, tmp_path: Path) -> None:
        """Test missing file."""
        doc = self.extractor.extract(tmp_path / "missing.txt")
        assert doc.text == ""
        assert doc.error is not None
        assert "Failed to read" in doc.error

    def test_latin1_encoding(self, tmp_path: Path) -> None:
        """Test latin1 encoding."""
        f = tmp_path / "latin.txt"
        f.write_bytes("Café crème résumé".encode("latin-1"))
        doc = self.extractor.extract(f)
        assert "Café" in doc.text or "Caf" in doc.text
        assert doc.word_count > 0

    def test_extraction_timing(self, sample_txt: Path) -> None:
        """Test extraction timing."""
        doc = self.extractor.extract(sample_txt)
        assert doc.extraction_time_ms >= 0

    def test_immutable_result(self, sample_txt: Path) -> None:
        """Test immutable result."""
        doc = self.extractor.extract(sample_txt)
        with pytest.raises(AttributeError):
            doc.text = "modified"  # type: ignore[misc]


# ==============================================================================
# DoclingExtractor Tests
# ==============================================================================


class TestDoclingExtractor:
    """Class docstring."""
    pytestmark = pytest.mark.integration
    extractor: DoclingExtractor

    @staticmethod
    def _skip_if_models_unavailable(doc) -> None:
        """Internal helper for skip if models unavailable."""
        error_text = str(getattr(doc, "error", "") or "")
        if "Full Privacy" in error_text or "required models are not cached" in error_text:
            pytest.skip("Docling models are not cached in this environment")

    def setup_method(self) -> None:
        """Setup method."""
        self.extractor = DoclingExtractor()

    def test_can_handle(self) -> None:
        """Test can handle."""
        assert self.extractor.can_handle(Path("doc.pdf"))
        assert self.extractor.can_handle(Path("image.jpg"))
        assert self.extractor.can_handle(Path("image.png"))
        assert self.extractor.can_handle(Path("image.tiff"))
        assert self.extractor.can_handle(Path("doc.docx"))
        assert self.extractor.can_handle(Path("slides.pptx"))
        assert self.extractor.can_handle(Path("data.xlsx"))
        assert self.extractor.can_handle(Path("data.csv"))
        assert self.extractor.can_handle(Path("page.html"))
        assert self.extractor.can_handle(Path("page.htm"))
        assert not self.extractor.can_handle(Path("doc.txt"))

    def test_extract_image_uses_ocr_fallback(self, monkeypatch, tmp_path: Path) -> None:
        """Test extract image uses ocr fallback."""
        image_file = tmp_path / "scan.png"
        image_file.write_bytes(b"not-a-real-image-but-good-enough-for-a-mocked-test")

        class _EmptyDocument:
            """Class docstring."""
            tables: list[object] = []
            form_items: list[object] = []
            key_value_items: list[object] = []
            pictures: list[object] = []
            pages: list[object] = []

            def iterate_items(self, with_groups: bool = True):  # type: ignore[no-untyped-def]
                """Iterate items."""
                return iter(())

            def export_to_markdown(self) -> str:
                """Export to markdown."""
                return ""

            def export_to_text(self) -> str:
                """Export to text."""
                return ""

        class _EmptyResult:
            """Class docstring."""
            def __init__(self) -> None:
                """Initialize the instance."""
                self.document = _EmptyDocument()
                self.input = type("Input", (), {"page_count": 1, "document_hash": "hash"})()

        class _OcrDocument:
            """Class docstring."""
            pages: list[object] = [object()]

            def export_to_markdown(self) -> str:
                """Export to markdown."""
                return "OCR text from image"

            def export_to_text(self) -> str:
                """Export to text."""
                return "OCR text from image"

        class _OcrResult:
            """Class docstring."""
            def __init__(self) -> None:
                """Initialize the instance."""
                self.document = _OcrDocument()
                self.input = type("Input", (), {"page_count": 1})()

        monkeypatch.setattr(
            self.extractor,
            "_get_converter",
            self._make_converter(_EmptyResult),
        )
        monkeypatch.setattr(
            self.extractor,
            "_create_ocr_converter",
            self._make_converter(_OcrResult),
        )

        doc = self.extractor.extract(image_file)

        assert doc.error is None
        assert doc.text == "OCR text from image"
        assert doc.metadata.get("ocr_used") == "true"
        assert doc.metadata.get("converter") == "docling+ocr"
        assert doc.word_count > 0
        assert doc.page_count == 1

    def test_extract_sparse_image_text_triggers_ocr_fallback(
        self, monkeypatch, tmp_path: Path
    ) -> None:
        """Test extract sparse image text triggers ocr fallback."""
        image_file = tmp_path / "scan-sparse.png"
        image_file.write_bytes(b"not-a-real-image-but-good-enough-for-a-mocked-test")

        class _SparseDocument:
            """Class docstring."""
            tables: list[object] = []
            form_items: list[object] = []
            key_value_items: list[object] = []
            pictures: list[object] = []
            pages: list[object] = [object()]

            def iterate_items(self, with_groups: bool = True):  # type: ignore[no-untyped-def]
                """Iterate items."""
                return iter(())

            def export_to_markdown(self) -> str:
                """Export to markdown."""
                return "Figure 1"

            def export_to_text(self) -> str:
                """Export to text."""
                return "Figure 1"

        class _SparseResult:
            """Class docstring."""
            def __init__(self) -> None:
                """Initialize the instance."""
                self.document = _SparseDocument()
                self.input = type("Input", (), {"page_count": 1, "document_hash": "hash"})()

        class _OcrDocument:
            """Class docstring."""
            pages: list[object] = [object()]

            def export_to_markdown(self) -> str:
                """Export to markdown."""
                return "OCR text from sparse image"

            def export_to_text(self) -> str:
                """Export to text."""
                return "OCR text from sparse image"

        class _OcrResult:
            """Class docstring."""
            def __init__(self) -> None:
                """Initialize the instance."""
                self.document = _OcrDocument()
                self.input = type("Input", (), {"page_count": 1})()

        monkeypatch.setattr(
            self.extractor,
            "_get_converter",
            self._make_converter(_SparseResult),
        )
        monkeypatch.setattr(
            self.extractor,
            "_create_ocr_converter",
            self._make_converter(_OcrResult),
        )

        doc = self.extractor.extract(image_file)

        assert doc.error is None
        assert doc.text == "OCR text from sparse image"
        assert doc.metadata.get("ocr_used") == "true"
        assert doc.metadata.get("converter") == "docling+ocr"
        assert doc.word_count > 0
        assert doc.page_count == 1

    @pytest.mark.parametrize("fixture_name", ["sample_ocr_png", "sample_ocr_jpeg"])
    def test_extract_real_image_uses_ocr(self, request, fixture_name: str) -> None:
        """Test extract real image uses ocr."""
        image_file: Path = request.getfixturevalue(fixture_name)

        doc = self.extractor.extract(image_file)
        self._skip_if_models_unavailable(doc)

        assert doc.error is None
        assert doc.metadata.get("ocr_used") == "true"
        assert doc.text.strip() != ""
        assert "ocr" in doc.text.lower() or "smoke" in doc.text.lower()
        assert doc.word_count > 0
        assert doc.preview_text.strip() != ""

    def test_extract_pdf(self, sample_pdf: Path) -> None:
        """Test extract pdf."""
        doc = self.extractor.extract(sample_pdf)
        self._skip_if_models_unavailable(doc)
        # Docling extracts text from PDFs
        assert len(doc.text) > 0
        assert doc.page_count == 2
        assert doc.word_count > 0
        assert doc.error is None
        assert doc.metadata.get("page_count") == "2"

    def test_extract_docx(self, sample_docx: Path) -> None:
        """Test extract docx."""
        doc = self.extractor.extract(sample_docx)
        self._skip_if_models_unavailable(doc)
        assert "Document Title" in doc.text or "First paragraph" in doc.text
        assert doc.word_count > 0
        assert doc.error is None

    def test_extract_contract_docx_preserves_structure(self, sample_contract_docx: Path) -> None:
        """Test extract contract docx preserves structure."""
        doc = self.extractor.extract(sample_contract_docx)
        self._skip_if_models_unavailable(doc)
        assert doc.error is None
        assert "Consulting Agreement" in doc.text
        assert "Confidentiality" in doc.text
        assert doc.text.count("\n\n") >= 3
        sections = routes_translate._split_text_for_translation(doc.text, max_tokens=50)
        assert len(sections) > 1
        assert all(routes_translate._count_tokens(section) <= 50 for section in sections)

    def test_extract_pptx(self, sample_pptx: Path) -> None:
        """Test extract pptx."""
        doc = self.extractor.extract(sample_pptx)
        self._skip_if_models_unavailable(doc)
        assert len(doc.text) > 0
        assert doc.page_count == 2
        assert doc.word_count > 0
        assert doc.error is None

    def test_extract_xlsx(self, sample_xlsx: Path) -> None:
        """Test extract xlsx."""
        doc = self.extractor.extract(sample_xlsx)
        self._skip_if_models_unavailable(doc)
        assert len(doc.text) > 0
        assert doc.word_count > 0
        assert doc.error is None

    def test_extract_csv(self, sample_csv: Path) -> None:
        """Test extract csv."""
        doc = self.extractor.extract(sample_csv)
        self._skip_if_models_unavailable(doc)
        assert "name" in doc.text.lower() or "alice" in doc.text.lower()
        assert doc.word_count > 0
        assert doc.error is None

    def test_extract_html(self, sample_html: Path) -> None:
        """Test extract html."""
        doc = self.extractor.extract(sample_html)
        self._skip_if_models_unavailable(doc)
        assert len(doc.text) > 0
        assert doc.word_count > 0
        assert doc.error is None

    def test_missing_file(self, tmp_path: Path) -> None:
        """Test missing file."""
        doc = self.extractor.extract(tmp_path / "missing.pdf")
        assert doc.text == ""
        assert doc.error is not None

    def test_corrupt_file(self, tmp_path: Path) -> None:
        """Test corrupt file."""
        f = tmp_path / "corrupt.pdf"
        f.write_bytes(b"this is not a pdf file at all")
        doc = self.extractor.extract(f)
        # Docling may return empty text or error for corrupt files
        assert doc.error is not None or doc.text == ""

    def test_extraction_timing(self, sample_pdf: Path) -> None:
        """Test extraction timing."""
        doc = self.extractor.extract(sample_pdf)
        self._skip_if_models_unavailable(doc)
        assert doc.extraction_time_ms >= 0

    def test_extraction_metadata(self, sample_pdf: Path) -> None:
        """Test extraction metadata."""
        doc = self.extractor.extract(sample_pdf)
        self._skip_if_models_unavailable(doc)
        # Docling provides metadata
        assert isinstance(doc.metadata, dict)


class TestEpubExtractor:
    """Class docstring."""
    extractor: EpubExtractor

    def setup_method(self) -> None:
        """Setup method."""
        self.extractor = EpubExtractor()

    def test_can_handle(self) -> None:
        """Test can handle."""
        assert self.extractor.can_handle(Path("book.epub"))
        assert not self.extractor.can_handle(Path("book.pdf"))

    def test_extract_epub(self, sample_epub: Path) -> None:
        """Test extract epub."""
        doc = self.extractor.extract(sample_epub)
        if doc.error and "dependency not available" in doc.error.lower():
            pytest.skip("ebooklib is not installed in this environment")
        assert doc.error is None
        assert "Hello from EPUB chapter one." in doc.text
        assert doc.word_count > 0
        assert doc.metadata.get("converter") == "ebooklib"
        assert doc.metadata.get("mime_type") == "application/epub+zip"
        assert doc.metadata.get("title") == "Test EPUB"

    def test_missing_file(self, tmp_path: Path) -> None:
        """Test missing file."""
        doc = self.extractor.extract(tmp_path / "missing.epub")
        assert doc.text == ""
        assert doc.error is not None

    def test_corrupt_epub(self, tmp_path: Path) -> None:
        """Test corrupt epub."""
        f = tmp_path / "corrupt.epub"
        f.write_bytes(b"not-a-valid-epub")
        doc = self.extractor.extract(f)
        if doc.error and "dependency not available" in doc.error.lower():
            pytest.skip("ebooklib is not installed in this environment")
        assert doc.error is not None
