from pathlib import Path

from informity.file_patterns import extract_year_from_text
from informity.indexer.classifier import extract_year


def test_extract_year_from_text_matches_embedded_year_digits() -> None:
    assert extract_year_from_text('gerasimenko2011annual.pdf') == 2011
    assert extract_year_from_text('foo_2024report.txt') == 2024


def test_extract_year_prefers_filename_embedded_year_before_text_fallback() -> None:
    path = Path('/tmp/gerasimenko2011annual.pdf')
    # Text intentionally contains a different year; filename year must win.
    text = 'Prepared in 2012 with supporting schedules.'
    assert extract_year(path, text) == 2011


def test_extract_year_ignores_longer_digit_runs() -> None:
    assert extract_year_from_text('invoice_20241_draft.pdf') is None
    assert extract_year_from_text('12024-rollup.txt') is None
