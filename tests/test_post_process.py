"""Test module for tests test post process."""

from informity.indexer.post_process import post_process_extracted_text


def test_post_process_splits_ocr_glued_numeric_field_values() -> None:
    """Test post process splits ocr glued numeric field values."""
    raw = (
        "IMPROVEMENTSTOTALASSESSED VALUES425000$1115000335151454891"
        "INCREASE IN ASSESSMENT89849660109LESS NEW EXEMPTIONS "
        "NET VALUE PRORATED(0.58) OF FISCAL YR6601092023-2024"
    )

    cleaned = post_process_extracted_text(raw)

    assert "VALUES 425000" in cleaned
    assert "1115000 " in cleaned
    assert "INCREASE IN ASSESSMENT 898496 60109" in cleaned
    assert "YR 660109 2023-2024" in cleaned


def test_post_process_splits_ocr_glued_numeric_field_run() -> None:
    """Test post process splits ocr glued numeric field run."""
    raw = "NET TAXABLE VALUE7000311216GERASIMENKO DENNIS"

    cleaned = post_process_extracted_text(raw)

    assert cleaned == "NET TAXABLE VALUE 700031 1216 GERASIMENKO DENNIS"


def test_post_process_does_not_split_non_field_long_digits() -> None:
    """Test post process does not split non field long digits."""
    raw = "Reference ID 12345678901234 for processing"

    cleaned = post_process_extracted_text(raw)

    assert cleaned == "Reference ID 12345678901234 for processing"
