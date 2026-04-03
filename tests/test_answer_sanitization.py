from informity.answer_sanitization import (
    DISPLAY_FALLBACK_MESSAGE,
    build_display_answer,
    sanitize_display_answer,
)


def test_sanitize_display_answer_strips_think_and_source_artifacts() -> None:
    raw = "<think>secret</think>Answer body (Source 1)\nSources: 1"
    assert sanitize_display_answer(raw) == "Answer body"


def test_build_display_answer_uses_fallback_for_reasoning_only_output() -> None:
    cleaned, reasoning_only = build_display_answer("<think>internal only</think>")
    assert reasoning_only is True
    assert cleaned == DISPLAY_FALLBACK_MESSAGE


def test_build_display_answer_preserves_non_reasoning_cleaned_text() -> None:
    raw = "Final answer starts.\n<think>incomplete"
    cleaned, reasoning_only = build_display_answer(raw)
    assert reasoning_only is False
    assert cleaned == "Final answer starts."


def test_sanitize_display_answer_normalizes_br_and_lowercase_source_markers() -> None:
    raw = "Row A<br/>Row B [source: 2]\nsource 2"
    assert sanitize_display_answer(raw) == "Row A; Row B"


def test_sanitize_display_answer_strips_double_angle_think_blocks() -> None:
    raw = "<<think>>internal reasoning</think>>Visible output"
    assert sanitize_display_answer(raw) == "Visible output"


def test_sanitize_display_answer_trims_truncated_trailing_markdown_table_row() -> None:
    raw = (
        "| Field | Value |\n"
        "|---|---|\n"
        "| A | 10 |\n"
        "| B | 20"
    )
    assert sanitize_display_answer(raw) == "| Field | Value |\n|---|---|\n| A | 10 |"


def test_sanitize_display_answer_strips_leading_answer_label() -> None:
    raw = "Answer: The declaration was signed in 1776."
    assert sanitize_display_answer(raw) == "The declaration was signed in 1776."


def test_sanitize_display_answer_strips_bold_inline_answer_label() -> None:
    raw = "The documents do not contain this information.\n\n**Answer:** 1776."
    assert sanitize_display_answer(raw) == "The documents do not contain this information.\n\n1776."


def test_sanitize_display_answer_removes_redundant_out_of_corpus_however_sentence() -> None:
    raw = (
        "The provided documents do not contain this information.\n\n"
        "Answer: The US Declaration of Independence was signed in 1776. "
        "However, this information is not contained in the provided documents."
    )
    assert sanitize_display_answer(raw) == (
        "The provided documents do not contain this information.\n\n"
        "The US Declaration of Independence was signed in 1776."
    )
