from informity.answer_sanitization import (
    DISPLAY_FALLBACK_MESSAGE,
    build_display_answer,
    count_words,
    extract_requested_max_words,
    normalize_assistant_identity_claim,
    sanitize_display_answer,
    truncate_to_word_limit,
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


def test_extract_requested_max_words_parses_common_contract_cues() -> None:
    assert extract_requested_max_words("Summarize in <= 180 words.") == 180
    assert extract_requested_max_words("Use at most 75 words.") == 75
    assert extract_requested_max_words("No limit specified.") is None


def test_truncate_to_word_limit_trims_overflow() -> None:
    text = "one two three four five six seven"
    truncated, applied = truncate_to_word_limit(text, 5)
    assert applied is True
    assert count_words(truncated) <= 5
    assert truncated == "one two three four five"


def test_truncate_to_word_limit_noop_within_limit() -> None:
    text = "one two three"
    truncated, applied = truncate_to_word_limit(text, 5)
    assert applied is False
    assert truncated == text


def test_normalize_assistant_identity_claim_rewrites_qwen_intro() -> None:
    raw = "My name is Qwen. I'm a large language model created by Alibaba Cloud. How can I help?"
    normalized = normalize_assistant_identity_claim(raw)
    assert normalized.startswith("I’m Informity AI, your local assistant.")
    assert "How can I help?" in normalized
    assert "Qwen" not in normalized


def test_normalize_assistant_identity_claim_keeps_regular_qwen_reference() -> None:
    raw = "Qwen is one of the available model families in this app."
    normalized = normalize_assistant_identity_claim(raw)
    assert normalized == raw


def test_build_display_answer_applies_identity_guard_before_cleaning() -> None:
    raw = "I am Qwen, created by Alibaba Cloud."
    cleaned, reasoning_only = build_display_answer(raw)
    assert reasoning_only is False
    assert cleaned == "I’m Informity AI, your local assistant."


def test_sanitize_display_answer_removes_overcautious_opening_without_meta_replacement() -> None:
    raw = (
        "Based on the provided text, a complete summary is not available. "
        "However, the following plot elements can be synthesized from context:\n\n"
        "- D'Artagnan travels to Paris."
    )
    assert sanitize_display_answer(raw) == "- D'Artagnan travels to Paris."


def test_sanitize_display_answer_strips_limitations_and_scope_meta_sections() -> None:
    raw = (
        "Character summary here.\n\n"
        "Limitations of the Provided Text\n"
        "The provided documents do not contain all early chapters.\n\n"
        "Note on Scope: The provided text contains excerpts from two works.\n\n"
        "Final grounded point."
    )
    assert sanitize_display_answer(raw) == "Character summary here.\n\nFinal grounded point."
