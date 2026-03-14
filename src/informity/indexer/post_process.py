# ==============================================================================
# Informity AI — Post-Processing (v2)
# Quality refinements on extracted text before chunking.
# Fixes extraction artifacts at index time (not query-time cleaning).
# ==============================================================================

import re

from informity.scanner.extractors.text_utils import clean_glyph_sequences

_ALNUM_BOUNDARY_PATTERN = re.compile(r'(?<=[A-Za-z\)])(?=\d)|(?<=\d)(?=[A-Za-z])')
_LONG_DIGIT_RUN_PATTERN = re.compile(r'\b\d{10,}\b')
_NUMERIC_FIELD_HINTS = (
    'value',
    'amount',
    'total',
    'net',
    'balance',
    'increase',
    'decrease',
)


def _split_long_digit_run(token: str) -> str:
    """
    Split OCR-glued digit runs so key assessment amounts become standalone tokens.

    Examples:
    - 89849660109 -> 898496 60109
    - 7000311216  -> 700031 1216
    - 1115000335151454891 -> 1115000 335151454891
    """
    length = len(token)

    # Common OCR glue pattern in tax forms: 6-digit amount + 4/5-digit trailing code.
    if length in {10, 11}:
        left = token[:6]
        right = token[6:]
        if 50_000 <= int(left) <= 9_999_999:
            return f'{left} {right}'

    # Very long runs often begin with the assessed value (7 digits), then extra code noise.
    if length >= 13:
        left = token[:7]
        right = token[7:]
        if 100_000 <= int(left) <= 9_999_999:
            return f'{left} {right}'

    return token


def _should_repair_numeric_glue_line(line: str) -> bool:
    """
    Generic line-level detector for OCR numeric glue.
    Avoids corpus-specific gates while limiting false positives.
    """
    if not _LONG_DIGIT_RUN_PATTERN.search(line):
        return False

    line_lower = line.casefold()
    has_field_hint = any(hint in line_lower for hint in _NUMERIC_FIELD_HINTS)
    has_currency_or_percent = '$' in line or '%' in line
    has_dense_numeric_tokens = len(re.findall(r'\b\d{4,}\b', line)) >= 2

    return has_field_hint or has_currency_or_percent or has_dense_numeric_tokens


def _normalize_ocr_numeric_glue_lines(text: str) -> str:
    """
    Repair OCR-concatenated numeric fields in generic field-like lines.
    """
    repaired_lines: list[str] = []
    for line in text.splitlines():
        normalized_line = _ALNUM_BOUNDARY_PATTERN.sub(' ', line)
        if _should_repair_numeric_glue_line(normalized_line):
            normalized_line = _LONG_DIGIT_RUN_PATTERN.sub(
                lambda match: _split_long_digit_run(match.group(0)),
                normalized_line,
            )
        repaired_lines.append(normalized_line)
    return '\n'.join(repaired_lines)


def post_process_extracted_text(text: str) -> str:
    """
    Apply quality refinements to extracted text before chunking.

    This is app-compliant because it fixes extraction artifacts at index time,
    not query-time cleaning that creates semantic drift.

    Current refinements:
    - Removes undecodable font glyph sequences (e.g., /g146/g146/g146...)
    - Removes HGLYPH escape sequences (e.g., %HGLYPH<c=3,font=...>)
    - Repairs OCR-concatenated numeric runs in generic field-like lines

    Args:
        text: Raw extracted text from extractor

    Returns:
        Cleaned text ready for chunking
    """
    if not text:
        return text

    # Remove glyph sequences that provide no semantic value
    # These are artifacts from PDFs with fonts docling cannot decode
    cleaned = clean_glyph_sequences(text)

    cleaned = _normalize_ocr_numeric_glue_lines(cleaned)

    return cleaned
