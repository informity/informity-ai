# ==============================================================================
# Informity AI — Text Utilities (v2)
# Shared utilities for text extraction and processing.
# ==============================================================================

import re
import time

from charset_normalizer import from_bytes

MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB
GLYPH_DOMINANCE_RATIO = 0.7


def elapsed_ms(start_time: float) -> float:
    # Calculate elapsed time in milliseconds.
    return (time.perf_counter() - start_time) * 1000


def decode_bytes(raw_bytes: bytes) -> tuple[str, str, str | None]:
    # Decode bytes to string, trying UTF-8 first, then charset-normalizer.
    # Returns (text, encoding, error).
    try:
        text = raw_bytes.decode('utf-8')
        return text, 'utf-8', None
    except UnicodeDecodeError:
        pass

    try:
        best_match = from_bytes(raw_bytes).best()
        if best_match is None:
            return '', 'unknown', 'Failed to decode: unable to detect character set'
        encoding = str(best_match.encoding or 'unknown')
        text = str(best_match)
        return text, encoding, None
    except (UnicodeDecodeError, LookupError, ValueError) as exc:
        return '', 'unknown', f'Failed to decode: {exc}'


def clean_glyph_sequences(text: str) -> str:
    """
    Remove undecodable font glyph sequences from extracted text.

    PDFs with fonts that docling cannot decode produce glyph escape sequences
    that have no semantic value and bloat token counts:
    - /g146/g146/g146... (repeated glyph references)
    - %HGLYPH<c=3,font=/BDJPPO+TimesNewRomanPSMT>... (glyph escape sequences)
    - GLYPH<c=3,font=...> (without % prefix)
    - %XWGLYPH<...> (variant prefixes)

    These sequences provide no value for RAG and can cause:
    - Token count inflation (16k+ tokens for unreadable content)
    - Embedding model confusion (content exceeds context window)

    This is app-compliant because it fixes extraction artifacts at index time,
    not query-time cleaning that creates semantic drift.

    Args:
        text: Text potentially containing glyph sequences

    Returns:
        Text with glyph sequences removed
    """
    if not text:
        return text

    # Pattern 1: Repeated glyph references like /g146/g146/g146...
    # Matches /g followed by digits, repeated 3+ times
    glyph_pattern = re.compile(r'(?:/g\d+){3,}')
    text = glyph_pattern.sub('', text)

    # Pattern 2: All GLYPH escape sequences (catch all variants):
    # - %HGLYPH<...>
    # - %XWGLYPH<...>
    # - GLYPH<...> (without % prefix)
    # These are docling's way of representing undecodable glyphs
    # Match: optional % + optional prefix chars + GLYPH<...> + optional trailing content
    glyph_escape_pattern = re.compile(r'%?[A-Z0-9]*GLYPH<[^>]+>[^%]*', re.IGNORECASE)
    text = glyph_escape_pattern.sub('', text)

    # Pattern 3: Standalone glyph references that might be noise
    # /g followed by digits, but only if they appear frequently (likely noise)
    # We're more conservative here - only remove if they dominate the text
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        # If line is mostly glyph references (70%+), remove it
        glyph_refs = re.findall(r'/g\d+', line)
        if len(glyph_refs) > 0 and len(glyph_refs) / max(len(line.split()), 1) > GLYPH_DOMINANCE_RATIO:
            continue  # Skip this line
        cleaned_lines.append(line)

    text = '\n'.join(cleaned_lines)

    # Clean up excessive whitespace left by removals
    text = re.sub(r'\n{3,}', '\n\n', text)  # Max 2 consecutive newlines
    text = text.strip()

    return text
