# ==============================================================================
# Informity AI — File Pattern Utilities (v2)
# Standardized patterns for file metadata extraction and matching
# Single source of truth for extension lists, filename patterns, year extraction
# ==============================================================================

import re
from re import Pattern

from informity.file_types import FILE_TYPE_OPTIONS

# ==============================================================================
# Extension Lists (aggregated from canonical source)
# ==============================================================================

def get_all_supported_extensions() -> list[str]:
    """
    Get flat list of all supported extensions from canonical FILE_TYPE_OPTIONS.

    Returns:
        Sorted list of unique extensions (e.g., ['.csv', '.docx', '.html', ...])
    """
    extensions: list[str] = []
    for opt in FILE_TYPE_OPTIONS:
        extensions.extend(opt['extensions'])
    return sorted(set(extensions))  # Deduplicate and sort


def get_extensions_without_dot() -> list[str]:
    """
    Get extension names without leading dot (for regex patterns).

    Returns:
        List of extension names (e.g., ['csv', 'docx', 'html', ...])
    """
    return [ext[1:] for ext in get_all_supported_extensions()]


# ==============================================================================
# Year Extraction Patterns
# ==============================================================================

# Year pattern: matches years 1900-2099 with digit boundaries.
# Uses digit-boundary guards so embedded years in filenames like
# "gerasimenko2011annual.pdf" are detected.
YEAR_PATTERN: Pattern[str] = re.compile(r'(?<!\d)(19|20)\d{2}(?!\d)')


def extract_year_from_text(text: str) -> int | None:
    """
    Extract year (1900-2099) from text using standardized pattern.

    Args:
        text: Text to search for year

    Returns:
        Year as integer if found, None otherwise
    """
    match = YEAR_PATTERN.search(text)
    return int(match.group(0)) if match else None


# ==============================================================================
# Extension Regex Pattern Builders
# ==============================================================================

def build_extension_regex_pattern(extensions: list[str] | None = None) -> str:
    """
    Build regex alternation pattern for extensions (without dots).

    Example: "csv|docx|html|pdf|txt"

    Args:
        extensions: Optional list of extensions (with dots). If None, uses all supported.

    Returns:
        Regex alternation pattern string
    """
    if extensions is None:
        ext_names = get_extensions_without_dot()
    else:
        ext_names = [ext[1:] if ext.startswith('.') else ext for ext in extensions]
    return '|'.join(ext_names)


def build_extension_pattern_regex(extensions: list[str] | None = None) -> Pattern[str]:
    """
    Build compiled regex pattern for matching extensions in text.

    Pattern matches: ".pdf", ".docx", etc. (with word boundaries)

    Args:
        extensions: Optional list of extensions (with dots). If None, uses all supported.

    Returns:
        Compiled regex pattern
    """
    ext_pattern = build_extension_regex_pattern(extensions)
    pattern = rf'\.({ext_pattern})\b'
    return re.compile(pattern, re.IGNORECASE)


# ==============================================================================
# Filename Pattern Builders
# ==============================================================================

def build_filename_pattern(extensions: list[str] | None = None) -> Pattern[str]:
    """
    Build compiled regex pattern for matching filenames with extensions.

    Pattern matches filenames with spaces, hyphens, dots, ending with extension.
    Example: "2025 Informity - San Diego Business Tax Certificate.pdf"

    The pattern matches filenames that start with a digit or uppercase letter,
    preventing matches from question words like "What" (which start with lowercase).
    For queries like "What does 2025 file.pdf", it correctly matches "2025 file.pdf"
    because it starts with a digit.

    Args:
        extensions: Optional list of extensions (with dots). If None, uses all supported.

    Returns:
        Compiled regex pattern (captures full filename including extension)
    """
    ext_pattern = build_extension_regex_pattern(extensions)
    # Pattern: filename starting with digit (common: "2025 file.pdf") or uppercase letter,
    # containing word chars/spaces/dots/hyphens, ending with extension.
    # To prevent false matches from question words, we require the match to either:
    # 1. Start with a digit (most common case)
    # 2. Be preceded by whitespace/punctuation (not mid-word)
    # Captures: full filename including extension
    # Note: Using negative lookbehind to ensure we don't match from middle of words
    pattern = rf'(?<!\w)([\dA-Z][\w .-]*\.({ext_pattern}))'
    return re.compile(pattern, re.IGNORECASE)


def build_filename_detection_patterns(extensions: list[str] | None = None) -> list[Pattern[str]]:
    """
    Build list of regex patterns for detecting filename references in queries.

    Patterns detect:
    - Explicit: "file named foo.pdf", "file called foo.pdf", "filename: foo.pdf"
    - Natural reference: "in foo.pdf", "about foo.pdf", "for foo.pdf", "of foo.pdf"
    - Question patterns: "what is in foo.pdf", "what does foo.pdf contain", "summarize foo.pdf"

    Args:
        extensions: Optional list of extensions (with dots). If None, uses all supported.

    Returns:
        List of compiled regex patterns
    """
    ext_pattern = build_extension_regex_pattern(extensions)

    patterns: list[Pattern[str]] = []

    # Explicit patterns: "file named", "file called", "filename:"
    patterns.append(re.compile(
        rf'\b(file|document)\s+(named|called|titled)\s+[\w .-]+\.({ext_pattern})\b',
        re.IGNORECASE
    ))

    patterns.append(re.compile(
        rf'\bfilename\s*[:=]\s*[\w .-]+\.({ext_pattern})\b',
        re.IGNORECASE
    ))

    # Natural reference patterns: "in/about/for/of/contains" followed by filename
    patterns.append(re.compile(
        rf'\b(in|about|for|of|contains?|contain)\b.{{0,60}}[\w .-]+\.({ext_pattern})\b',
        re.IGNORECASE
    ))

    # Question patterns: "what is in", "what does", "summarize", "what information is in"
    patterns.append(re.compile(
        rf'\b(what\s+(is|does|information\s+is)|summarize|describe)\b.{{0,60}}[\w .-]+\.({ext_pattern})\b',
        re.IGNORECASE
    ))

    return patterns


# ==============================================================================
# Extension Query Detection Patterns
# ==============================================================================

def build_extension_query_patterns(extensions: list[str] | None = None) -> list[Pattern[str]]:
    """
    Build regex patterns for detecting extension queries (not filename queries).

    Patterns detect:
    - "all PDFs", "every PDF", "each PDF"
    - ".pdf files", "PDF files"
    - "all .pdf files"

    Args:
        extensions: Optional list of extensions (with dots). If None, uses all supported.

    Returns:
        List of compiled regex patterns
    """
    ext_pattern = build_extension_regex_pattern(extensions)

    patterns: list[Pattern[str]] = []

    # Quantifier + extension: "all PDFs", "every PDF", "each PDF"
    patterns.append(re.compile(
        rf'\b(all|every|each|any)\s+({ext_pattern})\b',
        re.IGNORECASE
    ))

    # Extension + file type words: ".pdf files", "PDF files"
    patterns.append(re.compile(
        rf'\.({ext_pattern})\s+(files?|documents?|types?)\b',
        re.IGNORECASE
    ))

    patterns.append(re.compile(
        rf'\b({ext_pattern})\s+files?\b',
        re.IGNORECASE
    ))

    return patterns
