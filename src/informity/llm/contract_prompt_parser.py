# ==============================================================================
# Informity AI — Contract Prompt Parser
# Shared extraction/parsing helpers for prompt guidance and contract enforcement.
# ==============================================================================

from __future__ import annotations

import re

EXPLICIT_YEAR_PATTERN = re.compile(r'\b(?:19|20)\d{2}\b')
ORDERED_HEADING_CUES = (
    r'\bin order\b',
    r'\bin this order\b',
    r'\bin sequence\b',
    r'output\s+must\s+contain\s*:\s*##',
    r'sections?\s+must\s+contain\s*:\s*##',
    r'headings?\s+exactly',
    r'headings?\s+in\s+exact\s+order',
    r'exact headings in order',
)
YEAR_SUBSECTION_CUES = (
    r'one\s+subsection\s+per\s+(?:indexed|available|requested)?\s*year',
    r'for\s+each\s+year',
    r'findings\s+by\s+year',
    r'\b(?:by|per)\s+year\b',
)


def extract_required_years(question: str) -> list[int]:
    return sorted({int(match.group(0)) for match in EXPLICIT_YEAR_PATTERN.finditer(str(question or ''))})


def has_ordered_heading_cue(question: str) -> bool:
    text = str(question or '')
    return any(re.search(pattern, text, re.IGNORECASE) for pattern in ORDERED_HEADING_CUES)


def has_year_subsection_cue(question: str) -> bool:
    text = str(question or '')
    return any(re.search(pattern, text, re.IGNORECASE) for pattern in YEAR_SUBSECTION_CUES)


def extract_required_headings(question: str) -> list[str]:
    text = str(question or '')
    headings: list[str] = []
    seen: set[str] = set()

    markdown_headings = re.findall(r'##\s+([^\n#]+)', text)
    for raw_heading in markdown_headings:
        normalized = str(raw_heading).strip().rstrip(' .,;:')
        if not normalized:
            continue
        key = normalized.casefold()
        if key in seen:
            continue
        seen.add(key)
        headings.append(normalized)

    numbered_heads = re.findall(
        r'(?:^|:\s*|,\s*)(?:\d+\)\s*)(.+?)(?=(?:,\s*\d+\)\s)|(?:\.\s|$))',
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    for raw_heading in numbered_heads:
        heading = raw_heading.strip().rstrip(' .,;:')
        if not heading:
            continue
        key = heading.casefold()
        if key in seen:
            continue
        seen.add(key)
        headings.append(heading)

    return headings
