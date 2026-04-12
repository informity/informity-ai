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

    # 1) Parse explicit heading lists declared in prompt prose, for example:
    # "headings in exact order: ## A, ## B, ## C."
    inline_list_anchors = (
        r'(?:headings?|sections?)\s+in\s+(?:this\s+)?exact\s+order\s*:\s*',
        r'exactly\s+\d+\s+sections?\s*:\s*',
        r'(?:headings?|sections?)\s+exactly\s*:\s*',
        r'(?:with\s+)?exact\s+headings?\s*:\s*',
        r'with\s+headings?\s+exactly\s*:\s*',
        r'required\s+headings?\s+in\s+order\s*:\s*',
        r'required\s+headings?\s*:\s*',
    )
    for anchor in inline_list_anchors:
        for match in re.finditer(anchor, text, flags=re.IGNORECASE):
            tail = text[match.end():]
            stop_match = re.search(
                r'\b(?:Under|In\s+each|For\s+each|If|Keep|When|Where)\b',
                tail,
                flags=re.IGNORECASE,
            )
            if stop_match is not None:
                tail = tail[:stop_match.start()]
            for inline_heading in re.findall(r'##\s*([^,\n#]+)', tail):
                normalized = str(inline_heading).strip().rstrip(' .,;:')
                if not normalized:
                    continue
                key = normalized.casefold()
                if key in seen:
                    continue
                seen.add(key)
                headings.append(normalized)

    # 2) Parse markdown headings when they are written as standalone lines.
    markdown_headings = re.findall(r'(?im)^\s*##\s+([^\n#]+?)\s*$', text)
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


def extract_required_labels(question: str) -> list[str]:
    """
    Extract explicit label/column contracts from prompt prose.

    Examples:
    - "columns: Field, Value, Source Snippet"
    - "format: Field | Value | Source Snippet"
    """
    text = str(question or '')
    labels: list[str] = []
    seen: set[str] = set()

    anchor_patterns = (
        r'\bcolumns?\s*:\s*',
        r'\bformat\s*:\s*',
    )
    for anchor in anchor_patterns:
        for match in re.finditer(anchor, text, flags=re.IGNORECASE):
            tail = text[match.end():]
            stop_match = re.search(
                r'\b(?:Under|In\s+each|For\s+each|If|Keep|When|Where|Output|Return|Then)\b',
                tail,
                flags=re.IGNORECASE,
            )
            if stop_match is not None:
                tail = tail[:stop_match.start()]
            normalized_tail = re.sub(r'\s+', ' ', tail).strip().strip('.')
            if not normalized_tail:
                continue
            # Support both comma-separated and pipe-separated format declarations.
            if '|' in normalized_tail:
                candidates = [part.strip().strip('`').strip() for part in normalized_tail.split('|')]
            else:
                candidates = [part.strip().strip('`').strip() for part in normalized_tail.split(',')]
            for candidate in candidates:
                if not candidate:
                    continue
                if len(candidate.split()) > 5:
                    continue
                if not re.search(r'[A-Za-z]', candidate):
                    continue
                key = candidate.casefold()
                if key in seen:
                    continue
                seen.add(key)
                labels.append(candidate)
    return labels
