# ==============================================================================
# Informity AI — Metadata Filter Extraction (v2)
# Unified metadata filter extraction and WHERE clause building
# ==============================================================================

import re
from dataclasses import dataclass
from datetime import datetime

from dateparser.search import search_dates

from informity.file_patterns import (
    YEAR_PATTERN,
    build_extension_query_patterns,
    build_filename_detection_patterns,
    extract_year_from_text,
    get_all_supported_extensions,
)
from informity.llm.types import FilterOperator

_YEAR_RANGE_PATTERN = re.compile(
    r'\b((?:19|20)\d{2})\s*(?:-|–|—|to|through|thru)\s*((?:19|20)\d{2})\b',
    re.IGNORECASE,
)
_MAX_EXPANDED_YEAR_RANGE = 20
_SUPPORTED_YEAR_MIN = 1900
_SUPPORTED_YEAR_MAX = 2099
_FILTER_QUESTION_WORDS: frozenset[str] = frozenset({
    'what', 'summarize', 'describe', 'information', 'content', 'topic',
    'file', 'document', 'does', 'named', 'of', 'in', 'is', 'the', 'a', 'an',
})
_DATEPARSER_SETTINGS = {
    'STRICT_PARSING': True,
    'PREFER_DAY_OF_MONTH': 'first',
}


def _is_supported_year(year: int) -> bool:
    return _SUPPORTED_YEAR_MIN <= year <= _SUPPORTED_YEAR_MAX


def _extract_years_with_dateparser(query: str) -> set[int]:
    parsed_years: set[int] = set()
    date_hits = search_dates(query, settings=_DATEPARSER_SETTINGS)
    if not date_hits:
        return parsed_years
    for _, parsed_dt in date_hits:
        if not isinstance(parsed_dt, datetime):
            continue
        year = int(parsed_dt.year)
        if _is_supported_year(year):
            parsed_years.add(year)
    return parsed_years


@dataclass(frozen=True)
class MetadataFilter:
    """
    A single metadata filter extracted from a query.

    Attributes:
        field: Metadata field name ('year', 'category', 'file_type', etc.)
        operator: Comparison operator ('eq', 'ne', 'gt', 'gte', 'lt', 'lte', 'in')
        value: Filter value (int, str, or list for 'in' operator)
    """
    field: str
    operator: FilterOperator
    value: int | str | list[int] | list[str]

_ALLOWED_FILTER_FIELDS = {'year', 'category', 'extension', 'filename'}


def _build_filter_sql(
    filter_item: MetadataFilter,
    params: list[int | str],
) -> str | None:
    if filter_item.field not in _ALLOWED_FILTER_FIELDS:
        return None

    col = filter_item.field
    op = filter_item.operator
    value = filter_item.value

    if op == FilterOperator.EQ:
        params.append(value)
        return f'{col} = ?'
    if op == FilterOperator.NE:
        params.append(value)
        return f'{col} != ?'
    if op == FilterOperator.GT:
        params.append(value)
        return f'{col} > ?'
    if op == FilterOperator.GTE:
        params.append(value)
        return f'{col} >= ?'
    if op == FilterOperator.LT:
        params.append(value)
        return f'{col} < ?'
    if op == FilterOperator.LTE:
        params.append(value)
        return f'{col} <= ?'

    if op == FilterOperator.IN:
        if isinstance(value, list) and value:
            placeholders = ', '.join('?' * len(value))
            params.extend(value)
            return f'{col} IN ({placeholders})'
        return None

    if op == FilterOperator.LIKE:
        if isinstance(value, str) and value:
            params.append(value)
            return f'{col} LIKE ?'
        return None

    if op == FilterOperator.CONTAINS_ANY:
        if not isinstance(value, list):
            return None
        terms = [str(item).strip() for item in value if str(item).strip()]
        if not terms:
            return None
        params.extend(f'%{term}%' for term in terms)
        term_clauses = [f'{col} LIKE ?' for _ in terms]
        return f"({' OR '.join(term_clauses)})"

    return None


def build_where_clause_and_params(
    filters: list[MetadataFilter],
) -> tuple[str | None, list[int | str]]:
    """
    Build parameterized SQL WHERE clause and params from metadata filters.

    Returns:
        tuple of (where_clause_without_WHERE_keyword, params)
    """
    params: list[int | str] = []
    clauses: list[str] = []
    for filter_item in filters:
        if clause := _build_filter_sql(filter_item, params):
            clauses.append(clause)
    if not clauses:
        return None, []
    return ' AND '.join(clauses), params


def extract_metadata_filters(query: str) -> list[MetadataFilter]:
    """
    Extract all metadata filters from a query using unified patterns.

    Currently supports:
    - Year extraction (exact years 1900-2099)
    - Category extraction (word-boundary matching)
    - File type extraction (extension matching)
    - Filename extraction (exact filename matching: "in foo.pdf", "file named foo.pdf", etc.)

    Future: relative dates, ranges, multiple values

    Args:
        query: User query string

    Returns:
        List of MetadataFilter objects
    """
    query_lower = query.lower()
    filters: list[MetadataFilter] = []

    # Year extraction:
    # - Single explicit year -> eq
    # - Multiple explicit years or explicit ranges -> in
    # This is generic temporal list/range normalization, not query-specific handling.
    explicit_years = {int(match.group(0)) for match in YEAR_PATTERN.finditer(query)}
    range_years: set[int] = set()
    for match in _YEAR_RANGE_PATTERN.finditer(query):
        start = int(match.group(1))
        end = int(match.group(2))
        low = min(start, end)
        high = max(start, end)
        if (high - low) <= _MAX_EXPANDED_YEAR_RANGE:
            range_years.update(range(low, high + 1))
        else:
            range_years.update({low, high})
    years = sorted(explicit_years | range_years)
    if not years:
        years = sorted(_extract_years_with_dateparser(query))
    if len(years) == 1:
        year = extract_year_from_text(query)
        if year is not None:
            filters.append(MetadataFilter(field='year', operator=FilterOperator.EQ, value=year))
    elif len(years) > 1:
        filters.append(MetadataFilter(field='year', operator=FilterOperator.IN, value=years))

    # Category extraction removed - causes conflicts with extension filters
    # Category filtering only used for metadata queries, extracted by LLM classifier
    # For content queries, extension or semantic search handles filtering

    # File type extraction (extension matching)
    # Distinguish between:
    # - Extension queries: "all PDFs", "find .pdf files" → filter by extension
    # - Filename queries: "file named report.pdf" → filter by exact filename
    supported_extensions = get_all_supported_extensions()

    # Check for filename queries first (more specific than extension queries)
    filename_detection_patterns = build_filename_detection_patterns()
    has_filename_query = any(pattern.search(query) for pattern in filename_detection_patterns)

    # Generic filename-contains extraction for focused/coverage constraints:
    # - filename contains "FormA"
    # - filenames containing "FormA" or "Form-A"
    contains_phrase_match = re.search(
        r'filenames?\s+contain(?:s|ing)?\s+(.+?)(?:,|\.|$)',
        query,
        re.IGNORECASE,
    )
    if contains_phrase_match:
        contains_phrase = contains_phrase_match.group(1).strip()
        quoted_terms = re.findall(r'"([^"]+)"|\'([^\']+)\'', contains_phrase)
        contains_terms = [a or b for a, b in quoted_terms if (a or b)]

        if not contains_terms:
            split_terms = [t.strip() for t in re.split(r'\s+or\s+', contains_phrase, flags=re.IGNORECASE)]
            contains_terms = [t.strip(' "\'') for t in split_terms if t.strip(' "\'')]

        if contains_terms:
            filters.append(MetadataFilter(field='filename', operator=FilterOperator.CONTAINS_ANY, value=contains_terms))

    # Conversational form constraints:
    # - "from FormA forms"
    # - "from Form-A forms"
    # - "from FormB forms"
    # Convert detected form identifiers into filename contains filters.
    form_token_matches = re.findall(
        r'\b([A-Za-z0-9][A-Za-z0-9-]{0,10}\d[A-Za-z0-9-]{0,10})\s+forms?\b',
        query,
        flags=re.IGNORECASE,
    )
    if form_token_matches:
        form_terms: list[str] = []
        for token in form_token_matches:
            normalized = token.strip()
            if not normalized:
                continue
            # Exclude plain years (e.g., "2023 forms"), keep true form identifiers.
            if normalized.isdigit():
                numeric_year = int(normalized)
                if _is_supported_year(numeric_year):
                    continue
            if normalized not in form_terms:
                form_terms.append(normalized)
            # Add simple hyphen/no-hyphen variant for resilient matching.
            if '-' in normalized:
                alt = normalized.replace('-', '')
                if alt and alt not in form_terms:
                    form_terms.append(alt)
            else:
                # Only add hyphen variant when token starts with letters and ends with digits (e.g., Form1 -> Form-1).
                match = re.match(r'^([A-Za-z]+)(\d+)$', normalized)
                if match:
                    alt = f'{match.group(1)}-{match.group(2)}'
                    if alt not in form_terms:
                        form_terms.append(alt)
        if form_terms and not any(
            f.field == 'filename' and f.operator == FilterOperator.CONTAINS_ANY
            for f in filters
        ):
            filters.append(MetadataFilter(field='filename', operator=FilterOperator.CONTAINS_ANY, value=form_terms))

    if has_filename_query and not any(
        f.field == 'filename' and f.operator in {FilterOperator.LIKE, FilterOperator.CONTAINS_ANY}
        for f in filters
    ):
        # Extract exact filename from query (handles filenames with spaces, hyphens, dots)
        # Strategy: Use detection patterns to constrain search region, then extract filename precisely
        # This prevents false matches from question words by only searching where filename likely appears
        # Find detection pattern match to locate filename position
        detection_match = None
        for pattern in filename_detection_patterns:
            match = pattern.search(query)
            if match:
                detection_match = match
                break

        if detection_match:
            # Extract filename from the region matched by detection pattern
            # Detection patterns match phrases like "what does 2025 file.pdf" or "summarize Report.pdf"
            # Strategy: Extract filename starting from first digit (if present) or first uppercase letter
            detection_region = query[detection_match.start():detection_match.end()]

            # Find first digit in detection region (for filenames like "2025 file.pdf")
            digit_match = re.search(r'\d', detection_region)
            if digit_match:
                # Extract from first digit to end (handles "what does 2025 file.pdf")
                filename = detection_region[digit_match.start():]
                filters.append(MetadataFilter(field='filename', operator=FilterOperator.EQ, value=filename))
            else:
                # No digit - extract filename from end of detection region
                # Detection patterns match phrases like "what does Report.pdf" or "file named Presentation.pptx"
                # Filenames end with extensions, so find extension and extract backwards
                ext_names = [ext[1:] for ext in get_all_supported_extensions()]
                ext_pattern = '|'.join(ext_names)
                # Find extension at end of detection region, then extract filename backwards
                # Pattern: word chars/spaces/dots/hyphens ending with extension
                end_filename_pattern = rf'([\w .-]+\.({ext_pattern}))$'
                end_match = re.search(end_filename_pattern, detection_region, re.IGNORECASE)
                if end_match:
                    potential_filename = end_match.group(1)
                    # Filter out question words at the start
                    first_word = potential_filename.split()[0].lower() if potential_filename.split() else ''
                    if first_word not in _FILTER_QUESTION_WORDS:
                        filename = potential_filename
                        filters.append(MetadataFilter(field='filename', operator=FilterOperator.EQ, value=filename))
                    else:
                        # Question word at start - find where filename actually begins
                        # Split by spaces and find first token that starts filename (uppercase/digit, not question word)
                        tokens = potential_filename.split()
                        # Find first token that starts filename (uppercase/digit, not question word)
                        filename_start_idx = None
                        for i, token in enumerate(tokens):
                            if token and (token[0].isdigit() or token[0].isupper()) and token.lower() not in _FILTER_QUESTION_WORDS:
                                filename_start_idx = i
                                break
                        if filename_start_idx is not None:
                            # Extract from filename start token to end
                            filename = ' '.join(tokens[filename_start_idx:])
                            filters.append(MetadataFilter(field='filename', operator=FilterOperator.EQ, value=filename))

            # If no filename extracted, year filter will still be applied for retrieval
    else:
        # Check for extension queries (more common): "PDFs", ".pdf files", "all PDFs"
        extension_query_patterns = build_extension_query_patterns()
        has_extension_query = any(pattern.search(query) for pattern in extension_query_patterns)

        if has_extension_query or any(ext in query_lower for ext in supported_extensions):
            # Extension query: find matching extension
            for ext in supported_extensions:
                if ext in query_lower:
                    filters.append(MetadataFilter(field='extension', operator=FilterOperator.EQ, value=ext))
                    break  # Only one extension filter supported for now

    return filters


def build_where_clause(filters: list[MetadataFilter]) -> str | None:
    """
    Build SQL WHERE clause from a list of metadata filters.

    Args:
        filters: List of MetadataFilter objects

    Returns:
        WHERE clause string (e.g., "year = 2023 AND category = 'document'")
        or None if no filters
    """
    where_clause, _ = build_where_clause_and_params(filters)
    return where_clause
