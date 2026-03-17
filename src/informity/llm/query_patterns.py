# ==============================================================================
# Informity AI — Query Pattern Utilities (v2)
# Standardized patterns for query intent classification
# Building blocks for natural language query understanding
# ==============================================================================

import re
from re import Pattern

# ==============================================================================
# Pattern Building Blocks
# ==============================================================================

# Quantifiers: "all", "every", "each"
QUANTIFIERS: str = r'(all|every|each)'

# Document types: "files", "documents", "reports"
DOCUMENT_TYPES: str = r'(files?|documents?|reports?)'

# List-triggering verbs: "list", "display" (explicit list commands)
LIST_VERBS: str = r'(list|display)'

# Query verbs: "show", "give me" (information retrieval - require quantifiers/document types for list intent)
QUERY_VERBS: str = r'(show|give\s+me)'

# All imperative verbs used by imperative_quantifier_pattern
IMPERATIVE_VERBS: str = rf'({LIST_VERBS}|{QUERY_VERBS})'

# Question words: "what", "which"
QUESTION_WORDS: str = r'(what|which)'

# Coverage verbs: "summarize", "compare", "analyze", "review"
COVERAGE_VERBS: str = r'(summarize|compare|analyze|review)'

# Count queries: "how many"
COUNT_PATTERN: str = r'\bhow\s+many\b'

# Greeting patterns
GREETING_PATTERN: str = r'\b(hello|hi|hey|greetings|thanks|thank\s+you|thank\s+you\s+very\s+much)\b'

# Clarification patterns
CLARIFICATION_PATTERN: str = r'\b(can\s+you\s+clarify|what\s+do\s+you\s+mean|what\s+does\s+that\s+mean|explain\s+that)\b'

# Meta query patterns
META_QUERY_PATTERN: str = r'\b(what\s+can\s+you\s+do|how\s+does\s+this\s+work|what\s+are\s+your\s+capabilities|help)\b'

# Document-related keywords
DOCUMENT_KEYWORDS: str = r'\b(files?|documents?|reports?|search|find|index|indexed|content|data|information)\b'

# Aggregation keywords: date range, min/max, earliest/latest, per year
# Note: Using 'minimum'/'maximum' instead of standalone 'min'/'max' to avoid false positives
# in content queries like "what is the minimum salary" (should be focused, not metadata)
AGGREGATION_KEYWORDS: str = r'\b(date\s+range|range\s+of\s+dates?|earliest|latest|oldest|newest|minimum|maximum|per\s+year|from\s+each\s+year|grouped\s+by|aggregate|summary\s+statistics)\b'

# Aggregation semantics for routing override in classification (Phase 1)
AGGREGATION_SEMANTICS_KEYWORDS: str = (
    r'\b(by[\s-]*year|per[\s-]*year|year[\s-]*by[\s-]*year|aggregate|aggregated|total\s+by|grouped\s+by)\b'
)

# Extraction/task verbs for document-derived outputs (generic, data-agnostic)
EXTRACTION_TASK_VERBS: str = r'\b(create|produce|extract|calculate|sum|total|compare|compile|build)\b'

# Grouping semantics
GROUP_BY_YEAR_KEYWORDS: str = r'\b(by\s+year|per\s+year|group(?:ed)?\s+by\s+year)\b'
GROUP_BY_CATEGORY_KEYWORDS: str = r'\b(by\s+category|per\s+category|group(?:ed)?\s+by\s+category)\b'
GROUP_BY_FILE_KEYWORDS: str = r'\b(by\s+file|per\s+file|group(?:ed)?\s+by\s+file)\b'

# Field extraction hints (generic): "Field 1", "Line 2a", "Row 10"
STRUCTURED_FIELD_HINT_KEYWORDS: str = r'\b(box\s+\d+[a-z]?|line\s+\d+[a-z]?|field\s+\d+[a-z]?)\b'

# Section hints for structured retrieval targeting
SECTION_HINT_KEYWORDS: str = r'\b(section|part|schedule)\s+([a-z0-9][a-z0-9 _-]{0,30})\b'

# Conflict-on-amount semantics (for structured financial contradiction extraction routing).
CONFLICT_AMOUNT_KEYWORDS: str = (
    r'\bfinance[-\s]*related\b.*\bconflict\b.*\b(?:totals?|balances?)\b'
)


# ==============================================================================
# Compiled Pattern Builders
# ==============================================================================

def build_list_pattern() -> Pattern[str]:
    """
    Build regex pattern for list/enumeration queries.

    Matches: "list (all)?", "display (all)?"
    Note: "show me" and "give me" are handled by build_imperative_quantifier_pattern()
    which requires quantifiers/document types to avoid false positives in content queries.

    Returns:
        Compiled regex pattern
    """
    return re.compile(
        rf'\b({LIST_VERBS}(\s+all)?)\b',
        re.IGNORECASE
    )


def build_file_query_pattern() -> Pattern[str]:
    """
    Build regex pattern for file/document queries with quantifiers.

    Matches: "what/which/all files", "all documents", "every report", "each file"

    Returns:
        Compiled regex pattern
    """
    return re.compile(
        rf'\b({QUESTION_WORDS}|{QUANTIFIERS})(\s+(the\s+)?)?{DOCUMENT_TYPES}\b',
        re.IGNORECASE
    )


def build_imperative_quantifier_pattern() -> Pattern[str]:
    """
    Build regex pattern for imperative + quantifier queries.

    Matches: "give me all", "show me every", "list all files"

    Returns:
        Compiled regex pattern
    """
    return re.compile(
        rf'\b({IMPERATIVE_VERBS})(\s+({QUANTIFIERS}))?\s+(the\s+)?{DOCUMENT_TYPES}\b',
        re.IGNORECASE
    )


def build_coverage_pattern() -> Pattern[str]:
    """
    Build regex pattern for coverage queries (broad scope).

    Matches: "all years", "all annual reports", "every document", "each file"
    Also: "summarize all", "compare all", "analyze all"

    Returns:
        Compiled regex pattern
    """
    # Quantifier + optional article/adjective + document type
    quantifier_pattern = rf'\b{QUANTIFIERS}(\s+(the\s+)?)?(\w+\s+){{0,2}}(years?|{DOCUMENT_TYPES})\b'

    # Coverage verb + quantifier + document type
    verb_pattern = rf'\b{COVERAGE_VERBS}(\s+{QUANTIFIERS})(\s+(the\s+)?)?(\w+\s+){{0,2}}(years?|{DOCUMENT_TYPES})\b'

    # Combine both patterns
    combined = rf'({quantifier_pattern}|{verb_pattern})'
    return re.compile(combined, re.IGNORECASE)


def build_count_pattern() -> Pattern[str]:
    """
    Build regex pattern for count queries.

    Matches: "how many files", "how many PDFs"

    Returns:
        Compiled regex pattern
    """
    return re.compile(COUNT_PATTERN, re.IGNORECASE)


def build_greeting_pattern() -> Pattern[str]:
    """
    Build regex pattern for greeting queries.

    Matches: "hello", "hi", "hey", "thanks", etc.

    Returns:
        Compiled regex pattern
    """
    return re.compile(GREETING_PATTERN, re.IGNORECASE)


def build_clarification_pattern() -> Pattern[str]:
    """
    Build regex pattern for clarification queries.

    Matches: "can you clarify", "what do you mean", etc.

    Returns:
        Compiled regex pattern
    """
    return re.compile(CLARIFICATION_PATTERN, re.IGNORECASE)


def build_meta_query_pattern() -> Pattern[str]:
    """
    Build regex pattern for meta queries (about the system itself).

    Matches: "what can you do", "how does this work", "help"

    Returns:
        Compiled regex pattern
    """
    return re.compile(META_QUERY_PATTERN, re.IGNORECASE)


def build_document_keywords_pattern() -> Pattern[str]:
    """
    Build regex pattern for detecting document-related keywords.

    Used to distinguish document queries from off-topic queries.

    Returns:
        Compiled regex pattern
    """
    return re.compile(DOCUMENT_KEYWORDS, re.IGNORECASE)


def build_enumeration_pattern() -> Pattern[str]:
    """
    Build regex pattern for enumeration queries (what years, categories, file types).

    Matches: "what years", "what categories", "what file types", "what extensions",
             "how many years" (count of distinct years, not count of files)

    Returns:
        Compiled regex pattern
    """
    return re.compile(
        rf'\b({QUESTION_WORDS}|how\s+many)\s+(years?|categories?|file\s+types?|extensions?)\b',
        re.IGNORECASE
    )


def build_file_list_pattern() -> Pattern[str]:
    """
    Build regex pattern for file listing queries.

    Matches: "list all files", "show files", "what files", "which files"

    Returns:
        Compiled regex pattern
    """
    # Combine list/show patterns with file query patterns
    # Matches: (list|show|what|which) (all)? files?
    return re.compile(
        rf'\b({IMPERATIVE_VERBS}|{QUESTION_WORDS})\s+(all\s+)?{DOCUMENT_TYPES}\b',
        re.IGNORECASE
    )


def build_aggregation_pattern() -> Pattern[str]:
    """
    Build regex pattern for aggregation queries (date range, min/max, per year).

    Matches: "date range", "earliest", "latest", "minimum/maximum", "per year", "from each year"
    Note: Uses "minimum"/"maximum" instead of standalone "min"/"max" to avoid false positives
    in content queries (e.g., "what is the minimum salary" should be focused, not metadata).

    Returns:
        Compiled regex pattern
    """
    return re.compile(AGGREGATION_KEYWORDS, re.IGNORECASE)


def build_aggregation_semantics_pattern() -> Pattern[str]:
    """
    Build regex pattern for aggregation semantics used by routing override.

    Matches: "by year", "per year", "aggregate", "total by", "grouped by"

    Returns:
        Compiled regex pattern
    """
    return re.compile(AGGREGATION_SEMANTICS_KEYWORDS, re.IGNORECASE)


def build_extraction_task_pattern() -> Pattern[str]:
    """
    Build regex pattern for extraction/aggregation task verbs.

    Matches generic task wording: "create", "extract", "calculate", "compare", etc.

    Returns:
        Compiled regex pattern
    """
    return re.compile(EXTRACTION_TASK_VERBS, re.IGNORECASE)


def build_group_by_year_pattern() -> Pattern[str]:
    """
    Build regex pattern for grouping by year constraints.
    """
    return re.compile(GROUP_BY_YEAR_KEYWORDS, re.IGNORECASE)


def build_group_by_category_pattern() -> Pattern[str]:
    """
    Build regex pattern for grouping by category constraints.
    """
    return re.compile(GROUP_BY_CATEGORY_KEYWORDS, re.IGNORECASE)


def build_group_by_file_pattern() -> Pattern[str]:
    """
    Build regex pattern for grouping by file constraints.
    """
    return re.compile(GROUP_BY_FILE_KEYWORDS, re.IGNORECASE)


def build_structured_field_hint_pattern() -> Pattern[str]:
    """
    Build regex pattern for structured field hints (box/line/field).
    """
    return re.compile(STRUCTURED_FIELD_HINT_KEYWORDS, re.IGNORECASE)


def build_section_hint_pattern() -> Pattern[str]:
    """
    Build regex pattern for section/part/schedule hints.
    """
    return re.compile(SECTION_HINT_KEYWORDS, re.IGNORECASE)


def build_conflict_amount_pattern() -> Pattern[str]:
    """
    Build regex pattern for conflict-on-amount tasks.

    Matches prompts asking for finance-related conflicts on totals/balances.
    """
    return re.compile(CONFLICT_AMOUNT_KEYWORDS, re.IGNORECASE | re.DOTALL)
