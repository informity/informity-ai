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

# Continuation cues for follow-up generation on prior context.
CONTINUATION_KEYWORDS: str = (
    r'\b(continue|go\s+on|keep\s+going|next\s+section|the\s+rest)\b'
)
REFERENTIAL_FOLLOWUP_KEYWORDS: str = (
    r'\b('
    r'there|that|those|these|it|they|them|same|above|earlier|previous|prior|'
    r'as\s+discussed|as\s+mentioned|continue|follow[-\s]?up|again'
    r')\b'
)

# Structured output schema directives (format-first requests).
STRUCTURED_OUTPUT_SCHEMA_KEYWORDS: str = (
    r'\b(markdown\s+table|columns?|output\s+only|format|headings?\s+in\s+exact\s+order|'
    r'exact\s+column\s+names?|rows?\s+as)\b'
)

# Analysis/synthesis action directives that imply content generation (not inventory metadata).
ANALYSIS_ACTION_KEYWORDS: str = (
    r'\b(summarize|compare|analyze|synthesize|explain|evaluate|assess|review|'
    r'find\s+one|recommendation|implication|tradeoff|with\s+evidence)\b'
)

# Inventory/capability metadata phrasing for indexed corpus.
INVENTORY_CAPABILITY_KEYWORDS: str = (
    r'\b(what\s+kind\s+of\s+(files?|documents?)\s+do\s+you\s+have|'
    r'what\s+(files?|documents?)\s+are\s+indexed|what\s+is\s+indexed)\b'
)

# Evidence/value extraction cues that indicate content synthesis is needed, not
# metadata-only inventory listing.
EVIDENCE_VALUE_EXTRACTION_KEYWORDS: str = (
    r'\b(evidence|snippet|key\s+amounts?|key\s+values?|numeric|figures?|'
    r'financial|amounts?|values?|found|mentions?|contains?)\b'
)
AGGREGATE_LISTING_SCOPE_KEYWORDS: str = (
    r'\b('
    r'(which|what)\s+(indexed\s+)?(files?|documents?|records?)\b.*\b(contain|contains|mention|mentions|include|includes|list)\b'
    r'|'
    r'across\s+all\s+(indexed\s+)?(files?|documents?|records?)\b'
    r'|'
    r'(names?|dates?|amounts?|figures?|values?)\s+mentioned\s+across\b'
    r')'
)
EXTREME_VALUE_LOOKUP_KEYWORDS: str = (
    r'\b('
    r'(largest|highest|maximum|max|biggest|lowest|minimum|min)\s+'
    r'(amount|value|number|figure|total)s?'
    r'|'
    r'(amount|value|number|figure|total)\s+is\s+the\s+'
    r'(largest|highest|maximum|max|biggest|lowest|minimum|min)'
    r')\b'
)
CONTENT_ANALYSIS_KEYWORDS: str = (
    r'\b('
    r'summarize|summary|compare|contrast|contradictions?|conflicts?|overview|'
    r'main subject|describe|analy[sz]e|findings?|mentioned|tell me about|key fields?|'
    r'what does'
    r')\b'
)
PLURAL_CORPUS_SCOPE_KEYWORDS: str = r'\b(documents|files|records)\b'
SINGLE_TARGET_KEYWORDS: str = (
    r'\b(?:any|one|single|this|that|the)\s+'
    r'(?:[a-z0-9][a-z0-9\s-]{0,40}\s+)?'
    r'(?:document|file|record|receipt|statement|report|return|form|invoice|summary)\b'
)
YEAR_AGGREGATE_CUE_KEYWORDS: str = (
    r'\b('
    r'by year|year[-\s]*by[-\s]*year|year[-\s]*over[-\s]*year|cross[-\s]*year|'
    r'findings by year|evidence map by year|coverage matrix|largest increase|largest decrease|'
    r'deltas?|per indexed year|years covered'
    r')\b'
)
BROAD_SCOPE_EXTRA_KEYWORDS: str = r'\b(across|all|cross[\s-]*document|year[\s-]*by[\s-]*year)\b'
MULTI_DOCUMENT_LISTING_KEYWORDS: str = r'\b(which|list|show)\b.*\b(files?|documents?)\b'
GLOBAL_ENTITY_LISTING_KEYWORDS: str = (
    r'\b('
    r'names?\s+of\s+people|people\s+names?|people\s+mentioned|'
    r'important\s+dates?|key\s+dates?|'
    r'numeric\s+amounts?|key\s+amounts?|financial\s+figures?|financial\s+amounts?'
    r')\b'
)
GENERIC_CAPABILITY_KEYWORDS: str = (
    r'\b(can\s+you\s+help|help\s+me\s+understand|what\s+information\s+is\s+available)\b'
)
FACT_LOOKUP_KEYWORDS: str = r'^\s*(when|what\s+year|which\s+year|who|where|what\s+is|what\s+was|when\s+was)\b'
ANCHOR_DOCUMENT_TERM_KEYWORDS: str = (
    r'\b(?:19|20)\d{2}\s+[a-z0-9][a-z0-9\s-]{1,64}\b(?:receipt|statement|report|return|form|record|invoice|summary)\b'
)
QUOTED_PHRASE_KEYWORDS: str = r'["\']([^"\']{3,80})["\']'
CORPUS_DOCUMENT_SCOPE_KEYWORDS: str = r'\b(indexed\s+)?(files?|documents?|records?)\b'


# ==============================================================================
# Compiled Pattern Builders
# ==============================================================================

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


def build_meta_query_pattern() -> Pattern[str]:
    """
    Build regex pattern for meta queries (about the system itself).

    Matches: "what can you do", "how does this work", "help"

    Returns:
        Compiled regex pattern
    """
    return re.compile(META_QUERY_PATTERN, re.IGNORECASE)


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
    # Matches: (list|show|what|which) (all)? (optional_modifier)? files?
    # e.g. "list all files", "list all indexed documents", "show all PDF documents"
    return re.compile(
        rf'\b({IMPERATIVE_VERBS}|{QUESTION_WORDS})\s+(all\s+(?:\w+\s+)?)?{DOCUMENT_TYPES}\b',
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


def build_conflict_amount_pattern() -> Pattern[str]:
    """
    Build regex pattern for conflict-on-amount tasks.

    Matches prompts asking for finance-related conflicts on totals/balances.
    """
    return re.compile(CONFLICT_AMOUNT_KEYWORDS, re.IGNORECASE | re.DOTALL)


def build_continuation_pattern() -> Pattern[str]:
    """
    Build regex pattern for continuation follow-up requests.
    """
    return re.compile(CONTINUATION_KEYWORDS, re.IGNORECASE)


def build_referential_followup_pattern() -> Pattern[str]:
    """
    Build regex pattern for referential follow-up phrasing.
    """
    return re.compile(REFERENTIAL_FOLLOWUP_KEYWORDS, re.IGNORECASE)


def build_structured_output_schema_pattern() -> Pattern[str]:
    """
    Build regex pattern for explicit structured output schema requests.
    """
    return re.compile(STRUCTURED_OUTPUT_SCHEMA_KEYWORDS, re.IGNORECASE)


def build_analysis_action_pattern() -> Pattern[str]:
    """
    Build regex pattern for analysis/synthesis action directives.
    """
    return re.compile(ANALYSIS_ACTION_KEYWORDS, re.IGNORECASE)


def build_inventory_capability_pattern() -> Pattern[str]:
    """
    Build regex pattern for corpus inventory capability metadata requests.
    """
    return re.compile(INVENTORY_CAPABILITY_KEYWORDS, re.IGNORECASE)


def build_evidence_value_extraction_pattern() -> Pattern[str]:
    """
    Build regex pattern for evidence/value extraction cues.
    """
    return re.compile(EVIDENCE_VALUE_EXTRACTION_KEYWORDS, re.IGNORECASE)


def build_extreme_value_lookup_pattern() -> Pattern[str]:
    """
    Build regex pattern for superlative single-value lookup requests.
    """
    return re.compile(EXTREME_VALUE_LOOKUP_KEYWORDS, re.IGNORECASE)


def build_aggregate_listing_scope_pattern() -> Pattern[str]:
    """
    Build regex pattern for broad corpus listing/synthesis requests.
    """
    return re.compile(AGGREGATE_LISTING_SCOPE_KEYWORDS, re.IGNORECASE)


def build_content_analysis_pattern() -> Pattern[str]:
    """Build regex pattern for content-analysis style requests."""
    return re.compile(CONTENT_ANALYSIS_KEYWORDS, re.IGNORECASE)


def build_plural_corpus_scope_pattern() -> Pattern[str]:
    """Build regex pattern for plural corpus scope cues."""
    return re.compile(PLURAL_CORPUS_SCOPE_KEYWORDS, re.IGNORECASE)


def build_single_target_pattern() -> Pattern[str]:
    """Build regex pattern for single-document targeting cues."""
    return re.compile(SINGLE_TARGET_KEYWORDS, re.IGNORECASE)


def build_year_aggregate_cue_pattern() -> Pattern[str]:
    """Build regex pattern for year-aggregate intent cues."""
    return re.compile(YEAR_AGGREGATE_CUE_KEYWORDS, re.IGNORECASE)


def build_broad_scope_extra_pattern() -> Pattern[str]:
    """Build regex pattern for broad-scope lexical cues."""
    return re.compile(BROAD_SCOPE_EXTRA_KEYWORDS, re.IGNORECASE)


def build_multi_document_listing_pattern() -> Pattern[str]:
    """Build regex pattern for multi-document listing requests."""
    return re.compile(MULTI_DOCUMENT_LISTING_KEYWORDS, re.IGNORECASE)


def build_global_entity_listing_pattern() -> Pattern[str]:
    """Build regex pattern for global entity listing prompts."""
    return re.compile(GLOBAL_ENTITY_LISTING_KEYWORDS, re.IGNORECASE)


def build_generic_capability_pattern() -> Pattern[str]:
    """Build regex pattern for generic capability questions."""
    return re.compile(GENERIC_CAPABILITY_KEYWORDS, re.IGNORECASE)


def build_fact_lookup_pattern() -> Pattern[str]:
    """Build regex pattern for general-world fact lookup prompts."""
    return re.compile(FACT_LOOKUP_KEYWORDS, re.IGNORECASE)


def build_anchor_document_term_pattern() -> Pattern[str]:
    """Build regex pattern for anchored document-term cues."""
    return re.compile(ANCHOR_DOCUMENT_TERM_KEYWORDS, re.IGNORECASE)


def build_quoted_phrase_pattern() -> Pattern[str]:
    """Build regex pattern for quoted phrase extraction."""
    return re.compile(QUOTED_PHRASE_KEYWORDS)


def build_corpus_document_scope_pattern() -> Pattern[str]:
    """Build regex pattern for corpus document-scope references."""
    return re.compile(CORPUS_DOCUMENT_SCOPE_KEYWORDS, re.IGNORECASE)
