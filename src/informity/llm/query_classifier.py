# ==============================================================================
# Informity AI — Query Classifier (v2)
# LLM-only classification. Structured slot extraction via classifier model.
# ==============================================================================

from dataclasses import dataclass, field
from typing import Literal

import structlog

log = structlog.get_logger(__name__)

CONFIDENCE_HIGH_THRESHOLD = 0.80
CONFIDENCE_MEDIUM_THRESHOLD = 0.55


@dataclass
class QueryClassification:
    """
    Query classification result with intent and filters.

    Attributes:
        intent: Query intent ('metadata', 'focused', 'coverage', 'simple')
        subtype: Internal subtype for routing/diagnostics (not user-facing)
        group_by: Optional grouping dimension extracted from query ('year'|'category'|'file')
        field_hint: Optional field extraction hint (for example, 'box_1')
        source_terms: Source constraint terms extracted from query (for example, filename contains terms)
        year_filter: Extracted year filter (int | None)
        category_filter: Extracted category filter (str | None)
        file_type_filter: Extracted file type filter (str | None)
        filename_filter: Extracted filename filter (str | None)
        block_type_filter: Extracted structural block filter ('table'|'form'|'narrative')
        section_filter: Extracted section hint (str | None)
        is_metadata_query: Whether this is a metadata query (count/enumeration)
        is_file_list_query: Whether this is a file listing query
    """
    intent: Literal['metadata', 'focused', 'coverage', 'simple']
    response_shape: Literal['structured_extract', 'narrative_synthesis', 'metadata_table', 'hybrid'] = 'narrative_synthesis'
    route_candidate: Literal[
        'metadata_inventory',
        'targeted_fact_lookup',
        'structured_field_extraction',
        'cross_document_synthesis',
        'comparative_analysis',
        'audit_or_compliance_brief',
        'continuation_or_refinement',
        'clarification_or_disambiguation',
    ] = 'targeted_fact_lookup'
    confidence: float = 0.5
    alternatives: list[tuple[str, float]] = field(default_factory=list)
    reason_codes: list[str] = field(default_factory=list)
    missing_slots: list[str] = field(default_factory=list)
    subtype: Literal['extract_structured_values', 'aggregate_by_period', 'file_inventory'] | None = None
    has_multi_year_scope: bool = False
    group_by: Literal['year', 'category', 'file'] | None = None
    field_hint: str | None = None
    source_terms: list[str] = field(default_factory=list)
    year_filter: int | None = None
    category_filter: str | None = None
    file_type_filter: str | None = None
    filename_filter: str | None = None
    block_type_filter: Literal['table', 'form', 'narrative'] | None = None
    section_filter: str | None = None
    is_metadata_query: bool = False
    is_file_list_query: bool = False

    @property
    def confidence_band(self) -> Literal['high', 'medium', 'low']:
        if self.confidence >= CONFIDENCE_HIGH_THRESHOLD:
            return 'high'
        if self.confidence >= CONFIDENCE_MEDIUM_THRESHOLD:
            return 'medium'
        return 'low'


def classify_query(query: str) -> QueryClassification:
    """
    Classify query using LLM (Qwen2.5-3B). Always enabled.

    Args:
        query: User query string

    Returns:
        QueryClassification with intent and extracted filters

    Raises:
        LLMError: If classification fails
    """
    from informity.llm.query_classifier_llm import classify_query_llm
    return classify_query_llm(query)


__all__ = [
    'CONFIDENCE_HIGH_THRESHOLD',
    'CONFIDENCE_MEDIUM_THRESHOLD',
    'classify_query',
    'QueryClassification',
]
