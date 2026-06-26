from __future__ import annotations

import re
from dataclasses import dataclass, field

import structlog

from informity.config import settings
from informity.db.models import ChatMessage
from informity.llm.five_q_classifier import ClassifierContext, FiveQClassifier
from informity.llm.five_q_decision import FiveQDecision
from informity.llm.types import (
    BlockType,
    ConfidenceBand,
    GroupBy,
    IntentLabel,
    IntentProfileId,
    OutputFormat,
    OutputShape,
    QuerySubtype,
    QueryType,
)

log = structlog.get_logger(__name__)

_YEAR_PATTERN = re.compile(r'\b(19|20)\d{2}\b')
_FILENAME_PATTERN = re.compile(r'\b([a-z0-9][a-z0-9._-]*\.[a-z0-9]{1,10})\b', re.IGNORECASE)
_OUTPUT_PATTERNS = {
    OutputFormat.TABLE: re.compile(r'\b(table|markdown table|in columns?)\b', re.IGNORECASE),
    OutputFormat.BULLETS: re.compile(r'\b(bullet points?|bullets?)\b', re.IGNORECASE),
    OutputFormat.CSV: re.compile(r'\b(csv|comma-separated)\b', re.IGNORECASE),
    OutputFormat.LIST: re.compile(r'\b(list format|as a list)\b', re.IGNORECASE),
    OutputFormat.NARRATIVE: re.compile(r'\b(narrative|paragraphs?)\b', re.IGNORECASE),
}
_CURRENT_INFO_PATTERN = re.compile(r'\b(today|now|current|latest|recent|this week|this month)\b', re.IGNORECASE)


@dataclass
class QueryClassification:
    intent: IntentLabel
    response_shape: OutputShape = OutputShape.NARRATIVE_SYNTHESIS
    route_candidate: IntentProfileId = IntentProfileId.TARGETED_FACT_LOOKUP
    confidence: float = 0.5
    alternatives: list[tuple[IntentLabel, float]] = field(default_factory=list)
    reason_codes: list[str] = field(default_factory=list)
    missing_slots: list[str] = field(default_factory=list)
    subtype: QuerySubtype | None = None
    has_multi_year_scope: bool = False
    group_by: GroupBy | None = None
    field_hint: str | None = None
    source_terms: list[str] = field(default_factory=list)
    year_filter: int | None = None
    category_filter: str | None = None
    file_type_filter: str | None = None
    filename_filter: str | None = None
    block_type_filter: BlockType | None = None
    section_filter: str | None = None
    output_format: OutputFormat | None = None
    secondary_intent: IntentLabel | None = None
    filename_exclude: list[str] = field(default_factory=list)
    is_negation_query: bool = False
    is_metadata_query: bool = False
    is_file_list_query: bool = False
    is_continuation: bool = False
    is_scope_reset: bool = False
    needs_current_info: bool = False
    mentions_time: bool = False
    needs_chat_history: bool = False
    action_hints: dict[str, bool] = field(default_factory=dict)
    retrieval_content_query: str | None = None
    retrieval_content_confidence: float = 0.0
    retrieval_content_reasons: list[str] = field(default_factory=list)
    deterministic_override: bool = False
    llm_confidence: float = 0.0
    focus_has_referential_followup: bool = False
    focus_has_topic_shift_cue: bool = False
    focus_explicit_title_reference: bool = False
    focus_referential_title_anchor: str | None = None
    focus_prefer_title_alignment: bool = False
    focus_strict_title_alignment: bool = False
    focus_title_alignment_query: str | None = None
    focus_disable_term_expansion: bool = False
    focus_query_rewritten: bool = False
    focus_rewritten_query: str | None = None
    focus_resolved: bool = False
    shadow_classifier_raw_output: str | None = None
    shadow_classifier_model: str | None = None
    shadow_classifier_decision: dict[str, object] = field(default_factory=dict)

    @property
    def confidence_band(self) -> ConfidenceBand:
        if self.confidence >= settings.classification_confidence_high_threshold:
            return ConfidenceBand.HIGH
        if self.confidence >= settings.classification_confidence_medium_threshold:
            return ConfidenceBand.MEDIUM
        return ConfidenceBand.LOW


def _extract_year_filter(text: str) -> int | None:
    matches = [match.group(0) for match in _YEAR_PATTERN.finditer(text)]
    if len(matches) != 1:
        return None
    year = int(matches[0])
    return year if 1900 <= year <= 2099 else None


def _extract_file_type(text: str) -> str | None:
    lowered = text.casefold()
    if '.pdf' in lowered:
        return '.pdf'
    match = _FILENAME_PATTERN.search(lowered)
    if match and match.group(1).lower().endswith('.pdf'):
        return '.pdf'
    return None


def _extract_filename(text: str) -> str | None:
    match = _FILENAME_PATTERN.search(text)
    if not match:
        return None
    filename = match.group(1).strip()
    return filename or None


def _detect_output_format(text: str) -> OutputFormat | None:
    for output_format, pattern in _OUTPUT_PATTERNS.items():
        if pattern.search(text):
            return output_format
    return None


def _derive_group_by(text: str, decision: FiveQDecision) -> GroupBy | None:
    if decision.partitions and all(re.fullmatch(r'\d{4}', partition) for partition in decision.partitions):
        return GroupBy.YEAR
    if re.search(r'\b(year|by year)\b', text, re.IGNORECASE):
        return GroupBy.YEAR
    return None


def _build_context(
    *,
    history: list[ChatMessage] | None,
    chat_mode: str | None,
    scope_kind: str | None,
    prior_user_query: str | None,
) -> ClassifierContext:
    return ClassifierContext(
        chat_mode=chat_mode,
        scope_kind=scope_kind,
        has_prior_turns=bool(history),
        prior_user_query=prior_user_query,
    )


def _map_decision_to_classification(query: str, decision: FiveQDecision) -> QueryClassification:
    lowered = query.casefold()
    intent = decision.derive_intent()
    route_candidate = decision.derive_route_candidate()
    output_format = _detect_output_format(query)
    year_filter = _extract_year_filter(query)
    file_type_filter = _extract_file_type(query)
    filename_filter = _extract_filename(query)
    group_by = _derive_group_by(query, decision)

    is_metadata_query = decision.source == 'index_metadata'
    is_file_list_query = is_metadata_query and bool(re.search(r'\b(list|enumerate|show|what kinds|what type)\b', lowered))
    needs_chat_history = decision.source == 'chat_history'
    needs_current_info = bool(_CURRENT_INFO_PATTERN.search(query))
    mentions_time = needs_current_info
    is_continuation = bool(re.search(r'\b(continue|keep going|go on|the rest|next part)\b', lowered))

    response_shape = OutputShape.NARRATIVE_SYNTHESIS
    if is_metadata_query:
        response_shape = OutputShape.METADATA_TABLE
    elif decision.operation == 'compare':
        response_shape = OutputShape.HYBRID

    subtype = decision.derive_subtype()
    reason_codes = ['five_q_classifier']
    if decision.source == 'index_metadata':
        reason_codes.append('five_q_index_metadata')
    elif decision.source == 'chat_history':
        reason_codes.append('five_q_chat_history')
    elif decision.source == 'app_knowledge':
        reason_codes.append('five_q_app_knowledge')
    else:
        reason_codes.append('five_q_document_content')

    return QueryClassification(
        intent=intent,
        response_shape=response_shape,
        route_candidate=route_candidate,
        confidence=decision.confidence,
        alternatives=[],
        reason_codes=reason_codes,
        subtype=subtype,
        has_multi_year_scope=len(decision.partitions) > 1,
        group_by=group_by,
        source_terms=[],
        year_filter=year_filter,
        file_type_filter=file_type_filter,
        filename_filter=filename_filter,
        output_format=output_format,
        secondary_intent=None,
        filename_exclude=[],
        is_negation_query=bool(re.search(r'\b(no|not|without|exclude|excluding|except)\b', lowered)),
        is_metadata_query=is_metadata_query,
        is_file_list_query=is_file_list_query,
        is_continuation=is_continuation,
        needs_current_info=needs_current_info,
        mentions_time=mentions_time,
        needs_chat_history=needs_chat_history,
        action_hints={},
        retrieval_content_query=query,
        retrieval_content_confidence=decision.confidence,
        retrieval_content_reasons=['five_q_direct'],
        deterministic_override=False,
        llm_confidence=decision.confidence,
        focus_has_referential_followup=bool(re.search(r'\b(this|that|it|same|above|earlier|previous|prior)\b', lowered)),
        focus_has_topic_shift_cue=False,
        focus_explicit_title_reference=bool(filename_filter),
        focus_referential_title_anchor=None,
        focus_prefer_title_alignment=bool(filename_filter),
        focus_strict_title_alignment=bool(filename_filter),
        focus_title_alignment_query=query if filename_filter else None,
        focus_disable_term_expansion=bool(filename_filter),
        focus_query_rewritten=False,
        focus_rewritten_query=None,
        focus_resolved=True,
        shadow_classifier_decision={
            'source': decision.source,
            'scope': decision.scope,
            'operation': decision.operation,
            'partitions': list(decision.partitions),
            'exhaustive': decision.exhaustive,
            'confidence': decision.confidence,
        },
    )


def classify_query(
    query: str,
    *,
    history: list[ChatMessage] | None = None,
    chat_mode: str | None = None,
    scope_kind: str | None = None,
    prior_user_query: str | None = None,
) -> QueryClassification:
    text = str(query or '').strip()
    if not text:
        return QueryClassification(intent=QueryType.SIMPLE, confidence=0.0, shadow_classifier_raw_output='')

    context = _build_context(
        history=history,
        chat_mode=chat_mode,
        scope_kind=scope_kind,
        prior_user_query=prior_user_query,
    )
    classifier = FiveQClassifier()
    result = classifier.classify(text, context)
    classification = _map_decision_to_classification(text, result.decision)
    classification.shadow_classifier_raw_output = result.raw_output or None
    classification.shadow_classifier_model = result.model_name
    return classification


__all__ = ['QueryClassification', 'classify_query']
