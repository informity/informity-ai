from dataclasses import dataclass
from typing import Literal

from informity.llm.query_classifier import (
    CONFIDENCE_HIGH_THRESHOLD,
    CONFIDENCE_MEDIUM_THRESHOLD,
)

IntentProfileId = Literal[
    'metadata_inventory',
    'targeted_fact_lookup',
    'structured_field_extraction',
    'cross_document_synthesis',
    'comparative_analysis',
    'audit_or_compliance_brief',
    'continuation_or_refinement',
    'clarification_or_disambiguation',
]

RetrievalMode = Literal['focused', 'coverage']
OutputShape = Literal['structured_extract', 'narrative_synthesis', 'metadata_table', 'hybrid']
ConfidenceBand = Literal['high', 'medium', 'low']


@dataclass(frozen=True)
class IntentProfilePolicy:
    profile_id: IntentProfileId
    preferred_retrieval_mode: RetrievalMode
    allowed_output_shapes: tuple[OutputShape, ...]
    fallback_target_route: IntentProfileId


_PROFILE_CATALOG: dict[IntentProfileId, IntentProfilePolicy] = {
    'metadata_inventory': IntentProfilePolicy(
        profile_id='metadata_inventory',
        preferred_retrieval_mode='focused',
        allowed_output_shapes=('metadata_table', 'structured_extract'),
        fallback_target_route='clarification_or_disambiguation',
    ),
    'targeted_fact_lookup': IntentProfilePolicy(
        profile_id='targeted_fact_lookup',
        preferred_retrieval_mode='focused',
        allowed_output_shapes=('narrative_synthesis', 'hybrid'),
        fallback_target_route='cross_document_synthesis',
    ),
    'structured_field_extraction': IntentProfilePolicy(
        profile_id='structured_field_extraction',
        preferred_retrieval_mode='focused',
        allowed_output_shapes=('structured_extract', 'hybrid'),
        fallback_target_route='targeted_fact_lookup',
    ),
    'cross_document_synthesis': IntentProfilePolicy(
        profile_id='cross_document_synthesis',
        preferred_retrieval_mode='coverage',
        allowed_output_shapes=('narrative_synthesis', 'hybrid'),
        fallback_target_route='clarification_or_disambiguation',
    ),
    'comparative_analysis': IntentProfilePolicy(
        profile_id='comparative_analysis',
        preferred_retrieval_mode='coverage',
        allowed_output_shapes=('narrative_synthesis', 'metadata_table', 'hybrid'),
        fallback_target_route='cross_document_synthesis',
    ),
    'audit_or_compliance_brief': IntentProfilePolicy(
        profile_id='audit_or_compliance_brief',
        preferred_retrieval_mode='coverage',
        allowed_output_shapes=('narrative_synthesis', 'hybrid'),
        fallback_target_route='cross_document_synthesis',
    ),
    'continuation_or_refinement': IntentProfilePolicy(
        profile_id='continuation_or_refinement',
        preferred_retrieval_mode='focused',
        allowed_output_shapes=('narrative_synthesis', 'structured_extract', 'hybrid'),
        fallback_target_route='clarification_or_disambiguation',
    ),
    'clarification_or_disambiguation': IntentProfilePolicy(
        profile_id='clarification_or_disambiguation',
        preferred_retrieval_mode='focused',
        allowed_output_shapes=('narrative_synthesis',),
        fallback_target_route='targeted_fact_lookup',
    ),
}


def get_intent_profile_policy(profile_id: IntentProfileId) -> IntentProfilePolicy:
    return _PROFILE_CATALOG[profile_id]


def get_confidence_band(confidence: float) -> ConfidenceBand:
    if confidence >= CONFIDENCE_HIGH_THRESHOLD:
        return 'high'
    if confidence >= CONFIDENCE_MEDIUM_THRESHOLD:
        return 'medium'
    return 'low'


def rank_profile_candidates(
    *,
    route_candidate: IntentProfileId,
    confidence: float,
) -> list[tuple[IntentProfileId, float]]:
    ranking: list[tuple[IntentProfileId, float]] = [(route_candidate, max(0.0, min(confidence, 1.0)))]
    fallback = get_intent_profile_policy(route_candidate).fallback_target_route
    ranking.append((fallback, max(0.0, min(confidence - 0.12, 0.95))))
    if fallback != 'cross_document_synthesis':
        ranking.append(('cross_document_synthesis', max(0.0, min(confidence - 0.2, 0.9))))
    deduped: list[tuple[IntentProfileId, float]] = []
    seen: set[str] = set()
    for profile_id, score in ranking:
        if profile_id in seen:
            continue
        seen.add(profile_id)
        deduped.append((profile_id, round(score, 3)))
    return deduped


__all__ = [
    'ConfidenceBand',
    'IntentProfileId',
    'IntentProfilePolicy',
    'OutputShape',
    'RetrievalMode',
    'get_confidence_band',
    'get_intent_profile_policy',
    'rank_profile_candidates',
]
