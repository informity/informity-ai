from dataclasses import dataclass

from informity.llm.query_classifier import (
    CONFIDENCE_HIGH_THRESHOLD,
    CONFIDENCE_MEDIUM_THRESHOLD,
)
from informity.llm.types import ConfidenceBand, IntentProfileId, OutputShape, RetrievalMode


@dataclass(frozen=True)
class IntentProfilePolicy:
    profile_id: IntentProfileId
    preferred_retrieval_mode: RetrievalMode
    allowed_output_shapes: tuple[OutputShape, ...]
    fallback_target_route: IntentProfileId


_PROFILE_CATALOG: dict[IntentProfileId, IntentProfilePolicy] = {
    IntentProfileId.METADATA_INVENTORY: IntentProfilePolicy(
        profile_id=IntentProfileId.METADATA_INVENTORY,
        preferred_retrieval_mode=RetrievalMode.FOCUSED,
        allowed_output_shapes=(OutputShape.METADATA_TABLE, OutputShape.STRUCTURED_EXTRACT),
        fallback_target_route=IntentProfileId.CLARIFICATION_OR_DISAMBIGUATION,
    ),
    IntentProfileId.TARGETED_FACT_LOOKUP: IntentProfilePolicy(
        profile_id=IntentProfileId.TARGETED_FACT_LOOKUP,
        preferred_retrieval_mode=RetrievalMode.FOCUSED,
        allowed_output_shapes=(OutputShape.NARRATIVE_SYNTHESIS, OutputShape.HYBRID),
        fallback_target_route=IntentProfileId.CROSS_DOCUMENT_SYNTHESIS,
    ),
    IntentProfileId.STRUCTURED_FIELD_EXTRACTION: IntentProfilePolicy(
        profile_id=IntentProfileId.STRUCTURED_FIELD_EXTRACTION,
        preferred_retrieval_mode=RetrievalMode.FOCUSED,
        allowed_output_shapes=(OutputShape.STRUCTURED_EXTRACT, OutputShape.HYBRID),
        fallback_target_route=IntentProfileId.TARGETED_FACT_LOOKUP,
    ),
    IntentProfileId.CROSS_DOCUMENT_SYNTHESIS: IntentProfilePolicy(
        profile_id=IntentProfileId.CROSS_DOCUMENT_SYNTHESIS,
        preferred_retrieval_mode=RetrievalMode.COVERAGE,
        allowed_output_shapes=(OutputShape.NARRATIVE_SYNTHESIS, OutputShape.HYBRID),
        fallback_target_route=IntentProfileId.CLARIFICATION_OR_DISAMBIGUATION,
    ),
    IntentProfileId.COMPARATIVE_ANALYSIS: IntentProfilePolicy(
        profile_id=IntentProfileId.COMPARATIVE_ANALYSIS,
        preferred_retrieval_mode=RetrievalMode.COVERAGE,
        allowed_output_shapes=(OutputShape.NARRATIVE_SYNTHESIS, OutputShape.METADATA_TABLE, OutputShape.HYBRID),
        fallback_target_route=IntentProfileId.CROSS_DOCUMENT_SYNTHESIS,
    ),
    IntentProfileId.AUDIT_OR_COMPLIANCE_BRIEF: IntentProfilePolicy(
        profile_id=IntentProfileId.AUDIT_OR_COMPLIANCE_BRIEF,
        preferred_retrieval_mode=RetrievalMode.COVERAGE,
        allowed_output_shapes=(OutputShape.NARRATIVE_SYNTHESIS, OutputShape.HYBRID),
        fallback_target_route=IntentProfileId.CROSS_DOCUMENT_SYNTHESIS,
    ),
    IntentProfileId.CONTINUATION_OR_REFINEMENT: IntentProfilePolicy(
        profile_id=IntentProfileId.CONTINUATION_OR_REFINEMENT,
        preferred_retrieval_mode=RetrievalMode.FOCUSED,
        allowed_output_shapes=(OutputShape.NARRATIVE_SYNTHESIS, OutputShape.STRUCTURED_EXTRACT, OutputShape.HYBRID),
        fallback_target_route=IntentProfileId.CLARIFICATION_OR_DISAMBIGUATION,
    ),
    IntentProfileId.CLARIFICATION_OR_DISAMBIGUATION: IntentProfilePolicy(
        profile_id=IntentProfileId.CLARIFICATION_OR_DISAMBIGUATION,
        preferred_retrieval_mode=RetrievalMode.FOCUSED,
        allowed_output_shapes=(OutputShape.NARRATIVE_SYNTHESIS,),
        fallback_target_route=IntentProfileId.TARGETED_FACT_LOOKUP,
    ),
}


def get_intent_profile_policy(profile_id: IntentProfileId) -> IntentProfilePolicy:
    return _PROFILE_CATALOG[IntentProfileId(profile_id)]


def get_confidence_band(confidence: float) -> ConfidenceBand:
    if confidence >= CONFIDENCE_HIGH_THRESHOLD:
        return ConfidenceBand.HIGH
    if confidence >= CONFIDENCE_MEDIUM_THRESHOLD:
        return ConfidenceBand.MEDIUM
    return ConfidenceBand.LOW


def rank_profile_candidates(
    *,
    route_candidate: IntentProfileId,
    confidence: float,
) -> list[tuple[IntentProfileId, float]]:
    ranking: list[tuple[IntentProfileId, float]] = [(route_candidate, max(0.0, min(confidence, 1.0)))]
    fallback = get_intent_profile_policy(route_candidate).fallback_target_route
    ranking.append((fallback, max(0.0, min(confidence - 0.12, 0.95))))
    if fallback != IntentProfileId.CROSS_DOCUMENT_SYNTHESIS:
        ranking.append((IntentProfileId.CROSS_DOCUMENT_SYNTHESIS, max(0.0, min(confidence - 0.2, 0.9))))
    deduped: list[tuple[IntentProfileId, float]] = []
    seen: set[IntentProfileId] = set()
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
