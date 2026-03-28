# ==============================================================================
# Informity AI — Retrieval Pipeline
# Encapsulates: initial retrieval → fallback on empty → validation gates →
# gatekeeper recovery → multi-step retrieval → deterministic fallbacks.
#
# Interface: run_retrieval_pipeline(...) → RetrievalSuccess | RetrievalFailure
# RetrievalSuccess: chunks and all gate/metric state for the generation stage.
# RetrievalFailure: terminal response (no generation needed) with metrics payload.
# ==============================================================================

import re
from dataclasses import dataclass, field

import aiosqlite
import structlog

from informity.llm.intent_profiles import get_intent_profile_policy
from informity.llm.model_adapter import get_retrieval_top_k
from informity.llm.query_classifier import QueryClassification
from informity.llm.rag_runtime import deterministic_fallbacks as _deterministic_fallbacks
from informity.llm.rag_runtime import generation_terminal as _generation_terminal
from informity.llm.rag_runtime import retrieval_gatekeeper as _retrieval_gatekeeper
from informity.llm.rag_runtime import retrieval_plan as _retrieval_plan
from informity.llm.rag_runtime import retrieval_validation as _retrieval_validation
from informity.llm.rag_runtime import structured_numeric as _structured_numeric
from informity.llm.retrieval import retrieve_chunks
from informity.llm.types import (
    ConfidenceBand,
    FallbackReason,
    IntentProfileId,
    OutputShape,
    QueryType,
    RetrievalMode,
)

log = structlog.get_logger(__name__)

# ==============================================================================
# Constants — retrieval-specific fallback helpers
# ==============================================================================

_CHUNK_SNIPPET_MAX_LENGTH = 220
_CHUNK_SNIPPET_ELLIPSIS_LENGTH = 3
_FALLBACK_SOURCE_LIMIT = 8
_FILENAME_SUMMARY_MAX_SNIPPETS = 5
_INVENTORY_MATCH_SNIPPET_CONTEXT_BEFORE = 90
_INVENTORY_MATCH_SNIPPET_CONTEXT_AFTER = 170
_FILENAME_SUMMARY_FALLBACK_PATTERN = re.compile(
    r'\b(?:summari[sz]e|what\s+does|what\s+is\s+in|describe)\b.*\b(?:content|contain|contains|summary)\b',
    re.IGNORECASE,
)
_FILENAME_SUMMARY_DETERMINISTIC_EXTENSIONS = ('.md', '.txt')
_STRUCTURED_EXTRACTION_SUBTYPES = _structured_numeric._STRUCTURED_EXTRACTION_SUBTYPES

_INSUFFICIENT_CONTEXT_RESPONSE = (
    'The available documents do not contain enough information to answer this question.'
)
_CLARIFICATION_METADATA_FALLBACK = (
    'Could you clarify the scope (for example: target year, file type, or specific section) '
    'so I can answer accurately?'
)
_CLARIFICATION_GENERIC_FALLBACK = (
    "I couldn't find relevant information. Could you clarify what you're looking for, "
    'or specify the document or topic?'
)

# ==============================================================================
# Output types
# ==============================================================================


@dataclass
class RetrievalSuccess:
    """All retrieval state needed by the generation stage."""
    chunks: list[dict]
    effective_query_type: RetrievalMode
    effective_top_k: int
    effective_response_shape: OutputShape   # may be modified by deterministic fallback paths
    retrieval_relevance_score: float
    distinct_sources_count: int
    retrieval_quality_score: float
    validation_gates: dict
    fallback_events: list
    applied_degradations: list
    retrieval_elapsed_ms: float
    retrieve_timing: dict
    gatekeeper_demoted_query_type: bool
    plan_step_events: list[tuple[str, dict]] = field(default_factory=list)


@dataclass
class RetrievalFailure:
    """Terminal retrieval outcome — generation is skipped."""
    response_message: str           # string to yield as the answer
    sources: list                   # source references to yield
    metrics_payload: dict           # content for '__metrics__' event
    has_remaining_scope: bool = False
    plan_step_events: list[tuple[str, dict]] = field(default_factory=list)


# ==============================================================================
# Helpers (moved from rag.py — retrieval-specific only)
# ==============================================================================


def _truncate_snippet(text: str, max_length: int = _CHUNK_SNIPPET_MAX_LENGTH) -> str:
    if len(text) <= max_length:
        return text
    return text[:max_length - _CHUNK_SNIPPET_ELLIPSIS_LENGTH] + '...'


def _deduplicate_prompt_chunks(chunks: list[dict]) -> list[dict]:
    """Remove exact normalized duplicates for the same source only."""
    seen: set[tuple[str, str]] = set()
    result: list[dict] = []
    for chunk in chunks:
        source = str(chunk.get('file_path') or chunk.get('filename') or '').strip().casefold()
        text = ' '.join(str(chunk.get('chunk_text', '') or '').split()).strip().casefold()
        if not source or not text:
            result.append(chunk)
            continue
        key = (source, text)
        if key in seen:
            continue
        seen.add(key)
        result.append(chunk)
    return result


def _build_clarification_fallback_message(classification: QueryClassification) -> str:
    is_metadata_scope = (
        classification.intent == QueryType.METADATA
        or classification.route_candidate == IntentProfileId.METADATA_INVENTORY
        or classification.is_metadata_query
    )
    if is_metadata_scope:
        return _CLARIFICATION_METADATA_FALLBACK
    return _CLARIFICATION_GENERIC_FALLBACK


def _build_inventory_plus_content_fallback_answer(
    *,
    chunks: list[dict],
    source_terms: list[str],
) -> str | None:
    if not chunks:
        return None

    normalized_terms = [
        str(term).strip()
        for term in source_terms
        if str(term).strip()
    ]
    keywords = [term.casefold() for term in normalized_terms]

    file_snippets: list[tuple[str, str]] = []
    seen_files: set[str] = set()
    for chunk in chunks:
        filename = str(chunk.get('filename', 'unknown')).strip() or 'unknown'
        file_key = filename.casefold()
        if file_key in seen_files:
            continue
        seen_files.add(file_key)

        text = re.sub(r'\s+', ' ', str(chunk.get('chunk_text', '') or '')).strip()
        if not text:
            continue

        snippet = _truncate_snippet(text)
        lowered = text.casefold()
        for keyword in keywords:
            if not keyword:
                continue
            idx = lowered.find(keyword)
            if idx >= 0:
                start = max(0, idx - _INVENTORY_MATCH_SNIPPET_CONTEXT_BEFORE)
                end = min(len(text), idx + _INVENTORY_MATCH_SNIPPET_CONTEXT_AFTER)
                snippet = text[start:end].strip()
                break
        snippet = _truncate_snippet(snippet)
        file_snippets.append((filename, snippet))
        if len(file_snippets) >= _FALLBACK_SOURCE_LIMIT:
            break

    if not file_snippets:
        return None

    lines = ['Files that match the requested terms and relevant content:', '']
    if normalized_terms:
        lines.append(f"Requested terms: {', '.join(normalized_terms[:8])}")
        lines.append('')
    for filename, snippet in file_snippets:
        lines.append(f'- **{filename}**: {snippet}')
    lines.extend([
        '',
        'Summary: These files contain the requested term matches with evidence snippets from indexed content.',
    ])
    return '\n'.join(lines).strip()


def _build_focused_anchor_recovery_query(*, question: str, source_terms: list[str]) -> str | None:
    terms = [re.sub(r'\s+', ' ', str(term or '').strip()) for term in source_terms if str(term or '').strip()]
    if not terms:
        return None
    deduped: list[str] = []
    seen: set[str] = set()
    for term in terms:
        key = term.casefold()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(term)
        if len(deduped) >= 4:
            break
    if not deduped:
        return None
    prefix = 'Find document evidence for: '
    if re.search(r'\b(?:19|20)\d{2}\b', question):
        prefix = 'Find matching year-specific document evidence for: '
    return prefix + '; '.join(deduped)


def _build_filename_summary_fallback_answer(
    *,
    question: str,
    filename_filter: str | None,
    chunks: list[dict],
    has_explicit_output_contract_fn: object,
) -> str | None:
    normalized_filename = str(filename_filter or '').strip()
    if not normalized_filename:
        return None
    if not normalized_filename.casefold().endswith(_FILENAME_SUMMARY_DETERMINISTIC_EXTENSIONS):
        return None
    if not _FILENAME_SUMMARY_FALLBACK_PATTERN.search(question):
        return None
    if has_explicit_output_contract_fn(question):
        return None
    if not chunks:
        return None

    lines = [f'### Summary: {normalized_filename}', '']
    unique_snippets: list[str] = []
    seen: set[str] = set()
    for chunk in chunks:
        text = re.sub(r'\s+', ' ', str(chunk.get('chunk_text', '') or '')).strip()
        if not text:
            continue
        snippet = _truncate_snippet(text)
        key = snippet.casefold()
        if key in seen:
            continue
        seen.add(key)
        unique_snippets.append(snippet)
        if len(unique_snippets) >= _FILENAME_SUMMARY_MAX_SNIPPETS:
            break
    if not unique_snippets:
        return None

    lines.append('Key points extracted from indexed sections:')
    for snippet in unique_snippets:
        lines.append(f'- {snippet}')
    return '\n'.join(lines).strip()


# ==============================================================================
# Retrieval Pipeline
# ==============================================================================


async def run_retrieval_pipeline(
    *,
    question: str,
    retrieval_question: str,
    classification: QueryClassification,
    effective_query_type: RetrievalMode,
    effective_top_k: int,
    effective_max_tokens: int,
    effective_response_shape: OutputShape,
    timeout_seconds: int,
    continuation_source_terms: list[str],
    prior_has_remaining_scope: bool,
    scope_reset_detected: bool,
    prior_source_anchors: set,
    retrieval_filename_filter: str | None,
    selected_policy_profile_id: IntentProfileId,
    selected_policy_fallback_target_route: IntentProfileId,
    profile_rag_max_score: float,
    applied_degradations: list,
    fallback_events: list,
    preflight_projected_seconds: float,
    preflight_ratio: float,
    db: aiosqlite.Connection,
    trace: object | None,
    has_explicit_output_contract_fn: object,
    truncate_preview_fn: object,
    normalize_relevance_score_fn: object,
) -> 'RetrievalSuccess | RetrievalFailure':
    """
    Run the retrieval pipeline:
      initial retrieval → empty fallback → validation gates → gatekeeper recovery
      → gate failure check → deterministic fallbacks.

    Returns RetrievalSuccess (chunks + all gate/metric state) or RetrievalFailure
    (terminal: no LLM generation needed).
    """
    # Defensive copies to avoid mutating caller state
    applied_degradations = list(applied_degradations)
    fallback_events = list(fallback_events)

    plan_step_events: list[tuple[str, dict]] = []

    # -------------------------------------------------------------------------
    # 1. Initial retrieval
    # -------------------------------------------------------------------------

    retrieval_result = await _retrieval_plan.run_initial_retrieval_plan(
        retrieval_question=retrieval_question,
        classification=classification,
        selected_policy_profile_id=selected_policy_profile_id,
        effective_query_type=effective_query_type,
        effective_top_k=effective_top_k,
        profile_rag_max_score=profile_rag_max_score,
        continuation_source_terms=continuation_source_terms,
        prior_has_remaining_scope=prior_has_remaining_scope,
        scope_reset_detected=scope_reset_detected,
        retrieval_filename_filter=retrieval_filename_filter,
        db=db,
        trace=trace,
        fallback_events=fallback_events,
        retrieve_with_constraints_fn=_retrieval_validation._retrieve_with_staged_structural_constraints,
        retrieve_fn=retrieve_chunks,
    )
    chunks = retrieval_result.chunks
    constraint_relaxation_applied = retrieval_result.constraint_relaxation_applied
    fallback_events = retrieval_result.fallback_events
    retrieval_elapsed_ms = retrieval_result.retrieval_elapsed_ms
    retrieve_timing = retrieval_result.retrieve_timing

    log.debug('chunks_retrieved', count=len(chunks), query_type=effective_query_type)

    # -------------------------------------------------------------------------
    # 2. Empty chunk fallback (profile fallback retrieval)
    # -------------------------------------------------------------------------

    if not chunks:
        fallback_profile = get_intent_profile_policy(selected_policy_fallback_target_route)
        fallback_events.append({
            'fallback_from': selected_policy_profile_id,
            'fallback_to': fallback_profile.profile_id,
            'fallback_reason': FallbackReason.EMPTY_RETRIEVAL_RESULT,
        })
        fallback_chunks = await retrieve_chunks(
            query=retrieval_question,
            top_k=get_retrieval_top_k(fallback_profile.preferred_retrieval_mode),
            max_score=profile_rag_max_score,
            year_filter=classification.year_filter,
            category_filter=classification.category_filter,
            extension_filter=classification.file_type_filter,
            filename_filter=retrieval_filename_filter,
            block_type_filter=None,
            section_filter=None,
            query_type=fallback_profile.preferred_retrieval_mode,
            db=db,
            trace=trace,
        )
        if fallback_chunks:
            chunks = fallback_chunks
            effective_query_type = fallback_profile.preferred_retrieval_mode
            effective_top_k = min(effective_top_k, len(chunks))
        else:
            schema_contract_recovery_chunks: list[dict] = []
            if has_explicit_output_contract_fn(question):
                schema_contract_recovery_chunks = await retrieve_chunks(
                    query=retrieval_question,
                    top_k=get_retrieval_top_k(RetrievalMode.COVERAGE),
                    max_score=profile_rag_max_score,
                    year_filter=classification.year_filter,
                    category_filter=classification.category_filter,
                    extension_filter=classification.file_type_filter,
                    filename_filter=None,
                    block_type_filter=None,
                    section_filter=None,
                    query_type=RetrievalMode.COVERAGE,
                    db=db,
                    trace=trace,
                )
            if schema_contract_recovery_chunks:
                chunks = schema_contract_recovery_chunks
                effective_query_type = RetrievalMode.COVERAGE
                effective_top_k = min(effective_top_k, len(chunks))
                fallback_events.append({
                    'fallback_from': selected_policy_profile_id,
                    'fallback_to': selected_policy_profile_id,
                    'fallback_reason': FallbackReason.SCHEMA_CONTRACT_EMPTY_RETRIEVAL_RECOVERY,
                })
            else:
                focused_anchor_recovery_chunks: list[dict] = []
                focused_anchor_recovery_query = _build_focused_anchor_recovery_query(
                    question=question,
                    source_terms=classification.source_terms,
                )
                if (
                    focused_anchor_recovery_query
                    and effective_query_type == RetrievalMode.FOCUSED
                    and not classification.filename_filter
                ):
                    focused_anchor_recovery_chunks = await retrieve_chunks(
                        query=focused_anchor_recovery_query,
                        top_k=get_retrieval_top_k(RetrievalMode.FOCUSED),
                        max_score=profile_rag_max_score,
                        year_filter=classification.year_filter,
                        category_filter=classification.category_filter,
                        extension_filter=classification.file_type_filter,
                        filename_filter=None,
                        block_type_filter=None,
                        section_filter=None,
                        query_type=RetrievalMode.FOCUSED,
                        db=db,
                        trace=trace,
                    )
                if focused_anchor_recovery_chunks:
                    chunks = focused_anchor_recovery_chunks
                    effective_query_type = RetrievalMode.FOCUSED
                    effective_top_k = min(effective_top_k, len(chunks))
                    fallback_events.append({
                        'fallback_from': selected_policy_profile_id,
                        'fallback_to': selected_policy_profile_id,
                        'fallback_reason': FallbackReason.FOCUSED_ANCHOR_EMPTY_RETRIEVAL_RECOVERY,
                    })
                else:
                    focused_year_recovery_chunks: list[dict] = []
                    if (
                        classification.year_filter is not None
                        and effective_query_type == RetrievalMode.FOCUSED
                        and not classification.filename_filter
                    ):
                        focused_year_recovery_chunks = await retrieve_chunks(
                            query=retrieval_question,
                            top_k=get_retrieval_top_k(RetrievalMode.COVERAGE),
                            max_score=profile_rag_max_score,
                            year_filter=classification.year_filter,
                            category_filter=classification.category_filter,
                            extension_filter=classification.file_type_filter,
                            filename_filter=None,
                            block_type_filter=None,
                            section_filter=None,
                            query_type=RetrievalMode.COVERAGE,
                            db=db,
                            trace=trace,
                        )
                    if focused_year_recovery_chunks:
                        chunks = focused_year_recovery_chunks
                        effective_query_type = RetrievalMode.COVERAGE
                        effective_top_k = min(effective_top_k, len(chunks))
                        fallback_events.append({
                            'fallback_from': selected_policy_profile_id,
                            'fallback_to': selected_policy_profile_id,
                            'fallback_reason': FallbackReason.FOCUSED_YEAR_SCOPE_EMPTY_RETRIEVAL_RECOVERY,
                        })
                    else:
                        return RetrievalFailure(
                            response_message=_INSUFFICIENT_CONTEXT_RESPONSE,
                            sources=[],
                            metrics_payload=_generation_terminal.build_generation_skipped_metrics_payload(
                                query_type=effective_query_type,
                                timeout_seconds=timeout_seconds,
                                retrieval_elapsed_ms=retrieval_elapsed_ms,
                                preflight_projected_seconds=preflight_projected_seconds,
                                preflight_ratio=preflight_ratio,
                                applied_degradations=applied_degradations,
                                fallback_events=fallback_events,
                                has_remaining_scope=False,
                                validation_gates={'retrieval_relevance_gate': False, 'source_diversity_gate': False},
                            ),
                            has_remaining_scope=False,
                        )

    # -------------------------------------------------------------------------
    # 3. Validation gates
    # -------------------------------------------------------------------------

    retrieval_relevance_passed, retrieval_relevance_score = (
        _retrieval_validation._evaluate_retrieval_relevance_gate(
            chunks=chunks,
            query_type=effective_query_type,
            route_candidate=selected_policy_profile_id,
            has_strong_anchor=bool(classification.filename_filter),
        )
    )
    source_diversity_passed, distinct_sources_count = (
        _retrieval_validation._evaluate_source_diversity_gate(
            chunks=chunks,
            query_type=effective_query_type,
        )
    )
    retrieval_relevance_passed, fallback_events = (
        _retrieval_validation._apply_coverage_evidence_floor_override(
            retrieval_relevance_passed=retrieval_relevance_passed,
            query_type=effective_query_type,
            subtype=classification.subtype,
            group_by=classification.group_by,
            response_shape=effective_response_shape,
            distinct_sources_count=distinct_sources_count,
            chunk_count=len(chunks),
            fallback_events=fallback_events,
            route_profile_id=selected_policy_profile_id,
            retrieval_relevance_score=retrieval_relevance_score,
        )
    )
    current_source_keys = _retrieval_validation._extract_current_source_keys(chunks)
    continuation_anchor_passed, anchor_overlap_count = (
        _retrieval_validation._evaluate_continuation_anchor_gate(
            route_candidate=classification.route_candidate,
            scope_reset_detected=scope_reset_detected,
            prior_source_anchors=prior_source_anchors,
            current_source_keys=current_source_keys,
            prior_has_remaining_scope=prior_has_remaining_scope,
        )
    )
    validation_gates = {
        'retrieval_relevance_gate': retrieval_relevance_passed,
        'source_diversity_gate': source_diversity_passed,
        'continuation_anchor_gate': continuation_anchor_passed,
    }
    if trace is not None:
        trace.record('validation_gates', {
            'gates': validation_gates,
            'retrieval_relevance_score': round(retrieval_relevance_score, 3),
            'distinct_sources_count': distinct_sources_count,
            'scope_reset_detected': scope_reset_detected,
            'anchor_overlap_count': anchor_overlap_count,
            'constraint_relaxation_applied': constraint_relaxation_applied,
        })

    # -------------------------------------------------------------------------
    # 4. Gatekeeper recovery
    # -------------------------------------------------------------------------

    recovery_result = await _retrieval_gatekeeper.run_validation_recovery_when_failed(
        chunks=chunks,
        effective_query_type=effective_query_type,
        effective_top_k=effective_top_k,
        retrieval_relevance_passed=retrieval_relevance_passed,
        source_diversity_passed=source_diversity_passed,
        continuation_anchor_passed=continuation_anchor_passed,
        retrieval_relevance_score=retrieval_relevance_score,
        distinct_sources_count=distinct_sources_count,
        anchor_overlap_count=anchor_overlap_count,
        validation_gates=validation_gates,
        fallback_events=fallback_events,
        classification=classification,
                effective_response_shape=effective_response_shape,
                selected_policy_profile_id=selected_policy_profile_id,
                selected_policy_fallback_target_route=selected_policy_fallback_target_route,
                scope_reset_detected=scope_reset_detected,
        prior_source_anchors=prior_source_anchors,
        prior_has_remaining_scope=prior_has_remaining_scope,
        retrieval_question=retrieval_question,
        retrieval_filename_filter=retrieval_filename_filter,
        profile_rag_max_score=profile_rag_max_score,
        db=db,
        trace=trace,
        retrieve_fn=retrieve_chunks,
        get_retrieval_top_k_fn=get_retrieval_top_k,
        get_intent_profile_policy_fn=get_intent_profile_policy,
    )
    chunks = recovery_result.chunks
    effective_query_type = recovery_result.effective_query_type
    effective_top_k = recovery_result.effective_top_k
    retrieval_relevance_passed = recovery_result.retrieval_relevance_passed
    source_diversity_passed = recovery_result.source_diversity_passed
    continuation_anchor_passed = recovery_result.continuation_anchor_passed
    retrieval_relevance_score = recovery_result.retrieval_relevance_score
    distinct_sources_count = recovery_result.distinct_sources_count
    anchor_overlap_count = recovery_result.anchor_overlap_count
    validation_gates = recovery_result.validation_gates
    fallback_events = recovery_result.fallback_events
    retrieval_quality_score = recovery_result.quality_score
    gatekeeper_demoted_query_type = (
        recovery_result.original_query_type != recovery_result.effective_query_type
    )
    log.info(
        'retrieval_quality_score',
        quality_score=round(retrieval_quality_score, 3),
        retrieval_relevance_score=round(retrieval_relevance_score, 3),
        distinct_sources_count=distinct_sources_count,
        chunk_count=len(chunks),
        effective_query_type=effective_query_type,
        fallback_count=len([e for e in fallback_events if 'fallback_reason' in e]),
    )

    # -------------------------------------------------------------------------
    # 5. Gate failure → terminal
    # -------------------------------------------------------------------------

    validation_failed = (
        not retrieval_relevance_passed
        or not source_diversity_passed
        or not continuation_anchor_passed
    )
    schema_driven_structured_request = (
        effective_response_shape in {OutputShape.STRUCTURED_EXTRACT, OutputShape.METADATA_TABLE}
        and has_explicit_output_contract_fn(question)
    )
    allow_structured_gate_bypass = (
        validation_failed
        and schema_driven_structured_request
        and continuation_anchor_passed
        and bool(chunks)
    )
    allow_continuation_anchor_bypass = (
        validation_failed
        and (
            classification.route_candidate == IntentProfileId.CONTINUATION_OR_REFINEMENT
            or selected_policy_profile_id == IntentProfileId.CONTINUATION_OR_REFINEMENT
        )
        and continuation_anchor_passed
    )
    continuation_validation_fallback_seen = any(
        isinstance(event, dict)
        and event.get('fallback_reason') == FallbackReason.VALIDATION_GATE_FAILED
        and event.get('fallback_from') == IntentProfileId.CONTINUATION_OR_REFINEMENT
        and bool((event.get('validation_gates') or {}).get('continuation_anchor_gate'))
        for event in fallback_events
    )
    if validation_failed and continuation_validation_fallback_seen:
        allow_continuation_anchor_bypass = True
    if validation_failed and not (allow_structured_gate_bypass or allow_continuation_anchor_bypass):
        use_clarification = (
            not classification.is_continuation
            and (
                classification.route_candidate == IntentProfileId.CLARIFICATION_OR_DISAMBIGUATION
                or classification.confidence_band == ConfidenceBand.LOW
                or not continuation_anchor_passed
            )
        )
        return RetrievalFailure(
            response_message=(
                _build_clarification_fallback_message(classification)
                if use_clarification
                else _INSUFFICIENT_CONTEXT_RESPONSE
            ),
            sources=[],
            metrics_payload=_generation_terminal.build_generation_skipped_metrics_payload(
                query_type=effective_query_type,
                timeout_seconds=timeout_seconds,
                retrieval_elapsed_ms=retrieval_elapsed_ms,
                preflight_projected_seconds=preflight_projected_seconds,
                preflight_ratio=preflight_ratio,
                applied_degradations=applied_degradations,
                fallback_events=fallback_events,
                has_remaining_scope=False,
                validation_gates=validation_gates,
            ),
            has_remaining_scope=False,
        )
    if allow_structured_gate_bypass:
        fallback_events.append({
            'fallback_from': selected_policy_profile_id,
            'fallback_to': selected_policy_profile_id,
            'fallback_reason': FallbackReason.SCHEMA_DRIVEN_GATE_BYPASS,
        })
    if allow_continuation_anchor_bypass:
        fallback_events.append({
            'fallback_from': selected_policy_profile_id,
            'fallback_to': selected_policy_profile_id,
            'fallback_reason': FallbackReason.CONTINUATION_ANCHOR_GATE_BYPASS,
        })

    # -------------------------------------------------------------------------
    # 6. Deterministic fallbacks (structured, filename summary, inventory)
    # -------------------------------------------------------------------------

    deterministic_fallback = await _deterministic_fallbacks.try_structured_fallback(
        question=question,
        classification=classification,
        response_shape=effective_response_shape,
        chunks=chunks,
        db=db,
        trace=trace,
    )

    if deterministic_fallback.kind == 'structured':
        structured_answer = deterministic_fallback.answer or ''
        structured_sources = deterministic_fallback.sources or []
        structured_metrics = deterministic_fallback.structured_metrics or {}
        return RetrievalFailure(
            response_message=structured_answer,
            sources=structured_sources,
            metrics_payload=_generation_terminal.build_generation_skipped_metrics_payload(
                query_type=effective_query_type,
                timeout_seconds=timeout_seconds,
                retrieval_elapsed_ms=retrieval_elapsed_ms,
                preflight_projected_seconds=preflight_projected_seconds,
                preflight_ratio=preflight_ratio,
                applied_degradations=applied_degradations,
                fallback_events=fallback_events,
                has_remaining_scope=False,
                extra_fields=structured_metrics,
            ),
            has_remaining_scope=False,
            plan_step_events=plan_step_events,
        )

    filename_summary_fallback_answer = _build_filename_summary_fallback_answer(
        question=question,
        filename_filter=classification.filename_filter,
        chunks=chunks,
        has_explicit_output_contract_fn=has_explicit_output_contract_fn,
    )
    if filename_summary_fallback_answer is not None:
        applied_degradations.append({
            'step': 'filename_summary_deterministic_fallback',
            'filename_filter': classification.filename_filter,
            'reason': 'focused_filename_summary_latency_guard',
        })
        fallback_sources = _generation_terminal.build_limited_fallback_sources(
            chunks=chunks,
            limit=_FALLBACK_SOURCE_LIMIT,
            truncate_preview_fn=truncate_preview_fn,
            normalize_relevance_score_fn=normalize_relevance_score_fn,
        )
        return RetrievalFailure(
            response_message=filename_summary_fallback_answer,
            sources=fallback_sources,
            metrics_payload=_generation_terminal.build_generation_skipped_metrics_payload(
                query_type=effective_query_type,
                timeout_seconds=timeout_seconds,
                retrieval_elapsed_ms=retrieval_elapsed_ms,
                preflight_projected_seconds=preflight_projected_seconds,
                preflight_ratio=preflight_ratio,
                applied_degradations=applied_degradations,
                fallback_events=fallback_events,
                has_remaining_scope=False,
            ),
            has_remaining_scope=False,
            plan_step_events=plan_step_events,
        )

    if 'policy_inventory_plus_content_to_coverage' in classification.reason_codes:
        fallback_answer = _build_inventory_plus_content_fallback_answer(
            chunks=chunks,
            source_terms=classification.source_terms,
        )
        if fallback_answer:
            applied_degradations.append({
                'step': 'inventory_plus_content_deterministic_fallback',
                'reason': 'policy_inventory_plus_content_to_coverage',
            })
            fallback_sources = _generation_terminal.build_limited_fallback_sources(
                chunks=chunks,
                limit=_FALLBACK_SOURCE_LIMIT,
                truncate_preview_fn=truncate_preview_fn,
                normalize_relevance_score_fn=normalize_relevance_score_fn,
            )
            return RetrievalFailure(
                response_message=fallback_answer,
                sources=fallback_sources,
                metrics_payload=_generation_terminal.build_generation_skipped_metrics_payload(
                    query_type=effective_query_type,
                    timeout_seconds=timeout_seconds,
                    retrieval_elapsed_ms=retrieval_elapsed_ms,
                    preflight_projected_seconds=preflight_projected_seconds,
                    preflight_ratio=preflight_ratio,
                    applied_degradations=applied_degradations,
                    fallback_events=fallback_events,
                    has_remaining_scope=False,
                ),
                has_remaining_scope=False,
                plan_step_events=plan_step_events,
            )

    # Structured extraction response shape fallback:
    # If structured_extract was requested but the route cannot support it,
    # fall back to narrative_synthesis.
    if (
        effective_response_shape == OutputShape.STRUCTURED_EXTRACT
        and classification.subtype in _STRUCTURED_EXTRACTION_SUBTYPES
    ):
        fallback_events.append({
            'fallback_from': IntentProfileId.STRUCTURED_FIELD_EXTRACTION,
            'fallback_to': IntentProfileId.TARGETED_FACT_LOOKUP,
            'fallback_reason': FallbackReason.STRUCTURED_EXTRACTION_INSUFFICIENT,
        })
        effective_response_shape = OutputShape.NARRATIVE_SYNTHESIS

    # -------------------------------------------------------------------------
    # 7. Success — pass chunks and all gate state to generation
    # -------------------------------------------------------------------------

    return RetrievalSuccess(
        chunks=chunks,
        effective_query_type=effective_query_type,
        effective_top_k=effective_top_k,
        effective_response_shape=effective_response_shape,
        retrieval_relevance_score=retrieval_relevance_score,
        distinct_sources_count=distinct_sources_count,
        retrieval_quality_score=retrieval_quality_score,
        validation_gates=validation_gates,
        fallback_events=fallback_events,
        applied_degradations=applied_degradations,
        retrieval_elapsed_ms=retrieval_elapsed_ms,
        retrieve_timing=retrieve_timing,
        gatekeeper_demoted_query_type=gatekeeper_demoted_query_type,
        plan_step_events=plan_step_events,
    )


__all__ = [
    'RetrievalSuccess',
    'RetrievalFailure',
    'run_retrieval_pipeline',
    '_build_clarification_fallback_message',
    '_build_focused_anchor_recovery_query',
    '_build_filename_summary_fallback_answer',
    '_build_inventory_plus_content_fallback_answer',
    '_deduplicate_prompt_chunks',
    '_truncate_snippet',
]
