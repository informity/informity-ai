# ==============================================================================
# Informity AI — RAG Query Handler
# Handles focused and coverage queries using vector search → rerank → LLM
# ==============================================================================

import asyncio
import hashlib
import re
from collections.abc import AsyncGenerator

import aiosqlite
import structlog

from informity.api.schemas import ChatSourceReference
from informity.config import settings
from informity.db.models import ChatMessage
from informity.llm.fit_to_budget_tuning import resolve_fit_to_budget_policy
from informity.llm.intent_profiles import get_intent_profile_policy
from informity.llm.model_adapter import get_retrieval_top_k
from informity.llm.prompt_builder import build_messages
from informity.llm.query_classifier import QueryClassification
from informity.llm.rag_runtime import deterministic_fallbacks as _deterministic_fallbacks
from informity.llm.rag_runtime import execution_plan as _execution_plan
from informity.llm.rag_runtime import generation_closeout as _generation_closeout
from informity.llm.rag_runtime import generation_plan as _generation_plan
from informity.llm.rag_runtime import generation_runtime as _generation_runtime
from informity.llm.rag_runtime import generation_stream as _generation_stream
from informity.llm.rag_runtime import generation_terminal as _generation_terminal
from informity.llm.rag_runtime import retrieval_gatekeeper as _retrieval_gatekeeper
from informity.llm.rag_runtime import retrieval_plan as _retrieval_plan
from informity.llm.rag_runtime import retrieval_validation as _retrieval_validation
from informity.llm.rag_runtime import structured_numeric as _structured_numeric
from informity.llm.retrieval import retrieve_chunks
from informity.llm.streaming import stream_llm

log = structlog.get_logger(__name__)
_HANDLER_RUNTIME_EXCEPTIONS = (RuntimeError, ValueError, TypeError, OSError, asyncio.TimeoutError)

_STRUCTURED_EXTRACTION_SUBTYPES = _structured_numeric._STRUCTURED_EXTRACTION_SUBTYPES
_INSUFFICIENT_CONTEXT_RESPONSE = 'The available documents do not contain enough information to answer this question.'
_CHUNK_SNIPPET_MAX_LENGTH = 220
_CHUNK_SNIPPET_ELLIPSIS_LENGTH = 3
_CHUNK_PREVIEW_MAX_LENGTH = 200
_FALLBACK_SOURCE_LIMIT = 8
_FILENAME_SUMMARY_MAX_SNIPPETS = 5
_INVENTORY_MATCH_SNIPPET_CONTEXT_BEFORE = 90
_INVENTORY_MATCH_SNIPPET_CONTEXT_AFTER = 170
_FILENAME_SUMMARY_FALLBACK_PATTERN = re.compile(
    r'\b(?:summari[sz]e|what\s+does|what\s+is\s+in|describe)\b.*\b(?:content|contain|contains|summary)\b',
    re.IGNORECASE,
)
_FILENAME_SUMMARY_DETERMINISTIC_EXTENSIONS = ('.md', '.txt')
_OUTPUT_CONTRACT_WORD_LIMIT_PATTERN = re.compile(
    r'\b(?:<=?|at\s+most|max(?:imum)?)\s*\d+\s+words?\b',
    re.IGNORECASE,
)
_OUTPUT_CONTRACT_BULLET_LIMIT_PATTERN = re.compile(
    r'\bexactly\s+\d+\s+bullets?\b',
    re.IGNORECASE,
)
_CLARIFICATION_METADATA_FALLBACK = (
    'Could you clarify the scope (for example: target year, file type, or specific section) '
    'so I can answer accurately?'
)
_CLARIFICATION_GENERIC_FALLBACK = (
    "I couldn't find relevant information. Could you clarify what you're looking for, "
    'or specify the document or topic?'
)


def _has_explicit_output_contract(question: str) -> bool:
    if _OUTPUT_CONTRACT_WORD_LIMIT_PATTERN.search(question):
        return True
    if _OUTPUT_CONTRACT_BULLET_LIMIT_PATTERN.search(question):
        return True
    if 'output must contain' in question.casefold():
        return True
    return bool(_structured_numeric._derive_format_requirements(question))


def _is_continuation_request(question: str) -> bool:
    return _retrieval_validation._is_continuation_utterance(question or '')


def _build_clarification_fallback_message(classification: QueryClassification) -> str:
    is_metadata_scope = (
        classification.intent == 'metadata'
        or classification.route_candidate == 'metadata_inventory'
        or classification.is_metadata_query
    )
    if is_metadata_scope:
        return _CLARIFICATION_METADATA_FALLBACK
    return _CLARIFICATION_GENERIC_FALLBACK


def _compute_widened_retry_top_k(
    *,
    current_top_k: int,
    query_type: str,
    response_mode: str,
) -> int:
    base_top_k = max(current_top_k, 1)
    widened = int(round(base_top_k * float(settings.retrieval_widening_retry_multiplier)))
    widened += int(settings.retrieval_widening_retry_extra_k)
    floor_top_k = get_retrieval_top_k(query_type, response_mode=response_mode)
    widened = max(widened, floor_top_k, base_top_k + 1)
    return min(widened, int(settings.retrieval_widening_retry_cap))


def _collapse_duplicate_insufficient_context_message(
    answer: str,
    *,
    phrase: str = _INSUFFICIENT_CONTEXT_RESPONSE,
) -> tuple[str, bool]:
    if not answer:
        return answer, False
    escaped_phrase = re.escape(phrase)
    pattern = re.compile(rf'(?:{escaped_phrase}\s*){{2,}}')
    collapsed = pattern.sub(f'{phrase}\n', answer)
    if collapsed == answer:
        return answer, False
    return collapsed, True


def _truncate_snippet(text: str, max_length: int = _CHUNK_SNIPPET_MAX_LENGTH) -> str:
    if len(text) <= max_length:
        return text
    trim_length = max(0, max_length - _CHUNK_SNIPPET_ELLIPSIS_LENGTH)
    return f'{text[:trim_length]}...'


def _truncate_preview(text: str, max_length: int = _CHUNK_PREVIEW_MAX_LENGTH) -> str:
    return text[:max_length]


def _deduplicate_prompt_chunks(chunks: list[dict]) -> list[dict]:
    # Remove exact normalized duplicates for the same source only.
    # Avoid prefix-based signatures, which can collapse distinct evidence chunks
    # that share templated openings.
    deduped_chunks: list[dict] = []
    seen_signatures: set[tuple[str, str]] = set()
    for chunk in chunks:
        source_key = str(chunk.get('file_path') or chunk.get('filename') or '').strip().casefold()
        text = str(chunk.get('chunk_text', '')).strip()
        normalized_text = re.sub(r'\s+', ' ', text).casefold()
        if not source_key or not normalized_text:
            deduped_chunks.append(chunk)
            continue
        content_hash = hashlib.sha1(normalized_text.encode('utf-8')).hexdigest()
        signature = (source_key, content_hash)
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        deduped_chunks.append(chunk)
    return deduped_chunks


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


def _build_filename_summary_fallback_answer(
    *,
    question: str,
    filename_filter: str | None,
    chunks: list[dict],
) -> str | None:
    normalized_filename = str(filename_filter or '').strip()
    if not normalized_filename:
        return None
    if not normalized_filename.casefold().endswith(_FILENAME_SUMMARY_DETERMINISTIC_EXTENSIONS):
        return None
    if not _FILENAME_SUMMARY_FALLBACK_PATTERN.search(question):
        return None
    if _has_explicit_output_contract(question):
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


class RAGHandler:
    """
    Handler for focused and coverage queries.

    Uses vector search → rerank → LLM pipeline.
    """

    def matches(self, classification: QueryClassification) -> bool:
        """Match focused and coverage queries."""
        return classification.intent in ('focused', 'coverage')

    async def handle(
        self,
        question:       str,
        classification: QueryClassification,
        history:        list[ChatMessage] | None,
        db:             aiosqlite.Connection,
        trace:          object | None,
        response_mode:  str | None = None,
        diagnostics_context: dict[str, object] | None = None,
    ) -> AsyncGenerator[str | list[ChatSourceReference] | tuple[str, object]]:
        """
        Handle RAG query using vector search → rerank → LLM pipeline.

        This is the existing RAG logic extracted into a handler.
        """
        try:
            # 1. Determine query type for proper retrieval and LLM settings
            plan = await _execution_plan.build_execution_plan(
                question=question,
                classification=classification,
                response_mode=response_mode,
                diagnostics_context=diagnostics_context,
                db=db,
                resolve_fit_to_budget_policy_fn=resolve_fit_to_budget_policy,
            )
            profile = plan.profile
            selected_policy = plan.selected_policy
            effective_response_shape = plan.effective_response_shape
            response_mode_used = plan.response_mode_used
            timeout_seconds = plan.timeout_seconds
            max_tokens = plan.max_tokens
            mode_adjustments_applied = plan.mode_adjustments_applied
            diagnostics_min_words = plan.diagnostics_min_words
            policy = plan.policy
            effective_query_type = plan.effective_query_type
            effective_top_k = plan.effective_top_k
            effective_reasoning_enabled = plan.effective_reasoning_enabled
            effective_max_tokens = plan.effective_max_tokens
            fit_to_budget_enabled = plan.fit_to_budget_enabled
            output_constraints = plan.output_constraints
            applied_degradations = plan.applied_degradations
            strict_ordered_mode = plan.strict_ordered_mode
            fallback_events = plan.fallback_events

            if response_mode_used == 'research':
                research_fallback_fields = profile.get_research_fallback_fields()
                if research_fallback_fields:
                    log.warning(
                        'research_mode_profile_fallback',
                        profile_name=profile.name,
                        fallback_fields=research_fallback_fields,
                        fallback_to='analysis_or_base_profile_values',
                    )
            retrieval_filename_filter = classification.filename_filter
            retrieval_context = _retrieval_plan.build_retrieval_context(
                question=question,
                classification=classification,
                history=history,
            )
            scope_reset_detected = retrieval_context.scope_reset_detected
            prior_source_anchors = retrieval_context.prior_source_anchors
            prior_has_remaining_scope = retrieval_context.prior_has_remaining_scope
            continuation_source_terms = retrieval_context.continuation_source_terms
            source_terms_for_retrieval = retrieval_context.source_terms_for_retrieval
            retrieval_question = retrieval_context.retrieval_question
            (
                timeout_seconds,
                effective_top_k,
                effective_reasoning_enabled,
                effective_max_tokens,
                applied_degradations,
            ) = _generation_runtime._apply_source_scoped_coverage_guard(
                query_type=effective_query_type,
                route_candidate=selected_policy.profile_id,
                source_terms=source_terms_for_retrieval,
                timeout_seconds=timeout_seconds,
                top_k=effective_top_k,
                reasoning_enabled=effective_reasoning_enabled,
                max_tokens=effective_max_tokens,
                applied_degradations=applied_degradations,
            )

            (
                timeout_seconds,
                effective_top_k,
                effective_reasoning_enabled,
                effective_max_tokens,
                applied_degradations,
                strict_ordered_mode,
            ) = _generation_runtime._apply_strict_pre_retrieval_guard(
                question=question,
                query_type=effective_query_type,
                timeout_seconds=timeout_seconds,
                top_k=effective_top_k,
                reasoning_enabled=effective_reasoning_enabled,
                max_tokens=effective_max_tokens,
                applied_degradations=applied_degradations,
                derive_format_requirements_fn=_structured_numeric._derive_format_requirements,
                profile_name=profile.name,
                response_mode=response_mode_used,
            )

            preflight_projected_seconds, preflight_ratio = _generation_runtime._estimate_budget_ratio(
                profile_name=profile.name,
                query_type=effective_query_type,
                timeout_seconds=timeout_seconds,
                question_length=len(question),
                context_chunks=effective_top_k,
                context_chars=0,
                top_k=effective_top_k,
                reasoning_enabled=effective_reasoning_enabled,
                max_tokens=effective_max_tokens,
            )

            (
                effective_query_type,
                effective_top_k,
                effective_reasoning_enabled,
                effective_max_tokens,
                timeout_seconds,
                output_constraints,
                applied_degradations,
                preflight_projected_seconds,
                preflight_ratio,
            ) = _generation_runtime._apply_preflight_budget_degradations(
                fit_to_budget_enabled=fit_to_budget_enabled,
                policy_soft_top_k_threshold=policy.soft_top_k_threshold,
                policy_soft_reasoning_threshold=policy.soft_reasoning_threshold,
                policy_soft_output_cap_threshold=policy.soft_output_cap_threshold,
                policy_soft_coverage_to_focused_threshold=policy.soft_coverage_to_focused_threshold,
                profile_name=profile.name,
                question_length=len(question),
                query_type=effective_query_type,
                timeout_seconds=timeout_seconds,
                top_k=effective_top_k,
                reasoning_enabled=effective_reasoning_enabled,
                max_tokens=effective_max_tokens,
                subtype=classification.subtype,
                focused_max_tokens=profile.get_max_tokens('focused'),
                focused_timeout_seconds=profile.get_timeout_seconds('focused'),
                output_constraints=output_constraints,
                applied_degradations=applied_degradations,
                response_mode=response_mode_used,
                strict_ordered_mode=strict_ordered_mode,
            )

            if trace is not None:
                trace.record('intent', {
                    'model_profile':     profile.name,
                    'intent':            classification.intent,
                    'route_candidate':   classification.route_candidate,
                    'confidence':        classification.confidence,
                    'confidence_band':   classification.confidence_band,
                    'response_shape':    effective_response_shape,
                    'query_type':        effective_query_type,
                    'coverage_mode':     effective_query_type == 'coverage',
                    'reasoning_enabled':  effective_reasoning_enabled,
                    'top_k':              effective_top_k,
                    'rag_max_score':     getattr(profile, 'rag_max_score', None),
                    'budget_preflight_projected_seconds': round(preflight_projected_seconds, 1),
                    'budget_preflight_ratio': round(preflight_ratio, 3),
                    'fit_to_budget_rollout_stage': policy.rollout_stage,
                    'fit_to_budget_enabled': fit_to_budget_enabled,
                    'fit_to_budget_sample_count': policy.sample_count,
                    'fit_to_budget_timeout_rate': policy.timeout_rate,
                    'response_mode_used': response_mode_used,
                    'strict_ordered_mode': strict_ordered_mode,
                    'mode_adjustments_applied': mode_adjustments_applied,
                    'applied_degradations': applied_degradations,
                    'fallback_events': fallback_events,
                })

            # 2. Retrieve chunks with appropriate top_k (unified retrieval path)
            # For coverage queries, uses file-anchored retrieval (one chunk per file)
            retrieval_result = await _retrieval_plan.run_initial_retrieval_plan(
                retrieval_question=retrieval_question,
                classification=classification,
                selected_policy_profile_id=selected_policy.profile_id,
                effective_query_type=effective_query_type,
                effective_top_k=effective_top_k,
                profile_rag_max_score=profile.rag_max_score,
                source_terms_for_retrieval=source_terms_for_retrieval,
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

            log.debug('chunks_retrieved', count=len(chunks), query_type=effective_query_type)

            if not chunks:
                fallback_profile = get_intent_profile_policy(selected_policy.fallback_target_route)
                fallback_events.append({
                    'fallback_from': selected_policy.profile_id,
                    'fallback_to': fallback_profile.profile_id,
                    'fallback_reason': 'empty_retrieval_result',
                })
                fallback_chunks = await retrieve_chunks(
                    query=retrieval_question,
                    top_k=get_retrieval_top_k(
                        fallback_profile.preferred_retrieval_mode,
                        response_mode=response_mode_used,
                    ),
                    max_score=profile.rag_max_score,
                    year_filter=classification.year_filter,
                    category_filter=classification.category_filter,
                    extension_filter=classification.file_type_filter,
                    filename_filter=retrieval_filename_filter,
                    source_terms_filter=source_terms_for_retrieval,
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
                    has_remaining_scope = False
                    yield ('__metrics__', _generation_terminal.build_generation_skipped_metrics_payload(
                        query_type=effective_query_type,
                        response_mode_used=response_mode_used,
                        mode_adjustments_applied=mode_adjustments_applied,
                        timeout_seconds=timeout_seconds,
                        retrieval_elapsed_ms=retrieval_elapsed_ms,
                        preflight_projected_seconds=preflight_projected_seconds,
                        preflight_ratio=preflight_ratio,
                        applied_degradations=applied_degradations,
                        fallback_events=fallback_events,
                        has_remaining_scope=has_remaining_scope,
                        validation_gates={'retrieval_relevance_gate': False, 'source_diversity_gate': False},
                    ))
                    yield _INSUFFICIENT_CONTEXT_RESPONSE
                    yield []
                    return

            retrieval_relevance_passed, retrieval_relevance_score = _retrieval_validation._evaluate_retrieval_relevance_gate(
                chunks=chunks,
                query_type=effective_query_type,
                route_candidate=selected_policy.profile_id,
                has_strong_anchor=bool(
                    classification.filename_filter
                    or (classification.year_filter is not None and source_terms_for_retrieval)
                ),
            )
            source_diversity_passed, distinct_sources_count = _retrieval_validation._evaluate_source_diversity_gate(
                chunks=chunks,
                query_type=effective_query_type,
            )
            retrieval_relevance_passed, fallback_events = _retrieval_validation._apply_coverage_evidence_floor_override(
                retrieval_relevance_passed=retrieval_relevance_passed,
                query_type=effective_query_type,
                subtype=classification.subtype,
                group_by=classification.group_by,
                response_shape=classification.response_shape,
                distinct_sources_count=distinct_sources_count,
                chunk_count=len(chunks),
                fallback_events=fallback_events,
                route_profile_id=selected_policy.profile_id,
                retrieval_relevance_score=retrieval_relevance_score,
            )
            current_source_keys = _retrieval_validation._extract_current_source_keys(chunks)
            continuation_anchor_passed, anchor_overlap_count = _retrieval_validation._evaluate_continuation_anchor_gate(
                route_candidate=classification.route_candidate,
                scope_reset_detected=scope_reset_detected,
                prior_source_anchors=prior_source_anchors,
                current_source_keys=current_source_keys,
                prior_has_remaining_scope=prior_has_remaining_scope,
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
                selected_policy_profile_id=selected_policy.profile_id,
                selected_policy_fallback_target_route=selected_policy.fallback_target_route,
                source_terms_for_retrieval=source_terms_for_retrieval,
                scope_reset_detected=scope_reset_detected,
                prior_source_anchors=prior_source_anchors,
                prior_has_remaining_scope=prior_has_remaining_scope,
                retrieval_question=retrieval_question,
                retrieval_filename_filter=retrieval_filename_filter,
                response_mode_used=response_mode_used,
                profile_rag_max_score=profile.rag_max_score,
                db=db,
                trace=trace,
                retrieve_fn=retrieve_chunks,
                get_retrieval_top_k_fn=get_retrieval_top_k,
                get_intent_profile_policy_fn=get_intent_profile_policy,
                compute_widened_retry_top_k_fn=_compute_widened_retry_top_k,
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
            if not retrieval_relevance_passed or not source_diversity_passed or not continuation_anchor_passed:
                has_remaining_scope = False
                yield ('__metrics__', _generation_terminal.build_generation_skipped_metrics_payload(
                    query_type=effective_query_type,
                    response_mode_used=response_mode_used,
                    mode_adjustments_applied=mode_adjustments_applied,
                    timeout_seconds=timeout_seconds,
                    retrieval_elapsed_ms=retrieval_elapsed_ms,
                    preflight_projected_seconds=preflight_projected_seconds,
                    preflight_ratio=preflight_ratio,
                    applied_degradations=applied_degradations,
                    fallback_events=fallback_events,
                    has_remaining_scope=has_remaining_scope,
                    validation_gates=validation_gates,
                ))
                if (
                    not _is_continuation_request(question)
                    and (
                        classification.route_candidate == 'clarification_or_disambiguation'
                        or classification.confidence_band == 'low'
                        or not continuation_anchor_passed
                    )
                ):
                    yield _build_clarification_fallback_message(classification)
                    yield []
                    return
                yield _INSUFFICIENT_CONTEXT_RESPONSE
                yield []
                return

            deterministic_fallback = await _deterministic_fallbacks.try_strict_or_structured_fallback(
                question=question,
                classification=classification,
                response_shape=effective_response_shape,
                chunks=chunks,
                response_mode=response_mode_used,
                db=db,
                trace=trace,
            )
            if deterministic_fallback.kind == 'strict':
                strict_answer = deterministic_fallback.answer or ''
                strict_sources = deterministic_fallback.sources or []
                strict_metrics = deterministic_fallback.strict_metrics or {}
                if trace is not None:
                    trace.record('strict_composer', {
                        'applied': True,
                        'family': strict_metrics.get('strict_composer_family'),
                        'sources_count': strict_metrics.get('strict_composer_sources_count'),
                        'claim_count': strict_metrics.get('strict_claim_count'),
                        'fallback_claim_count': strict_metrics.get('strict_fallback_claim_count'),
                        'unsupported_claim_count': strict_metrics.get('strict_unsupported_claim_count'),
                        'evidence_coverage_rate': strict_metrics.get('strict_evidence_coverage_rate'),
                        'claim_emission_decisions_preview': strict_metrics.get('strict_claim_emission_decisions_preview'),
                    })
                    trace.record('llm', {
                        'token_count': 0,
                        'max_tokens': effective_max_tokens,
                        'first_token_ms': None,
                        'total_elapsed_ms': 0.0,
                        'model_profile': profile.name,
                        'stream_recovery_reason': None,
                        'output_contract_check': strict_metrics.get('output_contract_check'),
                    })
                yield ('__metrics__', _generation_terminal.build_generation_skipped_metrics_payload(
                    query_type=effective_query_type,
                    response_mode_used=response_mode_used,
                    mode_adjustments_applied=mode_adjustments_applied,
                    timeout_seconds=timeout_seconds,
                    retrieval_elapsed_ms=retrieval_elapsed_ms,
                    preflight_projected_seconds=preflight_projected_seconds,
                    preflight_ratio=preflight_ratio,
                    applied_degradations=applied_degradations,
                    fallback_events=fallback_events,
                    has_remaining_scope=False,
                    extra_fields=strict_metrics,
                ))
                yield strict_answer
                yield strict_sources
                return

            if deterministic_fallback.kind == 'structured':
                structured_answer = deterministic_fallback.answer or ''
                structured_sources = deterministic_fallback.sources or []
                structured_metrics = deterministic_fallback.structured_metrics or {}
                yield ('__metrics__', _generation_terminal.build_generation_skipped_metrics_payload(
                    query_type=effective_query_type,
                    response_mode_used=response_mode_used,
                    mode_adjustments_applied=mode_adjustments_applied,
                    timeout_seconds=timeout_seconds,
                    retrieval_elapsed_ms=retrieval_elapsed_ms,
                    preflight_projected_seconds=preflight_projected_seconds,
                    preflight_ratio=preflight_ratio,
                    applied_degradations=applied_degradations,
                    fallback_events=fallback_events,
                    has_remaining_scope=False,
                    extra_fields=structured_metrics,
                ))
                yield structured_answer
                yield structured_sources
                return
            filename_summary_fallback_answer = _build_filename_summary_fallback_answer(
                question=question,
                filename_filter=classification.filename_filter,
                chunks=chunks,
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
                    truncate_preview_fn=_truncate_preview,
                    normalize_relevance_score_fn=_retrieval_validation._normalize_relevance_score,
                )
                yield ('__metrics__', _generation_terminal.build_generation_skipped_metrics_payload(
                    query_type=effective_query_type,
                    response_mode_used=response_mode_used,
                    mode_adjustments_applied=mode_adjustments_applied,
                    timeout_seconds=timeout_seconds,
                    retrieval_elapsed_ms=retrieval_elapsed_ms,
                    preflight_projected_seconds=preflight_projected_seconds,
                    preflight_ratio=preflight_ratio,
                    applied_degradations=applied_degradations,
                    fallback_events=fallback_events,
                    has_remaining_scope=False,
                ))
                yield filename_summary_fallback_answer
                yield fallback_sources
                return
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
                        truncate_preview_fn=_truncate_preview,
                        normalize_relevance_score_fn=_retrieval_validation._normalize_relevance_score,
                    )
                    yield ('__metrics__', _generation_terminal.build_generation_skipped_metrics_payload(
                        query_type=effective_query_type,
                        response_mode_used=response_mode_used,
                        mode_adjustments_applied=mode_adjustments_applied,
                        timeout_seconds=timeout_seconds,
                        retrieval_elapsed_ms=retrieval_elapsed_ms,
                        preflight_projected_seconds=preflight_projected_seconds,
                        preflight_ratio=preflight_ratio,
                        applied_degradations=applied_degradations,
                        fallback_events=fallback_events,
                        has_remaining_scope=False,
                    ))
                    yield fallback_answer
                    yield fallback_sources
                    return
            if (
                effective_response_shape == 'structured_extract'
                and classification.subtype in _STRUCTURED_EXTRACTION_SUBTYPES
            ):
                fallback_events.append({
                    'fallback_from': 'structured_field_extraction',
                    'fallback_to': 'targeted_fact_lookup',
                    'fallback_reason': 'structured_extraction_insufficient',
                })
                effective_response_shape = 'narrative_synthesis'

            # 3. Build messages
            generation_plan = _generation_plan.build_generation_prompt_plan(
                question=question,
                chunks=chunks,
                history=history,
                profile_name=profile.name,
                profile_prepare_messages_fn=profile.prepare_messages,
                build_messages_fn=build_messages,
                fit_to_budget_enabled=fit_to_budget_enabled,
                policy_soft_top_k_threshold=policy.soft_top_k_threshold,
                policy_soft_coverage_to_focused_threshold=policy.soft_coverage_to_focused_threshold,
                policy_soft_output_cap_threshold=policy.soft_output_cap_threshold,
                retrieval_precloseout_min_relevance_score=float(settings.retrieval_precloseout_min_relevance_score),
                retrieval_relevance_score=float(retrieval_relevance_score),
                subtype=classification.subtype,
                focused_max_tokens=profile.get_max_tokens('focused'),
                focused_timeout_seconds=profile.get_timeout_seconds('focused'),
                applied_degradations=applied_degradations,
                min_output_budget_floor=diagnostics_min_words,
                output_constraints=output_constraints,
                effective_query_type=effective_query_type,
                effective_top_k=effective_top_k,
                effective_reasoning_enabled=effective_reasoning_enabled,
                effective_max_tokens=effective_max_tokens,
                timeout_seconds=timeout_seconds,
                response_mode=response_mode_used,
                route_candidate=classification.route_candidate,
                dedupe_prompt_chunks_fn=_deduplicate_prompt_chunks,
                derive_format_requirements_fn=_structured_numeric._derive_format_requirements,
            )
            chunks = generation_plan.chunks
            effective_query_type = generation_plan.effective_query_type
            effective_top_k = generation_plan.effective_top_k
            effective_reasoning_enabled = generation_plan.effective_reasoning_enabled
            effective_max_tokens = generation_plan.effective_max_tokens
            timeout_seconds = generation_plan.timeout_seconds
            applied_degradations = generation_plan.applied_degradations
            post_retrieval_projected_seconds = generation_plan.post_retrieval_projected_seconds
            post_retrieval_ratio = generation_plan.post_retrieval_ratio
            pre_closeout_quality_check_passed = generation_plan.pre_closeout_quality_check_passed
            pre_closeout_quality_check_reason = generation_plan.pre_closeout_quality_check_reason
            output_constraints = generation_plan.output_constraints
            format_requirements = generation_plan.format_requirements
            output_contract_plan = generation_plan.output_contract_plan
            messages = generation_plan.messages
            prompt_elapsed_ms = generation_plan.prompt_elapsed_ms
            diagnostics_depth_constraints_applied = generation_plan.diagnostics_depth_constraints_applied

            if not pre_closeout_quality_check_passed:
                fallback_events.append({
                    'fallback_from': selected_policy.profile_id,
                    'fallback_to': selected_policy.fallback_target_route,
                    'fallback_reason': pre_closeout_quality_check_reason,
                })
                has_remaining_scope = _generation_runtime._has_remaining_scope(
                    timeout_reason=None,
                    stream_recovery_reason='pre_closeout_quality_check_failed',
                    generation_skipped=True,
                    applied_degradations=applied_degradations,
                )
                yield ('__metrics__', _generation_terminal.build_generation_skipped_metrics_payload(
                    query_type=effective_query_type,
                    response_mode_used=response_mode_used,
                    mode_adjustments_applied=mode_adjustments_applied,
                    timeout_seconds=timeout_seconds,
                    retrieval_elapsed_ms=retrieval_elapsed_ms,
                    preflight_projected_seconds=preflight_projected_seconds,
                    preflight_ratio=preflight_ratio,
                    applied_degradations=applied_degradations,
                    fallback_events=fallback_events,
                    has_remaining_scope=has_remaining_scope,
                    suggested_completion_mode='scoped_complete',
                    post_retrieval_projected_seconds=post_retrieval_projected_seconds,
                    post_retrieval_ratio=post_retrieval_ratio,
                    validation_gates=validation_gates,
                    retrieval_relevance_score=retrieval_relevance_score,
                    pre_closeout_quality_check={
                        'passed': pre_closeout_quality_check_passed,
                        'reason': pre_closeout_quality_check_reason,
                    },
                ))
                if (
                    not _is_continuation_request(question)
                    and (
                        classification.route_candidate == 'clarification_or_disambiguation'
                        or classification.confidence_band == 'low'
                    )
                ):
                    yield _build_clarification_fallback_message(classification)
                    yield []
                    return
                yield _INSUFFICIENT_CONTEXT_RESPONSE
                yield []
                return

            history_for_prompt = history

            if trace is not None:
                trace.record('prompt', {
                    'messages_count':    len(messages),
                    'context_chunks':    len(chunks),
                    'history_messages':  len(history_for_prompt) if history_for_prompt else 0,
                    'reasoning_enabled':  effective_reasoning_enabled,
                    'output_constraints': output_constraints,
                    'format_requirements': format_requirements,
                    'response_mode_used': response_mode_used,
                    'mode_adjustments_applied': mode_adjustments_applied,
                    'applied_degradations': applied_degradations,
                    'diagnostics_depth_constraints_applied': diagnostics_depth_constraints_applied,
                    'duration_ms':        round(prompt_elapsed_ms, 1),
                })

            if fit_to_budget_enabled and post_retrieval_ratio >= policy.hard_pre_generation_threshold:
                has_remaining_scope = _generation_runtime._has_remaining_scope(
                    timeout_reason=None,
                    stream_recovery_reason='hard_pre_generation_scope_reduction',
                    generation_skipped=True,
                    applied_degradations=applied_degradations,
                )
                scoped_response = (
                    'I narrowed the request automatically to stay within the response time budget.\n\n'
                    f'- **Completed scope:** analyzed top {len(chunks)} most relevant context chunk(s).\n'
                    '- **Omitted scope:** exhaustive cross-document expansion was skipped for this turn.\n'
                    '- **Next step:** ask a follow-up for the remaining sections if you need full coverage.'
                )
                scoped_metrics_payload = _generation_terminal.build_generation_skipped_metrics_payload(
                    query_type=effective_query_type,
                    response_mode_used=response_mode_used,
                    mode_adjustments_applied=mode_adjustments_applied,
                    timeout_seconds=timeout_seconds,
                    retrieval_elapsed_ms=retrieval_elapsed_ms,
                    preflight_projected_seconds=preflight_projected_seconds,
                    preflight_ratio=preflight_ratio,
                    applied_degradations=applied_degradations,
                    fallback_events=fallback_events,
                    has_remaining_scope=has_remaining_scope,
                    suggested_completion_mode='scoped_complete',
                    post_retrieval_projected_seconds=post_retrieval_projected_seconds,
                    post_retrieval_ratio=post_retrieval_ratio,
                )
                scoped_metrics_payload['prompt_duration_ms'] = round(prompt_elapsed_ms, 1)
                scoped_metrics_payload['fit_to_budget_rollout_stage'] = policy.rollout_stage
                scoped_metrics_payload['fit_to_budget_enabled'] = fit_to_budget_enabled
                scoped_metrics_payload['fit_to_budget_sample_count'] = policy.sample_count
                scoped_metrics_payload['fit_to_budget_timeout_rate'] = policy.timeout_rate
                yield ('__metrics__', scoped_metrics_payload)
                yield scoped_response
                sources = _generation_closeout.build_source_references(
                    chunks=chunks,
                    truncate_preview_fn=_truncate_preview,
                    normalize_relevance_score_fn=_retrieval_validation._normalize_relevance_score,
                )
                yield sources
                return

            # 4. Get model profile settings for query type
            max_tokens = effective_max_tokens
            stop_sequences = profile.get_stop_sequences(effective_reasoning_enabled)
            soft_closeout_allowed = _generation_runtime._should_apply_soft_stream_closeout(format_requirements)

            # 5. Stream response; collect tokens for trace
            stream_summary: _generation_stream.StreamExecutionSummary | None = None
            async for item in _generation_stream.stream_generation_with_budget(
                messages=messages,
                max_tokens=max_tokens,
                temperature=profile.temperature,
                top_p=profile.top_p,
                timeout_seconds=timeout_seconds,
                stop_sequences=stop_sequences,
                fit_to_budget_enabled=fit_to_budget_enabled,
                stream_soft_limit_ratio=policy.stream_soft_limit_ratio,
                soft_closeout_allowed=soft_closeout_allowed,
                checkpoint_query_type=effective_query_type,
                dedupe_insufficient_context_after_stream=profile.dedupe_insufficient_context_after_stream,
                insufficient_context_response=_INSUFFICIENT_CONTEXT_RESPONSE,
                applied_degradations=applied_degradations,
                output_contract_plan=output_contract_plan,
                collapse_duplicate_message_fn=_collapse_duplicate_insufficient_context_message,
                stream_llm_fn=stream_llm,
            ):
                if (
                    isinstance(item, tuple)
                    and len(item) == 2
                    and item[0] == _generation_stream.STREAM_SUMMARY_EVENT
                ):
                    stream_summary = item[1] if isinstance(item[1], _generation_stream.StreamExecutionSummary) else None
                    continue
                yield item

            if stream_summary is None:
                stream_summary = _generation_stream.StreamExecutionSummary(
                    token_count=0,
                    first_token_ms=None,
                    total_elapsed_ms=0.0,
                    timeout_reason='missing_stream_summary',
                    stream_recovery_reason='missing_stream_summary',
                    soft_budget_checkpoints_hit=[],
                    completion_mode='scoped_complete',
                    has_remaining_scope=True,
                    output_contract_check={'passed': False, 'error': 'missing_stream_summary'},
                )
                applied_degradations.append({
                    'step': 'missing_stream_summary_guard',
                    'reason': 'stream_summary_not_emitted',
                })

            token_count = stream_summary.token_count
            first_token_ms = stream_summary.first_token_ms
            llm_elapsed_ms = stream_summary.total_elapsed_ms
            timeout_reason = stream_summary.timeout_reason
            stream_recovery_reason = stream_summary.stream_recovery_reason
            completion_mode = stream_summary.completion_mode
            has_remaining_scope = stream_summary.has_remaining_scope
            output_contract_check = stream_summary.output_contract_check
            checkpoints_hit = stream_summary.soft_budget_checkpoints_hit

            metrics_payload = _generation_closeout.build_generation_metrics_payload(
                query_type=effective_query_type,
                response_mode_used=response_mode_used,
                mode_adjustments_applied=mode_adjustments_applied,
                timeout_seconds=timeout_seconds,
                retrieval_elapsed_ms=retrieval_elapsed_ms,
                prompt_elapsed_ms=prompt_elapsed_ms,
                first_token_ms=first_token_ms,
                llm_elapsed_ms=llm_elapsed_ms,
                timeout_reason=timeout_reason,
                checkpoints_hit=checkpoints_hit,
                completion_mode=completion_mode,
                preflight_projected_seconds=preflight_projected_seconds,
                preflight_ratio=preflight_ratio,
                post_retrieval_projected_seconds=post_retrieval_projected_seconds,
                post_retrieval_ratio=post_retrieval_ratio,
                fit_to_budget_rollout_stage=policy.rollout_stage,
                fit_to_budget_enabled=fit_to_budget_enabled,
                fit_to_budget_sample_count=policy.sample_count,
                fit_to_budget_timeout_rate=policy.timeout_rate,
                fit_to_budget_first_token_p95_ms=policy.first_token_p95_ms,
                fit_to_budget_completion_p95_seconds=policy.completion_p95_seconds,
                applied_degradations=applied_degradations,
                fallback_events=fallback_events,
                has_remaining_scope=has_remaining_scope,
                stream_recovery_reason=stream_recovery_reason,
                output_contract_check=output_contract_check,
            )
            yield ('__metrics__', metrics_payload)
            _generation_closeout.record_generation_trace(
                trace=trace,
                token_count=token_count,
                max_tokens=max_tokens,
                first_token_ms=first_token_ms,
                llm_elapsed_ms=llm_elapsed_ms,
                profile_name=profile.name,
                stream_recovery_reason=stream_recovery_reason,
                output_contract_check=output_contract_check,
            )
            _generation_closeout.log_generation_completion(
                log=log,
                query_type=effective_query_type,
                question_length=len(question),
                context_chunks=len(chunks),
                history_messages=len(history_for_prompt) if history_for_prompt else 0,
                max_tokens=max_tokens,
                timeout_seconds=timeout_seconds,
                prompt_elapsed_ms=prompt_elapsed_ms,
                first_token_ms=first_token_ms,
                llm_elapsed_ms=llm_elapsed_ms,
                token_count=token_count,
                preflight_ratio=preflight_ratio,
                post_retrieval_ratio=post_retrieval_ratio,
                response_mode_used=response_mode_used,
                mode_adjustments_applied=mode_adjustments_applied,
                applied_degradations=applied_degradations,
                stream_recovery_reason=stream_recovery_reason,
            )

            # 6. Build source references and record in trace
            # Normalize CrossEncoder raw logits to 0-1 for display (sigmoid)
            sources = _generation_closeout.build_source_references(
                chunks=chunks,
                truncate_preview_fn=_truncate_preview,
                normalize_relevance_score_fn=_retrieval_validation._normalize_relevance_score,
            )
            _generation_closeout.record_sources_trace(trace=trace, sources=sources)

            yield sources
        except _HANDLER_RUNTIME_EXCEPTIONS as exc:
            log.error('rag_handler_failed', error=str(exc), exc_info=True)
            yield f"Error: {exc}"
            yield []
