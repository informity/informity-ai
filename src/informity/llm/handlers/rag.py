# ==============================================================================
# Informity AI — RAG Query Handler
# Handles focused and coverage queries using vector search → rerank → LLM
# ==============================================================================

import asyncio
import re
from collections.abc import AsyncGenerator

import aiosqlite
import structlog

from informity.api.schemas import ChatSourceReference
from informity.config import settings
from informity.db.models import ChatMessage
from informity.llm.fit_to_budget_tuning import resolve_fit_to_budget_policy
from informity.llm.prompt_builder import build_messages
from informity.llm.query_classifier import QueryClassification
from informity.llm.rag_runtime import execution_plan as _execution_plan
from informity.llm.rag_runtime import generation_closeout as _generation_closeout
from informity.llm.rag_runtime import generation_plan as _generation_plan
from informity.llm.rag_runtime import generation_runtime as _generation_runtime
from informity.llm.rag_runtime import generation_stream as _generation_stream
from informity.llm.rag_runtime import generation_terminal as _generation_terminal
from informity.llm.rag_runtime import retrieval_pipeline as _retrieval_pipeline
from informity.llm.rag_runtime import retrieval_plan as _retrieval_plan
from informity.llm.rag_runtime import retrieval_validation as _retrieval_validation
from informity.llm.rag_runtime import structured_numeric as _structured_numeric
from informity.llm.rag_runtime.retrieval_pipeline import _build_clarification_fallback_message
from informity.llm.streaming import stream_llm
from informity.llm.types import CompletionMode, ConfidenceBand, FallbackReason, IntentProfileId, QueryType, StreamSignalTag

log = structlog.get_logger(__name__)
_HANDLER_RUNTIME_EXCEPTIONS = (RuntimeError, ValueError, TypeError, OSError, asyncio.TimeoutError)

_INSUFFICIENT_CONTEXT_RESPONSE = _retrieval_pipeline._INSUFFICIENT_CONTEXT_RESPONSE
_CHUNK_PREVIEW_MAX_LENGTH = 200
_FALLBACK_SOURCE_LIMIT = 8
_OUTPUT_CONTRACT_WORD_LIMIT_PATTERN = re.compile(
    r'\b(?:<=?|at\s+most|max(?:imum)?)\s*\d+\s+words?\b',
    re.IGNORECASE,
)
_OUTPUT_CONTRACT_BULLET_LIMIT_PATTERN = re.compile(
    r'\bexactly\s+\d+\s+bullets?\b',
    re.IGNORECASE,
)


def _has_explicit_output_contract(question: str) -> bool:
    if _OUTPUT_CONTRACT_WORD_LIMIT_PATTERN.search(question):
        return True
    if _OUTPUT_CONTRACT_BULLET_LIMIT_PATTERN.search(question):
        return True
    if 'output must contain' in question.casefold():
        return True
    return bool(_structured_numeric._derive_format_requirements(question))


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


def _resolve_sampling_params(
    *,
    profile_temperature: float,
    profile_top_p: float,
    format_requirements: list[str] | None = None,
) -> tuple[float, float]:
    # Keep runtime simple and contract-compliant:
    # when strict format contracts are present, lower sampling variance so
    # heading/section structure is more reproducible across reruns.
    requirements_joined = ' '.join(str(item or '').casefold() for item in (format_requirements or []))
    strict_contract = (
        'requested order' in requirements_joined
        or 'include heading:' in requirements_joined
        or 'one subsection per year' in requirements_joined
    )
    if strict_contract:
        return min(profile_temperature, 0.2), min(profile_top_p, 0.8)
    return profile_temperature, profile_top_p


def _truncate_preview(text: str, max_length: int = _CHUNK_PREVIEW_MAX_LENGTH) -> str:
    return text[:max_length]




class RAGHandler:
    """
    Handler for focused and coverage queries.

    Uses vector search → rerank → LLM pipeline.
    """

    def matches(self, classification: QueryClassification) -> bool:
        """Match focused and coverage queries."""
        return classification.intent in {QueryType.FOCUSED, QueryType.COVERAGE}

    async def handle(
        self,
        question:       str,
        classification: QueryClassification,
        history:        list[ChatMessage] | None,
        db:             aiosqlite.Connection,
        trace:          object | None,
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
                diagnostics_context=diagnostics_context,
                db=db,
                resolve_fit_to_budget_policy_fn=resolve_fit_to_budget_policy,
            )
            profile = plan.profile
            selected_policy = plan.selected_policy
            effective_response_shape = plan.effective_response_shape
            timeout_seconds = plan.timeout_seconds
            max_tokens = plan.max_tokens
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
            retrieval_question = retrieval_context.retrieval_question
            source_terms_for_retrieval = list(classification.source_terms or [])
            for term in continuation_source_terms:
                if term not in source_terms_for_retrieval:
                    source_terms_for_retrieval.append(term)
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
                focused_max_tokens=profile.get_max_tokens(QueryType.FOCUSED),
                focused_timeout_seconds=profile.get_timeout_seconds(QueryType.FOCUSED),
                output_constraints=output_constraints,
                applied_degradations=applied_degradations,
                route_candidate=selected_policy.profile_id,
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
                    'coverage_mode':     effective_query_type == QueryType.COVERAGE,
                    'reasoning_enabled':  effective_reasoning_enabled,
                    'top_k':              effective_top_k,
                    'rag_max_score':     getattr(profile, 'rag_max_score', None),
                    'budget_preflight_projected_seconds': round(preflight_projected_seconds, 1),
                    'budget_preflight_ratio': round(preflight_ratio, 3),
                    'fit_to_budget_rollout_stage': policy.rollout_stage,
                    'fit_to_budget_enabled': fit_to_budget_enabled,
                    'fit_to_budget_sample_count': policy.sample_count,
                    'fit_to_budget_timeout_rate': policy.timeout_rate,
                    'strict_ordered_mode': strict_ordered_mode,
                    'applied_degradations': applied_degradations,
                    'fallback_events': fallback_events,
                })

            # 2. Retrieve chunks, validate, and apply all fallbacks via retrieval pipeline
            retrieval_outcome = await _retrieval_pipeline.run_retrieval_pipeline(
                question=question,
                retrieval_question=retrieval_question,
                classification=classification,
                effective_query_type=effective_query_type,
                effective_top_k=effective_top_k,
                effective_max_tokens=effective_max_tokens,
                effective_response_shape=effective_response_shape,
                timeout_seconds=timeout_seconds,
                continuation_source_terms=continuation_source_terms,
                prior_has_remaining_scope=prior_has_remaining_scope,
                scope_reset_detected=scope_reset_detected,
                prior_source_anchors=prior_source_anchors,
                retrieval_filename_filter=retrieval_filename_filter,
                selected_policy_profile_id=selected_policy.profile_id,
                selected_policy_fallback_target_route=selected_policy.fallback_target_route,
                profile_rag_max_score=profile.rag_max_score,
                applied_degradations=applied_degradations,
                fallback_events=fallback_events,
                preflight_projected_seconds=preflight_projected_seconds,
                preflight_ratio=preflight_ratio,
                db=db,
                trace=trace,
                has_explicit_output_contract_fn=_has_explicit_output_contract,
                truncate_preview_fn=_truncate_preview,
                normalize_relevance_score_fn=_retrieval_validation._normalize_relevance_score,
            )
            # Forward plan_step events to SSE stream
            for _evt in retrieval_outcome.plan_step_events:
                yield _evt
            # Terminal retrieval outcome: skip generation
            if isinstance(retrieval_outcome, _retrieval_pipeline.RetrievalFailure):
                has_remaining_scope = retrieval_outcome.has_remaining_scope
                yield (StreamSignalTag.METRICS, retrieval_outcome.metrics_payload)
                yield retrieval_outcome.response_message
                yield retrieval_outcome.sources
                return
            # Unpack retrieval success state for generation stage
            chunks = retrieval_outcome.chunks
            effective_query_type = retrieval_outcome.effective_query_type
            effective_top_k = retrieval_outcome.effective_top_k
            effective_response_shape = retrieval_outcome.effective_response_shape
            retrieval_relevance_score = retrieval_outcome.retrieval_relevance_score
            validation_gates = retrieval_outcome.validation_gates
            fallback_events = retrieval_outcome.fallback_events
            applied_degradations = retrieval_outcome.applied_degradations
            retrieval_elapsed_ms = retrieval_outcome.retrieval_elapsed_ms
            retrieve_timing = retrieval_outcome.retrieve_timing
            _gatekeeper_demoted_query_type = retrieval_outcome.gatekeeper_demoted_query_type

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
                focused_max_tokens=profile.get_max_tokens(QueryType.FOCUSED),
                focused_timeout_seconds=profile.get_timeout_seconds(QueryType.FOCUSED),
                applied_degradations=applied_degradations,
                min_output_budget_floor=diagnostics_min_words,
                output_constraints=output_constraints,
                effective_query_type=effective_query_type,
                effective_top_k=effective_top_k,
                effective_reasoning_enabled=effective_reasoning_enabled,
                effective_max_tokens=effective_max_tokens,
                timeout_seconds=timeout_seconds,
                route_candidate=classification.route_candidate,
                dedupe_prompt_chunks_fn=_retrieval_pipeline._deduplicate_prompt_chunks,
                derive_format_requirements_fn=_structured_numeric._derive_format_requirements,
                action_hints=classification.action_hints,
                # Skip the pre-closeout quality check when the gatekeeper demoted a coverage query
                # to focused mode via fallback. The check — "don't generate for a focused query with
                # uncertain relevance under budget pressure" — does not apply: the original query was
                # coverage-typed, so low reranker scores are expected (chunks are anchored by year
                # filter, not semantic similarity). The gatekeeper's fallback already confirms that
                # sufficient evidence exists.
                skip_precloseout_quality_check=_gatekeeper_demoted_query_type,
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
                    'fallback_reason': FallbackReason.PRE_CLOSEOUT_QUALITY_CHECK_FAILED,
                })
                has_remaining_scope = _generation_runtime._has_remaining_scope(
                    timeout_reason=None,
                    stream_recovery_reason='pre_closeout_quality_check_failed',
                    generation_skipped=True,
                    applied_degradations=applied_degradations,
                )
                yield (StreamSignalTag.METRICS, _generation_terminal.build_generation_skipped_metrics_payload(
                    query_type=effective_query_type,
                    timeout_seconds=timeout_seconds,
                    retrieval_elapsed_ms=retrieval_elapsed_ms,
                    preflight_projected_seconds=preflight_projected_seconds,
                    preflight_ratio=preflight_ratio,
                    applied_degradations=applied_degradations,
                    fallback_events=fallback_events,
                    has_remaining_scope=has_remaining_scope,
                    suggested_completion_mode=CompletionMode.SCOPED_COMPLETE,
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
                    not classification.is_continuation
                    and (
                        classification.route_candidate == IntentProfileId.CLARIFICATION_OR_DISAMBIGUATION
                        or classification.confidence_band == ConfidenceBand.LOW
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
                    'applied_degradations': applied_degradations,
                    'diagnostics_depth_constraints_applied': diagnostics_depth_constraints_applied,
                    'duration_ms':        round(prompt_elapsed_ms, 1),
                })

            # 4. Get model profile settings for query type
            max_tokens = effective_max_tokens
            stop_sequences = profile.get_stop_sequences(effective_reasoning_enabled)
            soft_closeout_allowed = _generation_runtime._should_apply_soft_stream_closeout(format_requirements)
            generation_temperature, generation_top_p = _resolve_sampling_params(
                profile_temperature=profile.temperature,
                profile_top_p=profile.top_p,
                format_requirements=format_requirements,
            )

            # 5. Stream response; collect tokens for trace
            stream_summary: _generation_stream.StreamExecutionSummary | None = None
            async for item in _generation_stream.stream_generation_with_budget(
                messages=messages,
                max_tokens=max_tokens,
                temperature=generation_temperature,
                top_p=generation_top_p,
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
                    completion_mode=CompletionMode.SCOPED_COMPLETE,
                    has_remaining_scope=True,
                )
                applied_degradations.append({
                    'step': 'missing_stream_summary_guard',
                    'reason': 'stream_summary_not_emitted',
                })

            # Populate per-stage timing fields on stream_summary.
            # retrieve_timing comes from the primary retrieval; prompt_build_ms and ttft_ms
            # are computed from existing stage measurements.
            stream_summary.embed_ms = retrieve_timing.get('embed_ms')
            stream_summary.vector_search_ms = retrieve_timing.get('vector_search_ms')
            stream_summary.rerank_ms = retrieve_timing.get('rerank_ms')
            stream_summary.prompt_build_ms = round(prompt_elapsed_ms, 1)
            stream_summary.ttft_ms = stream_summary.first_token_ms

            token_count = stream_summary.token_count
            first_token_ms = stream_summary.first_token_ms
            llm_elapsed_ms = stream_summary.total_elapsed_ms
            timeout_reason = stream_summary.timeout_reason
            stream_recovery_reason = stream_summary.stream_recovery_reason
            completion_mode = stream_summary.completion_mode
            has_remaining_scope = stream_summary.has_remaining_scope
            checkpoints_hit = stream_summary.soft_budget_checkpoints_hit

            metrics_payload = _generation_closeout.build_generation_metrics_payload(
                query_type=effective_query_type,
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
                embed_ms=stream_summary.embed_ms,
                vector_search_ms=stream_summary.vector_search_ms,
                rerank_ms=stream_summary.rerank_ms,
                prompt_build_ms=stream_summary.prompt_build_ms,
                ttft_ms=stream_summary.ttft_ms,
            )
            yield (StreamSignalTag.METRICS, metrics_payload)
            _generation_closeout.record_generation_trace(
                trace=trace,
                token_count=token_count,
                max_tokens=max_tokens,
                first_token_ms=first_token_ms,
                llm_elapsed_ms=llm_elapsed_ms,
                profile_name=profile.name,
                stream_recovery_reason=stream_recovery_reason,
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
                applied_degradations=applied_degradations,
                stream_recovery_reason=stream_recovery_reason,
            )

            # 6. Build source references and record in trace
            # Normalize CrossEncoder raw logits to 0-1 for display (sigmoid)
            sources = _generation_closeout.build_source_references(
                chunks=chunks,
                answer_text=stream_summary.final_answer,
                truncate_preview_fn=_truncate_preview,
                normalize_relevance_score_fn=_retrieval_validation._normalize_relevance_score,
            )
            _generation_closeout.record_sources_trace(trace=trace, sources=sources)

            yield sources
        except _HANDLER_RUNTIME_EXCEPTIONS as exc:
            log.error('rag_handler_failed', error=str(exc), exc_info=True)
            yield f"Error: {exc}"
            yield []
