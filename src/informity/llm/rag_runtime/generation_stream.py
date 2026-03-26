# ==============================================================================
# Informity AI — Generation Stream Runtime
# Streaming execution + completion summary extraction for RAG handler.
# ==============================================================================

from __future__ import annotations

import re
import time
from collections.abc import AsyncGenerator, Callable
from dataclasses import dataclass

import structlog

from informity.llm.rag_runtime import generation_runtime as _generation_runtime
from informity.llm.streaming import stream_llm

log = structlog.get_logger(__name__)

STREAM_SUMMARY_EVENT = '__stream_summary__'


@dataclass
class StreamExecutionSummary:
    token_count: int
    first_token_ms: float | None
    total_elapsed_ms: float
    timeout_reason: str | None
    stream_recovery_reason: str | None
    soft_budget_checkpoints_hit: list[int]
    completion_mode: str
    has_remaining_scope: bool
    final_answer: str = ''
    # Per-stage latency breakdown (set by rag.py after streaming completes).
    # All values are wall-clock milliseconds measured with perf_counter.
    embed_ms: float | None = None           # Query embedding time
    vector_search_ms: float | None = None   # Vector ANN search time
    rerank_ms: float | None = None          # Cross-encoder reranker time
    prompt_build_ms: float | None = None    # Context assembly + message build time
    ttft_ms: float | None = None            # Time to first generated token


def _is_section_boundary(token: str) -> bool:
    return (
        '\n\n' in token
        or token.rstrip().endswith('.')
        or token.rstrip().endswith('!')
        or token.rstrip().endswith('?')
        or token.rstrip().endswith(':')
    )


def _find_section_anchor_position(answer: str, section: str) -> int | None:
    section_core = re.escape(section.strip())
    section_core = section_core.replace(r'\ ', r'\s+')
    line_patterns = (
        rf'^\s*#{{1,6}}\s*(?:\d+[\).]\s*)?{section_core}(?:\s*\([^)\n]*\))?\s*$',
        rf'^\s*(?:\d+[\).]\s*)?{section_core}(?:\s*\([^)\n]*\))?\s*$',
    )
    for pattern in line_patterns:
        match = re.search(pattern, answer, flags=re.IGNORECASE | re.MULTILINE)
        if match:
            return match.start()
    return None


async def stream_generation_with_budget(
    *,
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float,
    top_p: float,
    timeout_seconds: int,
    stop_sequences: list[str],
    fit_to_budget_enabled: bool,
    stream_soft_limit_ratio: float,
    soft_closeout_allowed: bool,
    checkpoint_query_type: str | None,
    dedupe_insufficient_context_after_stream: bool,
    insufficient_context_response: str,
    applied_degradations: list[dict[str, object]],
    output_contract_plan: object | None,
    collapse_duplicate_message_fn: Callable[[str], tuple[str, bool]],
    stream_llm_fn: Callable[..., AsyncGenerator[str | tuple[str, object]]] = stream_llm,
) -> AsyncGenerator[str | tuple[str, object]]:
    checkpoint_targets = [0.6, 0.8]
    checkpoints_emitted: set[float] = set()
    timeout_reason: str | None = None
    stream_recovery_reason: str | None = None
    stream_soft_limit_ms = timeout_seconds * stream_soft_limit_ratio * 1000
    should_close_after_boundary = False

    llm_start = time.perf_counter()
    token_count = 0
    first_token_ms: float | None = None
    answer_parts: list[str] = []
    async for item in stream_llm_fn(
        messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        timeout_seconds=timeout_seconds,
        stop_sequences=stop_sequences,
    ):
        if isinstance(item, tuple) and len(item) == 2 and item[0] == '__timeout__':
            timeout_payload = item[1] if isinstance(item[1], dict) else {}
            timeout_reason = str(timeout_payload.get('reason') or 'unknown_timeout')
            yield ('__timeout__', timeout_payload)
            continue

        if not isinstance(item, str):
            continue

        if dedupe_insufficient_context_after_stream and answer_parts:
            projected_answer = ''.join(answer_parts) + item
            if projected_answer.count(insufficient_context_response) > 1:
                stream_recovery_reason = 'duplicate_insufficient_context_guard'
                applied_degradations.append({
                    'step': 'duplicate_insufficient_context_guard',
                    'reason': 'duplicate_insufficient_context_phrase_detected',
                })
                break

        stream_elapsed_ms = (time.perf_counter() - llm_start) * 1000
        for checkpoint_ratio in checkpoint_targets:
            checkpoint_ms = timeout_seconds * checkpoint_ratio * 1000
            if checkpoint_ratio in checkpoints_emitted:
                continue
            if stream_elapsed_ms >= checkpoint_ms:
                checkpoints_emitted.add(checkpoint_ratio)
                checkpoint_payload: dict[str, object] = {
                    'ratio': checkpoint_ratio,
                    'elapsed_seconds': round(stream_elapsed_ms / 1000, 1),
                    'timeout_seconds': timeout_seconds,
                }
                if checkpoint_query_type:
                    checkpoint_payload['query_type'] = checkpoint_query_type
                yield ('__budget_checkpoint__', checkpoint_payload)

        if first_token_ms is None:
            first_token_ms = stream_elapsed_ms

        if fit_to_budget_enabled and soft_closeout_allowed and stream_elapsed_ms >= stream_soft_limit_ms:
            should_close_after_boundary = True

        token_count += 1
        answer_parts.append(item)
        yield item

        if should_close_after_boundary and _is_section_boundary(item):
            stream_recovery_reason = 'soft_limit_section_closeout'
            applied_degradations.append({
                'step': 'mid_stream_recovery_soft_limit',
                'elapsed_seconds': round(stream_elapsed_ms / 1000, 1),
                'soft_limit_seconds': round(stream_soft_limit_ms / 1000, 1),
                'reason': 'soft_budget_limit_crossed',
            })
            break

    llm_elapsed_ms = (time.perf_counter() - llm_start) * 1000
    if dedupe_insufficient_context_after_stream and answer_parts:
        deduped_answer, dedup_applied = collapse_duplicate_message_fn(''.join(answer_parts))
        if dedup_applied:
            answer_parts = [deduped_answer]
            applied_degradations.append({
                'step': 'post_stream_duplicate_insufficient_context_dedup',
                'reason': 'duplicate_insufficient_context_phrase_collapsed',
            })
    min_year_subsections = 0
    expected_years: list[int] = []
    if isinstance(output_contract_plan, dict):
        raw_min_year_subsections = output_contract_plan.get('min_year_subsections')
        if isinstance(raw_min_year_subsections, int) and raw_min_year_subsections > 0:
            min_year_subsections = raw_min_year_subsections
        raw_expected_years = output_contract_plan.get('expected_years')
        if isinstance(raw_expected_years, list):
            expected_years = [int(year) for year in raw_expected_years if isinstance(year, int)]
    if min_year_subsections > 0:
        full_answer = ''.join(answer_parts)
        present_years = {int(match) for match in re.findall(r'\b(?:19|20)\d{2}\b', full_answer)}
        missing_years = [year for year in expected_years if year not in present_years]
        add_years: list[int] = []
        while len(present_years) + len(add_years) < min_year_subsections:
            if missing_years:
                add_years.append(missing_years.pop(0))
                continue
            baseline_year = max(present_years | set(add_years) | {2000})
            add_years.append(baseline_year + 1)
        if add_years:
            lines = ['']
            for year in add_years:
                lines.append(f'### {year}')
                lines.append('- Missing Evidence: no validated evidence surfaced for this year in retrieved context.')
            suffix = '\n'.join(lines)
            answer_parts.append(suffix)
            yield suffix
            applied_degradations.append({
                'step': 'post_stream_year_subsections_enforced',
                'reason': 'required_min_year_subsections_not_met',
                'years_appended': add_years,
            })
    requires_missing_evidence_callout = bool(
        isinstance(output_contract_plan, dict)
        and output_contract_plan.get('requires_missing_evidence_callout') is True
    )
    if requires_missing_evidence_callout:
        full_answer = ''.join(answer_parts).casefold()
        if 'missing evidence:' not in full_answer:
            suffix = '\n\nMissing Evidence: none explicitly identified in the retrieved context.'
            answer_parts.append(suffix)
            yield suffix
            applied_degradations.append({
                'step': 'post_stream_missing_evidence_callout_enforced',
                'reason': 'required_missing_evidence_callout_not_present',
            })
    required_terms: list[str] = []
    enforce_required_terms = False
    required_headings: list[str] = []
    enforce_required_headings = False
    enforce_heading_order = False
    if isinstance(output_contract_plan, dict):
        raw_required_terms = output_contract_plan.get('required_terms')
        if isinstance(raw_required_terms, list):
            required_terms = [
                str(term or '').strip().casefold()
                for term in raw_required_terms
                if str(term or '').strip()
            ]
        raw_required_headings = output_contract_plan.get('required_headings')
        if isinstance(raw_required_headings, list):
            required_headings = [
                str(heading or '').strip()
                for heading in raw_required_headings
                if str(heading or '').strip()
            ]
        enforce_required_terms = bool(output_contract_plan.get('enforce_required_terms') is True)
        enforce_required_headings = bool(output_contract_plan.get('enforce_required_headings') is True)
        enforce_heading_order = bool(output_contract_plan.get('enforce_heading_order') is True)
    if enforce_required_terms and required_terms:
        full_answer = ''.join(answer_parts).casefold()
        missing_terms = [
            term for term in required_terms
            if term not in full_answer
        ]
        if missing_terms:
            suffix = '\n\nRequired Terms: ' + ', '.join(missing_terms) + '.'
            answer_parts.append(suffix)
            yield suffix
            applied_degradations.append({
                'step': 'post_stream_required_terms_enforced',
                'reason': 'required_terms_not_present',
                'missing_terms': missing_terms,
            })
    if enforce_required_headings and required_headings:
        full_answer = ''.join(answer_parts)
        answer_casefold = full_answer.casefold()
        missing_headings = [
            heading for heading in required_headings
            if heading.casefold() not in answer_casefold
        ]
        if missing_headings:
            lines = ['', '']
            for heading in missing_headings:
                lines.append(f'## {heading}')
                lines.append('- Missing Evidence: no validated evidence surfaced for this section in retrieved context.')
            suffix = '\n'.join(lines)
            answer_parts.append(suffix)
            yield suffix
            applied_degradations.append({
                'step': 'post_stream_required_headings_enforced',
                'reason': 'required_headings_not_present',
                'missing_headings': missing_headings,
            })
            full_answer = ''.join(answer_parts)
        if enforce_heading_order:
            last_position = -1
            out_of_order = False
            for heading in required_headings:
                position = _find_section_anchor_position(full_answer, heading)
                if position is None:
                    continue
                if position < last_position:
                    out_of_order = True
                    break
                last_position = position
            if out_of_order:
                ordered_lines = ['', '', '## Ordered Sections (Contract Copy)']
                for heading in required_headings:
                    ordered_lines.append(f'## {heading}')
                    ordered_lines.append('- Missing Evidence: section restated to preserve requested order.')
                suffix = '\n'.join(ordered_lines)
                answer_parts.append(suffix)
                yield suffix
                applied_degradations.append({
                    'step': 'post_stream_heading_order_enforced',
                    'reason': 'section_order_violation_detected',
                    'required_headings': required_headings,
                })
    completion_mode = 'partial' if timeout_reason else 'complete'
    if stream_recovery_reason is not None:
        completion_mode = 'scoped_complete'
    has_remaining_scope = _generation_runtime._has_remaining_scope(
        timeout_reason=timeout_reason,
        stream_recovery_reason=stream_recovery_reason,
        generation_skipped=False,
        applied_degradations=applied_degradations,
    )

    yield (STREAM_SUMMARY_EVENT, StreamExecutionSummary(
        token_count=token_count,
        first_token_ms=first_token_ms,
        total_elapsed_ms=llm_elapsed_ms,
        timeout_reason=timeout_reason,
        stream_recovery_reason=stream_recovery_reason,
        soft_budget_checkpoints_hit=sorted(int(ratio * 100) for ratio in checkpoints_emitted),
        completion_mode=completion_mode,
        has_remaining_scope=has_remaining_scope,
        final_answer=''.join(answer_parts),
    ))
