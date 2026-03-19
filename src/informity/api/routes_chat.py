# ==============================================================================
# Informity AI — Chat API Routes
# Endpoints for RAG-based chat: send messages (SSE streaming), list
# chats, and retrieve chat history.
# ==============================================================================

import asyncio
import difflib
import re
import time
import uuid
from collections.abc import AsyncGenerator
from dataclasses import dataclass, replace

import aiosqlite
import structlog
from fastapi import APIRouter, Depends, HTTPException, Query
from sse_starlette.sse import EventSourceResponse
from structlog.contextvars import get_contextvars

from informity import answer_sanitization
from informity.answer_sanitization import build_display_answer, sanitize_display_answer
from informity.api.schemas import (
    ChatRequest,
    ChatSourceReference,
    ChatStopRequest,
)
from informity.api.security import EndpointGuard
from informity.chat_trace import flush_trace_writer, get_trace_writer
from informity.config import settings
from informity.db.models import ChatMessage, ContinuationPassArtifact
from informity.db.sqlite import (
    delete_chat,
    get_chat,
    get_chat_count,
    get_chat_message_by_id,
    get_chats,
    get_connection,
    get_db,
    get_distinct_categories,
    get_distinct_years,
    get_file_count,
    insert_chat_message,
    insert_continuation_pass_artifact,
    insert_diagnostics_metrics,
    set_chat_title,
)
from informity.diagnostics.grounding_verifier import run_grounding_verifier
from informity.diagnostics.observer import EvalMetrics, detect_issues
from informity.diagnostics.resource_snapshot import build_resource_delta, capture_resource_snapshot
from informity.llm.model_adapter import get_profile
from informity.llm.planner import (
    PLANNING_ELIGIBLE_ROUTES,
    QueryPlan,
    build_corpus_summary,
    build_plan,
)
from informity.llm.query_classifier import classify_query
from informity.llm.rag import answer_question
from informity.llm.rag_runtime import strict_output_contract as _strict_output_contract
from informity.llm.rag_runtime import structured_numeric as _structured_numeric
from informity.utils.json_utils import serialize_api_response

# Trace logging constants
MAX_ANSWER_PREVIEW_LENGTH = 1500  # Maximum length of answer preview in trace logs
DISPLAY_FALLBACK_MESSAGE = answer_sanitization.DISPLAY_FALLBACK_MESSAGE
SSE_PHASE_ORDER = {'chat': 1, 'plan_step': 1, 'token': 2, 'budget': 2, 'timeout': 2, 'sources': 3, 'cleaned': 4, 'error': 4, 'done': 5}
SSE_STATUS_ORDER = {
    'classifying': 1,
    'planning':    2,
    'retrieving':  3,
    'generating':  4,
    'continuing':  5,
    'finalizing':  6,
}
MAX_CHAT_MESSAGE_CHARS = 20000
_VALID_RESPONSE_MODES = {'analysis', 'research'}
_VALID_COMPLETION_MODES = {'complete', 'partial', 'scoped_complete', 'stopped'}
_PERSISTENCE_EXCEPTIONS = (aiosqlite.Error, ValueError, RuntimeError, OSError)
_STREAM_RUNTIME_EXCEPTIONS = (RuntimeError, ValueError, TypeError, OSError, ConnectionError, aiosqlite.Error)
_REFUSAL_PATTERNS = (
    'i cannot',
    "i can't",
    "i'm unable",
    "i don't have",
    'not enough information',
    'cannot answer',
    'unable to',
    "don't contain enough information",
)
_STRICT_NUMERIC_TOKEN_PATTERN = re.compile(r'\$?\d[\d,]*(?:\.\d{1,2})?')
_TABLE_SEPARATOR_PATTERN = re.compile(r'^\s*\|?\s*:?-{3,}(?:\s*\|\s*:?-{3,})+\s*\|?\s*$')
_EMPTY_ORDERED_LIST_ITEM_PATTERN = re.compile(r'^\s*\d+\.\s*$')


# ==============================================================================
# Utility Functions
# ==============================================================================


def build_display_blocks(cleaned_answer: str) -> list[dict[str, str]]:
    # Additive Phase 2 contract: structured display blocks (markdown-compatible text block first).
    if not cleaned_answer:
        return []
    return [{'type': 'text', 'markdown': cleaned_answer}]


def resolve_response_mode(request_mode: str | None) -> str:
    mode = str(request_mode or settings.default_response_mode or 'analysis').strip().lower()
    if mode not in _VALID_RESPONSE_MODES:
        return 'analysis'
    return mode


def enforce_response_mode_supported(response_mode: str) -> None:
    profile = get_profile()
    supported_modes = tuple(str(mode).strip().lower() for mode in getattr(profile, 'supported_modes', ()) if mode)
    if not supported_modes:
        supported_modes = ('analysis',)
    if response_mode in supported_modes:
        return
    raise HTTPException(
        status_code=422,
        detail=(
            f"response_mode '{response_mode}' is not supported by active model "
            f"'{profile.name}'. Supported modes: {', '.join(supported_modes)}."
        ),
    )


def _safe_int(value: object, default: int = 0) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    return default


def _safe_float(value: object, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    return default


def _detect_refusal_pattern(text: str) -> bool:
    lowered = text.lower()
    return any(pattern in lowered for pattern in _REFUSAL_PATTERNS)


def _normalize_numeric_token(raw_value: str) -> str:
    return re.sub(r'[^0-9.\-]', '', raw_value or '')


def _is_year_token(token: str) -> bool:
    return token.isdigit() and len(token) == 4 and 1900 <= int(token) <= 2099


def _is_actionable_numeric_token(raw_token: str, normalized_token: str) -> bool:
    if not normalized_token:
        return False
    digits_only = re.sub(r'[^0-9]', '', normalized_token)
    if len(digits_only) <= 1:
        return False
    if _is_year_token(normalized_token):
        return False
    has_currency_shape = (
        '$' in raw_token
        or ',' in raw_token
        or '.' in raw_token
        or raw_token.strip().startswith('(')
        or raw_token.strip().endswith(')')
    )
    if has_currency_shape:
        return True
    return not (normalized_token.isdigit() and len(normalized_token) < 5)


def _build_support_span(text: str, start_idx: int, end_idx: int, radius: int = 48) -> str:
    source = str(text or '')
    span_start = max(0, start_idx - radius)
    span_end = min(len(source), end_idx + radius)
    return re.sub(r'\s+', ' ', source[span_start:span_end]).strip()


def _build_canonical_numeric_fact_index(
    sources: list[dict[str, object]],
) -> tuple[dict[str, dict[str, object]], list[dict[str, object]]]:
    fact_by_token: dict[str, dict[str, object]] = {}
    facts: list[dict[str, object]] = []
    claim_id = 1
    for source in sources:
        filename = str(source.get('filename') or '').strip()
        preview = str(source.get('chunk_preview') or '')
        if not filename or not preview.strip():
            continue
        for match in _STRICT_NUMERIC_TOKEN_PATTERN.finditer(preview):
            raw = match.group(0)
            normalized = _normalize_numeric_token(raw)
            if not _is_actionable_numeric_token(raw, normalized):
                continue
            if normalized in fact_by_token:
                continue
            fact = {
                'claim_id': f'cf-{claim_id}',
                'metric': 'numeric_claim',
                'value': normalized,
                'unit': 'unknown',
                'period': None,
                'entity': None,
                'source_file': filename,
                'source_page': None,
                'confidence': 1.0,
                'support_span': _build_support_span(preview, match.start(), match.end()),
            }
            claim_id += 1
            fact_by_token[normalized] = fact
            facts.append(fact)
    return fact_by_token, facts


def _summarize_strict_claim_evidence_gate(
    *,
    sources: list[dict[str, object]],
    unsupported_claims: list[object],
) -> dict[str, object]:
    fact_index, canonical_facts = _build_canonical_numeric_fact_index(sources)
    unsupported_tokens = {
        token for token in (
            _normalize_numeric_token(str(item))
            for item in unsupported_claims
            if isinstance(item, (str, int, float))
        )
        if token
    }
    unsupported_tokens_with_facts = sum(1 for token in unsupported_tokens if token in fact_index)
    gate_summary = {
        'canonical_fact_count': len(canonical_facts),
        # The strict gate is diagnostics-only metadata and must not rewrite answer text.
        'replaced_line_count': 0,
        'bound_line_count': 0,
        'unsupported_token_count': len(unsupported_tokens),
        'unsupported_token_with_fact_count': unsupported_tokens_with_facts,
    }
    return gate_summary


def _resolve_completion_state(
    *,
    completion_mode_override: str | None,
    timeout_occurred: bool,
    has_remaining_scope: bool,
) -> tuple[str, bool]:
    default_mode = 'partial' if timeout_occurred else 'complete'
    completion_mode = completion_mode_override or default_mode
    if completion_mode not in _VALID_COMPLETION_MODES:
        completion_mode = default_mode
    resolved_remaining_scope = (
        has_remaining_scope
        or timeout_occurred
        or completion_mode == 'stopped'
    )
    if resolved_remaining_scope and completion_mode == 'complete':
        completion_mode = 'scoped_complete'
    if completion_mode == 'complete':
        resolved_remaining_scope = False
    return completion_mode, resolved_remaining_scope


def _resolve_next_action(
    *,
    stopped_by_user: bool,
    timeout_occurred: bool,
    has_remaining_scope: bool,
    output_contract_check: dict[str, object] | None,
    continuation_resolution_reason: str | None,
) -> tuple[str, str | None]:
    if stopped_by_user:
        return 'regenerate', 'stopped'
    normalized_reason = str(continuation_resolution_reason or '').strip().lower()
    if normalized_reason in {'duplicate_continuation_detected'}:
        return 'regenerate', 'stalled'
    if not has_remaining_scope:
        return 'none', None
    if normalized_reason in {'continuation_pass_budget_exhausted'}:
        return 'continue', 'budget_exhausted'
    if timeout_occurred:
        return 'continue', 'timeout'

    has_content_gap = False
    if isinstance(output_contract_check, dict):
        content_gap_value = output_contract_check.get('has_content_gap')
        if isinstance(content_gap_value, bool):
            has_content_gap = content_gap_value
    if has_content_gap:
        return 'continue', 'unresolved_content'
    return 'none', None


def _enforce_completion_action_consistency(
    *,
    completion_mode: str,
    has_remaining_scope: bool,
    next_action: str,
    next_action_reason: str | None,
) -> tuple[str, bool]:
    # Canonical terminal contract:
    # - next_action=none => complete + no remaining scope
    # - next_action=continue => remaining scope must be true
    # - next_action=regenerate may be terminal (e.g. duplicate continuation stall)
    normalized_completion_mode = completion_mode
    normalized_has_remaining_scope = has_remaining_scope
    if next_action == 'none':
        normalized_has_remaining_scope = False
        if normalized_completion_mode in {'scoped_complete', 'partial'}:
            normalized_completion_mode = 'complete'
        return normalized_completion_mode, normalized_has_remaining_scope
    if next_action == 'regenerate':
        # Duplicate-continuation stalls are terminal: no additional continuation
        # scope should remain once we decide to regenerate.
        if (next_action_reason or '').strip().lower() == 'stalled':
            normalized_has_remaining_scope = False
            if normalized_completion_mode in {'scoped_complete', 'partial'}:
                normalized_completion_mode = 'complete'
        return normalized_completion_mode, normalized_has_remaining_scope
    if not normalized_has_remaining_scope:
        log.warning(
            'chat_action_scope_inconsistency_autofixed',
            completion_mode=completion_mode,
            next_action=next_action,
            next_action_reason=next_action_reason,
        )
        normalized_has_remaining_scope = True
    if next_action == 'continue' and normalized_completion_mode == 'complete':
        normalized_completion_mode = 'scoped_complete'
    return normalized_completion_mode, normalized_has_remaining_scope


_CONTINUATION_DUPLICATE_SIMILARITY_THRESHOLD = 0.9
_CONTINUATION_DUPLICATE_MIN_LENGTH = 240


def _has_contract_requirements(plan: _strict_output_contract.OutputContractPlan) -> bool:
    return bool(
        plan.required_headings
        or plan.enforce_order
        or plan.required_bullet_depth is not None
        or plan.requires_missing_evidence_callout
        or plan.max_words is not None
        or plan.exact_top_level_bullets is not None
        or plan.requires_evidence_grounding
        or plan.requires_not_found_fallback
    )


def _is_auto_continue_contract_eligible(plan: _strict_output_contract.OutputContractPlan) -> bool:
    return bool(
        (plan.required_headings and plan.enforce_order)
        or plan.required_bullet_depth is not None
        or plan.requires_missing_evidence_callout
        or plan.max_words is not None
        or plan.exact_top_level_bullets is not None
        or plan.requires_evidence_grounding
        or plan.requires_not_found_fallback
    )


_CONTINUATION_PHRASES = frozenset({
    'continue', 'continue please', 'please continue', 'go on', 'carry on',
    'resume', 'proceed', 'keep going', 'go ahead', 'next', 'next section',
    'next part', 'more', 'more please', 'the rest', 'rest',
})
_CONTINUATION_PATTERN = re.compile(
    r'^\s*(continue|go on|carry on|resume|proceed|keep going|go ahead)\b',
    re.IGNORECASE,
)


def _is_continuation_request(question: str) -> bool:
    """Lightweight pre-classification continuation check for session binding and history resolution."""
    normalized = ' '.join((question or '').strip().split())
    if not normalized:
        return False
    if normalized.casefold().rstrip('.!?') in _CONTINUATION_PHRASES:
        return True
    return bool(_CONTINUATION_PATTERN.search(normalized))


def _resolve_continuation_anchor_question(*, question: str, history: list[ChatMessage]) -> str:
    normalized_question = (question or '').strip()
    if not _is_continuation_request(normalized_question):
        return normalized_question
    # If the continuation turn itself provides explicit heading contract cues,
    # treat this turn as the anchor to avoid carrying stale heading requirements
    # from the prior prompt.
    if '##' in normalized_question:
        return normalized_question
    for message in reversed(history):
        if message.role != 'user':
            continue
        candidate = str(message.content or '').strip()
        if not candidate:
            continue
        if _is_continuation_request(candidate):
            continue
        return candidate
    return normalized_question


def _count_continuation_chain_depth(history: list[ChatMessage]) -> int:
    """Count consecutive assistant messages with has_remaining_scope=True at the tail of history."""
    depth = 0
    for message in reversed(history):
        if message.role != 'assistant':
            continue
        if message.has_remaining_scope:
            depth += 1
        else:
            break
    return depth


def _extract_missing_headings(output_contract_check: dict[str, object]) -> list[str]:
    return [
        heading.strip()
        for heading in output_contract_check.get('missing_headings', [])
        if isinstance(heading, str) and heading.strip()
    ]


def _build_auto_continue_pass_prompt(
    *,
    auto_continue_prompt: str,
    original_question: str,
    missing_headings: list[str] | None = None,
) -> str:
    lines = [
        auto_continue_prompt.strip(),
        '',
        'Original request:',
        original_question.strip(),
    ]
    unresolved_headings = [heading for heading in (missing_headings or []) if heading.strip()]
    if unresolved_headings:
        lines.extend([
            '',
            'Remaining headings to complete in this pass:',
        ])
        lines.extend(f'- {heading}' for heading in unresolved_headings)
    return '\n'.join(lines).strip()


def _is_duplicate_continuation_pass(previous_answer: str | None, current_answer: str) -> bool:
    if not previous_answer:
        return False
    left = previous_answer.strip()
    right = (current_answer or '').strip()
    if not left or not right:
        return False
    if left == right:
        return True
    if min(len(left), len(right)) < _CONTINUATION_DUPLICATE_MIN_LENGTH:
        return False
    similarity = difflib.SequenceMatcher(a=left, b=right).ratio()
    return similarity >= _CONTINUATION_DUPLICATE_SIMILARITY_THRESHOLD


def _build_continuing_status_message(response_mode: str) -> str:
    mode = str(response_mode or 'analysis').strip().lower()
    if mode == 'analysis':
        return 'Continuing analysis...'
    if mode == 'research':
        return 'Continuing research...'
    return 'Continuing...'


def _enforce_continuation_chat_binding(*, question: str, chat_id: str | None) -> None:
    if chat_id:
        return
    if not _is_continuation_request(question):
        return
    raise HTTPException(
        status_code=409,
        detail='Continuation requests require an existing chat. Use Continue from the same chat thread.',
    )


def _build_contract_closure_prompt(
    *,
    output_contract_check: dict[str, object],
    original_question: str,
) -> str:
    missing_headings = [
        heading
        for heading in output_contract_check.get('missing_headings', [])
        if isinstance(heading, str) and heading.strip()
    ]
    lines = [
        'Close the remaining scope from your last answer using only previously cited evidence.',
        'Do not add preamble or repeat completed sections.',
        'Do not invent or infer unsupported evidence; when evidence is missing, emit "Missing Evidence:" lines.',
    ]
    if missing_headings:
        lines.append('Complete only these remaining headings exactly:')
        for heading in missing_headings:
            lines.append(f'- {heading}')
    else:
        lines.append('Complete only unresolved missing scope from the original request.')
    lines.extend([
        '',
        'Original request:',
        original_question.strip(),
    ])
    return '\n'.join(lines).strip()


def _build_continuation_anchor_retry_prompt(original_question: str) -> str:
    return (
        'Continue the unresolved scope from your last answer.\n'
        'Use only previously cited source documents and keep the same requested output format.\n'
        'Do not restart, do not broaden scope, and do not add clarifying questions.\n\n'
        f'Original request:\n{original_question.strip()}'
    )


def _build_targeted_contract_repair_prompt(
    *,
    plan: _strict_output_contract.OutputContractPlan,
    output_contract_check: dict[str, object],
    original_question: str,
) -> str | None:
    if not _has_contract_requirements(plan):
        return None
    if bool(output_contract_check.get('passed', True)):
        return None

    missing_headings = [
        heading
        for heading in output_contract_check.get('missing_headings', [])
        if isinstance(heading, str) and heading.strip()
    ]
    order_violations = [
        heading
        for heading in output_contract_check.get('order_violations', [])
        if isinstance(heading, str) and heading.strip()
    ]
    bullet_depth_ok = output_contract_check.get('bullet_depth_ok')
    missing_evidence_callout_ok = output_contract_check.get('missing_evidence_callout_ok')
    word_count_ok = output_contract_check.get('word_count_ok')
    top_level_bullet_count_ok = output_contract_check.get('top_level_bullet_count_ok')
    evidence_grounding_ok = output_contract_check.get('evidence_grounding_ok')
    not_found_fallback_ok = output_contract_check.get('not_found_fallback_ok')
    contradiction_placeholder_ok = output_contract_check.get('contradiction_placeholder_ok')
    uncited_delta_numeric_ok = output_contract_check.get('uncited_delta_numeric_ok')
    max_words = output_contract_check.get('max_words')
    exact_top_level_bullets = output_contract_check.get('exact_top_level_bullets')
    failure_reason = str(output_contract_check.get('failure_reason') or '').strip().lower()

    repair_lines = [
        'Repair only the missing/invalid output requirements from your last answer.',
        'Do not add new scope or unrelated sections.',
    ]
    full_rewrite_required = _requires_full_rewrite_for_contract_repair(output_contract_check)
    if full_rewrite_required:
        repair_lines.extend([
            'Rewrite the full response so all global output constraints are satisfied together.',
            'Output one complete corrected answer only.',
        ])
    else:
        repair_lines.extend([
            'Do not rewrite completed valid sections.',
            'Output only the missing or corrected sections.',
        ])
    if failure_reason == 'reasoning_only_output':
        repair_lines.extend([
            'Your previous pass produced reasoning-only output with no visible final answer.',
            'Return a visible user-facing final answer now; do not emit <think> blocks.',
            'Do not return generic fallback/refusal sentences when required output structure is specified.',
        ])
    if missing_headings:
        repair_lines.append('Include these missing headings exactly:')
        for heading in missing_headings:
            repair_lines.append(f'- {heading}')
    if order_violations:
        repair_lines.append('Fix section order for these headings:')
        for heading in order_violations:
            repair_lines.append(f'- {heading}')
    if bullet_depth_ok is False:
        repair_lines.append('Add the required nested bullet depth in the relevant section.')
    if missing_evidence_callout_ok is False:
        repair_lines.append('Add explicit missing-evidence callouts where evidence is absent.')
    if word_count_ok is False and isinstance(max_words, int):
        repair_lines.append(f'Shorten output to <= {max_words} words while preserving existing evidence-grounded claims.')
    if top_level_bullet_count_ok is False and isinstance(exact_top_level_bullets, int):
        repair_lines.append(
            f'Rewrite to exactly {exact_top_level_bullets} top-level bullets in total (no extras).'
        )
    if evidence_grounding_ok is False:
        repair_lines.append(
            'Add explicit evidence metadata to every claim-bearing bullet/list item using canonical '
            '"Evidence: {filename}, page {N}" references.'
        )
        repair_lines.append(
            'Parenthetical page-only shorthand like "(Page 1)" is not sufficient without canonical "Evidence:" metadata.'
        )
        repair_lines.append(
            'For narrative claim paragraphs, include at least one canonical evidence line per paragraph block.'
        )
        repair_lines.append(
            'If evidence is unavailable for a required claim, do not fabricate details; emit "Missing Evidence:" for that claim.'
        )
        missing_previews = output_contract_check.get('evidence_missing_blocks_preview', [])
        if isinstance(missing_previews, list):
            normalized_previews = [
                str(item).strip()
                for item in missing_previews
                if isinstance(item, str) and item.strip()
            ]
            if normalized_previews:
                repair_lines.append('Rewrite these failing claim blocks with canonical evidence metadata or Missing Evidence:')
                for preview in normalized_previews[:8]:
                    repair_lines.append(f'- {preview}')
    if contradiction_placeholder_ok is False:
        repair_lines.append(
            'Under contradictions sections, replace bare "No contradictions found" lines with lines that include '
            'canonical "Evidence:" metadata or explicit "Missing Evidence:".'
        )
        repair_lines.append(
            'Use exact replacement shape when evidence is absent: "Missing Evidence: no verified contradiction record for the requested period."'
        )
    if uncited_delta_numeric_ok is False:
        repair_lines.append(
            'Under Largest Increase/Largest Decrease, every numeric delta line must include canonical "Evidence:" '
            'metadata or explicit "Missing Evidence:".'
        )
    if not_found_fallback_ok is False:
        repair_lines.append(
            'For unsupported required claims, use "Not found" instead of inferred values.'
        )

    if len(repair_lines) <= 3:
        return None
    repair_lines.extend([
        '',
        'Original request:',
        original_question.strip(),
    ])
    return '\n'.join(repair_lines)


def _requires_full_rewrite_for_contract_repair(output_contract_check: dict[str, object]) -> bool:
    # Global constraints cannot be safely fixed via partial append patches.
    return bool(
        output_contract_check.get('word_count_ok') is False
        or output_contract_check.get('top_level_bullet_count_ok') is False
        or output_contract_check.get('evidence_grounding_ok') is False
    )


def _has_unresolved_contract_targets(output_contract_check: dict[str, object]) -> bool:
    missing_headings = output_contract_check.get('missing_headings')
    if isinstance(missing_headings, list) and any(
        isinstance(item, str) and item.strip() for item in missing_headings
    ):
        return True
    order_violations = output_contract_check.get('order_violations')
    if isinstance(order_violations, list) and any(
        isinstance(item, str) and item.strip() for item in order_violations
    ):
        return True
    return any(
        output_contract_check.get(flag_key) is False
        for flag_key in (
            'bullet_depth_ok',
            'missing_evidence_callout_ok',
            'word_count_ok',
            'top_level_bullet_count_ok',
            'evidence_grounding_ok',
            'contradiction_placeholder_ok',
            'uncited_delta_numeric_ok',
            'not_found_fallback_ok',
        )
    )


def _has_continue_worthy_gap(
    *,
    has_remaining_scope_signal: bool,
    output_contract_check: dict[str, object] | None,
) -> bool:
    if isinstance(output_contract_check, dict):
        content_gap = output_contract_check.get('has_content_gap')
        if isinstance(content_gap, bool):
            # Contract-derived content gap is authoritative when present.
            return content_gap
    return has_remaining_scope_signal


def _detect_structural_incomplete_reason(answer: str) -> str | None:
    text = str(answer or '')
    if not text.strip():
        return None
    if text.count('```') % 2 != 0:
        return 'unclosed_code_fence'
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    if not lines:
        return None
    last_line = lines[-1].rstrip()
    has_markdown_table = any(_TABLE_SEPARATOR_PATTERN.match(line) for line in lines)
    if has_markdown_table and last_line.lstrip().startswith('|') and not last_line.rstrip().endswith('|'):
        return 'truncated_markdown_table_row'
    if _EMPTY_ORDERED_LIST_ITEM_PATTERN.match(last_line):
        return 'truncated_markdown_list_item'
    return None


def _mark_structural_output_gap(
    output_contract_check: dict[str, object] | None,
    *,
    answer: str,
) -> dict[str, object]:
    normalized = dict(output_contract_check) if isinstance(output_contract_check, dict) else {}
    structural_reason = _detect_structural_incomplete_reason(answer)
    if not structural_reason:
        return normalized
    normalized['passed'] = False
    normalized['has_content_gap'] = True
    normalized.setdefault('failure_reason', structural_reason)
    normalized['structural_incomplete_reason'] = structural_reason
    return normalized


def _mark_reasoning_only_contract_gap(
    output_contract_check: dict[str, object] | None,
) -> dict[str, object]:
    normalized = dict(output_contract_check) if isinstance(output_contract_check, dict) else {}
    normalized['passed'] = False
    normalized['has_content_gap'] = True
    normalized.setdefault('failure_reason', 'reasoning_only_output')
    return normalized


def _evaluate_grounding_repair_gate(
    *,
    plan: _strict_output_contract.OutputContractPlan,
    grounding_verifier: dict[str, object],
) -> tuple[bool, list[str]]:
    if not isinstance(grounding_verifier, dict):
        return False, []
    if not (
        plan.requires_evidence_grounding
        or plan.requires_not_found_fallback
    ):
        return False, []
    unsupported_claim_count = _safe_int(
        grounding_verifier.get('unsupported_claim_count'),
        default=0,
    )
    evidence_coverage_rate = _safe_float(
        grounding_verifier.get('evidence_coverage_rate'),
        default=0.0,
    )
    not_found_count = _safe_int(
        grounding_verifier.get('not_found_count'),
        default=0,
    )
    reasons: list[str] = []
    if unsupported_claim_count > int(settings.chat_grounding_repair_max_unsupported_claims):
        reasons.append('unsupported_claims_detected')
    if (
        plan.requires_evidence_grounding
        and evidence_coverage_rate < float(settings.chat_grounding_repair_min_coverage_rate)
    ):
        reasons.append('low_evidence_coverage')
    if (
        plan.requires_not_found_fallback
        and not_found_count > int(settings.chat_grounding_repair_max_not_found_count)
    ):
        reasons.append('excessive_not_found_fallback')
    return bool(reasons), reasons


def _build_grounding_repair_prompt(
    *,
    original_question: str,
    grounding_reasons: list[str],
) -> str:
    lines = [
        'Repair your last answer to improve evidence grounding and unsupported-claim safety.',
        'Keep the same scope and output shape; do not add new topics.',
        'Every claim-bearing bullet/paragraph must include canonical "Evidence: {filename}, page {N}" metadata.',
        'If support is unavailable, replace the claim with "Not found" or "Missing Evidence:".',
    ]
    if 'unsupported_claims_detected' in grounding_reasons:
        lines.append('Remove or rewrite unsupported claims that are not present in cited source snippets.')
    if 'low_evidence_coverage' in grounding_reasons:
        lines.append('Increase evidence coverage for all remaining claim-bearing blocks.')
    if 'excessive_not_found_fallback' in grounding_reasons:
        lines.append('Use "Not found" only for truly unsupported required claims, not as a blanket fallback.')
    lines.extend([
        '',
        'Original request:',
        original_question.strip(),
    ])
    return '\n'.join(lines).strip()


def _build_section_progress_payload(
    *,
    plan: _strict_output_contract.OutputContractPlan,
    output_contract_check: dict[str, object],
) -> dict[str, object] | None:
    if not plan.required_headings:
        return None
    missing_normalized = {
        heading.casefold()
        for heading in output_contract_check.get('missing_headings', [])
        if isinstance(heading, str) and heading.strip()
    }
    completed = [
        heading
        for heading in plan.required_headings
        if heading.casefold() not in missing_normalized
    ]
    remaining = [
        heading
        for heading in plan.required_headings
        if heading.casefold() in missing_normalized
    ]
    return {
        'completed': completed,
        'remaining': remaining,
        'total': len(plan.required_headings),
    }


def _resolve_auto_continue_policy() -> tuple[bool, int, str]:
    enabled = bool(settings.chat_auto_continue_enabled)
    default_rounds = max(0, int(settings.chat_auto_continue_default_max_rounds))
    hard_cap = max(0, int(settings.chat_auto_continue_hard_cap))
    max_rounds = min(default_rounds, hard_cap)
    prompt = str(settings.chat_auto_continue_prompt or '').strip()
    if not prompt:
        prompt = (
            'Continue with the remaining sections from your last answer. '
            'Keep the same structure and avoid repeating completed sections.'
        )
    return enabled, max_rounds, prompt


class SseContractTracker:
    # Tracks expected SSE phase progression and reports out-of-order events.
    def __init__(self) -> None:
        self.current_phase = 0

    def update(self, event_name: str) -> bool:
        event_phase = SSE_PHASE_ORDER.get(event_name, 0)
        if event_phase < self.current_phase:
            return False
        self.current_phase = event_phase
        return True


# ==============================================================================
# Logger
# ==============================================================================

log = structlog.get_logger(__name__)

# ==============================================================================
# Router
# ==============================================================================

router = APIRouter(tags=['chat'])
CHAT_GUARD = EndpointGuard(
    name='chat',
    max_in_flight=1,
    max_requests_per_window=30,
    window_seconds=60,
)


@dataclass
class ActiveChatStream:
    chat_id: str
    stop_event: asyncio.Event
    stopped_by_user: bool = False


_ACTIVE_CHAT_STREAMS: dict[str, ActiveChatStream] = {}
_ACTIVE_CHAT_STREAMS_LOCK = asyncio.Lock()


async def _register_active_stream(stream_id: str, chat_id: str, stop_event: asyncio.Event) -> None:
    async with _ACTIVE_CHAT_STREAMS_LOCK:
        _ACTIVE_CHAT_STREAMS[stream_id] = ActiveChatStream(
            chat_id=chat_id,
            stop_event=stop_event,
        )


async def _unregister_active_stream(stream_id: str) -> None:
    async with _ACTIVE_CHAT_STREAMS_LOCK:
        _ACTIVE_CHAT_STREAMS.pop(stream_id, None)


async def _mark_stream_stopped_by_user(stream_id: str, chat_id: str | None) -> bool:
    async with _ACTIVE_CHAT_STREAMS_LOCK:
        stream = _ACTIVE_CHAT_STREAMS.get(stream_id)
        if stream is None:
            return False
        if chat_id and stream.chat_id != chat_id:
            return False
        stream.stopped_by_user = True
        stream.stop_event.set()
        return True


def _is_stream_stopped_by_user(stream_id: str) -> bool:
    stream = _ACTIVE_CHAT_STREAMS.get(stream_id)
    return bool(stream and stream.stopped_by_user)


class UserStopRequestedError(Exception):
    """Raised when the user explicitly requests to stop an in-flight stream."""



# ==============================================================================
# POST /api/chat — send a message and stream the response via SSE
# ==============================================================================

@router.post('/api/chat')
async def chat(
    request: ChatRequest,
    db: aiosqlite.Connection = Depends(get_db),
) -> EventSourceResponse:
    # Validate the incoming message
    message_text = request.message.strip()
    if not message_text:
        raise HTTPException(status_code=400, detail='Message cannot be empty')
    if len(message_text) > MAX_CHAT_MESSAGE_CHARS:
        raise HTTPException(
            status_code=413,
            detail=f'Message too large (max {MAX_CHAT_MESSAGE_CHARS} characters).',
        )
    await CHAT_GUARD.check_rate_limit()
    response_mode = resolve_response_mode(request.response_mode)
    requested_run_id = str(request.run_id or '').strip() or None
    enforce_response_mode_supported(response_mode)
    _enforce_continuation_chat_binding(question=message_text, chat_id=request.chat_id)

    # Resolve chat ID — create a new one if not provided
    chat_id = request.chat_id or str(uuid.uuid4())
    stream_id = str(uuid.uuid4())
    request_id = str(get_contextvars().get('request_id') or '')
    artifact_request_id = request_id or stream_id

    # Fetch chat history (excluding the current message we're about to add)
    history = await get_chat(db, chat_id)

    # Persist the user message
    user_message = ChatMessage(
        chat_id = chat_id,
        role    = 'user',
        content = message_text,
    )
    await insert_chat_message(db, user_message)

    log.info(
        'chat_message_received',
        chat_id          = chat_id,
        run_id           = requested_run_id,
        message_length   = len(message_text),
        history_messages = len(history),
    )

    message_id = str(uuid.uuid4())
    trace_writer = get_trace_writer(chat_id, message_id, run_id=requested_run_id)
    request_resource_snapshot = capture_resource_snapshot()
    if trace_writer is not None:
        trace_writer.record('request', {
            'chat_id':          chat_id,
            'question':         message_text,
            'question_length':  len(message_text),
            'history_messages': len(history),
            'response_mode':    response_mode,
            'resource_snapshot': request_resource_snapshot,
        })

    # Build the SSE event generator
    async def _event_stream() -> AsyncGenerator[dict]:
        async with CHAT_GUARD.slot(check_rate=False):
            start_time = time.time()
            sse_tracker = SseContractTracker()
            stop_event = asyncio.Event()
            await _register_active_stream(stream_id, chat_id, stop_event)

            def _update_sse_phase(event_name: str) -> None:
                if not sse_tracker.update(event_name):
                    log.warning(
                        'chat_sse_out_of_order',
                        chat_id=chat_id,
                        event=event_name,
                        current_phase=sse_tracker.current_phase,
                        event_phase=SSE_PHASE_ORDER.get(event_name, 0),
                    )

            _update_sse_phase('chat')
            yield {
                'event': 'chat',
                'data': serialize_api_response({
                    'chat_id': chat_id,
                    'stream_id': stream_id,
                    'request_id': request_id if request_id else None,
                }),
            }

            answer_parts: list[str] = []
            sources: list[ChatSourceReference] = []
            generation_seconds: float | None = None
            timeout_occurred = False
            timeout_reason: str | None = None
            assistant_message_id: int | None = None
            assistant_message_record: ChatMessage | None = None
            message_persisted = False
            cleaned_answer = ''
            completion_mode_override: str | None = None
            budget_metrics: dict[str, object] = {}
            budget_checkpoints: list[dict[str, object]] = []
            response_mode_used = response_mode
            mode_adjustments_applied: list[dict[str, object]] = []
            grounding_verifier: dict[str, object] = {}
            has_remaining_scope = False
            stopped_by_user = False
            finalized_sources = False
            finalized_cleaned = False
            metrics_query_type = 'unknown'
            metrics_raw_chunks_count = 0
            continuation_passes = 0
            repair_pass_applied = False
            pass_details: list[dict[str, object]] = []
            pending_repair_prompt: str | None = None
            latest_output_contract_check: dict[str, object] = {'passed': True}
            continuation_unresolved_headings: list[str] = []
            continuation_resolution_reason: str | None = None
            continuation_progress_state: str | None = None
            current_status_phase = 0
            current_status_state: str | None = None
            status_transitions: list[dict[str, object]] = []
            resource_end_snapshot: dict[str, object] | None = None
            resource_metrics: dict[str, object] = {}
            exhausted_pass_budget_with_remaining_scope = False
            previous_pass_raw_answer: str | None = None
            previous_pass_cleaned_answer: str | None = None
            active_query_plan: QueryPlan | None = None
            planning_latency_ms: float | None = None

            try:
                def _build_status_event(
                    state: str,
                    *,
                    message: str,
                    pass_index: int | None = None,
                    pass_total: int | None = None,
                    section_progress: dict[str, object] | None = None,
                    allow_same_state: bool = False,
                ) -> dict | None:
                    nonlocal current_status_phase, current_status_state
                    if state == current_status_state and not allow_same_state:
                        return None
                    state_phase = SSE_STATUS_ORDER.get(state)
                    if state_phase is None:
                        return None
                    if state_phase < current_status_phase:
                        log.warning(
                            'chat_status_out_of_order',
                            chat_id=chat_id,
                            status_state=state,
                            current_status_state=current_status_state,
                            current_status_phase=current_status_phase,
                            status_phase=state_phase,
                        )
                        return None
                    current_status_phase = state_phase
                    current_status_state = state
                    payload: dict[str, object] = {
                        'state': state,
                        'message': message,
                    }
                    if pass_index is not None:
                        payload['pass_index'] = pass_index
                    if pass_total is not None:
                        payload['pass_total'] = pass_total
                    if section_progress is not None:
                        payload['section_progress'] = section_progress
                    status_transitions.append({
                        'state': state,
                        'elapsed_seconds': round(time.time() - start_time, 3),
                        'pass_index': pass_index,
                        'pass_total': pass_total,
                        'section_progress_emitted': section_progress is not None,
                    })
                    return {'event': 'status', 'data': serialize_api_response(payload)}

                classifying_status = _build_status_event(
                    'classifying',
                    message='Analyzing your request...',
                )
                if classifying_status is not None:
                    yield classifying_status

                generation_started = False
                source_map: dict[tuple[str, str], ChatSourceReference] = {}
                continuation_request = _is_continuation_request(message_text)
                continuation_anchor_question = _resolve_continuation_anchor_question(
                    question=message_text,
                    history=history,
                )
                format_requirements = _structured_numeric._derive_format_requirements(continuation_anchor_question)
                output_contract_plan = _strict_output_contract._build_output_contract_plan(
                    question=continuation_anchor_question,
                    format_requirements=format_requirements,
                )

                # Pre-classify to gate planning and lock classification for all passes.
                # Planning adds structural value only for multi-section synthesis routes;
                # running it on focused/simple/metadata queries wastes 9-28s with no benefit.
                # On failure, falls through to None so planning runs unconditionally (safe fallback).
                locked_classification = None
                try:
                    locked_classification = await asyncio.to_thread(
                        classify_query, continuation_anchor_question,
                    )
                    log.info(
                        'query_pre_classified',
                        chat_id=chat_id,
                        route_candidate=locked_classification.route_candidate,
                        confidence=locked_classification.confidence,
                    )
                except (RuntimeError, ValueError, TypeError, OSError) as exc:
                    log.warning('pre_classification_failed', chat_id=chat_id, error=str(exc))
                    locked_classification = None

                # Override continuation_request with the authoritative classifier result.
                if locked_classification is not None:
                    continuation_request = locked_classification.is_continuation

                # Continuation chain depth guard.
                # Count consecutive assistant messages with has_remaining_scope=True at the tail of
                # history. When the chain reaches max_continuation_depth, terminate with a short
                # content-covered notice rather than re-entering the RAG pipeline.
                continuation_chain_depth = _count_continuation_chain_depth(history)
                continuation_depth_exceeded = (
                    continuation_request
                    and continuation_chain_depth >= int(settings.max_continuation_depth)
                )
                if continuation_depth_exceeded:
                    log.info(
                        'continuation_chain_depth_limit_reached',
                        chat_id=chat_id,
                        chain_depth=continuation_chain_depth,
                        max_depth=settings.max_continuation_depth,
                    )

                # Planning pass: build a QueryPlan using the active model.
                # If the planner produces answer_sections, override output_contract_plan
                # required_headings and enforce_order with the plan's answer outline.
                # Falls through with heuristic output_contract_plan on any failure.
                # Gated on planning-eligible routes to avoid latency on focused/simple queries.
                if (
                    response_mode in ('research', 'analysis')
                    and (
                        locked_classification is None
                        or locked_classification.route_candidate in PLANNING_ELIGIBLE_ROUTES
                    )
                ):
                    try:
                        _planning_status = _build_status_event('planning', message='Planning response...')
                        if _planning_status is not None:
                            yield _planning_status
                        _corpus_years = await get_distinct_years(db)
                        _corpus_categories = await get_distinct_categories(db)
                        _corpus_file_count = await get_file_count(db)
                        _corpus_summary = build_corpus_summary(
                            _corpus_years, _corpus_categories, _corpus_file_count,
                        )
                        _plan_start = time.time()
                        active_query_plan = await asyncio.to_thread(
                            build_plan, continuation_anchor_question, _corpus_summary,
                        )
                        planning_latency_ms = round((time.time() - _plan_start) * 1000, 2)
                    except (aiosqlite.Error, RuntimeError, ValueError, TypeError, OSError) as exc:
                        log.warning('planning_pass_failed', chat_id=chat_id, error=str(exc))
                        active_query_plan = None

                    if active_query_plan is not None and active_query_plan.answer_sections:
                        plan_headings = tuple(s.heading for s in active_query_plan.answer_sections)
                        output_contract_plan = replace(
                            output_contract_plan,
                            required_headings=plan_headings,
                            enforce_order=True,
                        )
                        log.info(
                            'planner_override_output_contract',
                            chat_id=chat_id,
                            answer_sections_count=len(active_query_plan.answer_sections),
                            planning_latency_ms=planning_latency_ms,
                        )

                if trace_writer is not None and active_query_plan is not None:
                    trace_writer.record('plan', {
                        'answer_sections_count': len(active_query_plan.answer_sections),
                        'steps_requested': len(active_query_plan.steps),
                        'aggregation_mode': active_query_plan.aggregation_mode,
                        'output_shape': active_query_plan.output_shape,
                        'planner_latency_ms': planning_latency_ms,
                    })

                finance_conflict_contract = (
                    isinstance(output_contract_plan.exact_top_level_bullets, int)
                    and output_contract_plan.exact_top_level_bullets > 0
                    and _structured_numeric._is_finance_conflict_prompt(continuation_anchor_question)
                )
                strict_deterministic_mode = _has_contract_requirements(output_contract_plan)
                strict_max_repair_passes = (
                    min(
                        max(1, len(output_contract_plan.required_headings) - 1),
                        max(1, settings.chat_auto_continue_hard_cap - 1),
                    )
                    if output_contract_plan.required_headings
                    else 1
                )
                auto_continue_enabled, max_auto_continue_rounds, auto_continue_prompt = _resolve_auto_continue_policy()
                auto_continue_contract_eligible = _is_auto_continue_contract_eligible(output_contract_plan)
                closure_pass_budget = strict_max_repair_passes if strict_deterministic_mode else (1 if auto_continue_contract_eligible else 0)
                closure_pass_used = False
                continuation_anchor_retry_used = False
                evidence_callout_retry_attempted = False
                base_history = list(history)
                max_total_passes = (
                    1 + strict_max_repair_passes
                    if strict_deterministic_mode
                    else 1 + max_auto_continue_rounds + closure_pass_budget
                )
                strict_reasoning_retry_extension_used = False
                strict_reasoning_retry_extra_passes = 1
                required_headings_last_missing: set[str] | None = None
                section_progress_enabled = (
                    response_mode == 'research'
                    and bool(output_contract_plan.required_headings)
                )
                section_progress_last_completed_count = 0
                pending_repair_requires_overwrite = False
                pass_index = 1
                while pass_index <= max_total_passes:
                    # Depth-exceeded short-circuit: emit a single notice token and exit.
                    # Downstream code (sources/cleaned/persist/done) handles the rest normally.
                    if continuation_depth_exceeded and pass_index == 1:
                        _gen_status = _build_status_event('generating', message='Generating response...')
                        if _gen_status is not None:
                            yield _gen_status
                        notice = 'All available content on this topic has been covered.'
                        _update_sse_phase('token')
                        yield {'event': 'token', 'data': notice}
                        answer_parts.append(notice)
                        generation_seconds = time.time() - start_time
                        has_remaining_scope = False
                        completion_mode_override = 'complete'
                        break

                    pass_question = (
                        _build_auto_continue_pass_prompt(
                            auto_continue_prompt=auto_continue_prompt,
                            original_question=continuation_anchor_question,
                        )
                        if continuation_request and not strict_deterministic_mode
                        else message_text
                    )
                    pass_history = history
                    pass_overwrites_prior_answer = False
                    if pass_index > 1:
                        if pending_repair_prompt:
                            pass_question = pending_repair_prompt
                            pass_overwrites_prior_answer = pending_repair_requires_overwrite
                        else:
                            pass_question = _build_auto_continue_pass_prompt(
                                auto_continue_prompt=auto_continue_prompt,
                                original_question=continuation_anchor_question,
                                missing_headings=_extract_missing_headings(latest_output_contract_check),
                            )
                        assistant_history_content = ''.join(answer_parts).strip()
                        unresolved_headings = _extract_missing_headings(latest_output_contract_check)
                        if active_query_plan is not None and active_query_plan.answer_sections:
                            # Plan-anchored verbatim context: plan anchor + breadcrumbs + last section.
                            # No compaction — the LLM receives the plan scope for remaining sections,
                            # heading-only breadcrumbs for completed sections, and the full verbatim
                            # text of the most recently completed section.
                            _unresolved_set = set(unresolved_headings)
                            _remaining_sections = [
                                s for s in active_query_plan.answer_sections
                                if s.heading in _unresolved_set
                            ]
                            _completed_sections = [
                                s for s in active_query_plan.answer_sections
                                if s.heading not in _unresolved_set
                            ]
                            _ctx_lines: list[str] = []
                            if _remaining_sections:
                                _ctx_lines.append('Sections to complete:')
                                for _sec in _remaining_sections:
                                    _ctx_lines.append(f'{_sec.heading}: {_sec.scope}')
                            if _completed_sections:
                                _ctx_lines.append('')
                                _ctx_lines.append('Completed sections (do not repeat):')
                                for _sec in _completed_sections:
                                    _ctx_lines.append(f'- {_sec.heading}')
                            if previous_pass_cleaned_answer:
                                _verbatim_budget = max(512, get_profile().context_length // 8)
                                _verbatim = previous_pass_cleaned_answer
                                if len(_verbatim) > _verbatim_budget:
                                    _truncated = _verbatim[:_verbatim_budget]
                                    _last_para = _truncated.rfind('\n\n')
                                    _verbatim = (
                                        _truncated[:_last_para].strip()
                                        if _last_para > 0
                                        else _truncated.strip()
                                    )
                                _ctx_lines.extend([
                                    '',
                                    'Most recent completed section (continue from this point):',
                                    _verbatim,
                                ])
                            assistant_history_content = '\n'.join(_ctx_lines).strip()
                        pass_history = [
                            *base_history,
                            user_message,
                            ChatMessage(
                                chat_id=chat_id,
                                role='assistant',
                                content=assistant_history_content,
                                sources=[s.model_dump(mode='json') for s in source_map.values()],
                                completion_mode='scoped_complete',
                                has_remaining_scope=True,
                            ),
                        ]
                        continuing_status = _build_status_event(
                            'continuing',
                            message=_build_continuing_status_message(response_mode_used),
                            pass_index=pass_index,
                            pass_total=max_total_passes,
                        )
                        if continuing_status is not None:
                            yield continuing_status

                    pass_sources: list[ChatSourceReference] = []
                    pass_has_remaining_scope = False
                    pass_completion_mode_override: str | None = None
                    pass_answer_parts: list[str] = []
                    answer_length_before_pass = len(''.join(answer_parts))

                    # Emit "retrieving" status for pass 1 when classification is pre-computed.
                    # Normally this fires from the __classification__ intercept below, but when
                    # locked_classification is pre-set answer_question() skips classification
                    # and never yields ('__classification__', ...).
                    if pass_index == 1 and locked_classification is not None:
                        _retrieval_message = (
                            'Checking document index...'
                            if locked_classification.is_metadata_query
                            else 'Searching for relevant information...'
                        )
                        retrieving_status = _build_status_event('retrieving', message=_retrieval_message)
                        if retrieving_status is not None:
                            yield retrieving_status

                    async for item in answer_question(
                        question=pass_question,
                        chat_id=chat_id,
                        history=pass_history,
                        db=db,
                        trace=trace_writer,
                        response_mode=response_mode,
                        classification=locked_classification,
                        diagnostics_context=(
                            {'query_plan': active_query_plan} if active_query_plan is not None else None
                        ),
                    ):
                        if stop_event.is_set() and _is_stream_stopped_by_user(stream_id):
                            raise UserStopRequestedError

                        if isinstance(item, tuple) and len(item) == 2 and item[0] == '__classification__':
                            locked_classification = item[1]
                            _classification = item[1]
                            _retrieval_message = (
                                'Checking document index...'
                                if _classification.is_metadata_query
                                else 'Searching for relevant information...'
                            )
                            retrieving_status = _build_status_event(
                                'retrieving',
                                message=_retrieval_message,
                            )
                            if retrieving_status is not None:
                                yield retrieving_status
                            continue

                        if isinstance(item, tuple) and len(item) == 2 and item[0] == '__timeout__':
                            timeout_occurred = True
                            timeout_payload = item[1] if isinstance(item[1], dict) else {}
                            timeout_seconds = float(timeout_payload.get('timeout_seconds') or 0.0)
                            timeout_reason = str(timeout_payload.get('reason') or 'unknown_timeout')
                            _update_sse_phase('timeout')
                            yield {
                                'event': 'timeout',
                                'data': serialize_api_response({
                                    'message': f'Response truncated: generation time limit ({int(timeout_seconds) if timeout_seconds else "unknown"}s) reached',
                                    'elapsed_seconds': round(time.time() - start_time, 1),
                                    'timeout_seconds': timeout_seconds if timeout_seconds else None,
                                    'timeout_reason': timeout_reason,
                                }),
                            }
                            continue

                        if isinstance(item, tuple) and len(item) == 2 and item[0] == '__budget_checkpoint__':
                            checkpoint_payload = item[1] if isinstance(item[1], dict) else {}
                            budget_checkpoints.append(checkpoint_payload)
                            _update_sse_phase('budget')
                            yield {
                                'event': 'budget',
                                'data': serialize_api_response(checkpoint_payload),
                            }
                            continue

                        if isinstance(item, tuple) and len(item) == 2 and item[0] == '__plan_step__':
                            step_payload = item[1] if isinstance(item[1], dict) else {}
                            _update_sse_phase('plan_step')
                            yield {
                                'event': 'plan_step',
                                'data': serialize_api_response(step_payload),
                            }
                            continue

                        if isinstance(item, tuple) and len(item) == 2 and item[0] == '__metrics__':
                            metrics_payload = item[1] if isinstance(item[1], dict) else {}
                            budget_metrics = metrics_payload
                            metrics_query_value = metrics_payload.get('query_type')
                            if isinstance(metrics_query_value, str) and metrics_query_value.strip():
                                metrics_query_type = metrics_query_value.strip().lower()
                            metrics_raw_chunks_count = _safe_int(
                                metrics_payload.get('raw_chunks_count'),
                                default=metrics_raw_chunks_count,
                            )
                            mode_value = metrics_payload.get('response_mode_used')
                            if isinstance(mode_value, str) and mode_value in {'analysis', 'research'}:
                                response_mode_used = mode_value
                            adjustments_value = metrics_payload.get('mode_adjustments_applied')
                            if isinstance(adjustments_value, list):
                                mode_adjustments_applied = [
                                    value for value in adjustments_value if isinstance(value, dict)
                                ]
                            remaining_scope_value = metrics_payload.get('has_remaining_scope')
                            if isinstance(remaining_scope_value, bool):
                                pass_has_remaining_scope = remaining_scope_value
                                has_remaining_scope = remaining_scope_value
                            suggested_mode = metrics_payload.get('suggested_completion_mode')
                            if isinstance(suggested_mode, str):
                                pass_completion_mode_override = suggested_mode
                                completion_mode_override = suggested_mode
                            continue

                        if isinstance(item, str):
                            if not generation_started:
                                generation_started = True
                                generating_status = _build_status_event(
                                    'generating',
                                    message='Generating response...',
                                )
                                if generating_status is not None:
                                    yield generating_status
                            answer_parts.append(item)
                            pass_answer_parts.append(item)
                            _update_sse_phase('token')
                            yield {'event': 'token', 'data': item}
                        elif isinstance(item, list):
                            pass_sources = item

                    for source in pass_sources:
                        source_key = (source.path, source.filename)
                        source_map[source_key] = source

                    pass_raw_answer = ''.join(pass_answer_parts).strip()
                    pass_cleaned_answer = sanitize_display_answer(pass_raw_answer) if pass_raw_answer else ''
                    pass_reasoning_only_output = bool(pass_raw_answer) and not pass_cleaned_answer and (
                        '<think>' in pass_raw_answer.lower() or '<<think>>' in pass_raw_answer.lower()
                    )
                    force_overwrite_finance_conflict = (
                        pass_index > 1
                        and finance_conflict_contract
                        and bool(pass_raw_answer)
                    )
                    if force_overwrite_finance_conflict and not pass_overwrites_prior_answer:
                        log.info(
                            'chat_finance_conflict_continuation_overwrite_applied',
                            chat_id=chat_id,
                            pass_index=pass_index,
                        )
                    if (pass_overwrites_prior_answer or force_overwrite_finance_conflict) and pass_raw_answer:
                        combined_answer = pass_raw_answer
                    else:
                        combined_answer = ''.join(answer_parts).strip()
                    if _has_contract_requirements(output_contract_plan):
                        latest_output_contract_check = _strict_output_contract._evaluate_output_contract(
                            answer=combined_answer,
                            plan=output_contract_plan,
                        )
                        if pass_reasoning_only_output:
                            latest_output_contract_check = _mark_reasoning_only_contract_gap(
                                latest_output_contract_check,
                            )
                            if strict_deterministic_mode and not strict_reasoning_retry_extension_used:
                                max_total_passes += strict_reasoning_retry_extra_passes
                                strict_reasoning_retry_extension_used = True
                                log.info(
                                    'chat_strict_reasoning_retry_budget_extended',
                                    chat_id=chat_id,
                                    pass_index=pass_index,
                                    new_max_total_passes=max_total_passes,
                                )
                            log.warning(
                                'chat_pass_reasoning_only_output_detected',
                                chat_id=chat_id,
                                pass_index=pass_index,
                            )
                    latest_output_contract_check = _mark_structural_output_gap(
                        latest_output_contract_check,
                        answer=pass_cleaned_answer or combined_answer,
                    )
                    if section_progress_enabled:
                        section_progress_payload = _build_section_progress_payload(
                            plan=output_contract_plan,
                            output_contract_check=latest_output_contract_check,
                        )
                        if section_progress_payload is not None:
                            completed_headings = section_progress_payload.get('completed')
                            completed_count = len(completed_headings) if isinstance(completed_headings, list) else 0
                            if completed_count > section_progress_last_completed_count:
                                section_progress_last_completed_count = completed_count
                                section_progress_status = _build_status_event(
                                    'continuing' if pass_index > 1 else 'generating',
                                    message='Still processing, almost there...' if pass_index > 1 else 'Generating response...',
                                    pass_index=pass_index,
                                    pass_total=max_total_passes,
                                    section_progress=section_progress_payload,
                                    allow_same_state=True,
                                )
                                if section_progress_status is not None:
                                    yield section_progress_status
                    repair_prompt = _build_targeted_contract_repair_prompt(
                        plan=output_contract_plan,
                        output_contract_check=latest_output_contract_check,
                        original_question=continuation_anchor_question,
                    )

                    if (pass_overwrites_prior_answer or force_overwrite_finance_conflict) and pass_raw_answer:
                        answer_parts = [pass_raw_answer]
                        added_answer_chars = len(pass_raw_answer)
                    else:
                        added_answer_chars = len(''.join(answer_parts)) - answer_length_before_pass
                    if pass_completion_mode_override is not None:
                        completion_mode_override = pass_completion_mode_override

                    pass_refusal_detected = _detect_refusal_pattern(pass_cleaned_answer or pass_raw_answer)
                    pass_sources_exhausted = pass_refusal_detected and len(pass_sources) == 0
                    pass_grounding_verifier = run_grounding_verifier(
                        question=continuation_anchor_question,
                        answer=combined_answer,
                        sources=[s.model_dump(mode='json') for s in source_map.values()],
                    )
                    weak_grounding_detected, weak_grounding_reasons = _evaluate_grounding_repair_gate(
                        plan=output_contract_plan,
                        grounding_verifier=pass_grounding_verifier,
                    )
                    grounding_repair_forced = False
                    if weak_grounding_detected and not pass_sources_exhausted and pass_index < max_total_passes:
                        repair_prompt = _build_grounding_repair_prompt(
                            original_question=continuation_anchor_question,
                            grounding_reasons=weak_grounding_reasons,
                        )
                        normalized_output_contract = (
                            dict(latest_output_contract_check)
                            if isinstance(latest_output_contract_check, dict)
                            else {}
                        )
                        normalized_output_contract['passed'] = False
                        normalized_output_contract['evidence_grounding_ok'] = False
                        normalized_output_contract['failure_reason'] = 'weak_evidence_grounding'
                        normalized_output_contract['has_content_gap'] = True
                        normalized_output_contract['grounding_verifier'] = pass_grounding_verifier
                        normalized_output_contract['grounding_repair_reasons'] = weak_grounding_reasons
                        latest_output_contract_check = normalized_output_contract
                        has_remaining_scope = True
                        grounding_repair_forced = True
                    if (
                        bool(repair_prompt)
                        and (
                            latest_output_contract_check.get('missing_evidence_callout_ok') is False
                            or latest_output_contract_check.get('evidence_grounding_ok') is False
                            or grounding_repair_forced
                        )
                    ):
                        evidence_callout_retry_attempted = True
                    pass_continue_worthy_gap = _has_continue_worthy_gap(
                        has_remaining_scope_signal=pass_has_remaining_scope,
                        output_contract_check=latest_output_contract_check,
                    )
                    pass_detail = {
                        'pass_index': pass_index,
                        'is_continuation': pass_index > 1,
                        'raw_answer_length': len(pass_raw_answer),
                        'cleaned_answer_length': len(pass_cleaned_answer),
                        'reasoning_only_output_detected': pass_reasoning_only_output,
                        'sources_count': len(pass_sources),
                        'pass_requires_more_work': pass_continue_worthy_gap,
                        'raw_has_remaining_scope_signal': pass_has_remaining_scope,
                        'contract_repair_requested': bool(repair_prompt),
                        'grounding_repair_forced': grounding_repair_forced,
                        'closure_pass_used': closure_pass_used,
                        'continuation_anchor_retry_used': continuation_anchor_retry_used,
                        'evidence_callout_retry_attempted': evidence_callout_retry_attempted,
                    }
                    pass_details.append(pass_detail)
                    pass_artifact_has_remaining_scope = pass_continue_worthy_gap
                    pass_completion_mode = pass_completion_mode_override
                    if not isinstance(pass_completion_mode, str) or pass_completion_mode not in _VALID_COMPLETION_MODES:
                        pass_completion_mode = 'scoped_complete' if pass_continue_worthy_gap else 'complete'
                    if pass_continue_worthy_gap and pass_completion_mode == 'complete':
                        pass_completion_mode = 'scoped_complete'
                    elif (not pass_continue_worthy_gap) and pass_completion_mode == 'scoped_complete':
                        pass_completion_mode = 'complete'
                    pass_artifact_has_remaining_scope = pass_completion_mode in {'partial', 'scoped_complete', 'stopped'}
                    pass_next_action_reason: str | None = None
                    failure_reason = latest_output_contract_check.get('failure_reason')
                    if isinstance(failure_reason, str) and failure_reason.strip():
                        pass_next_action_reason = failure_reason.strip()
                    elif timeout_occurred and timeout_reason:
                        pass_next_action_reason = timeout_reason
                    pass_sources_payload = [source.model_dump(mode='json') for source in pass_sources]
                    pass_stitch_mode = (
                        'overwrite'
                        if (pass_overwrites_prior_answer or force_overwrite_finance_conflict)
                        else 'append'
                    )
                    pass_artifact = ContinuationPassArtifact(
                        chat_id=chat_id,
                        request_id=artifact_request_id,
                        pass_index=pass_index,
                        response_mode_used=response_mode_used,
                        stitch_mode=pass_stitch_mode,
                        raw_answer=pass_raw_answer,
                        cleaned_answer=pass_cleaned_answer,
                        has_remaining_scope=pass_artifact_has_remaining_scope,
                        completion_mode=pass_completion_mode,
                        next_action_reason=pass_next_action_reason,
                        sources=pass_sources_payload,
                        pass_details=pass_detail,
                        status_transitions=list(status_transitions),
                    )
                    try:
                        await insert_continuation_pass_artifact(db, pass_artifact)
                    except _PERSISTENCE_EXCEPTIONS as artifact_error:
                        log.warning(
                            'chat_pass_artifact_persist_failed',
                            chat_id=chat_id,
                            request_id=artifact_request_id,
                            pass_index=pass_index,
                            error=str(artifact_error),
                        )
                    pass_has_unresolved_targets = (
                        pass_continue_worthy_gap
                    )
                    pass_duplicate_of_previous = (
                        pass_index > 1
                        and _is_duplicate_continuation_pass(previous_pass_raw_answer, pass_raw_answer)
                    )
                    if pass_duplicate_of_previous and pass_has_unresolved_targets:
                        normalized_output_contract = (
                            dict(latest_output_contract_check)
                            if isinstance(latest_output_contract_check, dict)
                            else {}
                        )
                        normalized_output_contract['passed'] = False
                        normalized_output_contract['duplicate_continuation_detected'] = True
                        normalized_output_contract['failure_reason'] = 'duplicate_continuation_detected'
                        normalized_output_contract['has_content_gap'] = True
                        latest_output_contract_check = normalized_output_contract
                        continuation_resolution_reason = 'duplicate_continuation_detected'
                        continuation_progress_state = 'stalled'
                        can_retry_with_new_strategy = (
                            (not continuation_anchor_retry_used)
                            and pass_index < max_total_passes
                            and not pass_sources_exhausted
                        )
                        if can_retry_with_new_strategy:
                            previous_pass_raw_answer = pass_raw_answer
                            pending_repair_prompt = _build_continuation_anchor_retry_prompt(continuation_anchor_question)
                            pending_repair_requires_overwrite = finance_conflict_contract
                            continuation_anchor_retry_used = True
                            continuation_passes += 1
                            pass_index += 1
                            continue
                        completion_mode_override = 'scoped_complete'
                        has_remaining_scope = False
                        break
                    force_contract_closure_pass = (
                        (not strict_deterministic_mode)
                        and auto_continue_contract_eligible
                        and not closure_pass_used
                        and pass_has_unresolved_targets
                        and not pass_sources_exhausted
                    )
                    force_continuation_anchor_retry = (
                        (not strict_deterministic_mode)
                        and continuation_request
                        and not continuation_anchor_retry_used
                        and (pass_has_unresolved_targets or pass_refusal_detected)
                        and not pass_sources_exhausted
                    )
                    if strict_deterministic_mode:
                        can_auto_continue = (
                            pass_index < max_total_passes
                            and bool(repair_prompt)
                            and not pass_sources_exhausted
                        )
                    else:
                        can_auto_continue = (
                            pass_index < max_total_passes
                            and pass_has_unresolved_targets
                            and not pass_sources_exhausted
                            and (
                                (
                                    auto_continue_enabled
                                    and auto_continue_contract_eligible
                                    and pass_has_unresolved_targets
                                )
                                or force_contract_closure_pass
                                or force_continuation_anchor_retry
                            )
                        )
                    if not can_auto_continue:
                        if pass_has_unresolved_targets and pass_index >= max_total_passes:
                            exhausted_pass_budget_with_remaining_scope = True
                        break
                    if added_answer_chars <= 0:
                        break

                    if output_contract_plan.required_headings:
                        missing_headings = {
                            heading.casefold()
                            for heading in latest_output_contract_check.get('missing_headings', [])
                            if isinstance(heading, str)
                        }
                        if required_headings_last_missing is not None and missing_headings >= required_headings_last_missing:
                            break
                        required_headings_last_missing = missing_headings
                        if not missing_headings and not repair_prompt and not pass_continue_worthy_gap:
                            break

                    previous_pass_raw_answer = pass_raw_answer
                    previous_pass_cleaned_answer = pass_cleaned_answer
                    pending_repair_prompt = repair_prompt
                    pending_repair_requires_overwrite = False
                    if force_continuation_anchor_retry:
                        pending_repair_prompt = _build_continuation_anchor_retry_prompt(continuation_anchor_question)
                        continuation_anchor_retry_used = True
                    elif force_contract_closure_pass and not pending_repair_prompt:
                        pending_repair_prompt = _build_contract_closure_prompt(
                            output_contract_check=latest_output_contract_check,
                            original_question=continuation_anchor_question,
                        )
                    elif pending_repair_prompt:
                        pending_repair_requires_overwrite = _requires_full_rewrite_for_contract_repair(
                            latest_output_contract_check
                        )
                    if force_contract_closure_pass:
                        closure_pass_used = True
                    if repair_prompt:
                        repair_pass_applied = True

                    continuation_passes += 1
                    pass_index += 1
                    continue

                sources = list(source_map.values())

                finalizing_status = _build_status_event(
                    'finalizing',
                    message='Finalizing answer...',
                )
                if finalizing_status is not None:
                    yield finalizing_status

                source_dicts = [s.model_dump(mode='json') for s in sources]
                _update_sse_phase('sources')
                yield {'event': 'sources', 'data': serialize_api_response(source_dicts)}
                finalized_sources = True

                full_answer = ''.join(answer_parts).strip()
                if not full_answer:
                    full_answer = 'I could not find enough information to answer your question.'
                    log.warning('chat_empty_after_cleaning', chat_id=chat_id)
                generation_seconds = time.time() - start_time
                cleaned_answer, reasoning_only_output = build_display_answer(full_answer)
                if _has_contract_requirements(output_contract_plan):
                    latest_output_contract_check = _strict_output_contract._evaluate_output_contract(
                        answer=cleaned_answer,
                        plan=output_contract_plan,
                    )
                    if reasoning_only_output:
                        latest_output_contract_check = _mark_reasoning_only_contract_gap(
                            latest_output_contract_check,
                        )
                    contract_passed = bool(latest_output_contract_check.get('passed', True))
                    if not contract_passed:
                        contract_has_content_gap = _has_continue_worthy_gap(
                            has_remaining_scope_signal=has_remaining_scope,
                            output_contract_check=latest_output_contract_check,
                        )
                        if contract_has_content_gap:
                            log.info(
                                'contract_content_gap_triggers_continuation',
                                source='has_content_gap',
                                missing_headings=latest_output_contract_check.get('missing_headings', []),
                            )
                            completion_mode_override = 'scoped_complete'
                            has_remaining_scope = True
                        elif not timeout_occurred and completion_mode_override != 'stopped':
                            has_remaining_scope = False
                            if completion_mode_override == 'scoped_complete':
                                completion_mode_override = None
                        if continuation_request:
                            continuation_unresolved_headings = [
                                heading
                                for heading in latest_output_contract_check.get('missing_headings', [])
                                if isinstance(heading, str) and heading.strip()
                            ]
                    elif not timeout_occurred and completion_mode_override != 'stopped':
                        has_remaining_scope = False
                        # Prevent stale scoped-complete overrides from re-deriving remaining scope.
                        if completion_mode_override == 'scoped_complete':
                            completion_mode_override = None
                latest_output_contract_check = _mark_structural_output_gap(
                    latest_output_contract_check,
                    answer=cleaned_answer or full_answer,
                )
                structural_incomplete_reason = latest_output_contract_check.get('structural_incomplete_reason')
                if (
                    isinstance(structural_incomplete_reason, str)
                    and structural_incomplete_reason.strip()
                    and completion_mode_override != 'stopped'
                ):
                    log.info(
                        'structural_gap_triggers_continuation',
                        source='structural_incomplete_reason',
                        reason=str(structural_incomplete_reason).strip(),
                    )
                    has_remaining_scope = True
                    completion_mode_override = 'scoped_complete'
                    if continuation_resolution_reason is None:
                        continuation_resolution_reason = structural_incomplete_reason.strip()
                # Structural-first continuation closure: if required headings are all present,
                # treat the continuation as complete unless timeout/stop explicitly says otherwise.
                if (
                    continuation_request
                    and output_contract_plan.required_headings
                    and not timeout_occurred
                    and completion_mode_override != 'stopped'
                ):
                    missing_headings = latest_output_contract_check.get('missing_headings', [])
                    structural_incomplete_reason = latest_output_contract_check.get('structural_incomplete_reason')
                    if isinstance(missing_headings, list) and not any(
                        isinstance(item, str) and item.strip() for item in missing_headings
                    ) and not (
                        isinstance(structural_incomplete_reason, str) and structural_incomplete_reason.strip()
                    ):
                        has_remaining_scope = False
                        if completion_mode_override == 'scoped_complete':
                            completion_mode_override = None
                if exhausted_pass_budget_with_remaining_scope and has_remaining_scope:
                    normalized_output_contract = (
                        dict(latest_output_contract_check)
                        if isinstance(latest_output_contract_check, dict)
                        else {}
                    )
                    normalized_output_contract['passed'] = False
                    normalized_output_contract['continuation_pass_budget_exhausted'] = True
                    normalized_output_contract.setdefault(
                        'failure_reason',
                        'continuation_pass_budget_exhausted',
                    )
                    normalized_output_contract['has_content_gap'] = True
                    latest_output_contract_check = normalized_output_contract
                    completion_mode_override = 'scoped_complete'
                    continuation_resolution_reason = 'continuation_pass_budget_exhausted'
                    continuation_progress_state = 'budget_exhausted'

                grounding_verifier = run_grounding_verifier(
                    question=continuation_anchor_question,
                    answer=cleaned_answer,
                    sources=source_dicts,
                )
                if strict_deterministic_mode:
                    gate_summary = _summarize_strict_claim_evidence_gate(
                        sources=source_dicts,
                        unsupported_claims=grounding_verifier.get('unsupported_claims', []),
                    )
                    latest_output_contract_check['strict_claim_evidence_gate'] = gate_summary
                    log.info(
                        'strict_claim_evidence_gate_observed',
                        chat_id=chat_id,
                        replaced_line_count=_safe_int(gate_summary.get('replaced_line_count')),
                        bound_line_count=_safe_int(gate_summary.get('bound_line_count')),
                        canonical_fact_count=_safe_int(gate_summary.get('canonical_fact_count')),
                        unsupported_token_count=_safe_int(gate_summary.get('unsupported_token_count')),
                    )
                latest_output_contract_check['grounding_verifier'] = grounding_verifier
                enforce_grounding_failure = requested_run_id is not None
                if enforce_grounding_failure and grounding_verifier.get('required') and not grounding_verifier.get('passed', True):
                    latest_output_contract_check['passed'] = False
                    latest_output_contract_check['evidence_grounding_ok'] = False
                    latest_output_contract_check['failure_reason'] = 'grounding_verifier_failed'
                    completion_mode_override = 'scoped_complete'
                    has_remaining_scope = True
                budget_metrics['grounding_verifier'] = grounding_verifier
                budget_metrics['unsupported_claim_count'] = int(grounding_verifier.get('unsupported_claim_count', 0) or 0)
                budget_metrics['evidence_coverage_rate'] = float(grounding_verifier.get('evidence_coverage_rate', 0.0) or 0.0)
                budget_metrics['not_found_count'] = int(grounding_verifier.get('not_found_count', 0) or 0)
                query_type = 'unknown'
                if trace_writer is not None and hasattr(trace_writer, 'get_sections'):
                    sections = trace_writer.get_sections()
                    query_type = str(sections.get('intent', {}).get('query_type', 'unknown'))

                if reasoning_only_output:
                    log.warning(
                        'chat_reasoning_only_output_detected',
                        chat_id=chat_id,
                        query_type=query_type,
                        answer_length=len(full_answer),
                        sources_count=len(source_dicts),
                    )

                if metrics_query_type == 'unknown':
                    metrics_query_type = query_type.strip().lower() if query_type else 'unknown'
                if continuation_passes > 0 and continuation_progress_state is None:
                    continuation_progress_state = 'progressed' if not has_remaining_scope else 'budget_exhausted'
                if continuation_passes > 0 and continuation_resolution_reason is None:
                    failure_reason = latest_output_contract_check.get('failure_reason')
                    if isinstance(failure_reason, str) and failure_reason.strip():
                        continuation_resolution_reason = failure_reason.strip()

                _update_sse_phase('cleaned')
                yield {'event': 'cleaned', 'data': cleaned_answer}
                finalized_cleaned = True

                message_completion_mode, message_has_remaining_scope = _resolve_completion_state(
                    completion_mode_override=completion_mode_override,
                    timeout_occurred=timeout_occurred,
                    has_remaining_scope=has_remaining_scope,
                )
                message_next_action, message_next_action_reason = _resolve_next_action(
                    stopped_by_user=False,
                    timeout_occurred=timeout_occurred,
                    has_remaining_scope=message_has_remaining_scope,
                    output_contract_check=latest_output_contract_check,
                    continuation_resolution_reason=continuation_resolution_reason,
                )
                message_completion_mode, message_has_remaining_scope = _enforce_completion_action_consistency(
                    completion_mode=message_completion_mode,
                    has_remaining_scope=message_has_remaining_scope,
                    next_action=message_next_action,
                    next_action_reason=message_next_action_reason,
                )
                assistant_message = ChatMessage(
                    chat_id=chat_id,
                    role='assistant',
                    content=full_answer,
                    sources=source_dicts,
                    generation_seconds=generation_seconds,
                    response_mode_used=response_mode_used,
                    completion_mode=message_completion_mode,
                    has_remaining_scope=message_has_remaining_scope,
                    next_action=message_next_action,
                    next_action_reason=message_next_action_reason,
                )
                persist_db = await get_connection()
                try:
                    assistant_message = await insert_chat_message(persist_db, assistant_message)
                    assistant_message_record = assistant_message
                    assistant_message_id = assistant_message.id
                    message_persisted = assistant_message_id is not None
                except _PERSISTENCE_EXCEPTIONS as persist_error:
                    log.warning('chat_response_persist_retry', chat_id=chat_id, error=str(persist_error))
                    assistant_message = await insert_chat_message(db, assistant_message)
                    assistant_message_record = assistant_message
                    assistant_message_id = assistant_message.id
                    message_persisted = assistant_message_id is not None
                finally:
                    await persist_db.close()

                if trace_writer is not None:
                    resource_end_snapshot = capture_resource_snapshot()
                    resource_metrics = {
                        'before': request_resource_snapshot,
                        'after': resource_end_snapshot,
                        'delta': build_resource_delta(before=request_resource_snapshot, after=resource_end_snapshot),
                    }
                    trace_writer.record('response', {
                        'answer_length': len(full_answer),
                        'display_answer_length': len(cleaned_answer),
                        'answer_preview': cleaned_answer[:MAX_ANSWER_PREVIEW_LENGTH] if cleaned_answer else '',
                        'display_answer_preview': cleaned_answer[:MAX_ANSWER_PREVIEW_LENGTH] if cleaned_answer else '',
                        'raw_answer_preview': full_answer[:MAX_ANSWER_PREVIEW_LENGTH] if full_answer else '',
                        'sources_count': len(sources),
                        'sources': source_dicts,
                        'continuation_passes': continuation_passes,
                        'contract_repair_pass_applied': repair_pass_applied,
                        'output_contract_check': latest_output_contract_check,
                        'grounding_verifier': grounding_verifier,
                        'pass_details': pass_details,
                        'status_transitions': status_transitions,
                        'resource_metrics': resource_metrics,
                    })
                    await flush_trace_writer(trace_writer)

            except UserStopRequestedError:
                stopped_by_user = True
                generation_seconds = time.time() - start_time
                partial_answer = ''.join(answer_parts).strip()
                partial_sources = [s.model_dump(mode='json') for s in sources] if sources else []
                has_remaining_scope = True
                completion_mode_override = 'stopped'
                cleaned_answer = sanitize_display_answer(partial_answer) if partial_answer else ''

                finalizing_status = _build_status_event(
                    'finalizing',
                    message='Finalizing answer...',
                )
                if finalizing_status is not None:
                    yield finalizing_status

                if not finalized_sources:
                    _update_sse_phase('sources')
                    yield {'event': 'sources', 'data': serialize_api_response(partial_sources)}
                    finalized_sources = True
                if not finalized_cleaned:
                    _update_sse_phase('cleaned')
                    yield {'event': 'cleaned', 'data': cleaned_answer}
                    finalized_cleaned = True

                assistant_message = ChatMessage(
                    chat_id=chat_id,
                    role='assistant',
                    content=partial_answer,
                    sources=partial_sources,
                    generation_seconds=generation_seconds,
                    response_mode_used=response_mode_used,
                    completion_mode='stopped',
                    stopped_by_user=True,
                    has_remaining_scope=True,
                    next_action='regenerate',
                    next_action_reason='stopped',
                )
                try:
                    persist_db = await get_connection()
                    try:
                        assistant_message = await insert_chat_message(persist_db, assistant_message)
                        assistant_message_record = assistant_message
                        assistant_message_id = assistant_message.id
                        message_persisted = assistant_message_id is not None
                    finally:
                        await persist_db.close()
                except _PERSISTENCE_EXCEPTIONS as persist_err:
                    log.warning('chat_stop_persist_failed', chat_id=chat_id, error=str(persist_err))
                if trace_writer is not None:
                    resource_end_snapshot = capture_resource_snapshot()
                    resource_metrics = {
                        'before': request_resource_snapshot,
                        'after': resource_end_snapshot,
                        'delta': build_resource_delta(before=request_resource_snapshot, after=resource_end_snapshot),
                    }
                    trace_writer.record('response_cancelled', {
                        'generation_seconds': generation_seconds,
                        'tokens_generated': len(answer_parts),
                        'stopped_by_user': True,
                        'resource_metrics': resource_metrics,
                    })
                    await flush_trace_writer(trace_writer)

            except asyncio.CancelledError:
                generation_seconds = time.time() - start_time
                partial_answer = ''.join(answer_parts).strip()
                stream_stopped_by_user = _is_stream_stopped_by_user(stream_id)
                if partial_answer or stream_stopped_by_user:
                    cancelled_next_action, cancelled_next_action_reason = _resolve_next_action(
                        stopped_by_user=stream_stopped_by_user,
                        timeout_occurred=False,
                        has_remaining_scope=stream_stopped_by_user,
                        output_contract_check=None,
                        continuation_resolution_reason=None,
                    )
                    assistant_message = ChatMessage(
                        chat_id=chat_id,
                        role='assistant',
                        content=partial_answer,
                        sources=[s.model_dump(mode='json') for s in sources] if sources else [],
                        generation_seconds=generation_seconds,
                        response_mode_used=response_mode_used,
                        completion_mode='stopped' if stream_stopped_by_user else 'partial',
                        stopped_by_user=stream_stopped_by_user,
                        has_remaining_scope=stream_stopped_by_user,
                        next_action=cancelled_next_action,
                        next_action_reason=cancelled_next_action_reason,
                    )
                    try:
                        persist_db = await get_connection()
                        try:
                            assistant_message_record = await insert_chat_message(persist_db, assistant_message)
                        finally:
                            await persist_db.close()
                    except _PERSISTENCE_EXCEPTIONS as persist_err:
                        log.warning('chat_partial_persist_failed', chat_id=chat_id, error=str(persist_err))
                if trace_writer is not None:
                    resource_end_snapshot = capture_resource_snapshot()
                    resource_metrics = {
                        'before': request_resource_snapshot,
                        'after': resource_end_snapshot,
                        'delta': build_resource_delta(before=request_resource_snapshot, after=resource_end_snapshot),
                    }
                    trace_writer.record('response_cancelled', {
                        'generation_seconds': generation_seconds,
                        'tokens_generated': len(answer_parts),
                        'stopped_by_user': stream_stopped_by_user,
                        'resource_metrics': resource_metrics,
                    })
                    await flush_trace_writer(trace_writer)
                await _unregister_active_stream(stream_id)
                raise

            except _STREAM_RUNTIME_EXCEPTIONS as exc:
                generation_seconds = time.time() - start_time
                log.error(
                    'chat_stream_error',
                    chat_id=chat_id,
                    error=str(exc),
                    generation_seconds=generation_seconds,
                    exc_info=True,
                )
                if trace_writer is not None:
                    resource_end_snapshot = capture_resource_snapshot()
                    resource_metrics = {
                        'before': request_resource_snapshot,
                        'after': resource_end_snapshot,
                        'delta': build_resource_delta(before=request_resource_snapshot, after=resource_end_snapshot),
                    }
                    trace_writer.record('response_error', {
                        'error': str(exc),
                        'resource_metrics': resource_metrics,
                    })
                    await flush_trace_writer(trace_writer)
                _update_sse_phase('error')
                yield {'event': 'error', 'data': serialize_api_response({'error': str(exc)})}

            completion_mode, done_has_remaining_scope = _resolve_completion_state(
                completion_mode_override=completion_mode_override,
                timeout_occurred=timeout_occurred,
                has_remaining_scope=has_remaining_scope,
            )
            if not isinstance(latest_output_contract_check, dict):
                latest_output_contract_check = {}
            if not isinstance(latest_output_contract_check.get('has_content_gap'), bool):
                missing_headings = latest_output_contract_check.get('missing_headings')
                has_missing_headings = isinstance(missing_headings, list) and any(
                    isinstance(heading, str) and heading.strip()
                    for heading in missing_headings
                )
                latest_output_contract_check['has_content_gap'] = (
                    has_missing_headings
                    if isinstance(missing_headings, list)
                    else done_has_remaining_scope
                )
            next_action, next_action_reason = _resolve_next_action(
                stopped_by_user=stopped_by_user,
                timeout_occurred=timeout_occurred,
                has_remaining_scope=done_has_remaining_scope,
                output_contract_check=latest_output_contract_check,
                continuation_resolution_reason=continuation_resolution_reason,
            )
            completion_mode, done_has_remaining_scope = _enforce_completion_action_consistency(
                completion_mode=completion_mode,
                has_remaining_scope=done_has_remaining_scope,
                next_action=next_action,
                next_action_reason=next_action_reason,
            )
            resolved_completion_mode = completion_mode
            resolved_has_remaining_scope = done_has_remaining_scope
            resolved_next_action = next_action
            resolved_next_action_reason = next_action_reason
            if assistant_message_record is not None:
                resolved_completion_mode = assistant_message_record.completion_mode or resolved_completion_mode
                resolved_has_remaining_scope = bool(assistant_message_record.has_remaining_scope)
                resolved_next_action = assistant_message_record.next_action
                resolved_next_action_reason = assistant_message_record.next_action_reason

            if metrics_raw_chunks_count <= 0 and trace_writer is not None and hasattr(trace_writer, 'get_sections'):
                sections = trace_writer.get_sections()
                retrieval = sections.get('retrieval', {}) if isinstance(sections, dict) else {}
                metrics_raw_chunks_count = _safe_int(retrieval.get('raw_chunks_count'), default=0)
                if metrics_query_type == 'unknown':
                    intent = sections.get('intent', {}) if isinstance(sections, dict) else {}
                    inferred_query_type = intent.get('query_type') if isinstance(intent, dict) else None
                    if isinstance(inferred_query_type, str) and inferred_query_type.strip():
                        metrics_query_type = inferred_query_type.strip().lower()

            final_answer = ''.join(answer_parts).strip()
            refusal_text = cleaned_answer if cleaned_answer else final_answer
            metrics_model = EvalMetrics(
                chat_id=chat_id,
                question=message_text,
                model_filename=settings.llm_model_filename,
                query_type=metrics_query_type,
                raw_chunks_count=metrics_raw_chunks_count,
                sources_count=len(sources),
                generation_seconds=_safe_float(generation_seconds, default=0.0),
                answer_length=len(final_answer),
                timeout_occurred=bool(timeout_occurred),
                has_empty_answer=not final_answer,
                has_refusal_pattern=_detect_refusal_pattern(refusal_text),
                unsupported_claim_count=_safe_int(grounding_verifier.get('unsupported_claim_count'), default=0),
                evidence_coverage_rate=_safe_float(grounding_verifier.get('evidence_coverage_rate'), default=0.0),
                not_found_count=_safe_int(grounding_verifier.get('not_found_count'), default=0),
            )
            detected_issues = detect_issues(refusal_text, metrics_model)
            issue_strings = [issue.value for issue in detected_issues]
            try:
                await insert_diagnostics_metrics(
                    db=db,
                    metrics=metrics_model,
                    detected_issues=issue_strings,
                    run_id=requested_run_id,
                )
            except _PERSISTENCE_EXCEPTIONS as metrics_exc:
                log.warning(
                    'chat_metrics_persist_failed',
                    chat_id=chat_id,
                    error=str(metrics_exc),
                )

            display_blocks = build_display_blocks(cleaned_answer)
            done_data: dict = {
                'elapsed_seconds': generation_seconds,
                'request_id': request_id if request_id else None,
                'timeout_occurred': timeout_occurred,
                'timeout_reason': timeout_reason,
                'completion_mode': resolved_completion_mode,
                'has_remaining_scope': resolved_has_remaining_scope,
                'stopped_by_user': stopped_by_user,
                'next_action': resolved_next_action,
                'next_action_reason': resolved_next_action_reason,
                'response_mode_used': response_mode_used,
                'mode_adjustments_applied': mode_adjustments_applied,
                'sources_count': len(sources),
                'message_persisted': message_persisted,
                'display_blocks': display_blocks,
                'budget_metrics': budget_metrics,
                'grounding_verifier': grounding_verifier,
                'budget_checkpoints': budget_checkpoints,
                'continuation_passes': continuation_passes,
                'contract_repair_pass_applied': repair_pass_applied,
                'output_contract_check': latest_output_contract_check,
                'continuation_unresolved_headings': continuation_unresolved_headings,
                'continuation_resolution_reason': continuation_resolution_reason,
                'continuation_progress_state': continuation_progress_state,
                'pass_details': pass_details,
                'status_transitions': status_transitions,
                'query_plan': {
                    'planned': active_query_plan is not None,
                    'answer_sections_count': (
                        len(active_query_plan.answer_sections) if active_query_plan else 0
                    ),
                    'steps_count': (
                        len(active_query_plan.steps) if active_query_plan else 0
                    ),
                    'aggregation_mode': (
                        active_query_plan.aggregation_mode if active_query_plan else None
                    ),
                    'output_shape': (
                        active_query_plan.output_shape if active_query_plan else None
                    ),
                    'planning_latency_ms': planning_latency_ms,
                },
            }
            if resource_end_snapshot is None:
                resource_end_snapshot = capture_resource_snapshot()
            if not resource_metrics:
                resource_metrics = {
                    'before': request_resource_snapshot,
                    'after': resource_end_snapshot,
                    'delta': build_resource_delta(before=request_resource_snapshot, after=resource_end_snapshot),
                }
            done_data['resource_metrics'] = resource_metrics
            if assistant_message_id is not None:
                done_data['message_id'] = assistant_message_id
            log.info(
                'chat_response_completed',
                chat_id=chat_id,
                completion_mode=done_data.get('completion_mode'),
                has_remaining_scope=done_data.get('has_remaining_scope'),
                next_action=done_data.get('next_action'),
                next_action_reason=done_data.get('next_action_reason'),
                timeout_occurred=timeout_occurred,
                timeout_reason=timeout_reason,
                response_mode_used=response_mode_used,
                message_persisted=message_persisted,
                sources_count=len(sources),
                tokens_streamed=len(answer_parts),
                continuation_passes=continuation_passes,
                contract_repair_pass_applied=repair_pass_applied,
                duration_ms=round((generation_seconds or 0.0) * 1000, 1),
                process_rss_mb=resource_end_snapshot.get('process_rss_mb') if isinstance(resource_end_snapshot, dict) else None,
                process_cpu_percent=resource_end_snapshot.get('process_cpu_percent') if isinstance(resource_end_snapshot, dict) else None,
                system_cpu_percent=resource_end_snapshot.get('system_cpu_percent') if isinstance(resource_end_snapshot, dict) else None,
            )
            await _unregister_active_stream(stream_id)
            _update_sse_phase('done')
            yield {'event': 'done', 'data': serialize_api_response(done_data)}

    return EventSourceResponse(_event_stream())


@router.post('/api/chat/stop')
async def stop_chat(request: ChatStopRequest) -> dict:
    stopped = await _mark_stream_stopped_by_user(
        stream_id=request.stream_id,
        chat_id=request.chat_id,
    )
    return {
        'stopped': stopped,
        'stream_id': request.stream_id,
    }


# ==============================================================================
# GET /api/chat/messages/{message_id}/raw — fetch raw content for a message
# ==============================================================================

@router.get('/api/chat/messages/{message_id}/raw')
async def get_message_raw(
    message_id: int,
    db: aiosqlite.Connection = Depends(get_db),
) -> dict:
    # Return raw content for a message (with <think> blocks). Used for on-demand
    # display when enable_raw_output_control is enabled. Only assistant messages
    # have meaningful raw content; user messages return their content as-is.
    message = await get_chat_message_by_id(db, message_id)
    if message is None:
        raise HTTPException(status_code=404, detail='Message not found')
    return {'content': message.content}


# ==============================================================================
# GET /api/chat/chats — list recent chats
# ==============================================================================

@router.get('/api/chat/chats')
async def list_chats(
    limit:  int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    search: str | None = Query(default=None),
    db: aiosqlite.Connection = Depends(get_db),
) -> dict:
    search_param = search.strip() if search and search.strip() else None
    chats = await get_chats(db, limit=limit, offset=offset, search=search_param)
    total = await get_chat_count(db, search=search_param)

    return {
        'chats':   chats,
        'total':   total,
        'limit':   limit,
        'offset':  offset,
    }


# ==============================================================================
# GET /api/chat/chats/{chat_id} — get chat messages
# ==============================================================================

@router.get('/api/chat/chats/{chat_id}')
async def get_chat_messages(
    chat_id: str,
    db: aiosqlite.Connection = Depends(get_db),
) -> dict:
    messages = await get_chat(db, chat_id)

    if not messages:
        raise HTTPException(status_code=404, detail='Chat not found')

    _, _, auto_continue_prompt = _resolve_auto_continue_policy()
    normalized_auto_continue_prompt = auto_continue_prompt.strip()
    serialized_messages: list[dict[str, object]] = []
    for message in messages:
        payload = message.model_dump(mode='json')
        if message.role == 'assistant':
            cleaned_content, _ = build_display_answer(message.content)
            payload['content'] = cleaned_content
            payload['display_blocks'] = build_display_blocks(cleaned_content)
        elif message.role == 'user':
            payload['is_internal'] = (
                bool(normalized_auto_continue_prompt)
                and str(message.content or '').strip() == normalized_auto_continue_prompt
            )
        serialized_messages.append(payload)

    return {
        'chat_id':  chat_id,
        'messages': serialized_messages,
        'total':    len(messages),
    }


# ==============================================================================
# PUT /api/chat/chats/{chat_id}/title — rename chat
# ==============================================================================

@router.put('/api/chat/chats/{chat_id}/title')
async def update_chat_title(
    chat_id: str,
    title: str = Query(..., min_length=1, max_length=200),
    db: aiosqlite.Connection = Depends(get_db),
) -> dict:
    messages = await get_chat(db, chat_id)
    if not messages:
        raise HTTPException(status_code=404, detail='Chat not found')

    await set_chat_title(db, chat_id, title.strip())
    log.info('chat_title_updated', chat_id=chat_id, title=title)

    return {
        'chat_id': chat_id,
        'title':   title.strip(),
    }


# ==============================================================================
# DELETE /api/chat/chats/{chat_id} — delete chat
# ==============================================================================

@router.delete('/api/chat/chats/{chat_id}')
async def delete_chat_endpoint(
    chat_id: str,
    db: aiosqlite.Connection = Depends(get_db),
) -> dict:
    deleted = await delete_chat(db, chat_id)
    if not deleted:
        raise HTTPException(status_code=404, detail='Chat not found')

    log.info('chat_deleted', chat_id=chat_id)

    return {
        'chat_id':  chat_id,
        'deleted':  True,
    }
