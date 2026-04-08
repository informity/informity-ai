# ==============================================================================
# Informity AI — Chat Closeout Helpers
# Done-payload assembly and closeout-only numeric evidence helpers.
# ==============================================================================

import re

_STRICT_NUMERIC_TOKEN_PATTERN = re.compile(r'\$?\d[\d,]*(?:\.\d{1,2})?')


def build_display_blocks(cleaned_answer: str) -> list[dict[str, str]]:
    if not cleaned_answer:
        return []
    return [{'type': 'text', 'markdown': cleaned_answer}]


def build_done_payload(
    *,
    elapsed_seconds: float | None,
    request_id: str | None,
    chat_mode: str,
    timeout_occurred: bool,
    timeout_reason: str | object | None,
    completion_mode: str | object,
    has_remaining_scope: bool,
    stopped_by_user: bool,
    next_action: str | object,
    next_action_reason: str | None,
    sources_count: int,
    message_persisted: bool,
    cleaned_answer: str,
    budget_metrics: dict[str, object],
    budget_checkpoints: list[dict[str, object]],
    continuation_passes: int,
    continuation_resolution_reason: str | object | None,
    continuation_progress_state: str | None,
    pass_details: list[dict[str, object]],
    status_transitions: list[dict[str, object]],
    resource_metrics: dict[str, object],
    message_id: int | None,
) -> dict:
    payload: dict[str, object] = {
        'elapsed_seconds': elapsed_seconds,
        'request_id': request_id if request_id else None,
        'chat_mode': chat_mode,
        'timeout_occurred': timeout_occurred,
        'timeout_reason': timeout_reason,
        'completion_mode': completion_mode,
        'has_remaining_scope': has_remaining_scope,
        'stopped_by_user': stopped_by_user,
        'next_action': next_action,
        'next_action_reason': next_action_reason,
        'sources_count': sources_count,
        'message_persisted': message_persisted,
        'display_blocks': build_display_blocks(cleaned_answer),
        'budget_metrics': budget_metrics,
        'web_search_used': bool(budget_metrics.get('web_search_used') is True),
        'budget_checkpoints': budget_checkpoints,
        'continuation_passes': continuation_passes,
        'continuation_resolution_reason': continuation_resolution_reason,
        'continuation_progress_state': continuation_progress_state,
        'pass_details': pass_details,
        'status_transitions': status_transitions,
        'resource_metrics': resource_metrics,
    }
    if message_id is not None:
        payload['message_id'] = message_id
    return payload


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


def build_canonical_numeric_fact_index(
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


def summarize_strict_claim_evidence_gate(
    *,
    sources: list[dict[str, object]],
    unsupported_claims: list[object],
) -> dict[str, object]:
    fact_index, canonical_facts = build_canonical_numeric_fact_index(sources)
    unsupported_tokens = {
        token for token in (
            _normalize_numeric_token(str(item))
            for item in unsupported_claims
            if isinstance(item, (str, int, float))
        )
        if token
    }
    unsupported_tokens_with_facts = sum(1 for token in unsupported_tokens if token in fact_index)
    return {
        'canonical_fact_count': len(canonical_facts),
        'replaced_line_count': 0,
        'bound_line_count': 0,
        'unsupported_token_count': len(unsupported_tokens),
        'unsupported_token_with_fact_count': unsupported_tokens_with_facts,
    }


__all__ = [
    'build_display_blocks',
    'build_done_payload',
    'build_canonical_numeric_fact_index',
    'summarize_strict_claim_evidence_gate',
]
