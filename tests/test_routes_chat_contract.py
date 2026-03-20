import json
from pathlib import Path

import pytest
from fastapi import HTTPException

from informity.api.routes_chat import (
    DISPLAY_FALLBACK_MESSAGE,
    SseContractTracker,
    _build_continuation_anchor_retry_prompt,
    _build_contract_closure_prompt,
    _build_grounding_repair_prompt,
    _build_section_progress_payload,
    _build_targeted_contract_repair_prompt,
    _detect_structural_incomplete_reason,
    _enforce_completion_action_consistency,
    _enforce_continuation_chat_binding,
    _evaluate_grounding_repair_gate,
    _has_continue_worthy_gap,
    _has_unresolved_contract_targets,
    _is_continuation_request,
    _mark_reasoning_only_contract_gap,
    _mark_structural_output_gap,
    _requires_full_rewrite_for_contract_repair,
    _resolve_completion_state,
    _resolve_next_action,
    _summarize_strict_claim_evidence_gate,
    build_display_answer,
    build_display_blocks,
    enforce_response_mode_supported,
    get_chat_messages,
    resolve_response_mode,
)
from informity.config import settings
from informity.db.models import ChatMessage
from informity.llm.model_adapter import QWEN3_14B_PROFILE, QWEN3_30B_A3B_PROFILE
from informity.llm.rag_runtime.strict_output_contract import _build_output_contract_plan

FIXTURES_PATH = Path(__file__).parent / 'fixtures' / 'malformed_output_fixtures.json'
MALFORMED_OUTPUT_FIXTURES = json.loads(FIXTURES_PATH.read_text(encoding='utf-8'))


@pytest.mark.parametrize('fixture', MALFORMED_OUTPUT_FIXTURES, ids=[f['id'] for f in MALFORMED_OUTPUT_FIXTURES])
def test_build_display_answer_matches_shared_fixtures(fixture: dict[str, object]) -> None:
    raw = str(fixture['raw'])
    cleaned, reasoning_only = build_display_answer(raw)
    assert cleaned == fixture['backend_cleaned']
    assert reasoning_only is fixture['backend_reasoning_only']
    # Raw output remains canonical for storage/retrieval paths.
    if 'Source' in raw or '<think>' in raw or '<<think>>' in raw:
        assert raw != cleaned


def test_reasoning_only_fixture_uses_contract_fallback_message() -> None:
    matching = [f for f in MALFORMED_OUTPUT_FIXTURES if f['id'] == 'reasoning_only_output']
    assert matching
    assert matching[0]['backend_cleaned'] == DISPLAY_FALLBACK_MESSAGE


def test_backend_frontend_display_parity_for_non_reasoning_fixtures() -> None:
    for fixture in MALFORMED_OUTPUT_FIXTURES:
        if fixture['backend_reasoning_only'] is True:
            continue
        assert fixture['backend_cleaned'] == fixture['frontend_cleaned'], fixture['id']


def test_reasoning_only_fixture_documents_parity_exception() -> None:
    matching = [f for f in MALFORMED_OUTPUT_FIXTURES if f['id'] == 'reasoning_only_output']
    assert matching
    fixture = matching[0]
    # Contract exception: backend emits fallback sentence, frontend fallback sanitizer returns empty.
    assert fixture['backend_reasoning_only'] is True
    assert fixture['backend_cleaned'] != fixture['frontend_cleaned']


def test_strict_claim_evidence_gate_summary_never_rewrites() -> None:
    summary = _summarize_strict_claim_evidence_gate(
        sources=[
            {
                'filename': 'sample.pdf',
                'chunk_preview': 'Revenue was 12345 and tax was $67.89 for the period.',
            },
        ],
        unsupported_claims=['12345', '88888'],
    )
    assert summary['canonical_fact_count'] >= 1
    assert summary['replaced_line_count'] == 0
    assert summary['bound_line_count'] == 0
    assert summary['unsupported_token_count'] == 2
    assert summary['unsupported_token_with_fact_count'] == 1


def test_sse_contract_tracker_rejects_out_of_order_events() -> None:
    tracker = SseContractTracker()
    assert tracker.update('chat') is True
    assert tracker.update('token') is True
    assert tracker.update('sources') is True
    assert tracker.update('done') is True
    assert tracker.update('token') is False


def test_build_display_blocks_returns_text_block() -> None:
    blocks = build_display_blocks('Hello **world**')
    assert blocks == [{'type': 'text', 'markdown': 'Hello **world**'}]


def test_targeted_contract_repair_prompt_lists_only_missing_headings() -> None:
    plan = _build_output_contract_plan(
        question='Output must contain: ## Scope, ## Method, ## Findings.',
        format_requirements=[
            'use the required headings exactly and in the requested order',
            'include heading: ## Scope',
            'include heading: ## Method',
            'include heading: ## Findings',
        ],
    )
    prompt = _build_targeted_contract_repair_prompt(
        plan=plan,
        output_contract_check={
            'passed': False,
            'missing_headings': ['## Method', '## Findings'],
            'order_violations': [],
            'bullet_depth_ok': None,
            'missing_evidence_callout_ok': None,
        },
        original_question='Output must contain: ## Scope, ## Method, ## Findings.',
    )
    assert prompt is not None
    assert 'Do not rewrite completed valid sections.' in prompt
    assert '- ## Method' in prompt
    assert '- ## Findings' in prompt


def test_targeted_contract_repair_prompt_includes_word_and_bullet_fixes() -> None:
    plan = _build_output_contract_plan(
        question='Include exactly 3 bullets and keep total <= 180 words.',
        format_requirements=[],
    )
    prompt = _build_targeted_contract_repair_prompt(
        plan=plan,
        output_contract_check={
            'passed': False,
            'missing_headings': [],
            'order_violations': [],
            'bullet_depth_ok': None,
            'missing_evidence_callout_ok': None,
            'word_count_ok': False,
            'max_words': 180,
            'top_level_bullet_count_ok': False,
            'exact_top_level_bullets': 3,
        },
        original_question='Include exactly 3 bullets and keep total <= 180 words.',
    )
    assert prompt is not None
    assert '<= 180 words' in prompt
    assert 'exactly 3 top-level bullets' in prompt


def test_targeted_contract_repair_prompt_includes_reasoning_only_guidance() -> None:
    plan = _build_output_contract_plan(
        question='Include exactly 3 bullets and keep total <= 180 words.',
        format_requirements=[],
    )
    prompt = _build_targeted_contract_repair_prompt(
        plan=plan,
        output_contract_check={
            'passed': False,
            'failure_reason': 'reasoning_only_output',
            'missing_headings': [],
            'order_violations': [],
            'bullet_depth_ok': None,
            'missing_evidence_callout_ok': None,
            'word_count_ok': False,
            'max_words': 180,
            'top_level_bullet_count_ok': False,
            'exact_top_level_bullets': 3,
        },
        original_question='Include exactly 3 bullets and keep total <= 180 words.',
    )
    assert prompt is not None
    assert 'reasoning-only output' in prompt
    assert 'do not emit <think> blocks' in prompt


def test_targeted_contract_repair_prompt_includes_missing_evidence_previews() -> None:
    plan = _build_output_contract_plan(
        question='Keep all claims evidence-grounded.',
        format_requirements=[
            (
                'require canonical evidence grounding for every claim-bearing bullet/list item '
                'and narrative claim paragraph'
            ),
        ],
    )
    prompt = _build_targeted_contract_repair_prompt(
        plan=plan,
        output_contract_check={
            'passed': False,
            'missing_headings': [],
            'order_violations': [],
            'bullet_depth_ok': None,
            'missing_evidence_callout_ok': None,
            'word_count_ok': None,
            'top_level_bullet_count_ok': None,
            'evidence_grounding_ok': False,
            'evidence_missing_blocks_preview': [
                '- Mortgage interest received in 2022: $5,331.34',
                '- No contradictions found in the 2023 documents.',
            ],
        },
        original_question='Build a reconciliation report.',
    )
    assert prompt is not None
    assert 'Rewrite these failing claim blocks' in prompt
    assert '- - Mortgage interest received in 2022: $5,331.34' in prompt


def test_targeted_contract_repair_prompt_includes_contradiction_and_delta_guidance() -> None:
    plan = _build_output_contract_plan(
        question='Keep all claims evidence-grounded.',
        format_requirements=[
            (
                'require canonical evidence grounding for every claim-bearing bullet/list item '
                'and narrative claim paragraph'
            ),
        ],
    )
    prompt = _build_targeted_contract_repair_prompt(
        plan=plan,
        output_contract_check={
            'passed': False,
            'missing_headings': [],
            'order_violations': [],
            'bullet_depth_ok': None,
            'missing_evidence_callout_ok': None,
            'word_count_ok': None,
            'top_level_bullet_count_ok': None,
            'evidence_grounding_ok': True,
            'contradiction_placeholder_ok': False,
            'uncited_delta_numeric_ok': False,
        },
        original_question='Build a report.',
    )
    assert prompt is not None
    assert 'replace bare "No contradictions found" lines' in prompt
    assert 'Largest Increase/Largest Decrease' in prompt


def test_has_unresolved_contract_targets_includes_new_contract_flags() -> None:
    assert _has_unresolved_contract_targets({'contradiction_placeholder_ok': False}) is True
    assert _has_unresolved_contract_targets({'uncited_delta_numeric_ok': False}) is True


def test_continuation_request_detector_matches_continue_verbs() -> None:
    assert _is_continuation_request('Continue with remaining sections.') is True
    assert _is_continuation_request('Go on with unresolved boxes only.') is True
    assert _is_continuation_request('More please.') is True
    assert _is_continuation_request('The rest') is True
    assert _is_continuation_request('Summarize the report.') is False


def test_continuation_chat_binding_requires_existing_chat_id() -> None:
    with pytest.raises(HTTPException) as exc_info:
        _enforce_continuation_chat_binding(
            question='Continue with remaining sections.',
            chat_id=None,
        )
    assert exc_info.value.status_code == 409


def test_continuation_chat_binding_allows_non_continuation_without_chat_id() -> None:
    _enforce_continuation_chat_binding(
        question='Summarize the report.',
        chat_id=None,
    )


def test_continuation_chat_binding_allows_continuation_with_chat_id() -> None:
    _enforce_continuation_chat_binding(
        question='Continue with remaining sections.',
        chat_id='chat-123',
    )


def test_contract_closure_prompt_includes_missing_headings() -> None:
    prompt = _build_contract_closure_prompt(
        output_contract_check={'missing_headings': ['## Scope', '## Method']},
        original_question='Build report with Scope/Method/Findings.',
    )
    assert 'Complete only these remaining headings exactly:' in prompt
    assert '- ## Scope' in prompt
    assert '- ## Method' in prompt


def test_build_section_progress_payload_returns_ordered_completed_and_remaining() -> None:
    plan = _build_output_contract_plan(
        question='Output must contain: ## Scope, ## Method, ## Findings.',
        format_requirements=[
            'use the required headings exactly and in the requested order',
            'include heading: ## Scope',
            'include heading: ## Method',
            'include heading: ## Findings',
        ],
    )
    payload = _build_section_progress_payload(
        plan=plan,
        output_contract_check={
            'passed': False,
            'missing_headings': ['## Method'],
        },
    )
    assert payload is not None
    assert payload['completed'] == ['## Scope', '## Findings']
    assert payload['remaining'] == ['## Method']
    assert payload['total'] == 3


def test_continuation_anchor_retry_prompt_scopes_to_prior_sources() -> None:
    prompt = _build_continuation_anchor_retry_prompt(
        'Continue with unresolved or missing boxes only.',
    )
    assert 'Use only previously cited source documents' in prompt
    assert 'do not broaden scope' in prompt.lower()


def test_evaluate_grounding_repair_gate_detects_low_coverage_and_unsupported_claims() -> None:
    plan = _build_output_contract_plan(
        question='Build an evidence-grounded brief.',
        format_requirements=[
            (
                'require canonical evidence grounding for every claim-bearing bullet/list item '
                'and narrative claim paragraph'
            ),
        ],
    )
    should_repair, reasons = _evaluate_grounding_repair_gate(
        plan=plan,
        grounding_verifier={
            'unsupported_claim_count': 2,
            'evidence_coverage_rate': 0.1,
            'not_found_count': 0,
        },
    )
    assert should_repair is True
    assert 'unsupported_claims_detected' in reasons
    assert 'low_evidence_coverage' in reasons


def test_build_grounding_repair_prompt_includes_reason_specific_guidance() -> None:
    prompt = _build_grounding_repair_prompt(
        original_question='Prepare a comparison report for 2022 and 2023.',
        grounding_reasons=['unsupported_claims_detected', 'low_evidence_coverage'],
    )
    assert 'unsupported claims' in prompt
    assert 'Increase evidence coverage' in prompt
    assert 'Original request:' in prompt


def test_resolve_response_mode_falls_back_to_settings_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, 'default_response_mode', 'analysis')
    assert resolve_response_mode(None) == 'analysis'
    # 'balanced' is no longer a valid response mode; invalid modes fall back to 'analysis'
    assert resolve_response_mode('balanced') == 'analysis'
    assert resolve_response_mode('research') == 'research'
    assert resolve_response_mode('invalid') == 'analysis'


def test_enforce_response_mode_supported_rejects_unsupported_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr('informity.api.routes_chat.get_profile', lambda: QWEN3_14B_PROFILE)
    with pytest.raises(HTTPException) as exc_info:
        enforce_response_mode_supported('research')
    assert exc_info.value.status_code == 422


def test_enforce_response_mode_supported_accepts_supported_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr('informity.api.routes_chat.get_profile', lambda: QWEN3_30B_A3B_PROFILE)
    enforce_response_mode_supported('research')


def test_resolve_completion_state_forces_scoped_complete_when_scope_remains() -> None:
    completion_mode, has_remaining_scope = _resolve_completion_state(
        completion_mode_override='complete',
        timeout_occurred=False,
        has_remaining_scope=True,
    )
    assert completion_mode == 'scoped_complete'
    assert has_remaining_scope is True


def test_resolve_next_action_returns_none_when_complete() -> None:
    next_action, next_action_reason = _resolve_next_action(
        stopped_by_user=False,
        timeout_occurred=False,
        has_remaining_scope=False,
        output_contract_check={'has_content_gap': False},
        continuation_resolution_reason=None,
    )
    assert next_action == 'none'
    assert next_action_reason is None


def test_resolve_next_action_prefers_regenerate_when_stopped() -> None:
    next_action, next_action_reason = _resolve_next_action(
        stopped_by_user=True,
        timeout_occurred=False,
        has_remaining_scope=True,
        output_contract_check={'has_content_gap': True},
        continuation_resolution_reason=None,
    )
    assert next_action == 'regenerate'
    assert next_action_reason == 'stopped'


def test_resolve_next_action_uses_content_gap_for_continue() -> None:
    next_action, next_action_reason = _resolve_next_action(
        stopped_by_user=False,
        timeout_occurred=False,
        has_remaining_scope=True,
        output_contract_check={'has_content_gap': True},
        continuation_resolution_reason=None,
    )
    assert next_action == 'continue'
    assert next_action_reason == 'unresolved_content'


def test_resolve_next_action_suppresses_continue_without_content_gap() -> None:
    next_action, next_action_reason = _resolve_next_action(
        stopped_by_user=False,
        timeout_occurred=False,
        has_remaining_scope=True,
        output_contract_check={'has_content_gap': False},
        continuation_resolution_reason=None,
    )
    assert next_action == 'none'
    assert next_action_reason is None


def test_resolve_next_action_maps_stalled_to_regenerate() -> None:
    next_action, next_action_reason = _resolve_next_action(
        stopped_by_user=False,
        timeout_occurred=False,
        has_remaining_scope=True,
        output_contract_check={'has_content_gap': True},
        continuation_resolution_reason='duplicate_continuation_detected',
    )
    assert next_action == 'regenerate'
    assert next_action_reason == 'stalled'


def test_resolve_next_action_maps_budget_exhausted_to_continue() -> None:
    next_action, next_action_reason = _resolve_next_action(
        stopped_by_user=False,
        timeout_occurred=False,
        has_remaining_scope=True,
        output_contract_check={'has_content_gap': True},
        continuation_resolution_reason='continuation_pass_budget_exhausted',
    )
    assert next_action == 'continue'
    assert next_action_reason == 'budget_exhausted'


def test_resolve_next_action_maps_timeout_to_continue() -> None:
    next_action, next_action_reason = _resolve_next_action(
        stopped_by_user=False,
        timeout_occurred=True,
        has_remaining_scope=True,
        output_contract_check={'has_content_gap': False},
        continuation_resolution_reason=None,
    )
    assert next_action == 'continue'
    assert next_action_reason == 'timeout'


def test_has_continue_worthy_gap_prefers_explicit_contract_flag_false() -> None:
    assert _has_continue_worthy_gap(
        has_remaining_scope_signal=True,
        output_contract_check={'has_content_gap': False},
    ) is False


def test_mark_reasoning_only_contract_gap_sets_continue_worthy_flags() -> None:
    normalized = _mark_reasoning_only_contract_gap({'passed': True, 'has_content_gap': False})
    assert normalized['passed'] is False
    assert normalized['has_content_gap'] is True
    assert normalized['failure_reason'] == 'reasoning_only_output'


def test_resolve_next_action_duplicate_continuation_prefers_regenerate() -> None:
    next_action, next_action_reason = _resolve_next_action(
        stopped_by_user=False,
        timeout_occurred=False,
        has_remaining_scope=False,
        output_contract_check={'has_content_gap': True},
        continuation_resolution_reason='duplicate_continuation_detected',
    )
    assert next_action == 'regenerate'
    assert next_action_reason == 'stalled'


def test_has_continue_worthy_gap_prefers_explicit_contract_flag_true() -> None:
    assert _has_continue_worthy_gap(
        has_remaining_scope_signal=False,
        output_contract_check={'has_content_gap': True},
    ) is True


def test_detect_structural_incomplete_reason_for_truncated_table_row() -> None:
    answer = (
        '| Field | Value |\n'
        '|-----------|-------|\n'
        '| Item A | Value A |\n'
        '| Item B | Value B\n'
    )
    assert _detect_structural_incomplete_reason(answer) == 'truncated_markdown_table_row'


def test_mark_structural_output_gap_sets_continue_worthy_flags() -> None:
    answer = (
        '| A | B |\n'
        '|---|---|\n'
        '| x | y\n'
    )
    normalized = _mark_structural_output_gap({'passed': True}, answer=answer)
    assert normalized['passed'] is False
    assert normalized['has_content_gap'] is True
    assert normalized['failure_reason'] == 'truncated_markdown_table_row'
    assert normalized['structural_incomplete_reason'] == 'truncated_markdown_table_row'


def test_enforce_completion_action_consistency_none_action_normalizes_complete() -> None:
    completion_mode, has_remaining_scope = _enforce_completion_action_consistency(
        completion_mode='scoped_complete',
        has_remaining_scope=True,
        next_action='none',
        next_action_reason=None,
    )
    assert completion_mode == 'complete'
    assert has_remaining_scope is False


def test_enforce_completion_action_consistency_continue_requires_scope() -> None:
    completion_mode, has_remaining_scope = _enforce_completion_action_consistency(
        completion_mode='complete',
        has_remaining_scope=False,
        next_action='continue',
        next_action_reason='unresolved_content',
    )
    assert completion_mode == 'scoped_complete'
    assert has_remaining_scope is True


def test_enforce_completion_action_consistency_regenerate_allows_terminal_scope() -> None:
    completion_mode, has_remaining_scope = _enforce_completion_action_consistency(
        completion_mode='scoped_complete',
        has_remaining_scope=True,
        next_action='regenerate',
        next_action_reason='stalled',
    )
    assert completion_mode == 'complete'
    assert has_remaining_scope is False


def test_requires_full_rewrite_for_contract_repair_on_evidence_grounding_failure() -> None:
    assert _requires_full_rewrite_for_contract_repair(
        {
            'word_count_ok': None,
            'top_level_bullet_count_ok': None,
            'evidence_grounding_ok': False,
        }
    ) is True


def test_has_unresolved_contract_targets_detects_missing_heading() -> None:
    assert _has_unresolved_contract_targets(
        {
            'passed': False,
            'missing_headings': ['## Scope'],
            'order_violations': [],
        }
    ) is True


def test_has_unresolved_contract_targets_detects_no_remaining_targets() -> None:
    assert _has_unresolved_contract_targets(
        {
            'passed': True,
            'missing_headings': [],
            'order_violations': [],
            'bullet_depth_ok': True,
            'missing_evidence_callout_ok': True,
            'word_count_ok': True,
            'top_level_bullet_count_ok': True,
        }
    ) is False


@pytest.mark.asyncio
async def test_get_chat_messages_returns_cleaned_display_blocks_for_assistant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _fake_get_chat(_db, _chat_id: str) -> list[ChatMessage]:
        return [
            ChatMessage(chat_id='c1', role='user', content='Hello'),
            ChatMessage(
                chat_id='c1',
                role='assistant',
                content='<think>internal</think>\n## Scope\nDone',
                next_action='none',
                next_action_reason=None,
            ),
        ]

    monkeypatch.setattr('informity.api.routes_chat.get_chat', _fake_get_chat)
    payload = await get_chat_messages(chat_id='c1', db=None)
    messages = payload['messages']
    assert isinstance(messages, list)
    assert messages[0]['role'] == 'user'
    assert messages[0].get('display_blocks') is None
    assert messages[1]['role'] == 'assistant'
    assert '<think>' not in messages[1]['content']
    assert messages[1].get('display_blocks') == [{'type': 'text', 'markdown': messages[1]['content']}]
    assert messages[1].get('next_action') == 'none'
