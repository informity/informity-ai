import pytest

from informity.config import settings
from informity.db.models import ChatMessage
from informity.llm.rag_runtime.retrieval_validation import (
    _apply_coverage_evidence_floor_override,
    _build_continuation_retrieval_query,
    _derive_continuation_source_terms,
    _evaluate_continuation_anchor_gate,
    _evaluate_source_diversity_gate,
    _extract_prior_has_remaining_scope,
)
from informity.llm.rag_runtime.strict_composers import try_compose_strict_contract_answer
from informity.llm.rag_runtime.structured_numeric import (
    _build_finance_conflict_placeholder_bullet,
    _derive_format_requirements,
    _evidence_overlap_tokens,
    _extract_exact_top_level_bullet_limit,
    _parse_numeric_token,
    _render_finance_conflict_bullets,
    _render_structured_rows_bullets_answer,
)


def test_retrieval_validation_source_diversity_coverage_gate() -> None:
    passed, distinct = _evaluate_source_diversity_gate(
        chunks=[
            {'file_id': 1},
            {'file_id': 2},
            {'file_id': 2},
        ],
        query_type='coverage',
    )
    assert passed is True
    assert distinct == 2


def test_structured_numeric_parse_currency_and_ignore_year() -> None:
    parsed = _parse_numeric_token('$3,187.30')
    assert parsed is not None
    assert parsed[0] == 3187.30
    assert _parse_numeric_token('2024') is None


def test_structured_numeric_derives_heading_order_requirement() -> None:
    requirements = _derive_format_requirements(
        'Create a brief with sections in order: 1) Scope, 2) Method, 3) Findings.'
    )
    assert any('requested order' in requirement for requirement in requirements)
    assert any('include heading: Scope' in requirement for requirement in requirements)


def test_structured_numeric_finance_conflict_bullets_truncates_excess() -> None:
    conflicts = [
        {'statement': 'A', 'docs': 'A1', 'values': '1', 'reason': 'x', 'rows': []},
        {'statement': 'B', 'docs': 'B1', 'values': '2', 'reason': 'y', 'rows': []},
        {'statement': 'C', 'docs': 'C1', 'values': '3', 'reason': 'z', 'rows': []},
        {'statement': 'D', 'docs': 'D1', 'values': '4', 'reason': 'q', 'rows': []},
        {'statement': 'E', 'docs': 'E1', 'values': '5', 'reason': 'r', 'rows': []},
    ]
    answer = _render_finance_conflict_bullets(selected_conflicts=conflicts, bullet_limit=4)
    assert answer.count('\n- ') == 3
    assert answer.startswith('- Conflict Statement: A;')
    assert 'Conflict Statement: E;' not in answer


def test_structured_numeric_finance_conflict_bullets_pads_missing() -> None:
    conflicts = [{'statement': 'A', 'docs': 'A1', 'values': '1', 'reason': 'x', 'rows': []}]
    answer = _render_finance_conflict_bullets(selected_conflicts=conflicts, bullet_limit=4)
    assert answer.count('\n- ') == 3
    placeholder = _build_finance_conflict_placeholder_bullet()
    assert answer.splitlines()[-1] == placeholder


def test_structured_numeric_evidence_overlap_tokens_counts_shared_terms() -> None:
    overlap = _evidence_overlap_tokens(
        'Mortgage interest total reported on Schedule A line 8a.',
        'Schedule A interest total differs from mortgage statement.',
    )
    assert overlap >= 2


def test_structured_numeric_evidence_overlap_tokens_ignores_noise_words() -> None:
    overlap = _evidence_overlap_tokens(
        'the and for this values were found in document',
        'this and the values were found for documents',
    )
    assert overlap == 0


def test_structured_numeric_derives_numbered_headings_with_parenthetical_commas() -> None:
    requirements = _derive_format_requirements(
        'Create a brief with sections in order: 1) Executive Summary (max 140 words), '
        '2) Year-by-Year Evidence Map (2022, 2023, 2024), 3) Action Checklist.'
    )
    assert any('include heading: Executive Summary (max 140 words)' in requirement for requirement in requirements)
    assert any('include heading: Year-by-Year Evidence Map (2022, 2023, 2024)' in requirement for requirement in requirements)
    assert any('include heading: Action Checklist' in requirement for requirement in requirements)


def test_structured_numeric_treats_output_must_contain_headings_as_ordered() -> None:
    requirements = _derive_format_requirements(
        'Output must contain: ## Scope, ## Method, ## Findings by Year, ## Cross-Year Deltas.'
    )
    assert any('requested order' in requirement for requirement in requirements)
    assert any('include heading: ## Scope' in requirement for requirement in requirements)


def test_structured_numeric_heading_extraction_ignores_trailing_instruction_prose() -> None:
    requirements = _derive_format_requirements(
        'Output must contain: ## Scope, ## Method, ## Findings by Year, ## Cross-Year Deltas, '
        '## Confidence Notes, ## Next Verification Steps. Under "Findings by Year", create subsections.'
    )
    assert any('include heading: ## Next Verification Steps' in requirement for requirement in requirements)
    assert not any(
        'include heading: ## Next Verification Steps. Under "Findings by Year"' in requirement
        for requirement in requirements
    )


def test_retrieval_validation_coverage_evidence_floor_override() -> None:
    passed, events = _apply_coverage_evidence_floor_override(
        retrieval_relevance_passed=False,
        query_type='coverage',
        subtype='aggregate_by_period',
        group_by='year',
        response_shape='narrative_synthesis',
        distinct_sources_count=3,
        chunk_count=8,
        fallback_events=[],
        route_profile_id='comparative_analysis',
        retrieval_relevance_score=0.11,
    )
    assert passed is True
    assert events
    assert events[0].get('fallback_reason') == 'coverage_evidence_floor_override'


def test_retrieval_validation_coverage_evidence_floor_override_for_narrative_brief() -> None:
    passed, events = _apply_coverage_evidence_floor_override(
        retrieval_relevance_passed=False,
        query_type='coverage',
        subtype=None,
        group_by=None,
        response_shape='narrative_synthesis',
        distinct_sources_count=4,
        chunk_count=9,
        fallback_events=[],
        route_profile_id='audit_or_compliance_brief',
        retrieval_relevance_score=0.14,
    )
    assert passed is True
    assert events
    assert events[0].get('fallback_reason') == 'coverage_evidence_floor_override'


def test_retrieval_validation_coverage_evidence_floor_override_respects_hard_floor(monkeypatch) -> None:
    monkeypatch.setattr(settings, 'retrieval_coverage_evidence_floor_hard_floor_enabled', True)
    monkeypatch.setattr(settings, 'retrieval_coverage_evidence_floor_min_score', 0.05)
    passed, events = _apply_coverage_evidence_floor_override(
        retrieval_relevance_passed=False,
        query_type='coverage',
        subtype=None,
        group_by='year',
        response_shape='narrative_synthesis',
        distinct_sources_count=6,
        chunk_count=14,
        fallback_events=[],
        route_profile_id='cross_document_synthesis',
        retrieval_relevance_score=0.01,
    )
    assert passed is False
    assert events == []


def test_retrieval_validation_coverage_evidence_floor_override_allows_low_score_when_hard_floor_disabled(monkeypatch) -> None:
    monkeypatch.setattr(settings, 'retrieval_coverage_evidence_floor_hard_floor_enabled', False)
    passed, events = _apply_coverage_evidence_floor_override(
        retrieval_relevance_passed=False,
        query_type='coverage',
        subtype='aggregate_by_period',
        group_by='year',
        response_shape='narrative_synthesis',
        distinct_sources_count=6,
        chunk_count=14,
        fallback_events=[],
        route_profile_id='cross_document_synthesis',
        retrieval_relevance_score=0.01,
    )
    assert passed is True
    assert events


def test_retrieval_validation_continuation_anchor_gate_requires_overlap_when_not_reset() -> None:
    passed, overlap = _evaluate_continuation_anchor_gate(
        route_candidate='continuation_or_refinement',
        scope_reset_detected=False,
        prior_source_anchors={'/docs/a.pdf'},
        current_source_keys={'/docs/b.pdf'},
    )
    assert passed is False
    assert overlap == 0


def test_retrieval_validation_continuation_anchor_gate_requires_overlap_even_with_remaining_scope() -> None:
    passed, overlap = _evaluate_continuation_anchor_gate(
        route_candidate='continuation_or_refinement',
        scope_reset_detected=False,
        prior_source_anchors={'/docs/a.pdf'},
        current_source_keys={'/docs/b.pdf'},
        prior_has_remaining_scope=True,
    )
    assert passed is False
    assert overlap == 0


def test_retrieval_validation_extracts_prior_remaining_scope() -> None:
    history = [
        ChatMessage(chat_id='c1', role='user', content='q1'),
        ChatMessage(chat_id='c1', role='assistant', content='a1', has_remaining_scope=True),
    ]
    assert _extract_prior_has_remaining_scope(history) is True


def test_retrieval_validation_derives_continuation_source_terms() -> None:
    terms = _derive_continuation_source_terms(
        route_candidate='continuation_or_refinement',
        prior_has_remaining_scope=True,
        scope_reset_detected=False,
        prior_source_anchors={
            '/docs/2024 Tax Return.pdf',
            '2023 Taxes - Completed and Signed.pdf',
        },
    )
    assert terms
    assert any('2024 Tax Return' in term for term in terms)
    assert any('2023 Taxes - Completed and Signed' in term for term in terms)


def test_retrieval_validation_builds_continuation_query_from_prior_user_scope() -> None:
    history = [
        ChatMessage(chat_id='c1', role='user', content='Build a report for 2022-2024 with Scope, Method, Findings.'),
        ChatMessage(chat_id='c1', role='assistant', content='Partial answer', has_remaining_scope=True),
    ]
    query = _build_continuation_retrieval_query(
        question='Continue with the remaining sections.',
        route_candidate='continuation_or_refinement',
        prior_has_remaining_scope=True,
        scope_reset_detected=False,
        is_continuation=True,
        history=history,
    )
    assert 'Build a report for 2022-2024' in query


def test_retrieval_validation_uses_short_continuation_prompt_for_prior_context() -> None:
    history = [
        ChatMessage(chat_id='c1', role='user', content='Extract unresolved risks by year for 2022-2024.'),
        ChatMessage(chat_id='c1', role='assistant', content='Partial answer', has_remaining_scope=True),
    ]
    query = _build_continuation_retrieval_query(
        question='More please.',
        route_candidate='continuation_or_refinement',
        prior_has_remaining_scope=True,
        scope_reset_detected=False,
        is_continuation=True,
        history=history,
    )
    assert 'Extract unresolved risks by year for 2022-2024' in query


def test_structured_numeric_extracts_exact_top_level_bullet_limit() -> None:
    assert _extract_exact_top_level_bullet_limit('Output exactly 5 bullets.') == 5
    assert _extract_exact_top_level_bullet_limit('Include exactly three bullets with evidence.') == 3


def test_structured_numeric_bullet_renderer_outputs_exact_count() -> None:
    answer = _render_structured_rows_bullets_answer(
        [
            {'field_label': 'line item', 'raw_value': '$100', 'evidence_span': 'row one'},
            {'field_label': 'line item', 'raw_value': '$200', 'evidence_span': 'row two'},
        ],
        3,
    )
    assert answer.count('\n- ') == 3
    assert 'Missing Evidence' in answer


@pytest.mark.parametrize(
    ('question', 'expected_family'),
    [
        (
            'Create a long-form cross-document synthesis of indexed finance-related records with sections in this exact '
            'order: ## Coverage Snapshot, ## Amounts and Trends, ## Inconsistencies, ## Missing Evidence, ## Follow-up Plan. '
            'Under ## Amounts and Trends include one markdown table with columns: Year, Document Group, Key Amount, Evidence Note. '
            'Under ## Missing Evidence include exactly 5 bullets and each bullet must name one missing item and one verification action.',
            'research_cross_document_synthesis',
        ),
        (
            'Using only indexed records from 2022-2024, produce one response with headings in exact order: '
            '## Executive Summary, ## Evidence Map by Year, ## Financial Deltas, ## Contradictions and Gaps, '
            '## Verification Actions.',
            'research_long_synthesis',
        ),
        (
            'Build a forensic reconciliation report across 2022-2024 with exact headings: '
            '## Scope, ## Method, ## Findings by Year, ## Cross-Year Deltas, ## Confidence Notes, '
            '## Next Verification Steps.',
            'research_forensic_report',
        ),
        (
            'Produce an audit-style narrative with headings in order: ## Scope, ## Evidence Coverage Matrix, '
            '## Largest Increase, ## Largest Decrease, ## Ambiguities, ## Recommended Verification.',
            'research_yearly_delta_matrix',
        ),
        (
            'Generate a comprehensive verification brief with headings exactly: ## Scope and Constraints, '
            '## Source Inventory, ## Structured Findings, ## Conflicts, ## Unknowns, ## Verification Checklist.',
            'research_verification_brief',
        ),
        (
            'Using only indexed finance/insurance/lending documents from 2022-2024, create a structured compliance brief '
            'with these sections in order: 1) Executive Summary (max 140 words), 2) Year-by-Year Evidence Map '
            '(2022, 2023, 2024), 3) Document Group Deep Dive (Group A finance docs, Group B insurance docs, '
            'Group C lending docs, agency confirmations), 4) Risks and Gaps, 5) Action Checklist. '
            'In section 3, use nested bullets with exactly 3 levels.',
            'research_structured_compliance_brief',
        ),
    ],
)
def test_strict_composer_emits_contract_metrics_for_all_families(question: str, expected_family: str) -> None:
    chunks = [
        {
            'filename': 'sample-evidence.pdf',
            'file_path': '/docs/sample-evidence.pdf',
            'chunk_text': 'sample evidence text with year and values',
            'score': 0.91,
        },
        {
            'filename': 'secondary-evidence.pdf',
            'file_path': '/docs/secondary-evidence.pdf',
            'chunk_text': 'secondary evidence text',
            'score': 0.88,
        },
    ]
    result = try_compose_strict_contract_answer(
        question=question,
        chunks=chunks,
        response_mode='research',
    )
    assert result is not None
    answer, sources, metrics = result
    assert answer
    assert sources
    assert metrics.get('strict_composer_family') == expected_family
    assert metrics.get('strict_claim_count', 0) > 0
    assert metrics.get('strict_fallback_claim_count', 0) >= 0
    assert metrics.get('strict_unsupported_claim_count') == 0
    assert 0.0 <= float(metrics.get('strict_evidence_coverage_rate', 0.0)) <= 1.0
    decisions_preview = metrics.get('strict_claim_emission_decisions_preview')
    assert isinstance(decisions_preview, list)
    assert decisions_preview
    for decision in decisions_preview:
        assert decision.get('emitted') is True
        assert decision.get('dropped') is False
        assert decision.get('decision') in {'grounded', 'fallback', 'unsupported'}
    output_contract_check = metrics.get('output_contract_check')
    assert isinstance(output_contract_check, dict)
    assert output_contract_check.get('passed') is True


def test_strict_composer_applies_for_compliance_brief_in_analysis_mode() -> None:
    question = (
        'Using only indexed finance/insurance/lending documents from 2022-2024, create a structured compliance brief '
        'with these sections in order: 1) Executive Summary (max 140 words), 2) Year-by-Year Evidence Map '
        '(2022, 2023, 2024), 3) Document Group Deep Dive (Group A finance docs, Group B insurance docs, '
        'Group C lending docs, agency confirmations), 4) Risks and Gaps, 5) Action Checklist. '
        'In section 3, use nested bullets with exactly 3 levels.'
    )
    chunks = [
        {
            'filename': 'sample-evidence.pdf',
            'file_path': '/docs/sample-evidence.pdf',
            'chunk_text': 'sample evidence text with year and values',
            'score': 0.91,
        },
    ]
    result = try_compose_strict_contract_answer(
        question=question,
        chunks=chunks,
        response_mode='analysis',
    )
    assert result is not None
    _, _, metrics = result
    assert metrics.get('strict_composer_family') == 'research_structured_compliance_brief'
    output_contract_check = metrics.get('output_contract_check')
    assert isinstance(output_contract_check, dict)
    assert output_contract_check.get('passed') is True


def test_strict_composer_applies_for_compliance_brief_in_analysis_mode() -> None:
    question = (
        'Using only indexed finance/insurance/lending documents from 2022-2024, create a structured compliance brief '
        'with these sections in order: 1) Executive Summary (max 140 words), 2) Year-by-Year Evidence Map '
        '(2022, 2023, 2024), 3) Document Group Deep Dive (Group A finance docs, Group B insurance docs, '
        'Group C lending docs, agency confirmations), 4) Risks and Gaps, 5) Action Checklist. '
        'In section 3, use nested bullets with exactly 3 levels.'
    )
    chunks = [
        {
            'filename': 'sample-evidence.pdf',
            'file_path': '/docs/sample-evidence.pdf',
            'chunk_text': 'sample evidence text with year and values',
            'score': 0.91,
        },
    ]
    result = try_compose_strict_contract_answer(
        question=question,
        chunks=chunks,
        response_mode='analysis',
    )
    assert result is not None
    _, _, metrics = result
    assert metrics.get('strict_composer_family') == 'research_structured_compliance_brief'
    output_contract_check = metrics.get('output_contract_check')
    assert isinstance(output_contract_check, dict)
    assert output_contract_check.get('passed') is True


def test_strict_composer_skips_non_contract_balanced_query() -> None:
    result = try_compose_strict_contract_answer(
        question='How many files are indexed?',
        chunks=[],
        response_mode='balanced',
    )
    assert result is None


def test_strict_composer_skips_non_contract_analysis_query() -> None:
    result = try_compose_strict_contract_answer(
        question='How many files are indexed?',
        chunks=[],
        response_mode='analysis',
    )
    assert result is None


def test_strict_composer_enforces_max_words_for_analysis_format_contract() -> None:
    question = (
        'Using only indexed finance/insurance/lending documents from 2022-2024, produce ONE response with headings '
        'in this exact order: ## Executive Summary, ## Year-by-Year Evidence Map, ## Document Group Deep Dive, '
        '## Risks and Gaps, ## Action Checklist. Constraints: total <= 520 words; no preamble; no closing commentary. '
        'Under "Year-by-Year Evidence Map", include exactly three subsections: ### 2022, ### 2023, ### 2024.'
    )
    chunks = [
        {
            'filename': 'sample-evidence.pdf',
            'file_path': '/docs/sample-evidence.pdf',
            'chunk_text': 'sample evidence text with year and values',
            'score': 0.91,
        },
        {
            'filename': 'secondary-evidence.pdf',
            'file_path': '/docs/secondary-evidence.pdf',
            'chunk_text': 'secondary evidence text',
            'score': 0.88,
        },
    ]
    result = try_compose_strict_contract_answer(
        question=question,
        chunks=chunks,
        response_mode='analysis',
    )
    assert result is not None
    answer, _, metrics = result
    output_contract_check = metrics.get('output_contract_check')
    assert isinstance(output_contract_check, dict)
    assert output_contract_check.get('passed') is True
    assert len(answer.split()) <= 520


def test_strict_composer_cross_document_synthesis_meets_length_and_missing_evidence_shape() -> None:
    question = (
        'Create a long-form cross-document synthesis of indexed finance-related records with sections in this exact order: '
        '## Coverage Snapshot, ## Amounts and Trends, ## Inconsistencies, ## Missing Evidence, ## Follow-up Plan. '
        'Under ## Amounts and Trends include one markdown table with columns: Year, Document Group, Key Amount, Evidence Note. '
        'Under ## Missing Evidence include exactly 5 bullets and each bullet must name one missing item and one verification action.'
    )
    chunks = [
        {
            'filename': 'sample-evidence.pdf',
            'file_path': '/docs/sample-evidence.pdf',
            'chunk_text': 'sample evidence text with year and values',
            'score': 0.91,
        },
        {
            'filename': 'secondary-evidence.pdf',
            'file_path': '/docs/secondary-evidence.pdf',
            'chunk_text': 'secondary evidence text',
            'score': 0.88,
        },
    ]
    result = try_compose_strict_contract_answer(
        question=question,
        chunks=chunks,
        response_mode='analysis',
    )
    assert result is not None
    answer, _, metrics = result
    assert len(answer.split()) >= 450
    assert metrics.get('strict_composer_family') == 'research_cross_document_synthesis'
    output_contract_check = metrics.get('output_contract_check')
    assert isinstance(output_contract_check, dict)
    assert output_contract_check.get('passed') is True
    assert '## Missing Evidence' in answer

    section = answer.split('## Missing Evidence', 1)[1].split('## Follow-up Plan', 1)[0]
    bullet_lines = [line for line in section.splitlines() if line.startswith('- ')]
    assert len(bullet_lines) == 5
