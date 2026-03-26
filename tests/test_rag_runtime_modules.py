from informity.config import settings
from informity.db.models import ChatMessage
from informity.llm.rag_runtime import generation_closeout as _generation_closeout
from informity.llm.rag_runtime import generation_runtime as _generation_runtime
from informity.llm.rag_runtime.retrieval_pipeline import _build_focused_anchor_recovery_query
from informity.llm.rag_runtime.retrieval_validation import (
    _apply_coverage_evidence_floor_override,
    _build_continuation_retrieval_query,
    _derive_continuation_source_terms,
    _evaluate_continuation_anchor_gate,
    _evaluate_source_diversity_gate,
    _extract_prior_has_remaining_scope,
)
from informity.llm.rag_runtime.structured_numeric import (
    _build_finance_conflict_placeholder_bullet,
    _derive_format_requirements,
    _evidence_overlap_tokens,
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


def test_structured_numeric_derives_include_clause_required_terms() -> None:
    requirements = _derive_format_requirements(
        'Compare records from different years and include conflict statement, involved documents, '
        'conflicting values, and likely reason grounded in evidence.'
    )
    assert 'include term: conflict' in requirements
    assert 'include term: documents' in requirements
    assert 'include term: values' in requirements
    assert 'include term: reason' in requirements
    assert 'include term: evidence' in requirements


def test_structured_numeric_derives_cover_clause_required_terms() -> None:
    requirements = _derive_format_requirements(
        'Summarize cross-year changes. Cover biggest increase, biggest decrease, and ambiguous deltas, with evidence.'
    )
    assert 'include term: increase' in requirements
    assert 'include term: decrease' in requirements
    assert 'include term: evidence' in requirements


def test_structured_numeric_uses_action_hints_for_enumeration() -> None:
    requirements = _derive_format_requirements(
        'Summarize key findings.',
        action_hints={'should_enumerate': True},
    )
    assert any('numbered or bulleted list' in requirement for requirement in requirements)


def test_structured_numeric_uses_action_hints_for_comparison() -> None:
    requirements = _derive_format_requirements(
        'Summarize key findings.',
        action_hints={'should_compare': True},
    )
    assert any('side-by-side or structured comparison format' in requirement for requirement in requirements)


def test_structured_numeric_action_hints_do_not_duplicate_existing_requirements() -> None:
    requirements = _derive_format_requirements(
        'Provide findings by year and compare key changes across all indexed records.',
        action_hints={'should_compare': True},
    )
    comparison_requirements = [
        requirement for requirement in requirements
        if 'side-by-side or structured comparison format' in requirement
    ]
    assert len(comparison_requirements) == 1


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


def test_retrieval_pipeline_builds_focused_anchor_recovery_query() -> None:
    query = _build_focused_anchor_recovery_query(
        question='What does the 2020 property tax receipt contain?',
        source_terms=['2020 property tax receipt'],
    )
    assert isinstance(query, str)
    assert 'year-specific' in query
    assert '2020 property tax receipt' in query


def test_generation_runtime_has_remaining_scope_false_for_terminal_timeout() -> None:
    assert _generation_runtime._has_remaining_scope(
        timeout_reason='queue_wait_timeout',
        stream_recovery_reason=None,
        generation_skipped=False,
        applied_degradations=[],
    ) is False


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


def test_generation_closeout_source_references_filter_to_used_chunks() -> None:
    chunks = [
        {
            'filename': 'tax_2024.pdf',
            'file_path': '/docs/tax_2024.pdf',
            'chunk_text': 'Property tax receipt shows total paid 2024 county bill.',
            'score': 0.81,
        },
        {
            'filename': 'bank.pdf',
            'file_path': '/docs/bank.pdf',
            'chunk_text': 'Checking account transfer history and unrelated debit card rows.',
            'score': 0.74,
        },
    ]
    sources = _generation_closeout.build_source_references(
        chunks=chunks,
        answer_text='The property tax receipt confirms total paid for 2024.',
        truncate_preview_fn=lambda text: text,
        normalize_relevance_score_fn=lambda score: float(score),
    )
    assert len(sources) == 1
    assert sources[0].filename == 'tax_2024.pdf'


def test_generation_closeout_source_references_fallback_to_top_when_no_overlap() -> None:
    chunks = [
        {
            'filename': f'doc_{idx}.pdf',
            'file_path': f'/docs/doc_{idx}.pdf',
            'chunk_text': f'Chunk text {idx} with archive metadata and unrelated content.',
            'score': 0.9 - idx * 0.01,
        }
        for idx in range(7)
    ]
    sources = _generation_closeout.build_source_references(
        chunks=chunks,
        answer_text='This final answer discusses topics absent from retrieved chunks.',
        truncate_preview_fn=lambda text: text,
        normalize_relevance_score_fn=lambda score: float(score),
    )
    assert len(sources) == 5
    assert sources[0].filename == 'doc_0.pdf'
    assert sources[-1].filename == 'doc_4.pdf'


def test_generation_closeout_source_references_keep_all_when_answer_empty() -> None:
    chunks = [
        {
            'filename': 'a.pdf',
            'file_path': '/docs/a.pdf',
            'chunk_text': 'Alpha chunk',
            'score': 0.5,
        },
        {
            'filename': 'b.pdf',
            'file_path': '/docs/b.pdf',
            'chunk_text': 'Beta chunk',
            'score': 0.4,
        },
    ]
    sources = _generation_closeout.build_source_references(
        chunks=chunks,
        answer_text='',
        truncate_preview_fn=lambda text: text,
        normalize_relevance_score_fn=lambda score: float(score),
    )
    assert len(sources) == 2
    assert {source.filename for source in sources} == {'a.pdf', 'b.pdf'}
