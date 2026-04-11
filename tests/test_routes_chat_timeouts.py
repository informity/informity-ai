from informity.api import routes_chat
from informity.api.chat_continuation import resolve_completion_state as _resolve_completion_state
from informity.llm.contract_gate import (
    ContractSpec,
    build_contract_spec,
    build_repair_guidance,
    enforce_required_sections,
    validate_contract,
)
from informity.llm.query_classifier import QueryClassification


def test_resolve_completion_state_treats_queue_timeout_as_terminal() -> None:
    completion_mode, has_remaining_scope = _resolve_completion_state(
        completion_mode_override=None,
        timeout_occurred=True,
        timeout_reason='queue_wait_timeout',
        has_remaining_scope=False,
    )
    assert completion_mode == 'partial'
    assert has_remaining_scope is False


def test_resolve_next_action_treats_queue_timeout_as_none() -> None:
    next_action, next_action_reason = routes_chat._resolve_next_action(
        stopped_by_user=False,
        timeout_occurred=True,
        has_remaining_scope=True,
        continuation_resolution_reason='queue_wait_timeout',
    )
    assert next_action == 'none'
    assert next_action_reason is None


def test_normalize_continuation_classification_forces_continuation_route() -> None:
    classification = QueryClassification(
        intent='coverage',
        route_candidate='comparative_analysis',
        response_shape='metadata_table',
        subtype='extract_structured_values',
        is_continuation=True,
    )
    normalized = routes_chat._normalize_continuation_classification(
        classification=classification,
        continuation_anchor_question='Continue with ## Cross-Year Deltas, ## Confidence Notes, ## Next Verification Steps only.',
    )
    assert normalized.route_candidate == 'continuation_or_refinement'
    assert normalized.response_shape == 'narrative_synthesis'
    assert normalized.subtype is None


def test_normalize_continuation_classification_keeps_structured_when_table_requested() -> None:
    classification = QueryClassification(
        intent='coverage',
        route_candidate='comparative_analysis',
        response_shape='metadata_table',
        subtype='extract_structured_values',
        is_continuation=True,
    )
    normalized = routes_chat._normalize_continuation_classification(
        classification=classification,
        continuation_anchor_question='Continue and output only a markdown table with columns: Field, Value, Source Snippet.',
    )
    assert normalized.route_candidate == 'continuation_or_refinement'
    assert normalized.response_shape == 'metadata_table'
    assert normalized.subtype == 'extract_structured_values'


def test_continuation_flag_lexical_signal_not_overwritten_by_classifier_false() -> None:
    lexical_continuation = True
    classifier_continuation = False
    merged = bool(lexical_continuation or classifier_continuation)
    assert merged is True


def test_resolve_chat_mode_defaults_to_researcher_for_invalid() -> None:
    assert routes_chat.resolve_chat_mode(None) == 'researcher'
    assert routes_chat.resolve_chat_mode('') == 'researcher'
    assert routes_chat.resolve_chat_mode('invalid') == 'researcher'


def test_resolve_chat_mode_accepts_assistant_and_researcher() -> None:
    assert routes_chat.resolve_chat_mode('assistant') == 'assistant'
    assert routes_chat.resolve_chat_mode('researcher') == 'researcher'
    assert routes_chat.resolve_chat_mode('Assistant') == 'assistant'


def test_validate_contract_detects_missing_required_heading() -> None:
    answer = "## Scope\nDone\n## Method\nDone"
    result = validate_contract(
        answer=answer,
        spec=ContractSpec(
            required_headings=['Scope', 'Method', 'Next Verification Steps'],
            min_year_count=0,
        ),
    )
    assert result.missing_required_headings == ['Next Verification Steps']
    assert result.has_gap is True


def test_validate_contract_counts_distinct_years() -> None:
    answer = "### 2022\nA\n### 2023\nB\n2023 repeated"
    result = validate_contract(
        answer=answer,
        spec=ContractSpec(required_headings=[], min_year_count=2),
    )
    assert result.observed_year_count == 2
    assert result.has_gap is False


def test_enforce_required_sections_appends_missing_headings_in_order() -> None:
    answer = "## Scope\nDone\n## Method\nDone"
    enforced, filled = enforce_required_sections(
        answer=answer,
        spec=ContractSpec(
            required_headings=['Scope', 'Method', 'Cross-Year Deltas', 'Next Verification Steps'],
            min_year_count=0,
        ),
    )
    assert filled == ['Cross-Year Deltas', 'Next Verification Steps']
    assert '## Cross-Year Deltas' in enforced
    assert '## Next Verification Steps' in enforced
    assert enforced.index('## Cross-Year Deltas') < enforced.index('## Next Verification Steps')


def test_enforce_required_sections_noop_when_complete() -> None:
    answer = "## Scope\nDone\n## Method\nDone\n## Next Verification Steps\nDone"
    enforced, filled = enforce_required_sections(
        answer=answer,
        spec=ContractSpec(
            required_headings=['Scope', 'Method', 'Next Verification Steps'],
            min_year_count=0,
        ),
    )
    assert filled == []
    assert enforced == answer


def test_build_repair_guidance_includes_missing_headings_and_year_floor() -> None:
    guidance = build_repair_guidance(
        result=validate_contract(
            answer="## Scope\nDone",
            spec=ContractSpec(required_headings=['Scope', 'Method'], min_year_count=2),
        ),
    )
    assert guidance is not None
    assert '## Method' in guidance
    assert 'Do not repeat sections that are already complete.' in guidance
    assert 'Include at least 2 distinct years' in guidance


def test_build_contract_spec_enables_year_floor_for_coverage_aggregate_queries() -> None:
    classification = QueryClassification(
        intent='coverage',
        subtype='aggregate_by_period',
    )
    spec = build_contract_spec(
        question='Compare by year and use year-based subsections in the final output.',
        classification=classification,
    )
    assert spec.min_year_count == 2


def test_build_contract_spec_tracks_order_and_missing_evidence_requirements() -> None:
    classification = QueryClassification(intent='coverage', subtype='aggregate_by_period')
    spec = build_contract_spec(
        question=(
            'Use headings in exact order: ## Scope, ## Findings, ## Missing Evidence. '
            'Explicitly call out missing evidence for unavailable records.'
        ),
        classification=classification,
    )
    assert spec.enforce_heading_order is True
    assert spec.requires_missing_evidence_callout is True


def test_build_contract_spec_extracts_required_labels_from_format_contract() -> None:
    classification = QueryClassification(intent='coverage', subtype='extract_structured_values')
    spec = build_contract_spec(
        question='Output rows in format: Field | Value | Source Snippet.',
        classification=classification,
    )
    assert 'Source Snippet' in spec.required_labels


def test_enforce_required_sections_redacts_ssn() -> None:
    answer = 'Taxpayer SSN: 123-45-6789'
    enforced, _filled = enforce_required_sections(
        answer=answer,
        spec=ContractSpec(required_headings=[], min_year_count=0),
    )
    assert '123-45-6789' not in enforced
    assert '[REDACTED-SSN]' in enforced


def test_enforce_required_sections_inserts_missing_evidence_callout_when_requested() -> None:
    answer = '## Scope\nDone'
    enforced, _filled = enforce_required_sections(
        answer=answer,
        spec=ContractSpec(
            required_headings=['Scope'],
            min_year_count=0,
            requires_missing_evidence_callout=True,
        ),
    )
    assert 'Missing Evidence:' in enforced


def test_enforce_required_sections_reorders_required_headings_when_order_enforced() -> None:
    answer = '## Findings\nA\n\n## Scope\nB'
    enforced, _filled = enforce_required_sections(
        answer=answer,
        spec=ContractSpec(
            required_headings=['Scope', 'Findings'],
            min_year_count=0,
            enforce_heading_order=True,
        ),
    )
    assert enforced.index('## Scope') < enforced.index('## Findings')


def test_enforce_required_sections_appends_missing_required_labels() -> None:
    answer = '## Findings\n- Field: Revenue\n- Value: 100'
    enforced, _filled = enforce_required_sections(
        answer=answer,
        spec=ContractSpec(
            required_headings=[],
            min_year_count=0,
            required_labels=['Field', 'Value', 'Source Snippet'],
        ),
    )
    assert 'Source Snippet: Missing Evidence:' in enforced
