from informity.api import routes_chat
from informity.llm.contract_gate import ContractSpec, validate_contract
from informity.llm.query_classifier import QueryClassification


def test_resolve_completion_state_treats_queue_timeout_as_terminal() -> None:
    completion_mode, has_remaining_scope = routes_chat._resolve_completion_state(
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
    assert routes_chat._resolve_chat_mode(None) == 'researcher'
    assert routes_chat._resolve_chat_mode('') == 'researcher'
    assert routes_chat._resolve_chat_mode('invalid') == 'researcher'


def test_resolve_chat_mode_accepts_assistant_and_researcher() -> None:
    assert routes_chat._resolve_chat_mode('assistant') == 'assistant'
    assert routes_chat._resolve_chat_mode('researcher') == 'researcher'
    assert routes_chat._resolve_chat_mode('Assistant') == 'assistant'


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
