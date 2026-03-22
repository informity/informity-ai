import pytest

from informity.llm.intent_router import (
    IntentPrediction,
    get_intent_router,
    set_intent_router_for_testing,
)
from informity.llm.query_classifier import QueryClassification, classify_query


class _FakeIntentRouter:
    def classify_intent(self, query: str) -> IntentPrediction:
        lowered = query.casefold()
        if 'how many' in lowered or 'what file types' in lowered or 'what years' in lowered:
            return IntentPrediction('metadata', 0.95, [('metadata', 0.95)], ['test_fake_router'])
        if lowered.strip() == 'hello' or 'information is available' in lowered:
            return IntentPrediction('simple', 0.9, [('simple', 0.9)], ['test_fake_router'])
        if 'what kind of documents' in lowered:
            return IntentPrediction('metadata', 0.9, [('metadata', 0.9)], ['test_fake_router'])
        if 'compare' in lowered:
            return IntentPrediction('coverage', 0.9, [('coverage', 0.9)], ['test_fake_router'])
        return IntentPrediction('focused', 0.85, [('focused', 0.85)], ['test_fake_router'])


@pytest.fixture(autouse=True)
def _fake_router() -> None:
    original = get_intent_router()
    set_intent_router_for_testing(_FakeIntentRouter())
    try:
        yield
    finally:
        set_intent_router_for_testing(original)


def test_metadata_route_for_count_query() -> None:
    result = classify_query('How many PDF files from 2023 are indexed?')
    assert isinstance(result, QueryClassification)
    assert result.intent == 'metadata'
    assert result.route_candidate == 'metadata_inventory'
    assert result.year_filter == 2023
    assert result.file_type_filter == '.pdf'
    assert result.is_metadata_query is True


def test_simple_route_for_greeting() -> None:
    result = classify_query('hello')
    assert result.intent == 'simple'
    assert result.route_candidate == 'clarification_or_disambiguation'


def test_rag_route_for_domain_question() -> None:
    result = classify_query('Summarize lender discrepancies by year from the indexed records.')
    assert result.intent == 'coverage'
    assert result.route_candidate == 'cross_document_synthesis'
    assert result.is_metadata_query is False


def test_corpus_capability_query_routes_to_metadata() -> None:
    result = classify_query('What kind of documents do you have indexed?')
    assert result.intent == 'metadata'
    assert result.route_candidate == 'metadata_inventory'
    assert result.is_metadata_query is True


def test_general_capabilities_query_stays_simple() -> None:
    result = classify_query('Can you help me understand what information is available?')
    assert result.intent == 'simple'
    assert result.route_candidate == 'clarification_or_disambiguation'


def test_filename_slot_extraction() -> None:
    result = classify_query('Summarize content in sample-lender-statement.pdf')
    assert result.filename_filter == 'sample-lender-statement.pdf'


def test_structured_schema_overrides_metadata_prediction_to_rag() -> None:
    class _MetadataOnlyRouter:
        def classify_intent(self, _query: str) -> IntentPrediction:
            return IntentPrediction('metadata', 0.9, [('metadata', 0.9)], ['forced_metadata'])

    original = get_intent_router()
    set_intent_router_for_testing(_MetadataOnlyRouter())
    try:
        result = classify_query(
            'Compare numeric amounts across indexed records and output only a markdown table '
            'with columns: Line Item, Amount, Source Snippet.'
        )
    finally:
        set_intent_router_for_testing(original)

    assert result.intent in ('focused', 'coverage')
    assert result.route_candidate in ('structured_field_extraction', 'comparative_analysis')
    assert result.deterministic_override is True
    assert 'deterministic_override_structured_schema_request' in result.reason_codes


def test_inventory_capability_remains_metadata_without_override() -> None:
    class _MetadataOnlyRouter:
        def classify_intent(self, _query: str) -> IntentPrediction:
            return IntentPrediction('metadata', 0.9, [('metadata', 0.9)], ['forced_metadata'])

    original = get_intent_router()
    set_intent_router_for_testing(_MetadataOnlyRouter())
    try:
        result = classify_query('What kind of documents do you have indexed?')
    finally:
        set_intent_router_for_testing(original)

    assert result.intent == 'metadata'
    assert result.route_candidate == 'metadata_inventory'
    assert result.deterministic_override is False


def test_inventory_with_evidence_request_overrides_to_rag() -> None:
    class _MetadataOnlyRouter:
        def classify_intent(self, _query: str) -> IntentPrediction:
            return IntentPrediction('metadata', 0.9, [('metadata', 0.9)], ['forced_metadata'])

    original = get_intent_router()
    set_intent_router_for_testing(_MetadataOnlyRouter())
    try:
        result = classify_query(
            'Which indexed documents contain numeric amounts or financial figures? '
            'List the files and the key amounts found.'
        )
    finally:
        set_intent_router_for_testing(original)

    assert result.intent in ('focused', 'coverage')
    assert result.route_candidate in ('targeted_fact_lookup', 'cross_document_synthesis')
    assert result.deterministic_override is True
    assert 'deterministic_override_inventory_with_evidence_request' in result.reason_codes


def test_metadata_content_request_overrides_to_coverage() -> None:
    class _MetadataOnlyRouter:
        def classify_intent(self, _query: str) -> IntentPrediction:
            return IntentPrediction('metadata', 0.9, [('metadata', 0.9)], ['forced_metadata'])

    original = get_intent_router()
    set_intent_router_for_testing(_MetadataOnlyRouter())
    try:
        result = classify_query('Are there contradictions or conflicts between documents in the index?')
    finally:
        set_intent_router_for_testing(original)

    assert result.intent == 'coverage'
    assert result.route_candidate == 'cross_document_synthesis'
    assert result.deterministic_override is True
    assert 'deterministic_override_metadata_content_request' in result.reason_codes


def test_coverage_narrow_scope_overrides_to_focused() -> None:
    class _CoverageOnlyRouter:
        def classify_intent(self, _query: str) -> IntentPrediction:
            return IntentPrediction('coverage', 0.9, [('coverage', 0.9)], ['forced_coverage'])

    original = get_intent_router()
    set_intent_router_for_testing(_CoverageOnlyRouter())
    try:
        result = classify_query('What does the 2020 property tax receipt contain? Summarize the key fields.')
    finally:
        set_intent_router_for_testing(original)

    assert result.intent == 'focused'
    assert result.route_candidate == 'targeted_fact_lookup'
    assert result.deterministic_override is True
    assert 'deterministic_override_narrow_scope_to_focused' in result.reason_codes


def test_focused_plural_scope_analysis_overrides_to_coverage() -> None:
    class _FocusedOnlyRouter:
        def classify_intent(self, _query: str) -> IntentPrediction:
            return IntentPrediction('focused', 0.9, [('focused', 0.9)], ['forced_focused'])

    original = get_intent_router()
    set_intent_router_for_testing(_FocusedOnlyRouter())
    try:
        result = classify_query('Summarize the key findings from 2022 records only.')
    finally:
        set_intent_router_for_testing(original)

    assert result.intent == 'coverage'
    assert result.route_candidate == 'cross_document_synthesis'
    assert result.deterministic_override is True
    assert 'deterministic_override_plural_scope_analysis_request' in result.reason_codes


def test_focused_structured_schema_with_corpus_scope_overrides_to_coverage() -> None:
    class _FocusedOnlyRouter:
        def classify_intent(self, _query: str) -> IntentPrediction:
            return IntentPrediction('focused', 0.9, [('focused', 0.9)], ['forced_focused'])

    original = get_intent_router()
    set_intent_router_for_testing(_FocusedOnlyRouter())
    try:
        result = classify_query(
            'Using only indexed documents, output a markdown table with columns: Field, Value, Source File.'
        )
    finally:
        set_intent_router_for_testing(original)

    assert result.intent == 'coverage'
    assert result.route_candidate == 'comparative_analysis'
    assert result.response_shape == 'metadata_table'
    assert result.deterministic_override is True
    assert 'deterministic_override_structured_schema_for_coverage_scope' in result.reason_codes


def test_coverage_multi_year_prompt_sets_aggregate_subtype() -> None:
    class _CoverageOnlyRouter:
        def classify_intent(self, _query: str) -> IntentPrediction:
            return IntentPrediction('coverage', 0.9, [('coverage', 0.9)], ['forced_coverage'])

    original = get_intent_router()
    set_intent_router_for_testing(_CoverageOnlyRouter())
    try:
        result = classify_query(
            'Produce a year-over-year evidence matrix with findings by year across indexed records.'
        )
    finally:
        set_intent_router_for_testing(original)

    assert result.intent == 'coverage'
    assert result.subtype == 'aggregate_by_period'
    assert result.group_by == 'year'
    assert result.has_multi_year_scope is True
    assert 'deterministic_override_coverage_year_aggregate_subtype' in result.reason_codes
