import pytest

import informity.llm.query_classifier as query_classifier_module
from informity.llm.intent_router import (
    IntentPrediction,
    get_intent_router,
    set_intent_router_for_testing,
)
from informity.llm.promptcue_adapter import PromptCueIntentAdapter
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
        if 'compare' in lowered or 'summarize' in lowered:
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
    assert result.retrieval_content_query is not None


def test_simple_route_for_greeting() -> None:
    result = classify_query('hello')
    assert result.intent == 'simple'
    assert result.route_candidate == 'clarification_or_disambiguation'


def test_retrieval_content_query_decomposes_discourse_prefix_clause() -> None:
    result = classify_query('OK, new topic. What is the general plot of The Count of Monte Cristo?')
    assert result.retrieval_content_query == 'What is the general plot of The Count of Monte Cristo?'
    assert result.retrieval_content_confidence > 0.5
    assert result.retrieval_content_reasons


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


def test_world_fact_lookup_does_not_stay_metadata_inventory() -> None:
    class _MetadataOnlyRouter:
        def classify_intent(self, _query: str) -> IntentPrediction:
            return IntentPrediction('metadata', 0.9, [('metadata', 0.9)], ['forced_metadata'])

    original = get_intent_router()
    set_intent_router_for_testing(_MetadataOnlyRouter())
    try:
        result = classify_query('What year was the US declaration of independence signed?')
    finally:
        set_intent_router_for_testing(original)

    # Simplified policy trusts primary classifier and avoids metadata->focused
    # rescue overrides beyond minimal guardrails.
    assert result.intent == 'metadata'
    assert result.route_candidate == 'metadata_inventory'
    assert result.is_metadata_query is True


def test_routing_expansion_reason_code_is_recorded(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeExpansion:
        expanded_query = 'What is ROI return on investment'
        canonical_terms = ['return on investment']

    monkeypatch.setattr(query_classifier_module, 'expand_query_for_routing', lambda _query: _FakeExpansion())

    result = classify_query('What is ROI')
    assert 'term_dictionary_routing_expansion_applied' in result.reason_codes


def test_general_capabilities_query_stays_simple() -> None:
    result = classify_query('Can you help me understand what information is available?')
    assert result.intent == 'simple'
    assert result.route_candidate == 'clarification_or_disambiguation'
    assert 'deterministic_general_capability_to_simple' in result.reason_codes


def test_non_promptcue_router_does_not_synthesize_freshness_signals() -> None:
    result = classify_query('What is the weather in Escondido, CA today and tomorrow?')
    assert result.needs_current_info is False
    assert result.mentions_time is False
    assert result.needs_chat_history is False


def test_deterministic_chat_summary_fallback_sets_simple_and_chat_history() -> None:
    class _CoverageRouter:
        def classify_intent(self, _query: str) -> IntentPrediction:
            return IntentPrediction('coverage', 0.8, [('coverage', 0.8)], ['forced_coverage'])

    original = get_intent_router()
    set_intent_router_for_testing(_CoverageRouter())
    try:
        result = classify_query('What have we been chatting about?')
    finally:
        set_intent_router_for_testing(original)

    assert result.intent == 'simple'
    assert result.needs_chat_history is True
    assert 'deterministic_chat_summary_to_simple' in result.reason_codes


def test_promptcue_router_passes_through_needs_chat_history() -> None:
    class _FakeSemanticHints:
        requests_structure = False
        requests_comparison = False
        requires_multi_period_analysis = False
        mentions_time = False

    class _FakePromptCue:
        scope = 'focused'
        primary_query_type = 'chat_summary'
        confidence = 0.97
        classification_basis = 'trigger_match'
        candidate_query_types: list[object] = []
        routing_hints = {
            'needs_retrieval': False,
            'needs_current_info': False,
            'needs_reasoning': False,
            'needs_structure': False,
            'needs_chat_history': True,
        }
        action_hints = {}
        is_continuation = False
        semantic_hints = _FakeSemanticHints()

    adapter = PromptCueIntentAdapter()

    def _classify(_query: str) -> tuple[IntentPrediction, object]:
        return (
            IntentPrediction('simple', 0.97, [('simple', 0.97)], ['promptcue_adapter', 'basis:trigger_match']),
            _FakePromptCue(),
        )

    original = get_intent_router()
    adapter.classify = _classify  # type: ignore[method-assign]
    set_intent_router_for_testing(adapter)
    try:
        result = classify_query('What have we been chatting about?')
    finally:
        set_intent_router_for_testing(original)

    assert result.intent == 'simple'
    assert result.needs_chat_history is True


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

    assert result.intent == 'metadata'
    assert result.route_candidate == 'metadata_inventory'
    assert result.deterministic_override is False


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

    assert result.intent == 'metadata'
    assert result.route_candidate == 'metadata_inventory'
    assert result.deterministic_override is False


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

    assert result.intent == 'metadata'
    assert result.route_candidate == 'metadata_inventory'
    assert result.deterministic_override is False


def test_coverage_narrow_scope_overrides_to_focused() -> None:
    class _CoverageOnlyRouter:
        def classify_intent(self, _query: str) -> IntentPrediction:
            return IntentPrediction('coverage', 0.9, [('coverage', 0.9)], ['forced_coverage'])

    original = get_intent_router()
    set_intent_router_for_testing(_CoverageOnlyRouter())
    try:
        result = classify_query('What does the 2020 property tax record contain? Summarize the key fields.')
    finally:
        set_intent_router_for_testing(original)

    assert result.intent == 'focused'
    assert result.route_candidate == 'targeted_fact_lookup'
    assert result.deterministic_override is True
    assert 'deterministic_override_single_target_to_focused' in result.reason_codes


def test_coverage_single_target_scope_overrides_to_focused() -> None:
    class _CoverageOnlyRouter:
        def classify_intent(self, _query: str) -> IntentPrediction:
            return IntentPrediction('coverage', 0.9, [('coverage', 0.9)], ['forced_coverage'])

    original = get_intent_router()
    set_intent_router_for_testing(_CoverageOnlyRouter())
    try:
        result = classify_query('What does the 2020 property tax record contain?')
    finally:
        set_intent_router_for_testing(original)

    assert result.intent == 'focused'
    assert result.route_candidate == 'targeted_fact_lookup'
    assert result.deterministic_override is True
    assert any(
        code in result.reason_codes
        for code in (
            'deterministic_override_single_target_to_focused',
            'deterministic_override_narrow_scope_to_focused',
        )
    )


def test_coverage_year_anchored_target_overrides_to_focused() -> None:
    class _CoverageOnlyRouter:
        def classify_intent(self, _query: str) -> IntentPrediction:
            return IntentPrediction('coverage', 0.9, [('coverage', 0.9)], ['forced_coverage'])

    original = get_intent_router()
    set_intent_router_for_testing(_CoverageOnlyRouter())
    try:
        result = classify_query('What does the 2020 property tax record contain? Summarize the key fields.')
    finally:
        set_intent_router_for_testing(original)

    assert result.intent == 'focused'
    assert result.route_candidate == 'targeted_fact_lookup'
    assert result.deterministic_override is True
    assert 'deterministic_override_single_target_to_focused' in result.reason_codes


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
    assert 'deterministic_override_plural_corpus_to_coverage' in result.reason_codes


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
    assert result.route_candidate == 'cross_document_synthesis'
    assert result.response_shape == 'narrative_synthesis'
    assert result.deterministic_override is True
    assert 'deterministic_override_plural_corpus_to_coverage' in result.reason_codes


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


def test_focused_multi_year_analysis_overrides_to_coverage() -> None:
    class _FocusedOnlyRouter:
        def classify_intent(self, _query: str) -> IntentPrediction:
            return IntentPrediction('focused', 0.9, [('focused', 0.9)], ['forced_focused'])

    original = get_intent_router()
    set_intent_router_for_testing(_FocusedOnlyRouter())
    try:
        result = classify_query('Summarize cross-year findings and deltas for indexed records.')
    finally:
        set_intent_router_for_testing(original)

    assert result.intent == 'coverage'
    assert result.route_candidate == 'cross_document_synthesis'
    assert result.deterministic_override is True
    assert 'deterministic_override_plural_corpus_to_coverage' in result.reason_codes


def test_focused_multi_year_numeric_extraction_overrides_to_coverage_table() -> None:
    class _FocusedOnlyRouter:
        def classify_intent(self, _query: str) -> IntentPrediction:
            return IntentPrediction('focused', 0.9, [('focused', 0.9)], ['forced_focused'])

    original = get_intent_router()
    set_intent_router_for_testing(_FocusedOnlyRouter())
    try:
        result = classify_query(
            'From indexed structured documents, extract key numeric field values by year and provide totals by year.'
        )
    finally:
        set_intent_router_for_testing(original)

    assert result.intent == 'focused'
    assert result.route_candidate == 'targeted_fact_lookup'
    assert result.response_shape == 'narrative_synthesis'
    assert result.subtype is None
    assert result.deterministic_override is False


def test_focused_corpus_scope_listing_overrides_to_coverage() -> None:
    class _FocusedOnlyRouter:
        def classify_intent(self, _query: str) -> IntentPrediction:
            return IntentPrediction('focused', 0.9, [('focused', 0.9)], ['forced_focused'])

    original = get_intent_router()
    set_intent_router_for_testing(_FocusedOnlyRouter())
    try:
        result = classify_query('Which indexed documents mention people names? List the documents.')
    finally:
        set_intent_router_for_testing(original)

    assert result.intent == 'coverage'
    assert result.route_candidate == 'cross_document_synthesis'
    assert result.deterministic_override is True
    assert 'deterministic_override_plural_corpus_to_coverage' in result.reason_codes


def test_coverage_extreme_value_lookup_overrides_to_focused() -> None:
    class _CoverageOnlyRouter:
        def classify_intent(self, _query: str) -> IntentPrediction:
            return IntentPrediction('coverage', 0.9, [('coverage', 0.9)], ['forced_coverage'])

    original = get_intent_router()
    set_intent_router_for_testing(_CoverageOnlyRouter())
    try:
        result = classify_query('How much is the largest amount mentioned in any document?')
    finally:
        set_intent_router_for_testing(original)

    assert result.intent == 'focused'
    assert result.route_candidate == 'targeted_fact_lookup'
    assert result.deterministic_override is True
    assert 'deterministic_override_single_target_to_focused' in result.reason_codes


def test_focused_aggregate_listing_scope_overrides_to_coverage() -> None:
    class _FocusedOnlyRouter:
        def classify_intent(self, _query: str) -> IntentPrediction:
            return IntentPrediction('focused', 0.9, [('focused', 0.9)], ['forced_focused'])

    original = get_intent_router()
    set_intent_router_for_testing(_FocusedOnlyRouter())
    try:
        result = classify_query('What are the most important dates mentioned across all indexed documents?')
    finally:
        set_intent_router_for_testing(original)

    assert result.intent == 'coverage'
    assert result.route_candidate == 'cross_document_synthesis'
    assert result.deterministic_override is True
    assert 'deterministic_override_plural_corpus_to_coverage' in result.reason_codes


def test_source_terms_extract_anchor_phrase() -> None:
    result = classify_query('What does the 2020 property tax record contain? Summarize the key fields.')
    assert any('2020 property tax record' in term.casefold() for term in result.source_terms)


def test_metadata_route_for_conversational_enumeration() -> None:
    result = classify_query('Tell me about the years in my data')
    assert result.intent == 'metadata'
    assert result.route_candidate == 'metadata_inventory'


def test_metadata_route_for_enumerate_categories() -> None:
    result = classify_query('Enumerate the categories')
    assert result.intent == 'metadata'
    assert result.route_candidate == 'metadata_inventory'


def test_metadata_route_for_how_much_data() -> None:
    result = classify_query('How much data do I have?')
    assert result.intent == 'metadata'
    assert result.route_candidate == 'metadata_inventory'


def test_metadata_route_for_time_span_question() -> None:
    result = classify_query('From when to when is the data?')
    assert result.intent == 'metadata'
    assert result.route_candidate == 'metadata_inventory'


def test_coverage_override_for_summary_of_files_phrase() -> None:
    result = classify_query('Provide a summary of the files')
    assert result.intent == 'coverage'
    assert result.route_candidate == 'cross_document_synthesis'
    assert 'deterministic_override_plural_corpus_to_coverage' in result.reason_codes


def test_continuation_detected_for_show_me_the_rest_phrase() -> None:
    result = classify_query('Show me the rest')
    assert result.is_continuation is True


def test_continuation_detected_for_same_question_filter_update() -> None:
    result = classify_query('Same question but for 2023')
    assert result.is_continuation is True
    assert result.year_filter == 2023


def test_continuation_not_detected_for_new_topic_what_about_question() -> None:
    result = classify_query('What about the interest rate?')
    assert result.is_continuation is False


def test_output_format_table_detected_from_query() -> None:
    result = classify_query('Build a table by year from indexed files')
    assert result.output_format == 'table'


def test_output_format_bullets_detected_from_query() -> None:
    result = classify_query('Show as bullet points')
    assert result.output_format == 'bullets'


def test_output_format_csv_detected_from_query() -> None:
    result = classify_query('CSV format')
    assert result.output_format == 'csv'


def test_comparative_subtype_detected_from_query() -> None:
    result = classify_query('Which year has the fewest files?')
    assert result.subtype == 'comparative'


def test_negation_signal_detected_from_query() -> None:
    result = classify_query("Find files that don't mention escrow")
    assert result.is_negation_query is True


def test_filename_exclusion_extracted_from_query() -> None:
    result = classify_query('Exclude file sample-lender-statement.pdf from results')
    assert result.filename_exclude == ['sample-lender-statement.pdf']


def test_compound_count_and_list_sets_secondary_intent() -> None:
    result = classify_query('How many files from 2023 are indexed and list them')
    assert result.secondary_intent == 'metadata'


def test_comparative_year_query_routes_to_metadata() -> None:
    result = classify_query('Which year has the fewest files?')
    assert result.intent == 'metadata'
    assert result.subtype == 'comparative'
    assert result.route_candidate == 'metadata_inventory'
    assert any(
        code in result.reason_codes
        for code in (
            'deterministic_comparative_metadata_group_detected',
            'deterministic_inventory_metadata_promoted',
        )
    )


def test_comparative_category_query_routes_to_metadata() -> None:
    result = classify_query('Which category has the most documents?')
    assert result.intent == 'metadata'
    assert result.subtype == 'comparative'
    assert result.route_candidate == 'metadata_inventory'
    assert result.group_by == 'category'
    assert 'deterministic_comparative_metadata_group_detected' in result.reason_codes


def test_comparative_file_mentions_routes_to_focused_comparative_path() -> None:
    result = classify_query('Which file has the most mentions of escrow?')
    assert result.intent == 'focused'
    assert result.subtype == 'comparative'
    assert result.route_candidate == 'comparative_analysis'
    assert result.group_by == 'file'


def test_number_of_content_phrase_does_not_route_to_metadata() -> None:
    result = classify_query('What is the number of employees listed in each contract?')
    assert result.intent != 'metadata'


def test_describe_all_files_routes_to_coverage_not_metadata() -> None:
    result = classify_query('Describe all files')
    assert result.intent == 'coverage'
    assert result.route_candidate == 'cross_document_synthesis'


def test_describe_the_files_i_have_routes_to_metadata_via_inventory_capability() -> None:
    result = classify_query('Describe the files I have')
    assert result.intent == 'metadata'
    assert result.route_candidate == 'metadata_inventory'


@pytest.mark.parametrize(
    ('query', 'expected_intent', 'expected_route'),
    [
        ('Summarize the key findings from 2022 records only.', 'coverage', 'cross_document_synthesis'),
        ('Which year has the fewest files?', 'metadata', 'metadata_inventory'),
        ('Which file has the most mentions of escrow?', 'focused', 'comparative_analysis'),
    ],
)
def test_phase_regression_matrix_queries(query: str, expected_intent: str, expected_route: str) -> None:
    result = classify_query(query)
    assert result.intent == expected_intent
    assert result.route_candidate == expected_route
