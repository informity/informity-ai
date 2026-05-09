# ==============================================================================
# Informity AI — Handler Tests
# Tests QueryHandler implementations (MetadataHandler, RAGHandler, SimpleHandler)
# ==============================================================================

from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from informity.config import settings
from informity.db.models import ChatMessage
from informity.llm.handlers.metadata import MetadataHandler
from informity.llm.handlers.query_handler import QueryHandler
from informity.llm.handlers.rag import (
    RAGHandler,
    _apply_negation_preferences,
    _apply_output_format_preferences,
    _build_history_aware_retrieval_query,
    _build_history_aware_retrieval_query_with_classification,
    _resolve_exhaustive_inventory_term_type,
    _should_boost_coverage_top_k,
)
from informity.llm.handlers.simple import SimpleHandler
from informity.llm.personas import get_mode_prompt
from informity.llm.query_classifier import QueryClassification
from informity.llm.types import OutputFormat
from informity.llm.web_search import SearchResult, WebSearchOutcome


class TestHandlerProtocol:
    # Test that handlers implement QueryHandler protocol

    def test_metadata_handler_implements_protocol(self) -> None:
        handler = MetadataHandler()
        assert isinstance(handler, QueryHandler)
        assert hasattr(handler, 'matches')
        assert hasattr(handler, 'handle')

    def test_rag_handler_implements_protocol(self) -> None:
        handler = RAGHandler()
        assert isinstance(handler, QueryHandler)
        assert hasattr(handler, 'matches')
        assert hasattr(handler, 'handle')

    def test_simple_handler_implements_protocol(self) -> None:
        handler = SimpleHandler()
        assert isinstance(handler, QueryHandler)
        assert hasattr(handler, 'matches')
        assert hasattr(handler, 'handle')


def test_should_boost_coverage_top_k_for_corpus_wide_entity_listing() -> None:
    classification = QueryClassification(intent='coverage')
    assert _should_boost_coverage_top_k(
        'What are the names of people mentioned across all indexed documents?',
        classification,
    )


def test_should_not_boost_top_k_for_focused_queries() -> None:
    classification = QueryClassification(intent='focused')
    assert not _should_boost_coverage_top_k(
        'What are the names of people mentioned across all indexed documents?',
        classification,
    )


def test_resolve_exhaustive_inventory_term_type_for_people_names() -> None:
    classification = QueryClassification(intent='coverage')
    assert _resolve_exhaustive_inventory_term_type(
        'What are the names of people mentioned across all indexed documents?',
        classification,
    ) == 'person_name'


def test_resolve_exhaustive_inventory_term_type_none_without_corpus_scope() -> None:
    classification = QueryClassification(intent='coverage')
    assert _resolve_exhaustive_inventory_term_type(
        'What are the names of people mentioned in this file?',
        classification,
    ) is None


class TestMetadataHandler:

    def test_matches_metadata_queries(self) -> None:
        handler = MetadataHandler()
        classification = QueryClassification(intent='metadata', is_metadata_query=True)
        assert handler.matches(classification) is True

    def test_does_not_match_non_metadata(self) -> None:
        handler = MetadataHandler()
        classification = QueryClassification(intent='focused')
        assert handler.matches(classification) is False

    def test_matches_comparative_subtype_even_when_not_metadata_intent(self) -> None:
        handler = MetadataHandler()
        classification = QueryClassification(intent='focused', subtype='comparative', group_by='year')
        assert handler.matches(classification) is True

    def test_does_not_match_comparative_file_scope_when_not_metadata_intent(self) -> None:
        handler = MetadataHandler()
        classification = QueryClassification(intent='focused', subtype='comparative', group_by='file')
        assert handler.matches(classification) is False

    @pytest.mark.asyncio
    async def test_handle_count_query(self) -> None:
        handler = MetadataHandler()
        classification = QueryClassification(intent='metadata', is_metadata_query=True)
        mock_cursor = MagicMock()
        mock_cursor.fetchone = AsyncMock(return_value={'cnt': 5})
        mock_db = MagicMock()
        mock_db.execute = AsyncMock(return_value=mock_cursor)

        results = []
        async for item in handler.handle('how many files', classification, None, mock_db, None):
            results.append(item)

        assert len(results) >= 1
        assert '5' in results[0] or 'five' in results[0].lower()
        assert results[-1] == []  # No sources

    @pytest.mark.asyncio
    async def test_handle_enumeration_query(self) -> None:
        handler = MetadataHandler()
        classification = QueryClassification(intent='metadata')
        mock_db = MagicMock()

        with patch('informity.llm.handlers.metadata.get_distinct_years', new_callable=AsyncMock) as mock_years:
            mock_years.return_value = [2020, 2021, 2022, 2023]

            results = []
            async for item in handler.handle('what years', classification, None, mock_db, None):
                results.append(item)

            assert len(results) >= 1
            assert '2020' in results[0] or '2021' in results[0]

    @pytest.mark.asyncio
    async def test_handle_file_list_query_applies_year_fallback_when_classifier_year_missing(self) -> None:
        handler = MetadataHandler()
        classification = QueryClassification(intent='metadata', is_file_list_query=True, year_filter=None)

        count_cursor = MagicMock()
        count_cursor.fetchone = AsyncMock(return_value={'cnt': 1})
        list_cursor = MagicMock()
        list_cursor.fetchall = AsyncMock(return_value=[])

        mock_db = MagicMock()
        mock_db.execute = AsyncMock(side_effect=[count_cursor, list_cursor])

        results = []
        async for item in handler.handle('List all files from 2012', classification, None, mock_db, None):
            results.append(item)

        assert len(results) >= 1
        execute_calls = mock_db.execute.await_args_list
        assert len(execute_calls) == 2
        count_params = list(execute_calls[0].args[1])
        list_params = list(execute_calls[1].args[1])
        assert count_params == [2012]
        assert list_params[0] == 2012

    @pytest.mark.asyncio
    async def test_get_files_with_filters_applies_filename_filter(self) -> None:
        handler = MetadataHandler()
        classification = QueryClassification(intent='metadata', filename_filter='report.pdf')

        count_cursor = MagicMock()
        count_cursor.fetchone = AsyncMock(return_value={'cnt': 0})
        list_cursor = MagicMock()
        list_cursor.fetchall = AsyncMock(return_value=[])
        mock_db = MagicMock()
        mock_db.execute = AsyncMock(side_effect=[count_cursor, list_cursor])

        await handler._get_files_with_filters(mock_db, classification)

        execute_calls = mock_db.execute.await_args_list
        assert len(execute_calls) == 2
        assert 'filename = ?' in execute_calls[0].args[0]
        assert list(execute_calls[0].args[1]) == ['report.pdf']

    def test_format_file_list_response_includes_year_in_header_when_filtered(self) -> None:
        handler = MetadataHandler()
        classification = QueryClassification(intent='metadata', year_filter=2021)
        response = handler._format_file_list_response(
            files=[MagicMock(filename='example.txt')],
            total=1,
            classification=classification,
        )
        assert 'from 2021' in response

    @pytest.mark.asyncio
    async def test_handle_comparative_query_uses_sql_aggregation(self) -> None:
        handler = MetadataHandler()
        classification = QueryClassification(intent='focused', subtype='comparative', group_by='year')
        mock_cursor = MagicMock()
        mock_cursor.fetchone = AsyncMock(return_value={'bucket': 2023, 'cnt': 2})
        mock_db = MagicMock()
        mock_db.execute = AsyncMock(return_value=mock_cursor)

        results = []
        async for item in handler.handle('Which year has the fewest files?', classification, None, mock_db, None):
            results.append(item)

        assert any(isinstance(item, str) and 'fewest files' in item.lower() for item in results)
        assert results[-1] == []

    def test_format_enumeration_response_as_table_when_requested(self) -> None:
        handler = MetadataHandler()
        response = handler._format_enumeration_response(
            {'years': [2022, 2023], 'categories': ['tax']},
            'what years and categories',
            as_table=True,
        )
        assert '| Dimension | Value |' in response
        assert '2022, 2023' in response
        assert 'tax' in response


class TestRAGHandler:

    def test_query_rewrite_passes_through_non_referential_questions(self) -> None:
        rewritten, applied = _build_history_aware_retrieval_query(
            'Summarize tax returns by year',
            [
                ChatMessage(chat_id='chat', role='user', content='Show me my 2024 taxes'),
                ChatMessage(chat_id='chat', role='assistant', content='Here are the documents'),
            ],
        )
        assert applied is False
        assert rewritten == 'Summarize tax returns by year'

    def test_query_rewrite_adds_context_for_referential_followups(self) -> None:
        rewritten, applied = _build_history_aware_retrieval_query(
            'What about that one?',
            [
                ChatMessage(chat_id='chat', role='user', content='Summarize my retirement plans in Escondido'),
                ChatMessage(chat_id='chat', role='assistant', content='I found two retirement plan files in Escondido.'),
            ],
        )
        assert applied is True
        assert 'Follow-up context:' in rewritten
        assert 'Previous user question:' in rewritten
        assert 'Previous assistant answer:' not in rewritten

    def test_query_rewrite_can_be_disabled_via_settings(self) -> None:
        original_enabled = settings.rag_query_rewrite_enabled
        try:
            settings.rag_query_rewrite_enabled = False
            rewritten, applied = _build_history_aware_retrieval_query(
                'What about that one?',
                [
                    ChatMessage(chat_id='chat', role='user', content='Summarize my retirement plans in Escondido'),
                    ChatMessage(chat_id='chat', role='assistant', content='I found two relevant files.'),
                ],
            )
            assert applied is False
            assert rewritten == 'What about that one?'
        finally:
            settings.rag_query_rewrite_enabled = original_enabled

    def test_query_rewrite_adds_context_for_topical_followups_without_pronouns(self) -> None:
        rewritten, applied = _build_history_aware_retrieval_query_with_classification(
            question='give basic character description for each character',
            history=[
                ChatMessage(chat_id='chat', role='user', content='List all the main characters in The Three Musketeers'),
                ChatMessage(chat_id='chat', role='assistant', content='Here are the main characters.'),
            ],
            classification=QueryClassification(intent='focused'),
        )
        assert applied is True
        assert 'Follow-up context:' in rewritten
        assert 'Three Musketeers' in rewritten

    def test_query_rewrite_skips_when_scope_reset_is_explicit(self) -> None:
        rewritten, applied = _build_history_aware_retrieval_query_with_classification(
            question='Summarize this contract',
            history=[
                ChatMessage(chat_id='chat', role='user', content='List all the main characters in The Three Musketeers'),
                ChatMessage(chat_id='chat', role='assistant', content='Here are the main characters.'),
            ],
            classification=QueryClassification(intent='focused', is_scope_reset=True),
        )
        assert applied is False
        assert rewritten == 'Summarize this contract'

    def test_query_rewrite_skips_when_explicit_topic_shift_cue_present(self) -> None:
        rewritten, applied = _build_history_aware_retrieval_query_with_classification(
            question='Instead, new topic: summarize 2025 planning notes',
            history=[
                ChatMessage(chat_id='chat', role='user', content='List all the main characters in The Three Musketeers'),
                ChatMessage(chat_id='chat', role='assistant', content='Here are the main characters.'),
            ],
            classification=QueryClassification(intent='focused'),
        )
        assert applied is False
        assert rewritten == 'Instead, new topic: summarize 2025 planning notes'
    @pytest.fixture(autouse=True)
    def _force_minimal_mode_for_rag_tests(self) -> None:
        # RAG handler tests validate the minimal one-path runtime directly.
        original = settings.rag_minimal_mode
        settings.rag_minimal_mode = True
        try:
            yield
        finally:
            settings.rag_minimal_mode = original

    def test_matches_focused_queries(self) -> None:
        handler = RAGHandler()
        classification = QueryClassification(intent='focused')
        assert handler.matches(classification) is True

    def test_matches_coverage_queries(self) -> None:
        handler = RAGHandler()
        classification = QueryClassification(intent='coverage')
        assert handler.matches(classification) is True

    def test_does_not_match_metadata(self) -> None:
        handler = RAGHandler()
        classification = QueryClassification(intent='metadata')
        assert handler.matches(classification) is False

    def test_does_not_match_simple(self) -> None:
        handler = RAGHandler()
        classification = QueryClassification(intent='simple')
        assert handler.matches(classification) is False

    def test_normalize_relevance_accepts_non_builtin_numeric(self) -> None:
        from informity.llm.rag_runtime.retrieval_validation import _normalize_relevance_score

        score = _normalize_relevance_score(Decimal('0.75'))
        assert 0.67 < score < 0.69

    def test_retrieval_relevance_gate_allows_low_but_valid_coverage_signal(self) -> None:
        from informity.llm.rag_runtime.retrieval_validation import (
            _evaluate_retrieval_relevance_gate,
        )

        chunks = [
            {'score': -3.3},  # sigmoid ~= 0.035
            {'score': -3.4},
            {'score': -3.5},
        ]
        passed, mean_score = _evaluate_retrieval_relevance_gate(
            chunks=chunks,
            query_type='coverage',
            route_candidate='cross_document_synthesis',
        )
        assert passed is True
        assert mean_score > 0.0

    def test_resolve_sampling_params_reduces_temperature_for_strict_contracts(self) -> None:
        from informity.llm.handlers.rag import _resolve_sampling_params

        temperature, top_p = _resolve_sampling_params(
            profile_temperature=0.7,
            profile_top_p=0.95,
            format_requirements=[
                'use the required headings exactly and in the requested order',
                'include heading: Findings by Year',
                'for year-grouped sections, include one subsection per year using markdown headings like "### YYYY"',
            ],
        )
        assert temperature <= 0.2
        assert top_p <= 0.8

    def test_resolve_sampling_params_preserves_profile_defaults_without_strict_contract(self) -> None:
        from informity.llm.handlers.rag import _resolve_sampling_params

        temperature, top_p = _resolve_sampling_params(
            profile_temperature=0.7,
            profile_top_p=0.95,
            format_requirements=['use all headings explicitly requested by the user'],
        )
        assert temperature == 0.7
        assert top_p == 0.95

    def test_apply_output_format_preferences_adds_table_requirement(self) -> None:
        requirements: list[str] = []
        constraints: dict[str, int] = {}
        _apply_output_format_preferences(
            output_format=OutputFormat.TABLE,
            format_requirements=requirements,
            output_constraints=constraints,
        )
        assert 'markdown table' in ' '.join(requirements).lower()

    def test_apply_negation_preferences_adds_limitation_requirement(self) -> None:
        requirements: list[str] = []
        _apply_negation_preferences(
            is_negation_query=True,
            format_requirements=requirements,
        )
        assert 'exact negation cannot be guaranteed' in ' '.join(requirements).lower()

    @pytest.mark.asyncio
    async def test_handle_uses_deterministic_term_inventory_for_exhaustive_people_query(self) -> None:
        handler = RAGHandler()
        classification = QueryClassification(intent='coverage')
        mock_db = MagicMock()
        mock_inventory_cursor = MagicMock()
        mock_inventory_cursor.fetchall = AsyncMock(
            return_value=[
                {'canonical_term': 'Benjamin Bjork', 'confidence': 0.75, 'file_count': 1},
                {'canonical_term': 'Glenn Perez', 'confidence': 0.75, 'file_count': 1},
            ]
        )
        mock_sources_cursor = MagicMock()
        mock_sources_cursor.fetchall = AsyncMock(
            return_value=[
                {
                    'file_id': 42,
                    'filename': 'retirement-plan.pdf',
                    'path': '/docs/retirement-plan.pdf',
                    'chunk_preview': 'Benjamin Bjork reviewed projected retirement distributions.',
                    'relevance_score': 0.75,
                }
            ]
        )
        mock_db.execute = AsyncMock(side_effect=[mock_inventory_cursor, mock_sources_cursor])

        with patch('informity.llm.handlers.rag.retrieve_chunks', new_callable=AsyncMock) as mock_retrieve:
            results: list[object] = []
            async for item in handler.handle(
                'What are the names of people mentioned across all indexed documents?',
                classification,
                None,
                mock_db,
                None,
            ):
                results.append(item)

        mock_retrieve.assert_not_called()
        assert any(isinstance(item, str) and 'Benjamin Bjork' in item for item in results)
        assert any(isinstance(item, str) and 'Glenn Perez' in item for item in results)
        metrics_events = [item for item in results if isinstance(item, tuple) and item[0] == '__metrics__']
        assert metrics_events
        assert metrics_events[0][1].get('deterministic_inventory') is True
        assert isinstance(results[-1], list)
        assert results[-1]

    @pytest.mark.asyncio
    async def test_empty_retrieval_terminal_refusal_sets_no_remaining_scope(self) -> None:
        handler = RAGHandler()
        classification = QueryClassification(intent='focused')
        mock_db = MagicMock()
        mock_count_cursor = MagicMock()
        mock_count_cursor.fetchone = AsyncMock(return_value={'count': 1})
        mock_db.execute = AsyncMock(return_value=mock_count_cursor)

        with patch('informity.llm.handlers.rag.retrieve_chunks', new_callable=AsyncMock) as mock_retrieve:
            mock_retrieve.return_value = []
            results: list[object] = []
            async for item in handler.handle('Which files mention withholding?', classification, None, mock_db, None):
                results.append(item)

        metrics_events = [item for item in results if isinstance(item, tuple) and item[0] == '__metrics__']
        assert metrics_events
        metrics = metrics_events[0][1]
        assert metrics.get('generation_skipped') is True
        assert metrics.get('answerability_passed') is False
        assert any(isinstance(item, str) and 'do not contain enough information' in item.casefold() for item in results)
        assert results[-1] == []

    @pytest.mark.asyncio
    async def test_validation_gate_terminal_refusal_sets_no_remaining_scope(self) -> None:
        handler = RAGHandler()
        classification = QueryClassification(intent='focused')
        mock_db = MagicMock()
        test_chunks = [{
            'file_id': 1,
            'filename': 'alpha.pdf',
            'file_path': '/docs/alpha.pdf',
            'chunk_text': 'Some weak evidence text.',
            'score': -6.0,
        }]
        original_threshold = settings.rag_minimal_answerability_threshold_focused
        original_min_chunks = settings.rag_minimal_min_chunks_focused
        settings.rag_minimal_answerability_threshold_focused = 0.95
        settings.rag_minimal_min_chunks_focused = 1
        try:
            with patch('informity.llm.handlers.rag.retrieve_chunks', new_callable=AsyncMock) as mock_retrieve, \
                 patch('informity.llm.handlers.rag.stream_llm', new_callable=AsyncMock) as mock_stream:
                mock_retrieve.return_value = test_chunks
                results: list[object] = []
                async for item in handler.handle('Summarize unresolved records.', classification, None, mock_db, None):
                    results.append(item)
                mock_stream.assert_not_called()
        finally:
            settings.rag_minimal_answerability_threshold_focused = original_threshold
            settings.rag_minimal_min_chunks_focused = original_min_chunks

        metrics_events = [item for item in results if isinstance(item, tuple) and item[0] == '__metrics__']
        assert metrics_events
        metrics = metrics_events[0][1]
        assert metrics.get('generation_skipped') is True
        assert metrics.get('answerability_passed') is False

    @pytest.mark.asyncio
    async def test_validation_gate_widened_retry_recovers_before_terminal_refusal(self) -> None:
        handler = RAGHandler()
        classification = QueryClassification(intent='focused')
        mock_db = MagicMock()
        original_threshold = settings.rag_minimal_answerability_threshold_focused
        settings.rag_minimal_answerability_threshold_focused = 0.9
        try:
            with patch('informity.llm.handlers.rag.retrieve_chunks', new_callable=AsyncMock) as mock_retrieve, \
                 patch('informity.llm.handlers.rag.stream_llm', new_callable=AsyncMock) as mock_stream:
                mock_retrieve.side_effect = [
                    [{'score': -5.0, 'chunk_text': 'weak'}],
                    [{'score': 3.0, 'chunk_text': 'strong'}],
                ]
                results: list[object] = []
                async for item in handler.handle('Summarize unresolved records.', classification, None, mock_db, None):
                    results.append(item)
                mock_stream.assert_not_called()
        finally:
            settings.rag_minimal_answerability_threshold_focused = original_threshold

        assert mock_retrieve.await_count == 1
        assert any(isinstance(item, str) and 'do not contain enough information' in item.casefold() for item in results)

    @pytest.mark.asyncio
    async def test_handle_calls_retrieve_chunks(self) -> None:
        handler = RAGHandler()
        classification = QueryClassification(intent='focused')
        mock_db = MagicMock()

        with patch('informity.llm.handlers.rag.retrieve_chunks', new_callable=AsyncMock) as mock_retrieve:
            mock_retrieve.return_value = []
            results: list[object] = []
            async for item in handler.handle('test question', classification, None, mock_db, None):
                results.append(item)
            assert results[-1] == []
            assert mock_retrieve.await_count == 1

    @pytest.mark.asyncio
    async def test_handle_passes_file_scopes_to_retrieve_chunks(self) -> None:
        handler = RAGHandler()
        classification = QueryClassification(intent='focused')
        mock_db = MagicMock()

        with patch('informity.llm.handlers.rag.retrieve_chunks', new_callable=AsyncMock) as mock_retrieve:
            mock_retrieve.return_value = []
            async for _item in handler.handle(
                'test question',
                classification,
                None,
                mock_db,
                None,
                file_ids=[7],
            ):
                pass
            assert mock_retrieve.await_count == 1
            assert mock_retrieve.await_args.kwargs.get('file_ids_filter') == [7]

    @pytest.mark.asyncio
    async def test_handle_rewrites_referential_query_for_retrieval(self) -> None:
        handler = RAGHandler()
        classification = QueryClassification(intent='focused')
        mock_db = MagicMock()
        history = [
            ChatMessage(chat_id='chat', role='user', content='Summarize my retirement plans in Escondido'),
            ChatMessage(chat_id='chat', role='assistant', content='I found relevant retirement plan documents.'),
        ]
        with patch('informity.llm.handlers.rag.retrieve_chunks', new_callable=AsyncMock) as mock_retrieve:
            mock_retrieve.return_value = []
            results: list[object] = []
            async for item in handler.handle('What about that one?', classification, history, mock_db, None):
                results.append(item)
            assert results[-1] == []
            assert mock_retrieve.await_count == 1
            assert 'Follow-up context:' in mock_retrieve.await_args.kwargs['query']

    @pytest.mark.asyncio
    async def test_handle_enables_term_expansion_and_diversity_for_focused_explicit_title_query(self) -> None:
        handler = RAGHandler()
        classification = QueryClassification(intent='focused')
        mock_db = MagicMock()
        with patch('informity.llm.handlers.rag.retrieve_chunks', new_callable=AsyncMock) as mock_retrieve:
            mock_retrieve.return_value = []
            results: list[object] = []
            async for item in handler.handle(
                'What is the general plot of The Three Musketeers book?',
                classification,
                [],
                mock_db,
                None,
            ):
                results.append(item)
            assert results[-1] == []
            assert mock_retrieve.await_count == 1
            assert mock_retrieve.await_args.kwargs.get('query') == 'What is the general plot of The Three Musketeers book?'
            assert mock_retrieve.await_args.kwargs.get('enable_term_expansion') is True
            assert mock_retrieve.await_args.kwargs.get('prefer_within_file_diversity') is True
            assert mock_retrieve.await_args.kwargs.get('strict_title_alignment') is True

    @pytest.mark.asyncio
    async def test_handle_uses_decomposed_retrieval_content_query(self) -> None:
        handler = RAGHandler()
        classification = QueryClassification(
            intent='focused',
            retrieval_content_query='What is the general plot of The Count of Monte Cristo?',
            retrieval_content_confidence=0.8,
            retrieval_content_reasons=['question_mark', 'question_word'],
        )
        mock_db = MagicMock()
        with patch('informity.llm.handlers.rag.retrieve_chunks', new_callable=AsyncMock) as mock_retrieve:
            mock_retrieve.return_value = []
            results: list[object] = []
            async for item in handler.handle(
                'OK, new topic. What is the general plot of The Count of Monte Cristo?',
                classification,
                [],
                mock_db,
                None,
            ):
                results.append(item)
            assert results[-1] == []
            assert mock_retrieve.await_count == 1
            assert mock_retrieve.await_args.kwargs.get('query') == 'What is the general plot of The Count of Monte Cristo?'

    @pytest.mark.asyncio
    async def test_continuation_without_overlap_keeps_scope_without_clarification(self) -> None:
        handler = RAGHandler()
        classification = QueryClassification(
            intent='focused',
            route_candidate='continuation_or_refinement',
            is_continuation=True,
            confidence=0.86,
        )
        mock_db = MagicMock()
        history = [
            ChatMessage(
                chat_id='test-chat',
                role='assistant',
                content='Previous answer',
                sources=[{'path': '/docs/alpha.pdf', 'filename': 'alpha.pdf'}],
            ),
        ]

        with patch('informity.llm.handlers.rag.retrieve_chunks', new_callable=AsyncMock) as mock_retrieve:
            mock_retrieve.return_value = [
                {
                    'file_id': 1,
                    'filename': 'beta.pdf',
                    'file_path': '/docs/beta.pdf',
                    'chunk_text': 'Key facts with strong evidence.',
                    'score': 2.1,
                },
            ]

            async def _fake_stream_llm(*_args, **_kwargs):
                yield 'Continuation answer token.'

            results: list[object] = []
            with patch('informity.llm.handlers.rag.stream_llm', _fake_stream_llm):
                async for item in handler.handle('continue with the same structure', classification, history, mock_db, None):
                    results.append(item)

        assert mock_retrieve.await_count == 1
        assert any(isinstance(item, str) and 'continuation answer token' in item.casefold() for item in results)
        assert results[-1] != []

    @pytest.mark.asyncio
    async def test_continuation_with_anchor_overlap_bypasses_relevance_gate(self) -> None:
        handler = RAGHandler()
        classification = QueryClassification(
            intent='focused',
            route_candidate='continuation_or_refinement',
            is_continuation=True,
            confidence=0.86,
        )
        mock_db = MagicMock()
        history = [
            ChatMessage(
                chat_id='test-chat',
                role='assistant',
                content='Previous answer',
                has_remaining_scope=True,
                sources=[{'path': '/docs/alpha.pdf', 'filename': 'alpha.pdf'}],
            ),
            ChatMessage(
                chat_id='test-chat',
                role='user',
                content='Summarize the evidence by year.',
            ),
        ]
        original_threshold = settings.rag_minimal_answerability_threshold_focused
        settings.rag_minimal_answerability_threshold_focused = 0.9
        try:
            with patch('informity.llm.handlers.rag.retrieve_chunks', new_callable=AsyncMock) as mock_retrieve:
                mock_retrieve.return_value = [
                    {
                        'file_id': 1,
                        'filename': 'alpha.pdf',
                        'file_path': '/docs/alpha.pdf',
                        'chunk_text': 'Prior anchored evidence.',
                        'score': -5.0,
                    },
                ]

                results: list[object] = []
                async for item in handler.handle('continue with cross-year comparison', classification, history, mock_db, None):
                    results.append(item)
        finally:
            settings.rag_minimal_answerability_threshold_focused = original_threshold

        assert any(isinstance(item, str) and 'do not contain enough information' in item.casefold() for item in results)
        metrics_events = [item for item in results if isinstance(item, tuple) and item[0] == '__metrics__']
        assert metrics_events
        metrics = metrics_events[0][1]
        assert metrics.get('generation_skipped') is True
        assert 'fallback_events' not in metrics or metrics.get('fallback_events') in (None, [])

    @pytest.mark.asyncio
    async def test_budget_pressure_with_weak_relevance_skips_generation(self) -> None:
        handler = RAGHandler()
        classification = QueryClassification(
            intent='focused',
            route_candidate='targeted_fact_lookup',
            confidence=0.84,
        )
        mock_db = MagicMock()

        with patch('informity.llm.handlers.rag.retrieve_chunks', new_callable=AsyncMock) as mock_retrieve:
            mock_retrieve.return_value = [
                {
                    'file_id': 1,
                    'filename': 'alpha.pdf',
                    'file_path': '/docs/alpha.pdf',
                    'chunk_text': 'Context exists and should generate in minimal mode.',
                    'score': 0.28,
                },
            ]

            async def _fake_stream_llm(*_args, **_kwargs):
                yield 'Generated answer token.'

            results: list[object] = []
            with patch('informity.llm.handlers.rag.stream_llm', _fake_stream_llm):
                async for item in handler.handle('summarize this quickly', classification, None, mock_db, None):
                    results.append(item)

        assert any(isinstance(item, str) and 'generated answer token' in item.casefold() for item in results)
        metrics_events = [item for item in results if isinstance(item, tuple) and item[0] == '__metrics__']
        assert metrics_events
        metrics = metrics_events[0][1]
        assert metrics.get('generation_skipped') is False
        assert 'stream_recovery_reason' not in metrics

    @pytest.mark.asyncio
    async def test_continuation_budget_pressure_closeout_includes_contract_terms(self) -> None:
        handler = RAGHandler()
        classification = QueryClassification(
            intent='focused',
            route_candidate='continuation_or_refinement',
            is_continuation=True,
            confidence=0.84,
        )
        mock_db = MagicMock()
        original_threshold = settings.rag_minimal_answerability_threshold_focused
        settings.rag_minimal_answerability_threshold_focused = 0.9
        try:
            with patch('informity.llm.handlers.rag.retrieve_chunks', new_callable=AsyncMock) as mock_retrieve, \
                 patch('informity.llm.handlers.rag.stream_llm', new_callable=AsyncMock) as mock_stream:
                mock_retrieve.return_value = [
                    {
                        'file_id': 1,
                        'filename': 'alpha.pdf',
                        'file_path': '/docs/alpha.pdf',
                        'chunk_text': 'Weak context under strict threshold.',
                        'score': -4.0,
                    },
                ]
                results: list[object] = []
                async for item in handler.handle(
                    'Continue with ## Cross-Year Deltas, ## Confidence Notes, ## Verification Steps only.',
                    classification,
                    None,
                    mock_db,
                    None,
                ):
                    results.append(item)
                mock_stream.assert_not_called()
        finally:
            settings.rag_minimal_answerability_threshold_focused = original_threshold

        rendered = '\n'.join(item for item in results if isinstance(item, str))
        assert 'do not contain enough information' in rendered.casefold()
        assert 'cross-year deltas' not in rendered.casefold()
        assert 'confidence notes' not in rendered.casefold()
        assert results[-1] == []

    @pytest.mark.asyncio
    async def test_narrative_response_shape_does_not_trigger_structured_insufficient_path(self) -> None:
        handler = RAGHandler()
        classification = QueryClassification(
            intent='coverage',
            response_shape='narrative_synthesis',
            route_candidate='audit_or_compliance_brief',
            subtype='extract_structured_values',
            confidence=0.86,
        )
        mock_db = MagicMock()
        with patch('informity.llm.handlers.rag.retrieve_chunks', new_callable=AsyncMock) as mock_retrieve:
            mock_retrieve.return_value = [
                {
                    'file_id': 1,
                    'filename': 'alpha.pdf',
                    'file_path': '/docs/alpha.pdf',
                    'chunk_text': 'Compliance report evidence with key findings across years.',
                    'score': -3.2,
                },
                {
                    'file_id': 2,
                    'filename': 'beta.pdf',
                    'file_path': '/docs/beta.pdf',
                    'chunk_text': 'Additional evidence for cross-year analysis and risk summary.',
                    'score': -3.1,
                },
            ]

            async def _fake_stream_llm(*_args, **_kwargs):
                yield 'narrative output token'

            results: list[object] = []
            with patch('informity.llm.handlers.rag.stream_llm', _fake_stream_llm):
                async for item in handler.handle('build compliance brief', classification, None, mock_db, None):
                    results.append(item)

        assert any(isinstance(item, str) and 'narrative output token' in item for item in results)
        assert not any(isinstance(item, str) and 'I could not extract enough validated structured values' in item for item in results)

    @pytest.mark.asyncio
    async def test_aggregate_coverage_query_does_not_degrade_to_focused(self) -> None:
        handler = RAGHandler()
        classification = QueryClassification(
            intent='coverage',
            response_shape='narrative_synthesis',
            route_candidate='comparative_analysis',
            subtype='aggregate_by_period',
            group_by='year',
            confidence=0.86,
        )
        mock_db = MagicMock()
        with patch('informity.llm.handlers.rag.retrieve_chunks', new_callable=AsyncMock) as mock_retrieve:
            mock_retrieve.return_value = [
                {'file_id': 1, 'filename': 'y2022.pdf', 'file_path': '/docs/y2022.pdf', 'chunk_text': 'Box 1 wages $1,000.00', 'score': 0.1},
                {'file_id': 2, 'filename': 'y2023.pdf', 'file_path': '/docs/y2023.pdf', 'chunk_text': 'Box 1 wages $2,000.00', 'score': 0.1},
                {'file_id': 3, 'filename': 'y2024.pdf', 'file_path': '/docs/y2024.pdf', 'chunk_text': 'Box 1 wages $3,000.00', 'score': 0.1},
            ]

            async def _fake_stream_llm(*_args, **_kwargs):
                yield 'aggregate summary'

            results: list[object] = []
            with patch('informity.llm.handlers.rag.stream_llm', _fake_stream_llm):
                async for item in handler.handle('extract box 1 totals by year 2022-2024', classification, None, mock_db, None):
                    results.append(item)

        metrics_events = [item for item in results if isinstance(item, tuple) and item[0] == '__metrics__']
        assert metrics_events
        assert metrics_events[0][1].get('query_type') == 'coverage'

    @pytest.mark.asyncio
    async def test_structured_insufficient_falls_back_to_narrative_generation(self) -> None:
        handler = RAGHandler()
        classification = QueryClassification(
            intent='focused',
            response_shape='structured_extract',
            route_candidate='structured_field_extraction',
            subtype='extract_structured_values',
            field_hint=None,
            confidence=0.86,
        )
        mock_db = MagicMock()

        with patch('informity.llm.handlers.rag.retrieve_chunks', new_callable=AsyncMock) as mock_retrieve:
            mock_retrieve.return_value = [
                {
                    'file_id': 1,
                    'filename': 'alpha.pdf',
                    'file_path': '/docs/alpha.pdf',
                    'chunk_text': 'This text has no parseable numeric values for deterministic extraction.',
                    'score': 1.2,
                },
                {
                    'file_id': 2,
                    'filename': 'beta.pdf',
                    'file_path': '/docs/beta.pdf',
                    'chunk_text': 'Narrative context still supports synthesis.',
                    'score': 1.1,
                },
            ]

            async def _fake_stream_llm(*_args, **_kwargs):
                yield 'fallback narrative token'

            results: list[object] = []
            with patch('informity.llm.handlers.rag.stream_llm', _fake_stream_llm):
                async for item in handler.handle('extract key values and explain', classification, None, mock_db, None):
                    results.append(item)

        assert any(isinstance(item, str) and 'fallback narrative token' in item for item in results)
        assert not any(isinstance(item, str) and 'I could not extract enough validated structured values' in item for item in results)

    @pytest.mark.asyncio
    async def test_soft_limit_closeout_applies_for_non_strict_formats(self) -> None:
        handler = RAGHandler()
        classification = QueryClassification(
            intent='focused',
            response_shape='narrative_synthesis',
            route_candidate='targeted_fact_lookup',
            confidence=0.86,
        )
        mock_db = MagicMock()
        with patch('informity.llm.handlers.rag.retrieve_chunks', new_callable=AsyncMock) as mock_retrieve:
            mock_retrieve.return_value = [
                {
                    'file_id': 1,
                    'filename': 'alpha.pdf',
                    'file_path': '/docs/alpha.pdf',
                    'chunk_text': 'Focused evidence for concise response.',
                    'score': 2.2,
                },
            ]

            async def _fake_stream_llm(*_args, **_kwargs):
                yield 'First sentence.'
                yield 'Second sentence should not be emitted.'

            results: list[object] = []
            with patch('informity.llm.handlers.rag.stream_llm', _fake_stream_llm):
                async for item in handler.handle('summarize this', classification, None, mock_db, None):
                    results.append(item)

        text_tokens = [item for item in results if isinstance(item, str)]
        assert any('First sentence.' in item for item in text_tokens)
        assert any('Second sentence should not be emitted.' in item for item in text_tokens)
        metrics_events = [item for item in results if isinstance(item, tuple) and item[0] == '__metrics__']
        assert metrics_events
        metrics = metrics_events[0][1]
        assert metrics.get('generation_skipped') is False
        assert metrics.get('stream_recovery_reason') is None


class TestSimpleHandler:

    def test_matches_simple_queries(self) -> None:
        handler = SimpleHandler()
        classification = QueryClassification(intent='simple')
        assert handler.matches(classification) is True

    def test_does_not_match_focused(self) -> None:
        handler = SimpleHandler()
        classification = QueryClassification(intent='focused')
        assert handler.matches(classification) is False

    @pytest.mark.asyncio
    async def test_handle_skips_retrieval(self) -> None:
        handler = SimpleHandler()
        classification = QueryClassification(intent='simple')
        mock_db = MagicMock()

        async def _fake_stream_llm(*_args, **_kwargs):
            yield 'Hello!'

        with patch('informity.llm.handlers.simple.stream_llm', _fake_stream_llm):
            results = []
            async for item in handler.handle('hello', classification, None, mock_db, None):
                results.append(item)

            # Should have streamed tokens
            assert len(results) >= 1
            # Should not have called retrieval (verify by checking no retrieve_chunks import)
            # Simple handler doesn't import retrieve_chunks, so this is implicit

    @pytest.mark.asyncio
    async def test_handle_no_sources(self) -> None:
        handler = SimpleHandler()
        classification = QueryClassification(intent='simple')
        mock_db = MagicMock()

        async def _fake_stream_llm(*_args, **_kwargs):
            yield 'answer'

        with patch('informity.llm.handlers.simple.stream_llm', _fake_stream_llm):
            results = []
            async for item in handler.handle('hello', classification, None, mock_db, None):
                results.append(item)

            # Last item should be empty sources list
            assert results[-1] == []

    @pytest.mark.asyncio
    async def test_handle_uses_assistant_prompt_without_corpus_capabilities(self) -> None:
        handler = SimpleHandler()
        classification = QueryClassification(intent='simple')
        mock_db = MagicMock()
        captured_messages: list[dict[str, str]] = []

        async def _fake_stream_llm(messages, *_args, **_kwargs):
            captured_messages.extend(messages)
            yield 'answer'

        with patch('informity.llm.handlers.simple.stream_llm', _fake_stream_llm):
            async for _item in handler.handle(
                'what files can you search',
                classification,
                None,
                mock_db,
                None,
                chat_mode='assistant',
            ):
                pass

        assert captured_messages
        system_message = captured_messages[0]['content']
        lowered = system_message.lower()
        assert 'without document retrieval' in lowered
        assert 'if asked about document search' not in lowered
        assert 'you can:' not in lowered
        assert captured_messages[0]['content'] == f"{get_mode_prompt('assistant_default')}\n\nContext:\n"

    @pytest.mark.asyncio
    async def test_handle_uses_researcher_prompt_exactly_in_researcher_mode(self) -> None:
        handler = SimpleHandler()
        classification = QueryClassification(intent='simple')
        mock_db = MagicMock()
        captured_messages: list[dict[str, str]] = []

        async def _fake_stream_llm(messages, *_args, **_kwargs):
            captured_messages.extend(messages)
            yield 'answer'

        with patch('informity.llm.handlers.simple.stream_llm', _fake_stream_llm):
            async for _item in handler.handle(
                'hello',
                classification,
                None,
                mock_db,
                None,
                chat_mode='researcher',
            ):
                pass

        assert captured_messages
        assert captured_messages[0]['content'] == f"{get_mode_prompt('researcher_default')}\n\nContext:\n"

    @pytest.mark.asyncio
    async def test_handle_chat_summary_mode_disables_web_search_and_uses_chat_prompt(self) -> None:
        handler = SimpleHandler()
        classification = QueryClassification(intent='simple', needs_chat_history=True)
        mock_db = MagicMock()
        captured_messages: list[dict[str, str]] = []
        history = [
            ChatMessage(chat_id='c1', role='user', content='We discussed Plato and Aristotle.'),
            ChatMessage(chat_id='c1', role='assistant', content='Yes, and their views on forms and causality.'),
        ]

        async def _fake_stream_llm(messages, *_args, **_kwargs):
            captured_messages.extend(messages)
            yield 'summary'

        with (
            patch('informity.llm.handlers.simple.stream_llm', _fake_stream_llm),
            patch('informity.llm.handlers.simple.has_any_provider_api_key', return_value=True),
            patch('informity.llm.handlers.simple.search_web') as mock_search_web,
        ):
            async for _item in handler.handle(
                'What have we been chatting about?',
                classification,
                history,
                mock_db,
                None,
                chat_mode='assistant',
                chat_web_search_enabled=True,
            ):
                pass

        assert mock_search_web.called is False
        assert captured_messages
        system_message = captured_messages[0]['content'].lower()
        assert 'summarize this chat conversation only' in system_message
        assert captured_messages[0]['content'].startswith(get_mode_prompt('chat_summary'))

    @pytest.mark.asyncio
    async def test_handle_web_search_synthesis_uses_exact_web_persona_prompt(self) -> None:
        handler = SimpleHandler()
        classification = QueryClassification(intent='simple')
        mock_db = MagicMock()
        captured_messages: list[dict[str, str]] = []

        async def _fake_stream_llm(messages, *_args, **_kwargs):
            captured_messages.extend(messages)
            yield 'answer'

        fake_outcome = WebSearchOutcome(
            status='ok',
            results=[
                SearchResult(
                    title='Doc',
                    url='https://example.com',
                    snippet='Snippet',
                )
            ],
            provider_attempted='provider-a',
            provider_used='provider-a',
            failover_applied=False,
        )

        with (
            patch('informity.llm.handlers.simple.stream_llm', _fake_stream_llm),
            patch('informity.llm.handlers.simple.has_any_provider_api_key', return_value=True),
            patch('informity.llm.handlers.simple.search_web', return_value=fake_outcome),
        ):
            async for _item in handler.handle(
                'latest updates',
                classification,
                None,
                mock_db,
                None,
                chat_mode='assistant',
                chat_web_search_enabled=True,
                chat_web_search_privacy_override=True,
            ):
                pass

        assert captured_messages
        assert captured_messages[0]['content'] == f"{get_mode_prompt('assistant_web_search_synthesis')}\n\nContext:\n"

    @pytest.mark.asyncio
    async def test_handle_chat_summary_mode_loads_chat_id_history_and_excludes_internal(self) -> None:
        handler = SimpleHandler()
        classification = QueryClassification(intent='simple', needs_chat_history=True)
        mock_db = MagicMock()
        captured_messages: list[dict[str, str]] = []
        db_messages = [
            ChatMessage(chat_id='c77', role='user', content='Topic A'),
            ChatMessage(chat_id='c77', role='assistant', content='Reply A'),
            ChatMessage(chat_id='c77', role='user', content='internal continuation prompt', is_internal=True),
            ChatMessage(chat_id='c77', role='user', content='What have we been chatting about?'),
        ]

        async def _fake_stream_llm(messages, *_args, **_kwargs):
            captured_messages.extend(messages)
            yield 'summary'

        with (
            patch('informity.llm.handlers.simple.stream_llm', _fake_stream_llm),
            patch('informity.llm.handlers.simple.get_chat', new_callable=AsyncMock) as mock_get_chat,
        ):
            mock_get_chat.return_value = db_messages
            async for _item in handler.handle(
                'What have we been chatting about?',
                classification,
                [],
                mock_db,
                None,
                chat_id='c77',
            ):
                pass

        assert captured_messages
        user_prompt = captured_messages[1]['content']
        assert 'internal continuation prompt' not in user_prompt
        assert user_prompt.count('What have we been chatting about?') == 1
        assert 'Topic A' in user_prompt
        assert 'Reply A' in user_prompt

    @pytest.mark.asyncio
    async def test_handle_chat_summary_mode_hierarchical_for_long_history(self) -> None:
        handler = SimpleHandler()
        classification = QueryClassification(intent='simple', needs_chat_history=True)
        mock_db = MagicMock()
        call_count = 0

        async def _fake_stream_llm(*_args, **_kwargs):
            nonlocal call_count
            call_count += 1
            yield f's{call_count}'

        long_history: list[ChatMessage] = []
        for idx in range(60):
            role = 'user' if idx % 2 == 0 else 'assistant'
            long_history.append(ChatMessage(chat_id='c2', role=role, content=f'message {idx}'))

        with patch('informity.llm.handlers.simple.stream_llm', _fake_stream_llm):
            async for _item in handler.handle(
                'Summarize our chat',
                classification,
                long_history,
                mock_db,
                None,
            ):
                pass

        # hierarchical mode: multiple internal chunk summary calls + one final streamed response
        assert call_count > 1
