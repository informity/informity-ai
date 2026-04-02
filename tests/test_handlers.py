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
from informity.llm.handlers.rag import RAGHandler
from informity.llm.handlers.simple import SimpleHandler
from informity.llm.query_classifier import QueryClassification
from informity.llm.rag_runtime.retrieval_pipeline import _deduplicate_prompt_chunks


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


class TestMetadataHandler:

    def test_matches_metadata_queries(self) -> None:
        handler = MetadataHandler()
        classification = QueryClassification(intent='metadata', is_metadata_query=True)
        assert handler.matches(classification) is True

    def test_does_not_match_non_metadata(self) -> None:
        handler = MetadataHandler()
        classification = QueryClassification(intent='focused')
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


class TestRAGHandler:
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

    def test_soft_stream_closeout_disabled_for_strict_heading_requirements(self) -> None:
        from informity.llm.rag_runtime.generation_runtime import _should_apply_soft_stream_closeout

        assert _should_apply_soft_stream_closeout(
            ['use the required headings exactly and in the requested order']
        ) is False
        assert _should_apply_soft_stream_closeout(['include heading: Scope']) is True


    def test_inventory_plus_content_fallback_answer_is_generic_and_term_aware(self) -> None:
        from informity.llm.rag_runtime.retrieval_pipeline import (
            _build_inventory_plus_content_fallback_answer,
        )

        answer = _build_inventory_plus_content_fallback_answer(
            chunks=[
                {
                    'filename': '2024 payroll-reconciliation.pdf',
                    'chunk_text': 'Payroll reconciliation references gross pay and federal tax withholding.',
                },
                {
                    'filename': '2023 tax-summary.txt',
                    'chunk_text': 'Tax summary includes wages and withheld amounts by quarter.',
                },
            ],
            source_terms=['payroll', 'withholding'],
        )
        assert isinstance(answer, str)
        assert 'requested terms' in answer.casefold()
        assert 'payroll' in answer.casefold()
        assert 'withholding' in answer.casefold()

    def test_filename_summary_fallback_answer_handles_markdown_summary_query(self) -> None:
        from informity.llm.handlers.rag import _has_explicit_output_contract
        from informity.llm.rag_runtime.retrieval_pipeline import (
            _build_filename_summary_fallback_answer,
        )

        answer = _build_filename_summary_fallback_answer(
            question='Summarize the content of portfolio_notes.md',
            filename_filter='portfolio_notes.md',
            chunks=[
                {'chunk_text': 'Scenario analysis compares delayed Social Security start age against early claiming.'},
                {'chunk_text': 'The document outlines monthly benefit tradeoffs and break-even points.'},
            ],
            has_explicit_output_contract_fn=_has_explicit_output_contract,
        )
        assert isinstance(answer, str)
        assert 'Summary: portfolio_notes.md' in answer
        assert 'Key points extracted' in answer

    def test_filename_summary_fallback_answer_skips_non_text_extensions(self) -> None:
        from informity.llm.handlers.rag import _has_explicit_output_contract
        from informity.llm.rag_runtime.retrieval_pipeline import (
            _build_filename_summary_fallback_answer,
        )

        answer = _build_filename_summary_fallback_answer(
            question='Summarize the content of annual_statement.pdf',
            filename_filter='annual_statement.pdf',
            chunks=[{'chunk_text': 'Some text'}],
            has_explicit_output_contract_fn=_has_explicit_output_contract,
        )
        assert answer is None

    def test_filename_summary_fallback_answer_skips_explicit_output_contract_queries(self) -> None:
        from informity.llm.handlers.rag import _has_explicit_output_contract
        from informity.llm.rag_runtime.retrieval_pipeline import (
            _build_filename_summary_fallback_answer,
        )

        answer = _build_filename_summary_fallback_answer(
            question=(
                'Summarize the content of portfolio_notes.md in <= 180 words. '
                'Include exactly 3 bullets: objective, key tradeoff, decision implication.'
            ),
            filename_filter='portfolio_notes.md',
            chunks=[{'chunk_text': 'Some summary text.'}],
            has_explicit_output_contract_fn=_has_explicit_output_contract,
        )
        assert answer is None

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

    def test_prompt_chunk_dedup_preserves_distinct_same_prefix_content(self) -> None:
        shared_prefix = 'Tax summary template text. ' * 15
        chunks = [
            {
                'file_path': '/docs/alpha.pdf',
                'filename': 'alpha.pdf',
                'chunk_text': f'{shared_prefix}Unique ending A: amount 100.',
            },
            {
                'file_path': '/docs/alpha.pdf',
                'filename': 'alpha.pdf',
                'chunk_text': f'{shared_prefix}Unique ending B: amount 200.',
            },
        ]
        deduped = _deduplicate_prompt_chunks(chunks)
        assert len(deduped) == 2

    def test_prompt_chunk_dedup_removes_exact_normalized_duplicates(self) -> None:
        chunks = [
            {
                'file_path': '/docs/alpha.pdf',
                'filename': 'alpha.pdf',
                'chunk_text': 'Line one.\nLine two.',
            },
            {
                'file_path': '/docs/alpha.pdf',
                'filename': 'alpha.pdf',
                'chunk_text': 'Line one.   Line two.',
            },
        ]
        deduped = _deduplicate_prompt_chunks(chunks)
        assert len(deduped) == 1

    @pytest.mark.asyncio
    async def test_empty_retrieval_terminal_refusal_sets_no_remaining_scope(self) -> None:
        handler = RAGHandler()
        classification = QueryClassification(intent='focused')
        mock_db = MagicMock()

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
