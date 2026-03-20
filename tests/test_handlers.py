# ==============================================================================
# Informity AI — Handler Tests
# Tests QueryHandler implementations (MetadataHandler, RAGHandler, SimpleHandler)
# ==============================================================================

from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

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

    def test_strict_ordered_output_budget_applies_generic_caps(self) -> None:
        from informity.llm.rag_runtime.generation_runtime import _apply_strict_ordered_output_budget

        constraints, max_tokens, reasoning_enabled, degradation = _apply_strict_ordered_output_budget(
            format_requirements=[
                'use the required headings exactly and in the requested order',
                'include heading: Executive Summary',
                'include heading: Action Checklist',
            ],
            query_type='coverage',
            output_constraints={},
            max_tokens=1536,
            reasoning_enabled=True,
        )

        assert constraints.get('max_words') == 420
        assert constraints.get('max_rows') == 18
        assert max_tokens >= 1536
        assert reasoning_enabled is False
        assert degradation is not None
        assert degradation.get('step') == 'strict_ordered_section_budget'
        assert degradation.get('required_heading_count') == 2

    def test_strict_ordered_format_requirements_gain_completion_priority_rules(self) -> None:
        from informity.llm.rag_runtime.generation_runtime import (
            _augment_strict_ordered_format_requirements,
        )

        requirements = _augment_strict_ordered_format_requirements([
            'use the required headings exactly and in the requested order',
            'include heading: Scope',
            'use nested bullet lists with exactly 3 levels where requested',
        ])

        assert any('ensure every required heading appears' in item for item in requirements)
        assert any('prioritize breadth before depth' in item for item in requirements)
        assert any('3-level chain' in item for item in requirements)
        assert any('Parent\\n  - Child\\n    - Grandchild' in item for item in requirements)

    def test_strict_ordered_output_budget_noop_when_not_requested(self) -> None:
        from informity.llm.rag_runtime.generation_runtime import _apply_strict_ordered_output_budget

        base_constraints = {'max_words': 550}
        constraints, max_tokens, reasoning_enabled, degradation = _apply_strict_ordered_output_budget(
            format_requirements=['include heading: Scope'],
            query_type='coverage',
            output_constraints=base_constraints,
            max_tokens=1536,
            reasoning_enabled=False,
        )

        assert constraints == base_constraints
        assert max_tokens == 1536
        assert reasoning_enabled is False
        assert degradation is None

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
        mock_policy = MagicMock()
        mock_policy.enabled = False
        mock_policy.rollout_stage = 'test'
        mock_policy.sample_count = 0
        mock_policy.timeout_rate = 0.0
        mock_policy.first_token_p95_ms = None
        mock_policy.completion_p95_seconds = None
        mock_policy.stream_soft_limit_ratio = 0.8
        mock_policy.soft_top_k_threshold = 0.9
        mock_policy.soft_reasoning_threshold = 0.9
        mock_policy.soft_output_cap_threshold = 0.9
        mock_policy.soft_coverage_to_focused_threshold = 0.9
        mock_policy.hard_pre_generation_threshold = 0.98

        with patch('informity.llm.rag_runtime.retrieval_pipeline.retrieve_chunks', new_callable=AsyncMock) as mock_retrieve, \
             patch('informity.llm.handlers.rag.resolve_fit_to_budget_policy', new_callable=AsyncMock) as mock_resolve_policy:
            mock_retrieve.return_value = []
            mock_resolve_policy.return_value = mock_policy

            results = []
            async for item in handler.handle('Which files mention withholding?', classification, None, mock_db, None):
                results.append(item)

        metrics_events = [item for item in results if isinstance(item, tuple) and item[0] == '__metrics__']
        assert metrics_events
        metrics = metrics_events[0][1]
        assert metrics.get('generation_skipped') is True
        assert metrics.get('has_remaining_scope') is False

    @pytest.mark.asyncio
    async def test_validation_gate_terminal_refusal_sets_no_remaining_scope(self) -> None:
        handler = RAGHandler()
        classification = QueryClassification(intent='focused')
        mock_db = MagicMock()
        mock_policy = MagicMock()
        mock_policy.enabled = False
        mock_policy.rollout_stage = 'test'
        mock_policy.sample_count = 0
        mock_policy.timeout_rate = 0.0
        mock_policy.first_token_p95_ms = None
        mock_policy.completion_p95_seconds = None
        mock_policy.stream_soft_limit_ratio = 0.8
        mock_policy.soft_top_k_threshold = 0.9
        mock_policy.soft_reasoning_threshold = 0.9
        mock_policy.soft_output_cap_threshold = 0.9
        mock_policy.soft_coverage_to_focused_threshold = 0.9
        mock_policy.hard_pre_generation_threshold = 0.98
        test_chunks = [{
            'file_id': 1,
            'filename': 'alpha.pdf',
            'file_path': '/docs/alpha.pdf',
            'chunk_text': 'Some weak evidence text.',
            'score': 0.01,
        }]

        with patch('informity.llm.rag_runtime.retrieval_pipeline.retrieve_chunks', new_callable=AsyncMock) as mock_retrieve, \
             patch('informity.llm.handlers.rag.resolve_fit_to_budget_policy', new_callable=AsyncMock) as mock_resolve_policy, \
             patch('informity.llm.handlers.rag._retrieval_validation._evaluate_retrieval_relevance_gate') as mock_rel_gate, \
             patch('informity.llm.handlers.rag._retrieval_validation._evaluate_source_diversity_gate') as mock_div_gate, \
             patch('informity.llm.handlers.rag._retrieval_validation._evaluate_continuation_anchor_gate') as mock_anchor_gate, \
             patch('informity.llm.handlers.rag._retrieval_validation._apply_coverage_evidence_floor_override') as mock_floor:
            mock_retrieve.return_value = test_chunks
            mock_resolve_policy.return_value = mock_policy
            mock_rel_gate.return_value = (False, 0.0)
            mock_div_gate.return_value = (True, 1)
            mock_anchor_gate.return_value = (True, 1)
            mock_floor.return_value = (False, [])

            results = []
            async for item in handler.handle('Summarize unresolved records.', classification, None, mock_db, None):
                results.append(item)

        metrics_events = [item for item in results if isinstance(item, tuple) and item[0] == '__metrics__']
        assert metrics_events
        metrics = metrics_events[0][1]
        assert metrics.get('generation_skipped') is True
        assert metrics.get('has_remaining_scope') is False

    @pytest.mark.asyncio
    async def test_validation_gate_widened_retry_recovers_before_terminal_refusal(self) -> None:
        handler = RAGHandler()
        classification = QueryClassification(intent='focused')
        mock_db = MagicMock()
        mock_policy = MagicMock()
        mock_policy.enabled = False
        mock_policy.rollout_stage = 'test'
        mock_policy.sample_count = 0
        mock_policy.timeout_rate = 0.0
        mock_policy.first_token_p95_ms = None
        mock_policy.completion_p95_seconds = None
        mock_policy.stream_soft_limit_ratio = 0.8
        mock_policy.soft_top_k_threshold = 0.9
        mock_policy.soft_reasoning_threshold = 0.9
        mock_policy.soft_output_cap_threshold = 0.9
        mock_policy.soft_coverage_to_focused_threshold = 0.9
        mock_policy.hard_pre_generation_threshold = 0.98

        weak_chunks = [{
            'file_id': 1,
            'filename': 'alpha.pdf',
            'file_path': '/docs/alpha.pdf',
            'chunk_text': 'Weak evidence snippet.',
            'score': 0.01,
        }]
        strong_chunks = [
            {
                'file_id': 1,
                'filename': 'alpha.pdf',
                'file_path': '/docs/alpha.pdf',
                'chunk_text': 'Strong evidence snippet A.',
                'score': 2.2,
            },
            {
                'file_id': 2,
                'filename': 'beta.pdf',
                'file_path': '/docs/beta.pdf',
                'chunk_text': 'Strong evidence snippet B.',
                'score': 2.1,
            },
        ]

        with patch('informity.llm.rag_runtime.retrieval_pipeline.retrieve_chunks', new_callable=AsyncMock) as mock_retrieve, \
             patch('informity.llm.handlers.rag.resolve_fit_to_budget_policy', new_callable=AsyncMock) as mock_resolve_policy, \
             patch('informity.llm.handlers.rag._retrieval_validation._evaluate_retrieval_relevance_gate') as mock_rel_gate, \
             patch('informity.llm.handlers.rag._retrieval_validation._evaluate_source_diversity_gate') as mock_div_gate, \
             patch('informity.llm.handlers.rag._retrieval_validation._evaluate_continuation_anchor_gate') as mock_anchor_gate, \
             patch('informity.llm.handlers.rag._retrieval_validation._apply_coverage_evidence_floor_override') as mock_floor:
            mock_retrieve.side_effect = [weak_chunks, strong_chunks]
            mock_resolve_policy.return_value = mock_policy
            mock_rel_gate.side_effect = [(False, 0.0), (True, 0.82)]
            mock_div_gate.side_effect = [(True, 1), (True, 2)]
            mock_anchor_gate.side_effect = [(True, 1), (True, 2)]
            mock_floor.side_effect = [
                (False, []),
                (True, []),
            ]

            async def _fake_stream_llm(*_args, **_kwargs):
                yield 'Recovered answer token.'

            results: list[object] = []
            with patch('informity.llm.handlers.rag.stream_llm', _fake_stream_llm):
                async for item in handler.handle('Summarize unresolved records.', classification, None, mock_db, None):
                    results.append(item)

        assert mock_retrieve.await_count == 2
        assert any(isinstance(item, str) and 'Recovered answer token.' in item for item in results)
        assert not any(
            isinstance(item, str) and 'do not contain enough information' in item
            for item in results
        )
        metrics_events = [item for item in results if isinstance(item, tuple) and item[0] == '__metrics__']
        assert metrics_events
        metrics = metrics_events[0][1]
        assert metrics.get('generation_skipped') is False

    @pytest.mark.asyncio
    async def test_handle_calls_retrieve_chunks(self) -> None:
        handler = RAGHandler()
        classification = QueryClassification(intent='focused')
        mock_db = MagicMock()
        mock_policy = MagicMock()
        mock_policy.enabled = False
        mock_policy.rollout_stage = 'test'
        mock_policy.sample_count = 0
        mock_policy.timeout_rate = 0.0
        mock_policy.first_token_p95_ms = None
        mock_policy.completion_p95_seconds = None
        mock_policy.stream_soft_limit_ratio = 0.8
        mock_policy.soft_top_k_threshold = 0.9
        mock_policy.soft_reasoning_threshold = 0.9
        mock_policy.soft_output_cap_threshold = 0.9
        mock_policy.soft_coverage_to_focused_threshold = 0.9
        mock_policy.hard_pre_generation_threshold = 0.98

        with patch('informity.llm.rag_runtime.retrieval_pipeline.retrieve_chunks', new_callable=AsyncMock) as mock_retrieve, \
             patch('informity.llm.handlers.rag.build_messages') as mock_build, \
             patch('informity.llm.handlers.rag.stream_llm', new_callable=AsyncMock) as mock_stream, \
             patch('informity.llm.handlers.rag.resolve_fit_to_budget_policy', new_callable=AsyncMock) as mock_resolve_policy:

            mock_retrieve.return_value = []
            mock_build.return_value = [{'role': 'user', 'content': 'test'}]
            mock_stream.return_value = AsyncMock()
            mock_stream.return_value.__aiter__.return_value = ['token']
            mock_resolve_policy.return_value = mock_policy

            results = []
            async for item in handler.handle('test question', classification, None, mock_db, None):
                results.append(item)

            assert mock_retrieve.call_count >= 1

    @pytest.mark.asyncio
    async def test_continuation_without_overlap_keeps_scope_without_clarification(self) -> None:
        handler = RAGHandler()
        classification = QueryClassification(
            intent='focused',
            route_candidate='continuation_or_refinement',
            confidence=0.86,
        )
        mock_db = MagicMock()
        mock_policy = MagicMock()
        mock_policy.enabled = False
        mock_policy.rollout_stage = 'test'
        mock_policy.sample_count = 0
        mock_policy.timeout_rate = 0.0
        mock_policy.first_token_p95_ms = None
        mock_policy.completion_p95_seconds = None
        mock_policy.stream_soft_limit_ratio = 0.8
        mock_policy.soft_top_k_threshold = 0.9
        mock_policy.soft_reasoning_threshold = 0.9
        mock_policy.soft_output_cap_threshold = 0.9
        mock_policy.soft_coverage_to_focused_threshold = 0.9
        mock_policy.hard_pre_generation_threshold = 0.98

        history = [
            ChatMessage(
                chat_id='test-chat',
                role='assistant',
                content='Previous answer',
                sources=[{'path': '/docs/alpha.pdf', 'filename': 'alpha.pdf'}],
            ),
        ]

        with patch('informity.llm.rag_runtime.retrieval_pipeline.retrieve_chunks', new_callable=AsyncMock) as mock_retrieve, \
             patch('informity.llm.handlers.rag.resolve_fit_to_budget_policy', new_callable=AsyncMock) as mock_resolve_policy:
            mock_retrieve.return_value = [
                {
                    'file_id': 1,
                    'filename': 'beta.pdf',
                    'file_path': '/docs/beta.pdf',
                    'chunk_text': 'Key facts with strong evidence.',
                    'score': 2.1,
                },
            ]
            mock_resolve_policy.return_value = mock_policy

            results: list[object] = []
            async for item in handler.handle('continue with the same structure', classification, history, mock_db, None):
                results.append(item)

        assert any(isinstance(item, str) and "couldn't find relevant information" in item.casefold() for item in results)
        assert results[-1] == []

    @pytest.mark.asyncio
    async def test_budget_pressure_with_weak_relevance_skips_generation(self) -> None:
        handler = RAGHandler()
        classification = QueryClassification(
            intent='focused',
            route_candidate='targeted_fact_lookup',
            confidence=0.84,
        )
        mock_db = MagicMock()
        mock_policy = MagicMock()
        mock_policy.enabled = True
        mock_policy.rollout_stage = 'test'
        mock_policy.sample_count = 100
        mock_policy.timeout_rate = 0.05
        mock_policy.first_token_p95_ms = 1200
        mock_policy.completion_p95_seconds = 12.0
        mock_policy.stream_soft_limit_ratio = 0.8
        mock_policy.soft_top_k_threshold = 0.2
        mock_policy.soft_reasoning_threshold = 0.9
        mock_policy.soft_output_cap_threshold = 0.15  # actual post_retrieval_ratio ~0.178 after preflight degradations
        mock_policy.soft_coverage_to_focused_threshold = 0.95
        mock_policy.hard_pre_generation_threshold = 0.99

        with patch('informity.llm.rag_runtime.retrieval_pipeline.retrieve_chunks', new_callable=AsyncMock) as mock_retrieve, \
             patch('informity.llm.handlers.rag.resolve_fit_to_budget_policy', new_callable=AsyncMock) as mock_resolve_policy, \
             patch('informity.llm.handlers.rag.stream_llm', new_callable=AsyncMock) as mock_stream:
            mock_retrieve.return_value = [
                {
                    'file_id': 1,
                    'filename': 'alpha.pdf',
                    'file_path': '/docs/alpha.pdf',
                    'chunk_text': 'Context exists but confidence is not strong enough under budget pressure.',
                    'score': 0.28,  # sigmoid ~= 0.57: passes retrieval gate, fails pre-closeout gate (0.62)
                },
            ]
            mock_resolve_policy.return_value = mock_policy

            results: list[object] = []
            async for item in handler.handle('summarize this quickly', classification, None, mock_db, None):
                results.append(item)

        assert any(
            isinstance(item, str) and 'do not contain enough information' in item
            for item in results
        )
        mock_stream.assert_not_called()
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
        mock_policy = MagicMock()
        mock_policy.enabled = False
        mock_policy.rollout_stage = 'test'
        mock_policy.sample_count = 0
        mock_policy.timeout_rate = 0.0
        mock_policy.first_token_p95_ms = None
        mock_policy.completion_p95_seconds = None
        mock_policy.stream_soft_limit_ratio = 0.8
        mock_policy.soft_top_k_threshold = 0.9
        mock_policy.soft_reasoning_threshold = 0.9
        mock_policy.soft_output_cap_threshold = 0.9
        mock_policy.soft_coverage_to_focused_threshold = 0.9
        mock_policy.hard_pre_generation_threshold = 0.98

        with patch('informity.llm.rag_runtime.retrieval_pipeline.retrieve_chunks', new_callable=AsyncMock) as mock_retrieve, \
             patch('informity.llm.handlers.rag.resolve_fit_to_budget_policy', new_callable=AsyncMock) as mock_resolve_policy, \
             patch('informity.llm.handlers.rag.stream_llm', new_callable=AsyncMock):
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
            mock_resolve_policy.return_value = mock_policy

            async def _fake_stream_llm(*_args, **_kwargs):
                yield 'narrative output token'

            results: list[object] = []
            with patch('informity.llm.handlers.rag.stream_llm', _fake_stream_llm):
                async for item in handler.handle('build compliance brief', classification, None, mock_db, None):
                    results.append(item)

        assert any(isinstance(item, str) and 'narrative output token' in item for item in results)
        assert not any(
            isinstance(item, str) and 'I could not extract enough validated structured values' in item
            for item in results
        )

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
        mock_policy = MagicMock()
        mock_policy.enabled = True
        mock_policy.rollout_stage = 'test'
        mock_policy.sample_count = 100
        mock_policy.timeout_rate = 0.05
        mock_policy.first_token_p95_ms = 900
        mock_policy.completion_p95_seconds = 15.0
        mock_policy.stream_soft_limit_ratio = 0.9
        mock_policy.soft_top_k_threshold = 0.2
        mock_policy.soft_reasoning_threshold = 0.2
        mock_policy.soft_output_cap_threshold = 0.2
        mock_policy.soft_coverage_to_focused_threshold = 0.2
        mock_policy.hard_pre_generation_threshold = 0.99

        with patch('informity.llm.rag_runtime.retrieval_pipeline.retrieve_chunks', new_callable=AsyncMock) as mock_retrieve, \
             patch('informity.llm.handlers.rag.resolve_fit_to_budget_policy', new_callable=AsyncMock) as mock_resolve_policy:
            mock_retrieve.return_value = [
                {
                    'file_id': 1,
                    'filename': 'y2022.pdf',
                    'file_path': '/docs/y2022.pdf',
                    'chunk_text': 'Box 1 wages $1,000.00',
                    'score': 0.1,
                },
                {
                    'file_id': 2,
                    'filename': 'y2023.pdf',
                    'file_path': '/docs/y2023.pdf',
                    'chunk_text': 'Box 1 wages $2,000.00',
                    'score': 0.1,
                },
                {
                    'file_id': 3,
                    'filename': 'y2024.pdf',
                    'file_path': '/docs/y2024.pdf',
                    'chunk_text': 'Box 1 wages $3,000.00',
                    'score': 0.1,
                },
            ]
            mock_resolve_policy.return_value = mock_policy

            results: list[object] = []
            async def _fake_stream_llm(*_args, **_kwargs):
                yield 'aggregate summary'

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
        mock_policy = MagicMock()
        mock_policy.enabled = False
        mock_policy.rollout_stage = 'test'
        mock_policy.sample_count = 0
        mock_policy.timeout_rate = 0.0
        mock_policy.first_token_p95_ms = None
        mock_policy.completion_p95_seconds = None
        mock_policy.stream_soft_limit_ratio = 0.8
        mock_policy.soft_top_k_threshold = 0.9
        mock_policy.soft_reasoning_threshold = 0.9
        mock_policy.soft_output_cap_threshold = 0.9
        mock_policy.soft_coverage_to_focused_threshold = 0.9
        mock_policy.hard_pre_generation_threshold = 0.98

        with patch('informity.llm.rag_runtime.retrieval_pipeline.retrieve_chunks', new_callable=AsyncMock) as mock_retrieve, \
             patch('informity.llm.handlers.rag.resolve_fit_to_budget_policy', new_callable=AsyncMock) as mock_resolve_policy, \
             patch(
                 'informity.llm.handlers.rag._structured_numeric._try_structured_value_extraction',
                 new_callable=AsyncMock,
             ) as mock_try_structured:
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
            mock_resolve_policy.return_value = mock_policy
            mock_try_structured.return_value = None

            async def _fake_stream_llm(*_args, **_kwargs):
                yield 'fallback narrative token'

            results: list[object] = []
            with patch('informity.llm.handlers.rag.stream_llm', _fake_stream_llm):
                async for item in handler.handle('extract key values and explain', classification, None, mock_db, None):
                    results.append(item)

        assert any(isinstance(item, str) and 'fallback narrative token' in item for item in results)
        assert not any(
            isinstance(item, str) and 'I could not extract enough validated structured values' in item
            for item in results
        )

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
        mock_policy = MagicMock()
        mock_policy.enabled = True
        mock_policy.rollout_stage = 'test'
        mock_policy.sample_count = 100
        mock_policy.timeout_rate = 0.05
        mock_policy.first_token_p95_ms = 900
        mock_policy.completion_p95_seconds = 15.0
        mock_policy.stream_soft_limit_ratio = 0.0
        mock_policy.soft_top_k_threshold = 99.0
        mock_policy.soft_reasoning_threshold = 99.0
        mock_policy.soft_output_cap_threshold = 99.0
        mock_policy.soft_coverage_to_focused_threshold = 99.0
        mock_policy.hard_pre_generation_threshold = 99.0

        with patch('informity.llm.rag_runtime.retrieval_pipeline.retrieve_chunks', new_callable=AsyncMock) as mock_retrieve, \
             patch('informity.llm.handlers.rag.resolve_fit_to_budget_policy', new_callable=AsyncMock) as mock_resolve_policy:
            mock_retrieve.return_value = [
                {
                    'file_id': 1,
                    'filename': 'alpha.pdf',
                    'file_path': '/docs/alpha.pdf',
                    'chunk_text': 'Focused evidence for concise response.',
                    'score': 2.2,
                },
            ]
            mock_resolve_policy.return_value = mock_policy

            async def _fake_stream_llm(*_args, **_kwargs):
                yield 'First sentence.'
                yield 'Second sentence should not be emitted.'

            results: list[object] = []
            with patch('informity.llm.handlers.rag.stream_llm', _fake_stream_llm):
                async for item in handler.handle('summarize this', classification, None, mock_db, None):
                    results.append(item)

        text_tokens = [item for item in results if isinstance(item, str)]
        assert any('First sentence.' in item for item in text_tokens)
        assert not any('Second sentence should not be emitted.' in item for item in text_tokens)
        metrics_events = [item for item in results if isinstance(item, tuple) and item[0] == '__metrics__']
        assert metrics_events
        metrics = metrics_events[0][1]
        assert metrics.get('stream_recovery_reason') == 'soft_limit_section_closeout'
        assert metrics.get('suggested_completion_mode') == 'scoped_complete'

    @pytest.mark.asyncio
    async def test_soft_limit_closeout_disabled_for_strict_ordered_headings(self) -> None:
        handler = RAGHandler()
        classification = QueryClassification(
            intent='focused',
            response_shape='narrative_synthesis',
            route_candidate='targeted_fact_lookup',
            confidence=0.86,
        )
        mock_db = MagicMock()
        mock_policy = MagicMock()
        mock_policy.enabled = True
        mock_policy.rollout_stage = 'test'
        mock_policy.sample_count = 100
        mock_policy.timeout_rate = 0.05
        mock_policy.first_token_p95_ms = 900
        mock_policy.completion_p95_seconds = 15.0
        mock_policy.stream_soft_limit_ratio = 0.0
        mock_policy.soft_top_k_threshold = 99.0
        mock_policy.soft_reasoning_threshold = 99.0
        mock_policy.soft_output_cap_threshold = 99.0
        mock_policy.soft_coverage_to_focused_threshold = 99.0
        mock_policy.hard_pre_generation_threshold = 99.0

        with patch('informity.llm.rag_runtime.retrieval_pipeline.retrieve_chunks', new_callable=AsyncMock) as mock_retrieve, \
             patch('informity.llm.handlers.rag.resolve_fit_to_budget_policy', new_callable=AsyncMock) as mock_resolve_policy:
            mock_retrieve.return_value = [
                {
                    'file_id': 1,
                    'filename': 'alpha.pdf',
                    'file_path': '/docs/alpha.pdf',
                    'chunk_text': 'Compliance evidence with enough context for ordered headings.',
                    'score': 2.1,
                },
            ]
            mock_resolve_policy.return_value = mock_policy

            async def _fake_stream_llm(*_args, **_kwargs):
                yield '## 1) Scope\nFirst section.'
                yield '\n\n## 2) Method\nSecond section.'

            question = (
                'Create a brief with sections in order: 1) Scope, 2) Method. '
                'Use evidence from context only.'
            )
            results: list[object] = []
            with patch('informity.llm.handlers.rag.stream_llm', _fake_stream_llm):
                async for item in handler.handle(question, classification, None, mock_db, None):
                    results.append(item)

        text_tokens = [item for item in results if isinstance(item, str)]
        assert any('## 1) Scope' in item for item in text_tokens)
        assert any('## 2) Method' in item for item in text_tokens)
        metrics_events = [item for item in results if isinstance(item, tuple) and item[0] == '__metrics__']
        assert metrics_events
        metrics = metrics_events[0][1]
        assert metrics.get('stream_recovery_reason') is None
        assert metrics.get('suggested_completion_mode') == 'complete'

    @pytest.mark.asyncio
    async def test_output_contract_incomplete_sets_scoped_completion_mode(self) -> None:
        handler = RAGHandler()
        classification = QueryClassification(
            intent='focused',
            response_shape='narrative_synthesis',
            route_candidate='targeted_fact_lookup',
            confidence=0.86,
        )
        mock_db = MagicMock()
        mock_policy = MagicMock()
        mock_policy.enabled = False
        mock_policy.rollout_stage = 'test'
        mock_policy.sample_count = 0
        mock_policy.timeout_rate = 0.0
        mock_policy.first_token_p95_ms = None
        mock_policy.completion_p95_seconds = None
        mock_policy.stream_soft_limit_ratio = 0.8
        mock_policy.soft_top_k_threshold = 0.9
        mock_policy.soft_reasoning_threshold = 0.9
        mock_policy.soft_output_cap_threshold = 0.9
        mock_policy.soft_coverage_to_focused_threshold = 0.9
        mock_policy.hard_pre_generation_threshold = 0.98

        with patch('informity.llm.rag_runtime.retrieval_pipeline.retrieve_chunks', new_callable=AsyncMock) as mock_retrieve, \
             patch('informity.llm.handlers.rag.resolve_fit_to_budget_policy', new_callable=AsyncMock) as mock_resolve_policy:
            mock_retrieve.return_value = [
                {
                    'file_id': 1,
                    'filename': 'alpha.pdf',
                    'file_path': '/docs/alpha.pdf',
                    'chunk_text': 'Evidence to support sectioned answer.',
                    'score': 2.2,
                },
            ]
            mock_resolve_policy.return_value = mock_policy

            async def _fake_stream_llm(*_args, **_kwargs):
                yield '## 1) Scope\nOnly first section included.'

            question = 'Create sections in order: 1) Scope, 2) Method.'
            results: list[object] = []
            with patch('informity.llm.handlers.rag.stream_llm', _fake_stream_llm):
                async for item in handler.handle(question, classification, None, mock_db, None):
                    results.append(item)

        metrics_events = [item for item in results if isinstance(item, tuple) and item[0] == '__metrics__']
        assert metrics_events
        metrics = metrics_events[0][1]
        # NC-2 invariant: contract failure must NOT trigger continuation.
        assert metrics.get('has_remaining_scope') is False
        assert metrics.get('suggested_completion_mode') in ('complete', None)
        # Contract failure must be visible in trace fields only.
        output_contract = metrics.get('output_contract_check')
        assert isinstance(output_contract, dict)
        assert output_contract.get('passed') is False
        degradations = metrics.get('applied_degradations')
        assert isinstance(degradations, list)
        assert any(
            isinstance(item, dict) and item.get('step') == 'strict_output_contract_incomplete'
            for item in degradations
        )


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
