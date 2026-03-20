# ==============================================================================
# Informity AI — Planner Tests
# Tests for QueryPlan data model, parsing, validation, and helper functions.
# LLM-calling build_plan() is excluded (requires live engine).
# ==============================================================================

from __future__ import annotations

from informity.llm.planner import (
    RetrievalFilters,
    _filters_to_kwargs,
    _normalize_output_shape,
    _parse_answer_section,
    _parse_plan_step,
    _parse_retrieval_filters,
    _validate_and_build_plan,
    build_corpus_summary,
)

# ==============================================================================
# build_corpus_summary
# ==============================================================================

class TestBuildCorpusSummary:

    def test_basic(self) -> None:
        result = build_corpus_summary([2022, 2023], ['document', 'data'], 12)
        assert '12 files indexed' in result
        assert 'years: 2022' in result
        assert '2023' in result
        assert 'document' in result
        assert 'data' in result

    def test_single_year(self) -> None:
        result = build_corpus_summary([2024], ['plaintext'], 3)
        assert 'years: 2024' in result
        assert '(1 years)' not in result  # Single year shown without range

    def test_no_years_no_categories(self) -> None:
        result = build_corpus_summary([], [], 0)
        assert '0 files indexed' in result
        assert 'years' not in result
        assert 'categories' not in result

    def test_categories_sorted(self) -> None:
        result = build_corpus_summary([], ['web', 'code', 'data'], 5)
        categories_part = result.split('categories: ')[1]
        # Alphabetical order: code, data, web
        assert categories_part.index('code') < categories_part.index('data')
        assert categories_part.index('data') < categories_part.index('web')


# ==============================================================================
# _normalize_output_shape
# ==============================================================================

class TestNormalizeOutputShape:

    def test_valid_values_passthrough(self) -> None:
        assert _normalize_output_shape('structured_extract') == 'structured_extract'
        assert _normalize_output_shape('narrative_synthesis') == 'narrative_synthesis'
        assert _normalize_output_shape('metadata_table') == 'metadata_table'
        assert _normalize_output_shape('hybrid') == 'hybrid'

    def test_invalid_defaults_to_narrative(self) -> None:
        assert _normalize_output_shape('unknown') == 'narrative_synthesis'
        assert _normalize_output_shape('') == 'narrative_synthesis'
        assert _normalize_output_shape('NARRATIVE_SYNTHESIS') == 'narrative_synthesis'  # Not exact case match

    def test_strips_whitespace(self) -> None:
        assert _normalize_output_shape('  hybrid  ') == 'hybrid'


# ==============================================================================
# _filters_to_kwargs
# ==============================================================================

class TestFiltersToKwargs:

    def test_empty_filters_produces_empty_dict(self) -> None:
        filters = RetrievalFilters()
        assert _filters_to_kwargs(filters) == {}

    def test_populated_fields_included(self) -> None:
        filters = RetrievalFilters(
            year_filter=2023,
            category_filter='document',
            extension_filter='pdf',
            filename_filter='report',
            source_terms_filter=['q4', 'revenue'],
            block_type_filter='table',
            section_filter='financials',
        )
        kwargs = _filters_to_kwargs(filters)
        assert kwargs['year_filter'] == 2023
        assert kwargs['category_filter'] == 'document'
        assert kwargs['extension_filter'] == 'pdf'
        assert kwargs['filename_filter'] == 'report'
        assert kwargs['source_terms_filter'] == ['q4', 'revenue']
        assert kwargs['block_type_filter'] == 'table'
        assert kwargs['section_filter'] == 'financials'

    def test_none_fields_excluded(self) -> None:
        filters = RetrievalFilters(year_filter=2022)
        kwargs = _filters_to_kwargs(filters)
        assert 'year_filter' in kwargs
        assert 'category_filter' not in kwargs
        assert 'extension_filter' not in kwargs


# ==============================================================================
# _parse_retrieval_filters
# ==============================================================================

class TestParseRetrievalFilters:

    def test_non_dict_returns_empty_filters(self) -> None:
        result = _parse_retrieval_filters(None)
        assert result == RetrievalFilters()
        result = _parse_retrieval_filters('string')
        assert result == RetrievalFilters()
        result = _parse_retrieval_filters([1, 2])
        assert result == RetrievalFilters()

    def test_valid_year_filter(self) -> None:
        result = _parse_retrieval_filters({'year_filter': 2023})
        assert result.year_filter == 2023

    def test_year_filter_out_of_range_rejected(self) -> None:
        result = _parse_retrieval_filters({'year_filter': 1800})
        assert result.year_filter is None
        result = _parse_retrieval_filters({'year_filter': 2200})
        assert result.year_filter is None

    def test_year_filter_invalid_type_rejected(self) -> None:
        result = _parse_retrieval_filters({'year_filter': 'not_a_year'})
        assert result.year_filter is None

    def test_valid_category_filter(self) -> None:
        for cat in ('document', 'plaintext', 'data', 'web', 'code'):
            result = _parse_retrieval_filters({'category_filter': cat})
            assert result.category_filter == cat

    def test_invalid_category_rejected(self) -> None:
        result = _parse_retrieval_filters({'category_filter': 'spreadsheet'})
        assert result.category_filter is None

    def test_valid_block_type(self) -> None:
        for bt in ('table', 'form', 'narrative'):
            result = _parse_retrieval_filters({'block_type_filter': bt})
            assert result.block_type_filter == bt

    def test_invalid_block_type_rejected(self) -> None:
        result = _parse_retrieval_filters({'block_type_filter': 'chart'})
        assert result.block_type_filter is None

    def test_source_terms_filter_valid(self) -> None:
        result = _parse_retrieval_filters({'source_terms_filter': ['q4', 'revenue']})
        assert result.source_terms_filter == ['q4', 'revenue']

    def test_source_terms_filter_empty_list(self) -> None:
        result = _parse_retrieval_filters({'source_terms_filter': []})
        assert result.source_terms_filter is None

    def test_source_terms_filter_blank_items_stripped(self) -> None:
        result = _parse_retrieval_filters({'source_terms_filter': ['  ', 'revenue', '']})
        assert result.source_terms_filter == ['revenue']

    def test_null_fields_remain_none(self) -> None:
        result = _parse_retrieval_filters({'year_filter': None, 'category_filter': None})
        assert result.year_filter is None
        assert result.category_filter is None

    def test_extension_and_filename_filter(self) -> None:
        result = _parse_retrieval_filters({'extension_filter': 'xlsx', 'filename_filter': 'annual'})
        assert result.extension_filter == 'xlsx'
        assert result.filename_filter == 'annual'

    def test_blank_string_fields_rejected(self) -> None:
        result = _parse_retrieval_filters({'extension_filter': '  ', 'filename_filter': ''})
        assert result.extension_filter is None
        assert result.filename_filter is None


# ==============================================================================
# _parse_plan_step
# ==============================================================================

class TestParsePlanStep:

    def test_valid_step(self) -> None:
        raw = {
            'step_id': 1,
            'description': 'Get revenue from 2023 reports',
            'sub_query': 'What was the total revenue in 2023?',
            'filters': {'year_filter': 2023},
            'retrieval_mode': 'focused',
            'expected_output': 'Revenue figure',
        }
        step = _parse_plan_step(raw)
        assert step is not None
        assert step.step_id == 1
        assert step.sub_query == 'What was the total revenue in 2023?'
        assert step.retrieval_mode == 'focused'
        assert step.filters.year_filter == 2023

    def test_missing_sub_query_returns_none(self) -> None:
        raw = {'step_id': 1, 'description': 'x', 'retrieval_mode': 'focused'}
        assert _parse_plan_step(raw) is None

    def test_blank_sub_query_returns_none(self) -> None:
        raw = {'step_id': 1, 'sub_query': '  ', 'retrieval_mode': 'focused'}
        assert _parse_plan_step(raw) is None

    def test_non_dict_returns_none(self) -> None:
        assert _parse_plan_step(None) is None
        assert _parse_plan_step('string') is None
        assert _parse_plan_step([1, 2]) is None

    def test_invalid_retrieval_mode_defaults_to_coverage(self) -> None:
        raw = {'step_id': 1, 'sub_query': 'query', 'retrieval_mode': 'unknown'}
        step = _parse_plan_step(raw)
        assert step is not None
        assert step.retrieval_mode == 'coverage'

    def test_valid_coverage_mode(self) -> None:
        raw = {'step_id': 2, 'sub_query': 'broad query', 'retrieval_mode': 'coverage'}
        step = _parse_plan_step(raw)
        assert step is not None
        assert step.retrieval_mode == 'coverage'

    def test_invalid_step_id_defaults_to_zero(self) -> None:
        raw = {'step_id': 'bad', 'sub_query': 'query', 'retrieval_mode': 'focused'}
        step = _parse_plan_step(raw)
        assert step is not None
        assert step.step_id == 0

    def test_missing_step_id_defaults_to_zero(self) -> None:
        raw = {'sub_query': 'query', 'retrieval_mode': 'focused'}
        step = _parse_plan_step(raw)
        assert step is not None
        assert step.step_id == 0


# ==============================================================================
# _parse_answer_section
# ==============================================================================

class TestParseAnswerSection:

    def test_valid_section(self) -> None:
        raw = {
            'heading': '## Revenue Overview',
            'scope': 'Covers total revenue for 2023.',
            'estimated_complexity': 'simple',
        }
        section = _parse_answer_section(raw)
        assert section is not None
        assert section.heading == '## Revenue Overview'
        assert section.scope == 'Covers total revenue for 2023.'
        assert section.estimated_complexity == 'simple'

    def test_detailed_complexity(self) -> None:
        raw = {'heading': '## Analysis', 'scope': 'Deep analysis.', 'estimated_complexity': 'detailed'}
        section = _parse_answer_section(raw)
        assert section is not None
        assert section.estimated_complexity == 'detailed'

    def test_invalid_complexity_defaults_to_detailed(self) -> None:
        raw = {'heading': '## Test', 'scope': 'test scope', 'estimated_complexity': 'unknown'}
        section = _parse_answer_section(raw)
        assert section is not None
        assert section.estimated_complexity == 'detailed'

    def test_missing_heading_returns_none(self) -> None:
        raw = {'scope': 'Some scope', 'estimated_complexity': 'simple'}
        assert _parse_answer_section(raw) is None

    def test_blank_heading_returns_none(self) -> None:
        raw = {'heading': '  ', 'scope': 'Some scope'}
        assert _parse_answer_section(raw) is None

    def test_missing_scope_returns_none(self) -> None:
        raw = {'heading': '## Test', 'estimated_complexity': 'simple'}
        assert _parse_answer_section(raw) is None

    def test_blank_scope_returns_none(self) -> None:
        raw = {'heading': '## Test', 'scope': '  '}
        assert _parse_answer_section(raw) is None

    def test_non_dict_returns_none(self) -> None:
        assert _parse_answer_section(None) is None
        assert _parse_answer_section('string') is None


# ==============================================================================
# _validate_and_build_plan
# ==============================================================================

class TestValidateAndBuildPlan:

    def _minimal_valid_data(self) -> dict:
        return {
            'steps': [
                {
                    'step_id': 1,
                    'description': 'Get data',
                    'sub_query': 'What is the revenue?',
                    'filters': {},
                    'retrieval_mode': 'focused',
                    'expected_output': 'Revenue figure',
                },
            ],
            'answer_sections': [
                {
                    'heading': '## Revenue',
                    'scope': 'Covers revenue figures.',
                    'estimated_complexity': 'simple',
                },
            ],
            'aggregation_mode': 'synthesize',
            'output_shape': 'narrative_synthesis',
        }

    def test_valid_plan_built(self) -> None:
        plan = _validate_and_build_plan(self._minimal_valid_data(), 'test query')
        assert plan is not None
        assert len(plan.steps) == 1
        assert len(plan.answer_sections) == 1
        assert plan.aggregation_mode == 'synthesize'
        assert plan.output_shape == 'narrative_synthesis'

    def test_no_valid_sections_returns_none(self) -> None:
        data = self._minimal_valid_data()
        data['answer_sections'] = []
        assert _validate_and_build_plan(data, 'test query') is None

    def test_invalid_sections_only_returns_none(self) -> None:
        data = self._minimal_valid_data()
        data['answer_sections'] = [{'heading': '', 'scope': ''}]
        assert _validate_and_build_plan(data, 'test query') is None

    def test_invalid_steps_are_dropped_not_fatal(self) -> None:
        data = self._minimal_valid_data()
        data['steps'] = [
            {'step_id': 1, 'sub_query': '', 'retrieval_mode': 'focused'},  # Invalid: blank sub_query
            {'step_id': 2, 'sub_query': 'valid query', 'retrieval_mode': 'focused'},
        ]
        plan = _validate_and_build_plan(data, 'test query')
        assert plan is not None
        assert len(plan.steps) == 1
        assert plan.steps[0].step_id == 2

    def test_invalid_aggregation_mode_defaults_to_synthesize(self) -> None:
        data = self._minimal_valid_data()
        data['aggregation_mode'] = 'invalid_mode'
        plan = _validate_and_build_plan(data, 'test query')
        assert plan is not None
        assert plan.aggregation_mode == 'synthesize'

    def test_valid_aggregation_modes(self) -> None:
        for mode in ('merge', 'compare', 'synthesize'):
            data = self._minimal_valid_data()
            data['aggregation_mode'] = mode
            plan = _validate_and_build_plan(data, 'test query')
            assert plan is not None
            assert plan.aggregation_mode == mode

    def test_invalid_output_shape_defaults_to_narrative(self) -> None:
        data = self._minimal_valid_data()
        data['output_shape'] = 'bad_shape'
        plan = _validate_and_build_plan(data, 'test query')
        assert plan is not None
        assert plan.output_shape == 'narrative_synthesis'

    def test_steps_capped_at_max(self) -> None:
        data = self._minimal_valid_data()
        data['steps'] = [
            {'step_id': i, 'sub_query': f'query {i}', 'retrieval_mode': 'focused'}
            for i in range(1, 10)  # 9 steps — must be capped at planner_max_steps
        ]
        plan = _validate_and_build_plan(data, 'test query')
        assert plan is not None
        from informity.config import settings
        assert len(plan.steps) <= settings.planner_max_steps

    def test_sections_capped_at_max(self) -> None:
        data = self._minimal_valid_data()
        data['answer_sections'] = [
            {'heading': f'## Section {i}', 'scope': f'Scope {i}.', 'estimated_complexity': 'simple'}
            for i in range(1, 15)  # 14 sections — must be capped at planner_max_sections
        ]
        plan = _validate_and_build_plan(data, 'test query')
        assert plan is not None
        from informity.config import settings
        assert len(plan.answer_sections) <= settings.planner_max_sections

    def test_non_list_steps_treated_as_empty(self) -> None:
        data = self._minimal_valid_data()
        data['steps'] = 'not a list'
        plan = _validate_and_build_plan(data, 'test query')
        assert plan is not None
        assert plan.steps == []

    def test_non_list_sections_returns_none(self) -> None:
        data = self._minimal_valid_data()
        data['answer_sections'] = 'not a list'
        assert _validate_and_build_plan(data, 'test query') is None

    def test_multi_section_multi_step_plan(self) -> None:
        data = {
            'steps': [
                {'step_id': 1, 'sub_query': 'Q1 revenue?', 'retrieval_mode': 'focused', 'filters': {}},
                {'step_id': 2, 'sub_query': 'Q2 revenue?', 'retrieval_mode': 'coverage', 'filters': {'year_filter': 2023}},
            ],
            'answer_sections': [
                {'heading': '## Q1 Analysis', 'scope': 'Q1 details.', 'estimated_complexity': 'detailed'},
                {'heading': '## Q2 Analysis', 'scope': 'Q2 details.', 'estimated_complexity': 'simple'},
                {'heading': '## Comparison', 'scope': 'Compare Q1 vs Q2.', 'estimated_complexity': 'detailed'},
            ],
            'aggregation_mode': 'compare',
            'output_shape': 'hybrid',
        }
        plan = _validate_and_build_plan(data, 'compare Q1 and Q2 revenue')
        assert plan is not None
        assert len(plan.steps) == 2
        assert len(plan.answer_sections) == 3
        assert plan.aggregation_mode == 'compare'
        assert plan.output_shape == 'hybrid'
        assert plan.steps[1].filters.year_filter == 2023
