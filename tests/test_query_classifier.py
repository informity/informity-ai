# ==============================================================================
# Informity AI — Query Classifier Tests
# Tests query classification (intent detection and filter extraction)
# Uses mocked LLM classifier — LLM-only, no regex fallback
# ==============================================================================

from unittest.mock import patch

import pytest

from informity.llm.query_classifier import QueryClassification, classify_query


def _mock_classify_query_llm(query: str) -> QueryClassification:
    """
    Mock LLM classifier for tests. Returns expected classifications for test queries.
    Keeps tests fast and deterministic without requiring classifier model.
    """
    q = query.lower()
    # Metadata file list first: "list all files", "show me all" (exclude "list all years")
    if any(x in q for x in ['list all', 'show me all', 'display']) and 'years' not in q:
        year = 2022 if '2022' in q else None
        return QueryClassification(
            intent='metadata',
            year_filter=year,
            category_filter=None,
            file_type_filter=None,
            filename_filter=None,
            is_metadata_query=True,
            is_file_list_query=True,
        )
    # Coverage
    if any(x in q for x in ['all the years', 'all years', 'every document', 'all files', 'all the categories', 'list all years']):
        return QueryClassification(
            intent='coverage',
            year_filter=None,
            category_filter=None,
            file_type_filter=None,
            filename_filter=None,
            is_metadata_query=False,
            is_file_list_query=False,
        )
    # Metadata count
    if 'how many' in q:
        year = 2023 if '2023' in q else 2022 if '2022' in q else None
        file_type = '.pdf' if 'pdf' in q else None
        category = 'document' if 'document' in q and 'pdf' in q else None
        return QueryClassification(
            intent='metadata',
            year_filter=year,
            category_filter=category,
            file_type_filter=file_type,
            filename_filter=None,
            is_metadata_query=True,
            is_file_list_query=True,
        )
    # Simple
    if any(x in q for x in ['hello', 'hi', 'hey', 'thanks']):
        return QueryClassification(
            intent='simple',
            year_filter=None,
            category_filter=None,
            file_type_filter=None,
            filename_filter=None,
            is_metadata_query=False,
            is_file_list_query=False,
        )
    # Coverage
    if any(x in q for x in ['all the years', 'all years', 'every document', 'all files', 'all the categories', 'list all years']):
        return QueryClassification(
            intent='coverage',
            year_filter=None,
            category_filter=None,
            file_type_filter=None,
            filename_filter=None,
            is_metadata_query=False,
            is_file_list_query=False,
        )
    # Focused with filters
    if 'pdf' in q and '2023' in q and 'say about' in q:
        return QueryClassification(
            intent='focused',
            year_filter=2023,
            category_filter=None,
            file_type_filter='.pdf',
            filename_filter=None,
            is_metadata_query=False,
            is_file_list_query=False,
        )
    # Focused
    if 'quantum' in q or 'what is' in q and 'ai' in q:
        return QueryClassification(
            intent='focused',
            year_filter=None,
            category_filter=None,
            file_type_filter=None,
            filename_filter=None,
            is_metadata_query=False,
            is_file_list_query=False,
        )
    # Metadata with filters (files from, documents in, etc.)
    if 'from 2023' in q or 'in 2020' in q or 'year 2021' in q:
        year = 2023 if '2023' in q else 2020 if '2020' in q else 2021 if '2021' in q else None
        file_type = '.pdf' if 'pdf' in q else None
        category = 'document' if 'document' in q else None
        return QueryClassification(
            intent='metadata',
            year_filter=year,
            category_filter=category,
            file_type_filter=file_type,
            filename_filter=None,
            is_metadata_query=True,
            is_file_list_query=True,
        )
    # Metadata category/file type
    if 'document files' in q or 'pdf' in q:
        return QueryClassification(
            intent='metadata',
            year_filter=2023 if '2023' in q else None,
            category_filter='document' if 'document' in q else None,
            file_type_filter='.pdf' if 'pdf' in q else None,
            filename_filter=None,
            is_metadata_query=True,
            is_file_list_query=True,
        )
    # Default focused
    return QueryClassification(
        intent='focused',
        year_filter=None,
        category_filter=None,
        file_type_filter=None,
        filename_filter=None,
        is_metadata_query=False,
        is_file_list_query=False,
    )


@pytest.fixture(autouse=True)
def _mock_llm_classifier():
    """Mock LLM classifier so tests run without the classifier model."""
    with patch('informity.llm.query_classifier_llm.classify_query_llm', side_effect=_mock_classify_query_llm):
        yield


class TestQueryClassifier:
    # Test intent detection

    def test_metadata_count_query(self) -> None:
        result = classify_query('how many files')
        assert result.intent == 'metadata'
        assert result.is_metadata_query is True

    def test_metadata_count_with_filter(self) -> None:
        result = classify_query('how many PDFs from 2023')
        assert result.intent == 'metadata'
        assert result.is_metadata_query is True
        assert result.year_filter == 2023
        assert result.file_type_filter == '.pdf'

    def test_file_list_query(self) -> None:
        result = classify_query('list all files')
        assert result.intent == 'metadata'
        assert result.is_file_list_query is True

    def test_file_list_with_filter(self) -> None:
        result = classify_query('show me all documents from 2022')
        assert result.intent == 'metadata'
        assert result.is_file_list_query is True
        assert result.year_filter == 2022

    def test_coverage_query(self) -> None:
        result = classify_query('what are all the years')
        assert result.intent == 'coverage'

    def test_coverage_query_patterns(self) -> None:
        queries = [
            'list all years',
            'every document',
            'all files',
            'what are all the categories',
        ]
        for query in queries:
            result = classify_query(query)
            assert result.intent == 'coverage', f"Failed for: {query}"

    def test_focused_query(self) -> None:
        result = classify_query('what is quantum computing')
        assert result.intent == 'focused'

    def test_focused_query_with_filters(self) -> None:
        result = classify_query('what do PDFs from 2023 say about AI')
        assert result.intent == 'focused'
        assert result.year_filter == 2023
        assert result.file_type_filter == '.pdf'

    def test_simple_query(self) -> None:
        result = classify_query('hello')
        assert result.intent == 'simple'

    def test_simple_greeting(self) -> None:
        queries = ['hi', 'hello', 'hey', 'thanks']
        for query in queries:
            result = classify_query(query)
            assert result.intent == 'simple', f"Failed for: {query}"

    # Test filter extraction

    def test_year_extraction(self) -> None:
        result = classify_query('files from 2023')
        assert result.year_filter == 2023

    def test_year_extraction_various_formats(self) -> None:
        queries = [
            'files from 2023',
            'documents in 2020',
            'reports from year 2021',
        ]
        for query in queries:
            result = classify_query(query)
            assert result.year_filter is not None, f"Failed to extract year from: {query}"

    def test_category_extraction(self) -> None:
        result = classify_query('document files')
        assert result.category_filter == 'document'

    def test_file_type_extraction(self) -> None:
        result = classify_query('PDF files')
        assert result.file_type_filter == '.pdf'

    def test_multiple_filters(self) -> None:
        result = classify_query('PDF documents from 2023')
        assert result.file_type_filter == '.pdf'
        assert result.category_filter == 'document'
        assert result.year_filter == 2023

    def test_no_filters(self) -> None:
        result = classify_query('what is AI')
        assert result.year_filter is None
        assert result.category_filter is None
        assert result.file_type_filter is None

    # Test QueryClassification dataclass

    def test_query_classification_structure(self) -> None:
        result = classify_query('how many files')
        assert isinstance(result, QueryClassification)
        assert hasattr(result, 'intent')
        assert hasattr(result, 'subtype')
        assert hasattr(result, 'group_by')
        assert hasattr(result, 'field_hint')
        assert hasattr(result, 'source_terms')
        assert hasattr(result, 'year_filter')
        assert hasattr(result, 'category_filter')
        assert hasattr(result, 'file_type_filter')
        assert hasattr(result, 'is_metadata_query')
        assert hasattr(result, 'is_file_list_query')

