# ==============================================================================
# Informity AI — RAG Pipeline Tests (v2)
# Tests the QueryRouter (answer_question) with mocked dependencies.
# Tests integration behavior, not internal implementation details.
# ==============================================================================

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from informity.db.models import ChatMessage
from informity.llm.query_classifier import QueryClassification
from informity.llm.rag import answer_question

# ==============================================================================
# Fixtures — mock dependencies
# ==============================================================================


def _make_async_mock_db():
    mock_db = MagicMock()
    mock_db.execute = AsyncMock(return_value=MagicMock(fetchall=AsyncMock(return_value=[])))
    return mock_db


@pytest.fixture
def mock_db():
    return _make_async_mock_db()


@pytest.fixture
def mock_chunks():
    # Mock chunk retrieval results
    return [
        {
            'chunk_id': 1,
            'file_id': 1,
            'file_path': '/test/file1.txt',
            'filename': 'file1.txt',
            'chunk_text': 'Content from document 1 about topic 1.',
            'score': 0.8,
        },
        {
            'chunk_id': 2,
            'file_id': 2,
            'file_path': '/test/file2.txt',
            'filename': 'file2.txt',
            'chunk_text': 'Content from document 2 about topic 2.',
            'score': 0.7,
        },
    ]


# ==============================================================================
# answer_question — main public API (QueryRouter)
# ==============================================================================


@pytest.mark.asyncio
async def test_answer_question_no_db():
    # Should yield error when db is None
    results = []
    async for item in answer_question('test question', db=None):
        results.append(item)

    assert len(results) >= 1
    assert 'Error' in results[0]
    assert results[-1] == []  # Empty sources


@pytest.mark.asyncio
async def test_answer_question_calls_classify(mock_db):
    # Should call classify_query with the question
    with patch('informity.llm.rag.classify_query') as mock_classify:
        mock_classify.return_value = QueryClassification(
            intent='focused',
            year_filter=None,
            category_filter=None,
        )

        # Mock handler to avoid actual processing
        with patch('informity.llm.rag._HANDLER_REGISTRY', new=[]):
            results = []
            async for item in answer_question('test question', db=mock_db):
                results.append(item)

        # Should have called classify_query
        mock_classify.assert_called_once_with('test question')


@pytest.mark.asyncio
async def test_answer_question_routes_to_metadata_handler(mock_db):
    # Should route metadata queries to MetadataHandler
    classification = QueryClassification(
        intent='metadata',
        is_metadata_query=True,
    )

    async def mock_meta_gen():
        yield 'There are 5 files.'
        yield []

    with patch('informity.llm.rag.classify_query', return_value=classification), \
         patch('informity.llm.handlers.metadata.MetadataHandler.handle', new_callable=MagicMock) as mock_handler:
        mock_handler.return_value = mock_meta_gen()

        results = []
        async for item in answer_question('how many files', db=mock_db):
            results.append(item)

        # Should have called metadata handler
        mock_handler.assert_called_once()


@pytest.mark.asyncio
async def test_answer_question_routes_to_simple_handler(mock_db):
    # Should route simple queries to SimpleHandler
    classification = QueryClassification(
        intent='simple',
    )

    async def mock_simple_gen():
        yield 'Hello!'
        yield []

    with patch('informity.llm.rag.classify_query', return_value=classification), \
         patch('informity.llm.handlers.simple.SimpleHandler.handle', new_callable=MagicMock) as mock_handler:
        mock_handler.return_value = mock_simple_gen()

        results = []
        async for item in answer_question('hello', db=mock_db):
            results.append(item)

        # Should have called simple handler
        mock_handler.assert_called_once()


@pytest.mark.asyncio
async def test_answer_question_routes_to_rag_handler(mock_db, mock_chunks):
    # Should route focused/coverage queries to RAGHandler
    classification = QueryClassification(
        intent='focused',
        year_filter=None,
        category_filter=None,
    )

    async def mock_rag_gen():
        yield 'token1'
        yield 'token2'
        yield mock_chunks

    with patch('informity.llm.rag.classify_query', return_value=classification), \
         patch('informity.llm.handlers.rag.RAGHandler.handle', new_callable=MagicMock) as mock_handler:
        mock_handler.return_value = mock_rag_gen()

        results = []
        async for item in answer_question('test question', db=mock_db):
            results.append(item)

        # Should have called RAG handler
        mock_handler.assert_called_once()
        call_kwargs = mock_handler.call_args[1]
        assert call_kwargs['classification'] == classification


@pytest.mark.asyncio
async def test_answer_question_passes_filters_to_handler(mock_db):
    # Should pass classification filters to handler
    classification = QueryClassification(
        intent='focused',
        year_filter=2020,
        category_filter='document',
        file_type_filter='.pdf',
    )

    async def mock_rag_gen():
        yield 'answer'
        yield []

    with patch('informity.llm.rag.classify_query', return_value=classification), \
         patch('informity.llm.handlers.rag.RAGHandler.handle', new_callable=MagicMock) as mock_handler:
        mock_handler.return_value = mock_rag_gen()

        results = []
        async for item in answer_question('files from 2020', db=mock_db):
            results.append(item)

        # Should have passed filters to handler
        call_kwargs = mock_handler.call_args[1]
        assert call_kwargs['classification'].year_filter == 2020
        assert call_kwargs['classification'].category_filter == 'document'
        assert call_kwargs['classification'].file_type_filter == '.pdf'


@pytest.mark.asyncio
async def test_answer_question_passes_history_to_handler(mock_db):
    # Should pass chat history to handler
    classification = QueryClassification(intent='focused')
    history = [
        ChatMessage(chat_id='test-chat', role='user', content='previous question'),
        ChatMessage(chat_id='test-chat', role='assistant', content='previous answer'),
    ]

    async def mock_rag_gen():
        yield 'answer'
        yield []

    with patch('informity.llm.rag.classify_query', return_value=classification), \
         patch('informity.llm.handlers.rag.RAGHandler.handle', new_callable=MagicMock) as mock_handler:
        mock_handler.return_value = mock_rag_gen()

        results = []
        async for item in answer_question('follow-up question', history=history, db=mock_db):
            results.append(item)

        # Should have passed history to handler
        call_kwargs = mock_handler.call_args[1]
        assert call_kwargs['history'] == history


@pytest.mark.asyncio
async def test_answer_question_error_handling(mock_db):
    # Should yield error message on exception
    with patch('informity.llm.rag.classify_query') as mock_classify:
        mock_classify.side_effect = RuntimeError('Test error')

        results = []
        async for item in answer_question('test question', db=mock_db):
            results.append(item)

        # Should have yielded error
        assert len(results) >= 1
        assert 'Error' in results[0] or 'Test error' in results[0]
        assert results[-1] == []  # Empty sources


@pytest.mark.asyncio
async def test_answer_question_sources_structure(mock_db, mock_chunks):
    # Sources should have correct structure when returned from handler
    classification = QueryClassification(intent='focused')

    async def mock_handler_gen():
        yield 'token1'
        yield 'token2'
        sources = [
            {
                'filename': 'file1.txt',
                'path': '/test/file1.txt',
                'chunk_preview': 'Content preview',
                'relevance_score': 0.8,
            },
        ]
        yield sources

    with patch('informity.llm.rag.classify_query', return_value=classification), \
         patch('informity.llm.handlers.rag.RAGHandler.handle', new_callable=MagicMock) as mock_handler:
        mock_handler.return_value = mock_handler_gen()

        results = []
        async for item in answer_question('test question', db=mock_db):
            results.append(item)

        # Last result should be sources
        sources = results[-1]
        assert isinstance(sources, list)
        assert len(sources) > 0

        # Each source should have required fields
        for source in sources:
            assert 'filename' in source
            assert 'path' in source
            assert 'chunk_preview' in source
            assert 'relevance_score' in source
