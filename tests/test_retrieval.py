# ==============================================================================
# Informity AI — Retrieval Tests
# Tests the unified retrieval pipeline (embed → vector search → rerank)
# ==============================================================================

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from informity.llm.metadata_filters import extract_metadata_filters
from informity.llm.retrieval import retrieve_chunks


def _make_async_mock_db():
    """Create a mock db with async execute/fetchall for retrieve_chunks."""
    mock_cursor = MagicMock()
    mock_cursor.fetchall = AsyncMock(return_value=[
        {
            'chunk_id': 1, 'file_id': 1, 'file_path': '/f1', 'filename': 'f1.txt',
            'chunk_text': 'chunk 1', 'page_number': None, 'start_page': None, 'end_page': None,
            'section_path': 'Introduction', 'block_type': 'narrative', 'parent_id': None,
        },
        {
            'chunk_id': 2, 'file_id': 2, 'file_path': '/f2', 'filename': 'f2.txt',
            'chunk_text': 'chunk 2', 'page_number': None, 'start_page': None, 'end_page': None,
            'section_path': 'Conclusion', 'block_type': 'table', 'parent_id': None,
        },
    ])
    mock_db = MagicMock()
    mock_db.execute = AsyncMock(return_value=mock_cursor)
    return mock_db


@pytest.fixture
def mock_db():
    return _make_async_mock_db()


@pytest.mark.asyncio
async def test_retrieve_chunks_embeds_query(mock_db):
    # Should embed query before searching
    with patch('informity.llm.retrieval.embedder') as mock_embedder, \
         patch('informity.llm.retrieval.vector_store') as mock_vector_store, \
         patch('informity.llm.retrieval.reranker') as mock_reranker:

        mock_embedder.embed_query.return_value = [0.1] * 768
        mock_vector_store.search_similar.return_value = []
        mock_reranker.rerank.return_value = []

        await retrieve_chunks('test query', top_k=5, db=mock_db)

        mock_embedder.embed_query.assert_called_once_with('test query')


@pytest.mark.asyncio
async def test_retrieve_chunks_applies_filters(mock_db):
    # Should apply year, category, extension filters
    with patch('informity.llm.retrieval.embedder') as mock_embedder, \
         patch('informity.llm.retrieval.vector_store') as mock_vector_store, \
         patch('informity.llm.retrieval.reranker') as mock_reranker, \
         patch('informity.llm.retrieval.build_where_clause_and_params') as mock_build_where:

        mock_embedder.embed_query.return_value = [0.1] * 768
        mock_vector_store.search_similar.return_value = []
        mock_reranker.rerank.return_value = []
        mock_build_where.return_value = ('year = ?', [2023])

        await retrieve_chunks(
            'test query',
            top_k=5,
            year_filter=2023,
            category_filter='document',
            extension_filter='.pdf',
            db=mock_db,
        )

        # Should have built WHERE clause with filters
        assert mock_build_where.call_count >= 1
        call_args = mock_build_where.call_args_list[0][0][0]
        assert len(call_args) >= 3  # Should have year, category, extension filters


@pytest.mark.asyncio
async def test_retrieve_chunks_calls_reranker(mock_db):
    # Should call reranker after vector search
    with patch('informity.llm.retrieval.embedder') as mock_embedder, \
         patch('informity.llm.retrieval.vector_store') as mock_vector_store, \
         patch('informity.llm.retrieval.reranker') as mock_reranker:

        mock_embedder.embed_query.return_value = [0.1] * 768
        mock_chunks = [
            {'chunk_id': 1, 'chunk_text': 'chunk 1', 'score': 0.8},
            {'chunk_id': 2, 'chunk_text': 'chunk 2', 'score': 0.7},
        ]
        mock_vector_store.search_similar.return_value = mock_chunks
        mock_vector_store.fts5_augment_candidates.return_value = []
        mock_reranker.rerank.return_value = mock_chunks

        await retrieve_chunks('test query', top_k=5, db=mock_db)

        # Should have called reranker
        mock_reranker.rerank.assert_called_once()


@pytest.mark.asyncio
async def test_retrieve_chunks_uses_profile_candidate_top_k(mock_db):
    with patch('informity.llm.retrieval.embedder') as mock_embedder, \
         patch('informity.llm.retrieval.get_profile') as mock_get_profile, \
         patch('informity.llm.retrieval.vector_store') as mock_vector_store, \
         patch('informity.llm.retrieval.reranker') as mock_reranker:
        mock_embedder.embed_query.return_value = [0.1] * 768
        mock_get_profile.return_value = SimpleNamespace(retrieval_top_k_candidates=20)
        mock_vector_store.search_similar.return_value = []
        mock_reranker.rerank.return_value = []

        await retrieve_chunks('all files', top_k=10, query_type='coverage', db=mock_db)

        search_call = mock_vector_store.search_similar.call_args
        candidate_limit = search_call[0][1]
        assert candidate_limit == 20


@pytest.mark.asyncio
async def test_retrieve_chunks_returns_top_k(mock_db):
    # Should return top_k results after reranking
    with patch('informity.llm.retrieval.embedder') as mock_embedder, \
         patch('informity.llm.retrieval.vector_store') as mock_vector_store, \
         patch('informity.llm.retrieval.reranker') as mock_reranker:

        mock_embedder.embed_query.return_value = [0.1] * 768
        mock_chunks = [
            {'chunk_id': i, 'chunk_text': f'chunk {i}', 'score': 0.9 - i * 0.1}
            for i in range(10)
        ]
        mock_vector_store.search_similar.return_value = mock_chunks
        mock_reranker.rerank.return_value = mock_chunks

        results = await retrieve_chunks('test query', top_k=5, db=mock_db)

        # Should return top_k results
        assert len(results) <= 5


@pytest.mark.asyncio
async def test_retrieve_chunks_applies_block_type_filter(mock_db):
    with patch('informity.llm.retrieval.embedder') as mock_embedder, \
         patch('informity.llm.retrieval.vector_store') as mock_vector_store, \
         patch('informity.llm.retrieval.reranker') as mock_reranker:
        mock_embedder.embed_query.return_value = [0.1] * 768
        mock_vector_store.search_similar.return_value = [
            {'chunk_id': 1, 'chunk_text': 'chunk 1', 'score': 0.2},
            {'chunk_id': 2, 'chunk_text': 'chunk 2', 'score': 0.1},
        ]
        mock_vector_store.fts5_augment_candidates.return_value = []
        mock_reranker.rerank.side_effect = lambda _q, chunks: chunks

        await retrieve_chunks('show me table data', top_k=5, block_type_filter='table', db=mock_db)

        rerank_chunks = mock_reranker.rerank.call_args[0][1]
        assert len(rerank_chunks) == 1
        assert rerank_chunks[0]['block_type'] == 'table'


@pytest.mark.asyncio
async def test_retrieve_chunks_applies_section_filter(mock_db):
    with patch('informity.llm.retrieval.embedder') as mock_embedder, \
         patch('informity.llm.retrieval.vector_store') as mock_vector_store, \
         patch('informity.llm.retrieval.reranker') as mock_reranker:
        mock_embedder.embed_query.return_value = [0.1] * 768
        mock_vector_store.search_similar.return_value = [
            {'chunk_id': 1, 'chunk_text': 'chunk 1', 'score': 0.2},
            {'chunk_id': 2, 'chunk_text': 'chunk 2', 'score': 0.1},
        ]
        mock_vector_store.fts5_augment_candidates.return_value = []
        mock_reranker.rerank.side_effect = lambda _q, chunks: chunks

        await retrieve_chunks('what is in conclusion section', top_k=5, section_filter='conclusion', db=mock_db)

        rerank_chunks = mock_reranker.rerank.call_args[0][1]
        assert len(rerank_chunks) == 1
        assert rerank_chunks[0]['section_path'] == 'Conclusion'


@pytest.mark.asyncio
async def test_retrieve_chunks_preserves_filename_constraint_when_no_candidates(mock_db):
    with patch('informity.llm.retrieval.embedder') as mock_embedder, \
         patch('informity.llm.retrieval.vector_store') as mock_vector_store, \
         patch('informity.llm.retrieval.reranker') as mock_reranker:
        mock_embedder.embed_query.return_value = [0.1] * 768
        mock_vector_store.search_similar.return_value = []
        mock_reranker.rerank.side_effect = lambda _q, chunks: chunks

        results = await retrieve_chunks(
            'Summarize the content of sample-lender-statement.pdf',
            top_k=5,
            filename_filter='sample-lender-statement.pdf',
            query_type='focused',
            db=mock_db,
        )

        assert results == []
        assert mock_vector_store.search_similar.call_count == 1
        where_clause = mock_vector_store.search_similar.call_args[0][2]
        where_params = mock_vector_store.search_similar.call_args[0][3]
        assert 'filename LIKE' in where_clause
        assert '%sample-lender-statement.pdf%' in where_params


@pytest.mark.asyncio
async def test_retrieve_chunks_parent_fallback_does_not_inject_zero_score_when_missing(mock_db):
    mock_cursor = MagicMock()
    mock_cursor.fetchall = AsyncMock(return_value=[
        {
            'chunk_id': 10,
            'file_id': 1,
            'file_path': '/f1',
            'filename': 'f1.txt',
            'chunk_text': 'child chunk',
            'page_number': None,
            'start_page': None,
            'end_page': None,
            'section_path': None,
            'block_type': None,
            'parent_id': 100,
        },
    ])
    mock_db.execute = AsyncMock(return_value=mock_cursor)

    with patch('informity.llm.retrieval.embedder') as mock_embedder, \
         patch('informity.llm.retrieval.vector_store') as mock_vector_store, \
         patch('informity.llm.retrieval.reranker') as mock_reranker, \
         patch('informity.llm.retrieval.get_chunks_by_parent_ids', new_callable=AsyncMock) as mock_get_parents:
        mock_embedder.embed_query.return_value = [0.1] * 768
        mock_vector_store.search_similar.return_value = [{'chunk_id': 10, 'score': 0.1}]
        mock_vector_store.fts5_augment_candidates.return_value = []
        mock_reranker.rerank.return_value = [{'chunk_id': 10}]  # score intentionally missing
        mock_get_parents.return_value = [
            {
                'chunk_id': 100,
                'file_id': 1,
                'file_path': '/f1',
                'filename': 'f1.txt',
                'chunk_text': 'parent chunk',
            },
        ]

        results = await retrieve_chunks('test query', top_k=1, db=mock_db)

    assert len(results) == 1
    assert results[0]['chunk_id'] == 100
    assert 'score' not in results[0]


def test_extract_metadata_filters_does_not_collapse_multi_year_range() -> None:
    filters = extract_metadata_filters(
        'Build a forensic reconciliation report from records across 2022-2024.',
    )
    year_filters = [f for f in filters if f.field == 'year']
    assert len(year_filters) == 1
    assert year_filters[0].operator == 'in'
    assert year_filters[0].value == [2022, 2023, 2024]


def test_extract_metadata_filters_parses_multi_year_list_to_in_filter() -> None:
    filters = extract_metadata_filters(
        'Compare records for years 2022, 2023, and 2024.',
    )
    year_filters = [f for f in filters if f.field == 'year']
    assert len(year_filters) == 1
    assert year_filters[0].operator == 'in'
    assert year_filters[0].value == [2022, 2023, 2024]
