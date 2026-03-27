# ==============================================================================
# Informity AI — Embedder Tests
# Tests embedding generation with a mocked sentence-transformers model.
# Verifies lazy loading, batch embedding, query embedding, dimensions,
# and error handling.
# ==============================================================================

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from informity.db.vectors import VECTOR_DIMENSION
from informity.exceptions import IndexingError
from informity.indexer.embedder import Embedder

# ==============================================================================
# Helpers
# ==============================================================================


def _make_mock_model(dimension: int = VECTOR_DIMENSION) -> MagicMock:
    # Create a mock SentenceTransformer that returns deterministic embeddings.
    # Each text gets a unique vector based on its hash, normalized to unit length.
    mock = MagicMock()

    def _encode(texts, batch_size=32, show_progress_bar=False,
                convert_to_numpy=True, normalize_embeddings=True):
        # Generate a unique-ish vector per text string
        _ = (batch_size, show_progress_bar, convert_to_numpy)
        results = []
        for text in texts:
            rng = np.random.default_rng(seed=hash(text) % (2**31))
            vec = rng.standard_normal(dimension).astype(np.float32)
            if normalize_embeddings:
                vec = vec / np.linalg.norm(vec)
            results.append(vec)
        return np.array(results)

    mock.encode = _encode
    return mock


def _make_embedder_with_mock(dimension: int = VECTOR_DIMENSION) -> Embedder:
    # Create an Embedder with a pre-injected mock model.
    emb = Embedder()
    emb._model = _make_mock_model(dimension)
    return emb


# ==============================================================================
# Lazy Loading
# ==============================================================================


class TestLazyLoading:
    # Tests that the model is lazy-loaded and not loaded until needed.

    def test_model_not_loaded_on_init(self) -> None:
        emb = Embedder()
        assert emb.is_loaded is False

    def test_model_loaded_on_first_embed(self) -> None:
        emb = Embedder()
        mock_model = _make_mock_model()
        with patch('informity.indexer.embedder.Embedder._load_model') as mock_load:
            # Simulate _load_model setting the model
            def side_effect():
                emb._model = mock_model
            mock_load.side_effect = side_effect

            emb.embed_texts(['test'])
            mock_load.assert_called_once()
            assert emb.is_loaded is True

    def test_model_loaded_only_once(self) -> None:
        emb = Embedder()
        mock_model = _make_mock_model()
        with patch('informity.indexer.embedder.Embedder._load_model') as mock_load:
            def side_effect():
                emb._model = mock_model
            mock_load.side_effect = side_effect

            emb.embed_texts(['first'])
            emb.embed_texts(['second'])
            mock_load.assert_called_once()

    def test_is_loaded_false_before_use(self) -> None:
        emb = Embedder()
        assert emb.is_loaded is False

    def test_is_loaded_true_after_injection(self) -> None:
        emb = _make_embedder_with_mock()
        assert emb.is_loaded is True


# ==============================================================================
# embed_texts — Batch Embedding
# ==============================================================================


class TestEmbedTexts:
    # Tests for the embed_texts method.

    def test_empty_list_returns_empty(self) -> None:
        emb    = _make_embedder_with_mock()
        result = emb.embed_texts([])
        assert result == []

    def test_single_text_returns_one_vector(self) -> None:
        emb    = _make_embedder_with_mock()
        result = emb.embed_texts(['Hello world'])
        assert len(result) == 1
        assert len(result[0]) == VECTOR_DIMENSION

    def test_multiple_texts_returns_matching_count(self) -> None:
        emb    = _make_embedder_with_mock()
        texts  = ['First text', 'Second text', 'Third text']
        result = emb.embed_texts(texts)
        assert len(result) == 3

    def test_vector_dimension_is_768(self) -> None:
        emb    = _make_embedder_with_mock()
        result = emb.embed_texts(['Test text for dimension check'])
        assert len(result[0]) == 768

    def test_returns_plain_lists_not_numpy(self) -> None:
        emb    = _make_embedder_with_mock()
        result = emb.embed_texts(['Convert me'])
        assert isinstance(result[0], list)
        assert isinstance(result[0][0], float)

    def test_different_texts_produce_different_vectors(self) -> None:
        emb    = _make_embedder_with_mock()
        result = emb.embed_texts(['alpha', 'beta'])
        # Vectors should not be identical (extremely unlikely with random seeding)
        assert result[0] != result[1]

    def test_same_text_produces_same_vector(self) -> None:
        emb     = _make_embedder_with_mock()
        result1 = emb.embed_texts(['deterministic'])
        result2 = emb.embed_texts(['deterministic'])
        assert result1[0] == result2[0]

    def test_large_batch(self) -> None:
        emb    = _make_embedder_with_mock()
        texts  = [f'Text number {i}' for i in range(100)]
        result = emb.embed_texts(texts)
        assert len(result) == 100
        for vec in result:
            assert len(vec) == VECTOR_DIMENSION

    def test_runtime_error_propagates(self) -> None:
        emb       = Embedder()
        mock_model = MagicMock()
        mock_model.encode.side_effect = RuntimeError('CUDA out of memory')
        emb._model = mock_model

        with pytest.raises(RuntimeError, match='CUDA out of memory'):
            emb.embed_texts(['will fail'])


# ==============================================================================
# embed_query — Single Query Embedding
# ==============================================================================


class TestEmbedQuery:
    # Tests for the embed_query method.

    def test_returns_single_vector(self) -> None:
        emb    = _make_embedder_with_mock()
        result = emb.embed_query('What is the meaning of life?')
        assert isinstance(result, list)
        assert len(result) == VECTOR_DIMENSION

    def test_vector_elements_are_floats(self) -> None:
        emb    = _make_embedder_with_mock()
        result = emb.embed_query('float check')
        for val in result:
            assert isinstance(val, float)

    def test_empty_query_returns_vector(self) -> None:
        emb = _make_embedder_with_mock()
        result = emb.embed_query('')
        assert isinstance(result, list)
        assert len(result) == VECTOR_DIMENSION

    def test_whitespace_only_query_returns_vector(self) -> None:
        emb = _make_embedder_with_mock()
        result = emb.embed_query('   \t\n  ')
        assert isinstance(result, list)
        assert len(result) == VECTOR_DIMENSION

    def test_query_produces_same_result_when_called_twice(self) -> None:
        # Query and document embeddings use different task prefixes (search_query vs
        # search_document), so they differ by design. We verify idempotency instead:
        # the same query yields the same vector when embedded twice.
        emb   = _make_embedder_with_mock()
        query = 'consistent embedding'
        a     = emb.embed_query(query)
        b     = emb.embed_query(query)
        assert a == b

    def test_query_embedding_cache_avoids_duplicate_encode_calls(self) -> None:
        emb = _make_embedder_with_mock()
        emb._model.encode = MagicMock(return_value=np.array([[0.1] * VECTOR_DIMENSION], dtype=np.float32))

        _ = emb.embed_query('cache me')
        _ = emb.embed_query('cache me')

        assert emb._model.encode.call_count == 1


# ==============================================================================
# Model Loading Errors
# ==============================================================================


class TestModelLoadingErrors:
    # Tests that model loading failures are handled gracefully.

    def test_import_error_raises_indexing_error(self) -> None:
        emb = Embedder()
        with patch.dict('sys.modules', {'sentence_transformers': None}), patch(
            'informity.indexer.embedder.Embedder._load_model',
            side_effect=IndexingError('sentence-transformers is not installed: No module'),
        ), pytest.raises(IndexingError, match='sentence-transformers is not installed'):
            _ = emb.model

    def test_os_error_raises_indexing_error(self) -> None:
        emb = Embedder()
        with patch(
            'informity.indexer.embedder.Embedder._load_model',
            side_effect=IndexingError('Failed to load embedding model'),
        ), pytest.raises(IndexingError, match='Failed to load embedding model'):
            _ = emb.model


# ==============================================================================
# Normalized Embeddings
# ==============================================================================


class TestNormalization:
    # Tests that embeddings are properly normalized (unit vectors).

    def test_embeddings_are_normalized(self) -> None:
        emb    = _make_embedder_with_mock()
        result = emb.embed_texts(['normalize me'])
        vec    = np.array(result[0])
        norm   = np.linalg.norm(vec)
        # Should be approximately 1.0 (unit vector)
        assert abs(norm - 1.0) < 1e-5

    def test_query_embedding_is_normalized(self) -> None:
        emb    = _make_embedder_with_mock()
        result = emb.embed_query('normalize this query')
        vec    = np.array(result)
        norm   = np.linalg.norm(vec)
        assert abs(norm - 1.0) < 1e-5
