from __future__ import annotations

from unittest.mock import MagicMock

from informity.config import settings
from informity.db.vectors import _get_expected_vector_dimension
from informity.indexer import embedder as embedder_module


def test_get_embedding_model_dimension_uses_model_metadata_and_default() -> None:
    assert embedder_module.get_embedding_model_dimension('nomic-ai/nomic-embed-text-v1.5') == 768
    # Unknown model falls back to default metadata dimension.
    assert embedder_module.get_embedding_model_dimension('custom/unknown-embedder') == 768


def test_effective_embedding_dimension_uses_loaded_model_dimension_when_available() -> None:
    original_model = embedder_module.embedder._model
    try:
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        embedder_module.embedder._model = mock_model
        assert embedder_module.get_effective_embedding_dimension() == 384
    finally:
        embedder_module.embedder._model = original_model


def test_vectors_expected_dimension_tracks_active_embedding_model_setting(monkeypatch) -> None:
    monkeypatch.setattr(settings, 'embedding_model', 'nomic-ai/nomic-embed-text-v1.5')
    original_model = embedder_module.embedder._model
    try:
        # Ensure resolution comes from model metadata when model is not loaded.
        embedder_module.embedder._model = None
        assert _get_expected_vector_dimension() == 768
    finally:
        embedder_module.embedder._model = original_model
