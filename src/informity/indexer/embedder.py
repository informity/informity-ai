# ==============================================================================
# Informity AI — Embedding Generator (v2)
# nomic-embed-text-v1.5, search_document: prefix only (no metadata)
# Uses sentence-transformers (PyTorch) with MPS batch size cap to prevent OOM
# ==============================================================================

from __future__ import annotations

import threading
from collections import OrderedDict
from time import monotonic
from typing import TYPE_CHECKING

import structlog

from informity.config import settings

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

log = structlog.get_logger(__name__)
_MPS_DETECTION_EXCEPTIONS = (ImportError, AttributeError, RuntimeError, OSError, TypeError, ValueError)

_TASK_PREFIX_DOCUMENT = 'search_document: '
_TASK_PREFIX_QUERY    = 'search_query: '
_QUERY_EMBED_CACHE_MAX_SIZE = 128
_QUERY_EMBED_CACHE_TTL_SECONDS = 300.0
_MPS_MAX_BATCH_SIZE = 4
_EMBEDDING_MODEL_DIMENSIONS: dict[str, int] = {
    # Centralized embedding-model metadata for vector dimension resolution.
    'nomic-ai/nomic-embed-text-v1.5': 768,
}
_DEFAULT_EMBEDDING_DIMENSION = _EMBEDDING_MODEL_DIMENSIONS['nomic-ai/nomic-embed-text-v1.5']


def get_embedding_model_dimension(model_name: str) -> int:
    normalized = str(model_name or '').strip().casefold()
    if normalized in _EMBEDDING_MODEL_DIMENSIONS:
        return int(_EMBEDDING_MODEL_DIMENSIONS[normalized])
    return _DEFAULT_EMBEDDING_DIMENSION


class Embedder:
    def __init__(self) -> None:
        self._model: SentenceTransformer | None = None
        self._query_embed_cache: OrderedDict[str, tuple[list[float], float]] = OrderedDict()
        self._cache_lock = threading.Lock()
        self._mps_available: bool | None = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._load_model()
        return self._model

    def _load_model(self) -> None:
        from sentence_transformers import SentenceTransformer

        from informity.config import configure_hf_environment

        # Ensure HF environment is configured before loading model
        # This sets HF_HOME, HF_HUB_CACHE, and offline flags
        configure_hf_environment()

        model_name = settings.embedding_model

        log.info('loading_embedding_model', model=model_name)
        # sentence-transformers uses HuggingFace cache (configured via configure_hf_environment)
        # trust_remote_code=True required for nomic-ai/nomic-embed-text-v1.5
        # Task prefixes are applied manually (sentence-transformers doesn't support native prefixes)
        self._model = SentenceTransformer(
            model_name,
            trust_remote_code=True,
            device='mps' if self._is_mps_available() else 'cpu'
        )
        log.info('embedding_model_loaded', model=model_name, device=self._model.device.type if hasattr(self._model, 'device') else 'unknown')

    def _is_mps_available(self) -> bool:
        """Check if Apple Metal Performance Shaders (MPS) is available."""
        if self._mps_available is not None:
            return self._mps_available
        try:
            import torch
            self._mps_available = torch.backends.mps.is_available()
        except _MPS_DETECTION_EXCEPTIONS:
            self._mps_available = False
        return self._mps_available

    def _cache_query_embedding(self, query: str, embedding: list[float]) -> None:
        now = monotonic()
        with self._cache_lock:
            self._query_embed_cache[query] = (embedding, now)
            self._query_embed_cache.move_to_end(query)
            while len(self._query_embed_cache) > _QUERY_EMBED_CACHE_MAX_SIZE:
                self._query_embed_cache.popitem(last=False)

    def _get_cached_query_embedding(self, query: str) -> list[float] | None:
        with self._cache_lock:
            cached = self._query_embed_cache.get(query)
            if cached is None:
                return None
            embedding, ts = cached
            if monotonic() - ts > _QUERY_EMBED_CACHE_TTL_SECONDS:
                self._query_embed_cache.pop(query, None)
                return None
            self._query_embed_cache.move_to_end(query)
            return embedding

    def _get_safe_batch_size(self) -> int:
        """
        Get safe batch size for embedding.

        MPS (Metal) backend pre-allocates worst-case memory for entire batch,
        causing OOM with large batches. Cap batch size to 4 when using MPS.
        This fixes the root cause (MPS memory pre-allocation) at source.
        """
        if self._is_mps_available():
            safe_size = min(settings.embedding_batch_size, _MPS_MAX_BATCH_SIZE)
            if safe_size < settings.embedding_batch_size:
                log.debug(
                    'mps_batch_size_capped',
                    requested=settings.embedding_batch_size,
                    actual=safe_size,
                    reason='mps_memory_preallocation_limit'
                )
            return safe_size
        return settings.embedding_batch_size

    def get_effective_batch_size(self) -> int:
        """Return the effective embedding batch size after safety caps."""
        return max(1, int(self._get_safe_batch_size()))

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        # Embed multiple texts with search_document: prefix.
        # sentence-transformers doesn't support native task prefixes, so we prepend manually.
        prefixed = [f'{_TASK_PREFIX_DOCUMENT}{text}' for text in texts]

        # Use safe batch size (capped for MPS to prevent OOM)
        safe_batch_size = self.get_effective_batch_size()

        # Process in batches to respect batch size limit
        embeddings = []
        for i in range(0, len(prefixed), safe_batch_size):
            batch = prefixed[i:i + safe_batch_size]
            # encode() returns numpy array, convert to list of lists
            batch_embeddings = self.model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
            # Convert numpy array to list of lists
            if batch_embeddings.ndim == 1:
                embeddings.append(batch_embeddings.tolist())
            else:
                embeddings.extend(batch_embeddings.tolist())

        return embeddings

    def embed_query(self, query: str) -> list[float]:
        # Embed a single query with search_query: prefix.
        query_key = query.strip()
        cached = self._get_cached_query_embedding(query_key)
        if cached is not None:
            return cached
        prefixed = f'{_TASK_PREFIX_QUERY}{query}'
        # encode() returns numpy array, convert to list
        embedding = self.model.encode([prefixed], convert_to_numpy=True, show_progress_bar=False)
        result = embedding[0].tolist() if embedding.ndim == 2 else embedding.tolist()
        self._cache_query_embedding(query_key, result)
        return result

    def unload(self) -> None:
        # Release model resources.
        if self._model is not None:
            del self._model
            self._model = None
        with self._cache_lock:
            self._query_embed_cache.clear()

    def get_embedding_dimension(self) -> int:
        model = self._model
        if model is not None and hasattr(model, 'get_sentence_embedding_dimension'):
            try:
                dim = int(model.get_sentence_embedding_dimension())
                if dim > 0:
                    return dim
            except (TypeError, ValueError, RuntimeError, AttributeError):
                pass
        return get_embedding_model_dimension(settings.embedding_model)


embedder = Embedder()


def get_effective_embedding_dimension() -> int:
    return embedder.get_embedding_dimension()
