# ==============================================================================
# Informity AI — Reranker (v2)
# cross-encoder/ms-marco-MiniLM-L-6-v2, always used (no flag)
# Uses sentence-transformers CrossEncoder (PyTorch)
# ==============================================================================

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

from informity.config import settings

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder

log = structlog.get_logger(__name__)
_MPS_DETECTION_EXCEPTIONS = (ImportError, AttributeError, RuntimeError, OSError, TypeError, ValueError)


class Reranker:
    def __init__(self) -> None:
        self._model: CrossEncoder | None = None

    @property
    def model(self) -> CrossEncoder:
        if self._model is None:
            self._load_model()
        return self._model

    def _load_model(self) -> None:
        from sentence_transformers import CrossEncoder

        from informity.config import configure_hf_environment

        # Ensure HF environment is configured before loading model
        # This sets HF_HOME, HF_HUB_CACHE, and offline flags
        configure_hf_environment()

        model_name = settings.rag_reranker_model

        log.info('loading_reranker_model', model=model_name)
        # sentence-transformers uses HuggingFace cache (configured via configure_hf_environment)
        # Suppress transformers warnings (they're noisy and not actionable)
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning, module='transformers')
            self._model = CrossEncoder(
                model_name,
                device='mps' if self._is_mps_available() else 'cpu'
            )
        log.info('reranker_model_loaded', model=model_name, device=self._model.device.type if hasattr(self._model, 'device') else 'unknown')

    def _is_mps_available(self) -> bool:
        """Check if Apple Metal Performance Shaders (MPS) is available."""
        try:
            import torch
            return torch.backends.mps.is_available()
        except _MPS_DETECTION_EXCEPTIONS:
            return False

    def rerank(self, query: str, chunks: list[dict]) -> list[dict]:
        # Rerank chunks by relevance to query.
        if not chunks:
            return []

        # CrossEncoder.predict() takes list of [query, document] pairs
        pairs = [[query, chunk['chunk_text']] for chunk in chunks]
        # Disable progress bars for server/runtime usage to avoid stderr-bound tqdm I/O errors.
        # predict() returns numpy array of scores (higher = more relevant).
        scores = self.model.predict(pairs, show_progress_bar=False)

        # Convert numpy array to list if needed
        if hasattr(scores, 'tolist'):
            scores = scores.tolist()

        # Match chunks with scores and sort by score (higher = more relevant)
        ranked = sorted(zip(chunks, scores, strict=True), key=lambda x: x[1], reverse=True)

        return [
            {**chunk, 'score': float(score)}
            for chunk, score in ranked
        ]

    def unload(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None


reranker = Reranker()
