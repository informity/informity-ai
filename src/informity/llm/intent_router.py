# ==============================================================================
# Informity AI — Intent Router
# Pluggable intent routing backend with embedding-similarity primary path.
# ==============================================================================

from __future__ import annotations

import re
from dataclasses import dataclass
from threading import Lock
from typing import Literal, Protocol

import numpy as np
import structlog

from informity.indexer.embedder import embedder

log = structlog.get_logger(__name__)

IntentLabel = Literal['metadata', 'simple', 'focused', 'coverage']


@dataclass(frozen=True)
class IntentPrediction:
    intent: IntentLabel
    confidence: float
    alternatives: list[tuple[str, float]]
    reason_codes: list[str]


class IntentRouter(Protocol):
    def classify_intent(self, query: str) -> IntentPrediction:
        ...


@dataclass(frozen=True)
class _IntentSpec:
    intent: IntentLabel
    description: str
    examples: tuple[str, ...]
    negatives: tuple[str, ...] = ()


_INTENT_SPECS: tuple[_IntentSpec, ...] = (
    _IntentSpec(
        intent='metadata',
        description='Inventory and metadata requests about indexed files (counts, lists, years, categories, file types).',
        examples=(
            'How many documents are indexed?',
            'What file types are in the index?',
            'What years are covered in the indexed documents?',
            'List all indexed documents from 2021.',
            'What kind of documents do you have indexed?',
            'Show files from 2022.',
        ),
        negatives=(
            'Compare records from 2021 versus 2022 and explain what changed.',
            'Summarize the main subject of the employee performance review document.',
        ),
    ),
    _IntentSpec(
        intent='simple',
        description='General conversational or capability prompts that do not require corpus retrieval.',
        examples=(
            'hello',
            'hey there',
            'thanks',
            'Can you help me understand what information is available?',
            'What can you do?',
            'Help',
        ),
        negatives=(
            'List all indexed documents from 2021.',
            'Which indexed documents contain numeric amounts or financial figures?',
        ),
    ),
    _IntentSpec(
        intent='focused',
        description='Targeted question answered from one or a few documents.',
        examples=(
            'What does the onboarding policy document say about remote work?',
            'Summarize the section on incident response in the security handbook.',
            'Find documents that mention data retention and summarize the guidance.',
            'What deadline is stated in the project plan document?',
        ),
        negatives=(
            'How many documents are indexed?',
            'Compare indexed records from 2021 versus 2022. What changed?',
        ),
    ),
    _IntentSpec(
        intent='coverage',
        description='Broad cross-document synthesis, comparison, or aggregated findings across many records.',
        examples=(
            'Compare policy changes between 2021 and 2022 across the corpus.',
            'Summarize recurring themes across all indexed documents.',
            'Which indexed documents discuss compliance requirements? List files and key points.',
            'Provide a cross-document synthesis of major changes across all records.',
        ),
        negatives=(
            'How many documents are indexed?',
            'What deadline is stated in the project plan document?',
        ),
    ),
)

_ROUTE_CONFIDENCE_FLOOR = 0.35
_NEGATIVE_PENALTY_WEIGHT = 0.15
_MIN_TOKEN_PATTERN = re.compile(r'[a-z0-9]+')


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return vectors / norms


class EmbeddingSimilarityIntentRouter:
    def __init__(self) -> None:
        self._lock = Lock()
        self._ready = False
        self._candidate_texts: list[str] = []
        self._candidate_intents: list[IntentLabel] = []
        self._candidate_vectors: np.ndarray | None = None
        self._negative_vectors: dict[IntentLabel, np.ndarray] = {}

    def _ensure_index(self) -> None:
        if self._ready:
            return
        with self._lock:
            if self._ready:
                return

            texts: list[str] = []
            intents: list[IntentLabel] = []
            negatives: dict[IntentLabel, list[str]] = {}
            for spec in _INTENT_SPECS:
                desc_text = f'{spec.intent} intent: {spec.description}'
                texts.append(desc_text)
                intents.append(spec.intent)
                for example in spec.examples:
                    texts.append(f'{spec.intent} example: {example}')
                    intents.append(spec.intent)
                if spec.negatives:
                    negatives[spec.intent] = [f'negative: {x}' for x in spec.negatives]

            vectors = np.array(
                [embedder.embed_query(text) for text in texts],
                dtype=np.float32,
            )

            self._candidate_texts = texts
            self._candidate_intents = intents
            self._candidate_vectors = _normalize(vectors)

            negative_vectors: dict[IntentLabel, np.ndarray] = {}
            for intent, neg_texts in negatives.items():
                neg_vec = np.array(
                    [embedder.embed_query(text) for text in neg_texts],
                    dtype=np.float32,
                )
                negative_vectors[intent] = _normalize(neg_vec)
            self._negative_vectors = negative_vectors
            self._ready = True

    def _fallback_predict(self, query: str) -> IntentPrediction:
        query_tokens = set(_MIN_TOKEN_PATTERN.findall(query.casefold()))
        if not query_tokens:
            return IntentPrediction(
                intent='simple',
                confidence=_ROUTE_CONFIDENCE_FLOOR,
                alternatives=[('simple', _ROUTE_CONFIDENCE_FLOOR)],
                reason_codes=['router_embedding_unavailable', 'token_overlap_fallback'],
            )

        intent_scores: dict[IntentLabel, float] = {spec.intent: 0.0 for spec in _INTENT_SPECS}
        for spec in _INTENT_SPECS:
            candidate_texts = (spec.description, *spec.examples)
            best = 0.0
            for text in candidate_texts:
                tokens = set(_MIN_TOKEN_PATTERN.findall(text.casefold()))
                if not tokens:
                    continue
                overlap = len(query_tokens & tokens) / len(query_tokens | tokens)
                best = max(best, overlap)
            intent_scores[spec.intent] = best

        ranked = sorted(intent_scores.items(), key=lambda item: item[1], reverse=True)
        top_intent, top_score = ranked[0]
        alternatives = [(intent, round(max(_ROUTE_CONFIDENCE_FLOOR, score), 4)) for intent, score in ranked[:3]]
        return IntentPrediction(
            intent=top_intent,
            confidence=round(max(_ROUTE_CONFIDENCE_FLOOR, top_score), 4),
            alternatives=alternatives,
            reason_codes=['router_embedding_unavailable', 'token_overlap_fallback'],
        )

    def classify_intent(self, query: str) -> IntentPrediction:
        text = str(query or '').strip()
        if not text:
            return IntentPrediction(
                intent='simple',
                confidence=1.0,
                alternatives=[('simple', 1.0)],
                reason_codes=['empty_query_default'],
            )

        try:
            self._ensure_index()
            assert self._candidate_vectors is not None

            query_vec = np.array(embedder.embed_query(text), dtype=np.float32).reshape(1, -1)
            query_vec = _normalize(query_vec)

            scores = (self._candidate_vectors @ query_vec.T).reshape(-1)
            intent_max: dict[IntentLabel, float] = {}
            for intent, score in zip(self._candidate_intents, scores, strict=False):
                intent_max[intent] = max(intent_max.get(intent, -1.0), float(score))

            for intent, neg_vecs in self._negative_vectors.items():
                neg_scores = (neg_vecs @ query_vec.T).reshape(-1)
                if neg_scores.size:
                    penalty = max(0.0, float(np.max(neg_scores))) * _NEGATIVE_PENALTY_WEIGHT
                    intent_max[intent] = intent_max.get(intent, -1.0) - penalty

            ranked = sorted(intent_max.items(), key=lambda item: item[1], reverse=True)
            top_intent, top_raw = ranked[0]
            alt = ranked[:3]

            def _to_conf(x: float) -> float:
                return max(_ROUTE_CONFIDENCE_FLOOR, min(1.0, (x + 1.0) / 2.0))

            confidence = round(_to_conf(top_raw), 4)
            alternatives = [(intent, round(_to_conf(score), 4)) for intent, score in alt]
            return IntentPrediction(
                intent=top_intent,
                confidence=confidence,
                alternatives=alternatives,
                reason_codes=['embedding_similarity_router'],
            )
        except Exception as exc:  # noqa: BLE001 - fallback required to preserve routing availability
            log.warning('intent_router_embedding_failed', error=str(exc))
            return self._fallback_predict(text)


_intent_router: IntentRouter = EmbeddingSimilarityIntentRouter()


def get_intent_router() -> IntentRouter:
    return _intent_router


def set_intent_router_for_testing(router: IntentRouter) -> None:
    global _intent_router
    _intent_router = router
