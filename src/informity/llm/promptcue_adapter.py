# ==============================================================================
# Informity AI — PromptCue Intent Adapter
# Thin IntentRouter implementation backed by the PromptCue classifier.
# Eliminates the duplicate embedding model and 4-type taxonomy by delegating
# to PromptCue's 12-type classifier via the injectable embed_fn path.
# ==============================================================================

from __future__ import annotations

from threading import Lock
from typing import TYPE_CHECKING, cast

import structlog

from informity.indexer.embedder import embedder
from informity.llm.intent_router import IntentLabel, IntentPrediction

if TYPE_CHECKING:
    from promptcue import PromptCueAnalyzer, PromptCueQueryObject

log = structlog.get_logger(__name__)

_FALLBACK_CONFIDENCE = 0.35

# ==============================================================================
# Type → Intent mapping
# ==============================================================================

# Default informity-ai intent label for each PromptCue query type.
# 'metadata' is not in the map — it is detected deterministically by
# query_classifier.py and never returned by the adapter.
_DEFAULT_INTENT: dict[str, IntentLabel] = {
    'chitchat':        IntentLabel.SIMPLE,
    'lookup':          IntentLabel.FOCUSED,
    'procedure':       IntentLabel.FOCUSED,
    'troubleshooting': IntentLabel.FOCUSED,
    'recommendation':  IntentLabel.FOCUSED,
    'validation':      IntentLabel.FOCUSED,
    'update':          IntentLabel.FOCUSED,
    'analysis':        IntentLabel.COVERAGE,
    'comparison':      IntentLabel.COVERAGE,
    'summarization':   IntentLabel.COVERAGE,
    'chat_summary':    IntentLabel.SIMPLE,
    'coverage':        IntentLabel.COVERAGE,
    'generation':      IntentLabel.COVERAGE,
}

# BROAD scope flips these focused-default types to 'coverage'.
_BROAD_FLIPS: frozenset[str] = frozenset({
    'lookup', 'procedure', 'troubleshooting', 'recommendation', 'validation', 'update',
})

def _map_intent(query_type: str, scope: str) -> IntentLabel:
    """Map a PromptCue (query_type, scope) pair to an informity-ai IntentLabel."""
    base = _DEFAULT_INTENT.get(query_type, IntentLabel.FOCUSED)
    normalized_scope = str(scope or '').strip().casefold()
    if base == IntentLabel.SIMPLE:
        return IntentLabel.SIMPLE
    if normalized_scope == 'broad' and query_type in _BROAD_FLIPS:
        return IntentLabel.COVERAGE
    return base


def _pcue_to_intent_prediction(pcue: PromptCueQueryObject) -> IntentPrediction:
    """Convert a PromptCueQueryObject to an IntentPrediction."""
    scope      = str(pcue.scope)
    top_intent = _map_intent(pcue.primary_query_type, scope)

    # Build up to 3 alternatives from candidates, deduplicating by intent label.
    seen: set[IntentLabel]        = {top_intent}
    alts: list[tuple[IntentLabel, float]] = [(top_intent, round(pcue.confidence, 4))]
    for candidate in pcue.candidate_query_types[:5]:
        alt_intent = _map_intent(candidate.label, scope)
        if alt_intent not in seen:
            seen.add(alt_intent)
            alts.append((alt_intent, round(candidate.score, 4)))
        if len(alts) >= 3:
            break

    return IntentPrediction(
        intent       = top_intent,
        confidence   = round(pcue.confidence, 4),
        alternatives = alts,
        reason_codes = ['promptcue_adapter', f'basis:{pcue.classification_basis}'],
    )


# ==============================================================================
# Adapter
# ==============================================================================

class PromptCueIntentAdapter:
    """IntentRouter implementation backed by PromptCue.

    Reuses the already-loaded Nomic embedding model via the injectable embed_fn
    (PromptCue Item 1) so no second model is loaded at startup.

    Two public methods:

    classify_intent(query) → IntentPrediction
        Satisfies the IntentRouter protocol.  Used by the default routing flow
        and by the set_intent_router_for_testing swap in tests.

    classify(query) → (IntentPrediction, PromptCueQueryObject)
        Extended API used by classify_query() to read additional signals
        (scope, routing_hints, action_hints, is_continuation) without
        triggering a second analysis call.
    """

    def __init__(self) -> None:
        self._analyzer: PromptCueAnalyzer | None = None
        self._lock = Lock()

    def _get_analyzer(self) -> PromptCueAnalyzer:
        if self._analyzer is None:
            with self._lock:
                if self._analyzer is None:
                    try:
                        from promptcue import PromptCueAnalyzer, PromptCueConfig
                        self._analyzer = PromptCueAnalyzer(
                            PromptCueConfig(embed_fn=embedder.embed_query)
                        )
                    except Exception as exc:  # noqa: BLE001 - convert lazy init failure to deterministic runtime error
                        raise RuntimeError('PromptCue analyzer failed to initialize') from exc
        return cast('PromptCueAnalyzer', self._analyzer)

    def classify(self, query: str) -> tuple[IntentPrediction, PromptCueQueryObject]:
        """Analyze query and return both intent prediction and full PromptCue output.

        Raises on failure — callers that need error isolation should use
        classify_intent() which has a built-in fallback.
        """
        pcue = self._get_analyzer().analyze(str(query or '').strip())
        return _pcue_to_intent_prediction(pcue), pcue

    def classify_intent(self, query: str) -> IntentPrediction:
        """IntentRouter protocol — maps PromptCue output to IntentPrediction.

        Falls back to a 'focused' prediction when PromptCue is unavailable
        (model not loaded, import error) to preserve routing availability.
        """
        text = str(query or '').strip()
        if not text:
            return IntentPrediction(
                intent       = IntentLabel.SIMPLE,
                confidence   = 1.0,
                alternatives = [(IntentLabel.SIMPLE, 1.0)],
                reason_codes = ['empty_query_default'],
            )
        try:
            prediction, _ = self.classify(text)
            return prediction
        except Exception as exc:  # noqa: BLE001 — fallback required to preserve routing availability
            log.warning('promptcue_adapter_failed', error=str(exc))
            return IntentPrediction(
                intent       = IntentLabel.FOCUSED,
                confidence   = _FALLBACK_CONFIDENCE,
                alternatives = [(IntentLabel.FOCUSED, _FALLBACK_CONFIDENCE)],
                reason_codes = ['promptcue_adapter_failed'],
            )
