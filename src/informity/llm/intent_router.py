# ==============================================================================
# Informity AI — Intent Router
# IntentRouter protocol and pluggable routing infrastructure.
# Default implementation: PromptCueIntentAdapter (see promptcue_adapter.py).
# ==============================================================================

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Literal, Protocol

IntentLabel = Literal['metadata', 'simple', 'focused', 'coverage']


@dataclass(frozen=True)
class IntentPrediction:
    intent:       IntentLabel
    confidence:   float
    alternatives: list[tuple[str, float]]
    reason_codes: list[str]


class IntentRouter(Protocol):
    def classify_intent(self, query: str) -> IntentPrediction:
        ...


# ==============================================================================
# Router singleton
# ==============================================================================

# Lazy-initialized on first get_intent_router() call to avoid a circular import
# with promptcue_adapter.py (which imports IntentLabel / IntentPrediction from
# this module).
_intent_router_lock: Lock                = Lock()
_intent_router:      IntentRouter | None = None


def get_intent_router() -> IntentRouter:
    """Return the active intent router, initializing the default on first call."""
    global _intent_router
    if _intent_router is not None:
        return _intent_router
    with _intent_router_lock:
        if _intent_router is None:
            from informity.llm.promptcue_adapter import PromptCueIntentAdapter
            _intent_router = PromptCueIntentAdapter()
    assert _intent_router is not None
    return _intent_router


def set_intent_router_for_testing(router: IntentRouter) -> None:
    """Swap the active router.  Restore the original after the test."""
    global _intent_router
    _intent_router = router
