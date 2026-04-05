from __future__ import annotations

from informity.llm.classification_policy import resolve_assistant_forced_classification
from informity.llm.query_classifier import QueryClassification
from informity.llm.types import QueryType


def test_resolve_assistant_forced_classification_defaults_to_simple() -> None:
    classification = resolve_assistant_forced_classification(None)
    assert classification.intent == QueryType.SIMPLE


def test_resolve_assistant_forced_classification_preserves_existing_value() -> None:
    provided = QueryClassification(intent=QueryType.SIMPLE, confidence=0.5)
    classification = resolve_assistant_forced_classification(provided)
    assert classification is provided
