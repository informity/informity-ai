from __future__ import annotations

from informity.llm.classification_policy import (
    apply_scoped_file_chat_summary_precedence,
    resolve_assistant_forced_classification,
)
from informity.llm.query_classifier import QueryClassification
from informity.llm.types import IntentProfileId, QueryType


def test_resolve_assistant_forced_classification_defaults_to_simple() -> None:
    classification = resolve_assistant_forced_classification(None)
    assert classification.intent == QueryType.SIMPLE


def test_resolve_assistant_forced_classification_preserves_existing_value() -> None:
    provided = QueryClassification(intent=QueryType.SIMPLE, confidence=0.5)
    classification = resolve_assistant_forced_classification(provided)
    assert classification is provided


def test_scoped_file_precedence_promotes_document_deictic_summary_request_to_focused() -> None:
    classification = QueryClassification(
        intent=QueryType.SIMPLE,
        route_candidate=IntentProfileId.CLARIFICATION_OR_DISAMBIGUATION,
        needs_chat_history=True,
        reason_codes=['promptcue_adapter'],
    )
    updated = apply_scoped_file_chat_summary_precedence(
        question='What are top 5 topics from this document?',
        classification=classification,
        scoped_file_active=True,
    )
    assert updated.intent == QueryType.FOCUSED
    assert updated.route_candidate == IntentProfileId.TARGETED_FACT_LOOKUP
    assert updated.needs_chat_history is False
    assert 'policy_scoped_file_document_request_precedence' in updated.reason_codes


def test_scoped_file_precedence_keeps_explicit_chat_summary_requests() -> None:
    classification = QueryClassification(
        intent=QueryType.SIMPLE,
        route_candidate=IntentProfileId.CLARIFICATION_OR_DISAMBIGUATION,
        needs_chat_history=True,
        reason_codes=['promptcue_adapter'],
    )
    updated = apply_scoped_file_chat_summary_precedence(
        question='Summarize our chat topics so far.',
        classification=classification,
        scoped_file_active=True,
    )
    assert updated.intent == QueryType.SIMPLE
    assert updated.route_candidate == IntentProfileId.CLARIFICATION_OR_DISAMBIGUATION
    assert updated.needs_chat_history is True
    assert 'policy_scoped_file_document_request_precedence' not in updated.reason_codes


def test_scoped_file_precedence_noop_without_active_scope() -> None:
    classification = QueryClassification(
        intent=QueryType.SIMPLE,
        route_candidate=IntentProfileId.CLARIFICATION_OR_DISAMBIGUATION,
        needs_chat_history=True,
        reason_codes=['promptcue_adapter'],
    )
    updated = apply_scoped_file_chat_summary_precedence(
        question='What are top 5 topics from this document?',
        classification=classification,
        scoped_file_active=False,
    )
    assert updated.intent == QueryType.SIMPLE
    assert updated.route_candidate == IntentProfileId.CLARIFICATION_OR_DISAMBIGUATION
    assert updated.needs_chat_history is True
