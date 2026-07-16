"""Test module for tests test classification policy."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

import informity.llm.classification_policy as classification_policy_module
from informity.llm.classification_policy import classify_query_with_timing
from informity.llm.query_classifier import QueryClassification
from informity.llm.types import IntentProfileId, QueryType


@pytest.mark.asyncio
async def test_classify_query_with_timing_emits_shadow_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test classify query with timing emits shadow event."""
    classification = QueryClassification(
        intent=QueryType.METADATA,
        route_candidate=IntentProfileId.METADATA_INVENTORY,
        confidence=0.91,
    )

    monkeypatch.setattr(
        classification_policy_module,
        "classify_query",
        lambda *_args, **_kwargs: classification,
    )
    emit_event = AsyncMock()
    monkeypatch.setattr(classification_policy_module, "emit_log_event", emit_event)

    result, elapsed_ms = await classify_query_with_timing(
        "What kind of documents do you have indexed?"
    )

    assert result is classification
    assert elapsed_ms >= 0.0
    assert emit_event.await_count == 1
    assert emit_event.await_args.kwargs["event_name"] == "five_q_shadow"
    assert emit_event.await_args.kwargs["details"]["intent"] == str(QueryType.METADATA)
