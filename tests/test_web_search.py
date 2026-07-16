"""Test module for tests test web search."""

# pylint: disable=line-too-long

from __future__ import annotations

from informity.config import settings
from informity.llm.web_search import (
    SearchResult,
    WebSearchOutcome,
    has_any_provider_api_key,
    search_web,
)


def test_has_any_provider_api_key_checks_tavily_and_linkup() -> None:
    """Test has any provider api key checks tavily and linkup."""
    original_tavily = settings.tavily_api_key
    original_linkup = settings.linkup_api_key
    try:
        settings.tavily_api_key = ""
        settings.linkup_api_key = ""
        assert has_any_provider_api_key() is False

        settings.linkup_api_key = "lk_test"
        assert has_any_provider_api_key() is True

        settings.linkup_api_key = ""
        settings.tavily_api_key = "tvly_test"
        assert has_any_provider_api_key() is True
    finally:
        settings.tavily_api_key = original_tavily
        settings.linkup_api_key = original_linkup


def test_search_web_returns_privacy_blocked_without_override() -> None:
    """Test search web returns privacy blocked without override."""
    original_privacy = settings.full_privacy
    try:
        settings.full_privacy = True
        result = search_web("test query", allow_privacy_override=False)
        assert result.status == "privacy_blocked"
        assert not result.results
    finally:
        settings.full_privacy = original_privacy


def test_search_web_fails_over_when_primary_unavailable(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """Test search web fails over when primary unavailable."""
    original_primary = settings.web_search_primary_provider
    original_privacy = settings.full_privacy
    original_tavily = settings.tavily_api_key
    original_linkup = settings.linkup_api_key
    try:
        settings.web_search_primary_provider = "tavily"
        settings.full_privacy = False
        settings.tavily_api_key = "tvly_key"
        settings.linkup_api_key = "lk_key"

        responses = {
            "tavily": WebSearchOutcome(results=[], status="quota_exceeded"),
            "linkup": WebSearchOutcome(
                results=[
                    SearchResult(title="Example", url="https://example.com", snippet="Example")
                ],
                status="ok",
            ),
        }

        def _fake_search_with_provider(provider_name, *, query):  # type: ignore[no-untyped-def]
            """Internal helper for fake search with provider."""
            del query
            return responses[str(provider_name)]

        monkeypatch.setattr(
            "informity.llm.web_search._search_with_provider",
            _fake_search_with_provider,
        )

        result = search_web("test query")
        assert result.status == "ok"
        assert result.provider_attempted == "tavily->linkup"
        assert result.provider_used == "linkup"
        assert result.failover_applied is True
    finally:
        settings.web_search_primary_provider = original_primary
        settings.full_privacy = original_privacy
        settings.tavily_api_key = original_tavily
        settings.linkup_api_key = original_linkup


def test_search_web_returns_fallback_status_when_both_fail(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """Test search web returns fallback status when both fail."""
    original_primary = settings.web_search_primary_provider
    original_privacy = settings.full_privacy
    original_tavily = settings.tavily_api_key
    original_linkup = settings.linkup_api_key
    try:
        settings.web_search_primary_provider = "tavily"
        settings.full_privacy = False
        settings.tavily_api_key = "tvly_key"
        settings.linkup_api_key = "lk_key"

        responses = {
            "tavily": WebSearchOutcome(results=[], status="api_key_missing"),
            "linkup": WebSearchOutcome(results=[], status="auth_invalid"),
        }

        def _fake_search_with_provider(provider_name, *, query):  # type: ignore[no-untyped-def]
            """Internal helper for fake search with provider."""
            del query
            return responses[str(provider_name)]

        monkeypatch.setattr(
            "informity.llm.web_search._search_with_provider",
            _fake_search_with_provider,
        )

        result = search_web("test query")
        assert result.status == "auth_invalid"
        assert result.provider_attempted == "tavily->linkup"
        assert result.provider_used is None
        assert result.failover_applied is False
    finally:
        settings.web_search_primary_provider = original_primary
        settings.full_privacy = original_privacy
        settings.tavily_api_key = original_tavily
        settings.linkup_api_key = original_linkup


def test_search_web_respects_linkup_primary_order(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """Test search web respects linkup primary order."""
    original_primary = settings.web_search_primary_provider
    original_privacy = settings.full_privacy
    original_tavily = settings.tavily_api_key
    original_linkup = settings.linkup_api_key
    try:
        settings.web_search_primary_provider = "linkup"
        settings.full_privacy = False
        settings.tavily_api_key = "tvly_key"
        settings.linkup_api_key = "lk_key"

        calls: list[str] = []

        def _fake_search_with_provider(provider_name, *, query):  # type: ignore[no-untyped-def]
            """Internal helper for fake search with provider."""
            del query
            calls.append(str(provider_name))
            return WebSearchOutcome(results=[], status="provider_error")

        monkeypatch.setattr(
            "informity.llm.web_search._search_with_provider",
            _fake_search_with_provider,
        )

        result = search_web("test query")
        assert calls == ["linkup", "tavily"]
        assert result.status == "provider_error"
        assert result.provider_attempted == "linkup->tavily"
    finally:
        settings.web_search_primary_provider = original_primary
        settings.full_privacy = original_privacy
        settings.tavily_api_key = original_tavily
        settings.linkup_api_key = original_linkup


def test_search_web_preserves_primary_error_when_fallback_key_missing(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """Test search web preserves primary error when fallback key missing."""
    original_primary = settings.web_search_primary_provider
    original_privacy = settings.full_privacy
    original_tavily = settings.tavily_api_key
    original_linkup = settings.linkup_api_key
    try:
        settings.web_search_primary_provider = "tavily"
        settings.full_privacy = False
        settings.tavily_api_key = "tvly_key"
        settings.linkup_api_key = ""

        responses = {
            "tavily": WebSearchOutcome(results=[], status="auth_invalid"),
            "linkup": WebSearchOutcome(results=[], status="api_key_missing"),
        }

        def _fake_search_with_provider(provider_name, *, query):  # type: ignore[no-untyped-def]
            """Internal helper for fake search with provider."""
            del query
            return responses[str(provider_name)]

        monkeypatch.setattr(
            "informity.llm.web_search._search_with_provider", _fake_search_with_provider
        )

        result = search_web("test query")
        assert result.status == "auth_invalid"
        assert result.provider_attempted == "tavily->linkup"
    finally:
        settings.web_search_primary_provider = original_primary
        settings.full_privacy = original_privacy
        settings.tavily_api_key = original_tavily
        settings.linkup_api_key = original_linkup
