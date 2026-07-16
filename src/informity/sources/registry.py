"""Module for sources registry."""

from __future__ import annotations

from dataclasses import dataclass

from informity.sources.base import ContentSourceAdapter


@dataclass(frozen=True)
class AdapterHealth:
    """Class docstring."""
    provider: str
    available: bool
    reason: str | None = None


class SourceAdapterRegistry:
    """Class docstring."""
    def __init__(self) -> None:
        """Initialize the instance."""
        self._adapters: dict[str, ContentSourceAdapter] = {}

    def register(self, adapter: ContentSourceAdapter) -> None:
        """Register."""
        provider = adapter.provider.strip().lower()
        if not provider:
            raise ValueError("Adapter provider key cannot be empty")
        self._adapters[provider] = adapter

    def get(self, provider: str) -> ContentSourceAdapter:
        """Get."""
        key = provider.strip().lower()
        adapter = self._adapters.get(key)
        if adapter is None:
            raise KeyError(f"No source adapter registered for provider: {provider}")
        return adapter

    def list_providers(self) -> list[str]:
        """List providers."""
        return sorted(self._adapters.keys())

    def health(self) -> list[AdapterHealth]:
        """Health."""
        return [
            AdapterHealth(provider=provider, available=True) for provider in self.list_providers()
        ]
