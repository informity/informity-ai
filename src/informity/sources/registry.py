from __future__ import annotations

from dataclasses import dataclass

from informity.sources.base import ContentSourceAdapter


@dataclass(frozen=True)
class AdapterHealth:
    provider: str
    available: bool
    reason: str | None = None


class SourceAdapterRegistry:
    def __init__(self) -> None:
        self._adapters: dict[str, ContentSourceAdapter] = {}

    def register(self, adapter: ContentSourceAdapter) -> None:
        provider = adapter.provider.strip().lower()
        if not provider:
            raise ValueError('Adapter provider key cannot be empty')
        self._adapters[provider] = adapter

    def get(self, provider: str) -> ContentSourceAdapter:
        key = provider.strip().lower()
        adapter = self._adapters.get(key)
        if adapter is None:
            raise KeyError(f'No source adapter registered for provider: {provider}')
        return adapter

    def list_providers(self) -> list[str]:
        return sorted(self._adapters.keys())

    def health(self) -> list[AdapterHealth]:
        return [AdapterHealth(provider=provider, available=True) for provider in self.list_providers()]
