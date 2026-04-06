from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

SourceProvider = str
SourceItemId = str
FILESYSTEM_PROVIDER: SourceProvider = 'filesystem'


@dataclass(frozen=True)
class SourceItemRef:
    provider: SourceProvider
    item_id: SourceItemId
    locator: str
    item_type: str
    modified_at: datetime | None = None


@dataclass
class IngestionItem:
    provider: SourceProvider
    source_item_id: SourceItemId
    item_type: str
    title: str
    author: str | None
    modified_at: datetime | None
    content_text: str
    content_hash: str
    metadata: dict[str, object]
    attachments: list[Path]


class ContentSourceAdapter(Protocol):
    provider: str

    def discover(self, scope: dict[str, Any]) -> list[SourceItemRef]:
        """Enumerate candidate items for ingestion within the requested scope."""

    def fetch(self, ref: SourceItemRef) -> IngestionItem:
        """Read + normalize one item; returns only normalized ingestion payload."""

    def dedupe_key(self, item: IngestionItem) -> str:
        """Stable dedupe identity for provider item."""

    def canonical_id(self, item: IngestionItem) -> str:
        """Stable source identity used for storage uniqueness."""
