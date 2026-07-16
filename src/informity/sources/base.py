"""Module for sources base."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

SourceProvider = str
SourceItemId = str
FILESYSTEM_PROVIDER: SourceProvider = "filesystem"
SOURCE_ENTITY_FILE = "file"


@dataclass(frozen=True)
class SourceItemRef:
    """Class docstring."""
    provider: SourceProvider
    item_id: SourceItemId
    locator: str
    item_type: str
    modified_at: datetime | None = None


@dataclass
class IngestionItem:
    """Class docstring."""
    provider: SourceProvider
    source_item_id: SourceItemId
    item_type: str
    title: str
    author: str | None
    modified_at: datetime | None
    content_text: str
    size_bytes: int
    content_hash: str
    metadata: dict[str, object]
    attachments: list[Path]


class ContentSourceAdapter(Protocol):
    """Class docstring."""
    provider: str

    def discover(self, scope: dict[str, Any]) -> list[SourceItemRef]:
        """Enumerate candidate items for ingestion within the requested scope."""

    def fetch(self, ref: SourceItemRef) -> IngestionItem:
        """Read + normalize one item; returns only normalized ingestion payload."""
