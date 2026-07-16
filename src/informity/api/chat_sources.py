# ==============================================================================
# Informity AI — Chat Source Utilities
# ==============================================================================

"""Module for api chat sources."""

from __future__ import annotations

from informity.api.schemas import ChatSourceReference


def source_key(source: ChatSourceReference) -> tuple[str, str]:
    """Source key."""
    return source.path, source.filename


def merge_sources(
    source_map: dict[tuple[str, str], ChatSourceReference],
    incoming_sources: list[ChatSourceReference],
) -> None:
    """Merge sources."""
    for source in incoming_sources:
        source_map[source_key(source)] = source


def serialize_sources(sources: list[ChatSourceReference]) -> list[dict[str, object]]:
    """Serialize sources."""
    return [source.model_dump(mode="json") for source in sources]
