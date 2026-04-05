# ==============================================================================
# Informity AI — Chat Source Utilities
# ==============================================================================

from __future__ import annotations

from informity.api.schemas import ChatSourceReference


def source_key(source: ChatSourceReference) -> tuple[str, str]:
    return source.path, source.filename


def merge_sources(
    source_map: dict[tuple[str, str], ChatSourceReference],
    incoming_sources: list[ChatSourceReference],
) -> None:
    for source in incoming_sources:
        source_map[source_key(source)] = source


def serialize_sources(sources: list[ChatSourceReference]) -> list[dict[str, object]]:
    return [source.model_dump(mode='json') for source in sources]
