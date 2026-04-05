from __future__ import annotations

from informity.api.chat_sources import merge_sources, serialize_sources
from informity.api.schemas import ChatSourceReference


def test_merge_sources_deduplicates_by_path_and_filename() -> None:
    source_map: dict[tuple[str, str], ChatSourceReference] = {}
    merge_sources(
        source_map,
        [
            ChatSourceReference(filename='a.txt', path='/a.txt', chunk_preview='a', relevance_score=0.5),
            ChatSourceReference(filename='a.txt', path='/a.txt', chunk_preview='updated', relevance_score=0.9),
        ],
    )
    assert len(source_map) == 1
    merged = list(source_map.values())[0]
    assert merged.chunk_preview == 'updated'


def test_serialize_sources_returns_json_payloads() -> None:
    payload = serialize_sources([
        ChatSourceReference(filename='a.txt', path='/a.txt', chunk_preview='a', relevance_score=0.5),
    ])
    assert isinstance(payload, list)
    assert payload[0]['filename'] == 'a.txt'
