from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from informity.indexer.chunker import ChunkData
from informity.indexer.pipeline import _chunk_embed_store


@pytest.mark.asyncio
async def test_chunk_embed_store_fails_on_partial_vector_write(monkeypatch: pytest.MonkeyPatch) -> None:
    parent_chunk = ChunkData(content='parent', chunk_index=0, token_count=10)
    child_chunk = ChunkData(
        content='child',
        chunk_index=1,
        token_count=8,
        parent_chunk_index=0,
    )

    async def _to_thread(func, *args, **kwargs):  # type: ignore[no-untyped-def]
        return func(*args, **kwargs)

    deleted_vectors: list[int] = []
    deleted_chunks: list[int] = []

    monkeypatch.setattr('informity.indexer.pipeline.asyncio.to_thread', _to_thread)
    monkeypatch.setattr('informity.indexer.pipeline.chunk_text', lambda *args, **kwargs: [parent_chunk])
    monkeypatch.setattr('informity.indexer.pipeline.create_child_chunks', lambda *args, **kwargs: [child_chunk])
    monkeypatch.setattr(
        'informity.indexer.pipeline.insert_chunks_batch',
        AsyncMock(side_effect=[[100], [200]]),
    )
    monkeypatch.setattr(
        'informity.indexer.pipeline.delete_chunks_for_file',
        AsyncMock(side_effect=lambda _db, file_id: deleted_chunks.append(file_id) or 1),
    )
    monkeypatch.setattr('informity.indexer.pipeline.embedder.embed_texts', lambda texts: [[0.0] * 768 for _ in texts])
    monkeypatch.setattr(
        'informity.indexer.pipeline.vector_store.store_embeddings_async',
        AsyncMock(return_value=0),
    )
    monkeypatch.setattr(
        'informity.indexer.pipeline.vector_store.delete_by_file_id',
        lambda file_id: deleted_vectors.append(file_id),
    )

    result = await _chunk_embed_store(
        db=object(),  # Not used because DB functions are mocked
        file_id=42,
        text='sample',
        file_path=Path('/tmp/sample.txt'),
        filename='sample.txt',
        extension='.txt',
        category='plaintext',
        year=None,
    )

    assert result.success is False
    assert result.chunks_created == 0
    assert deleted_chunks == [42]
    assert deleted_vectors == [42]
