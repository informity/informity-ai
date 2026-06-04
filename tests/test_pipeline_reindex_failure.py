from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from informity.db.models import IndexedFile
from informity.indexer.pipeline import IndexResult, _chunk_embed_store, reindex_file
from informity.scanner.crawler import ScannedFile


@pytest.mark.asyncio
async def test_chunk_embed_store_fails_when_no_embeddable_child_chunks_exist(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    file_path = tmp_path / 'example.txt'
    parent_chunk = SimpleNamespace(
        chunk_index=0,
        content='parent',
        token_count=1,
        page_number=None,
        start_page=None,
        end_page=None,
        section_path=None,
        block_type=None,
        parent_chunk_index=None,
    )
    child_chunk = SimpleNamespace(
        chunk_index=1,
        content='child',
        token_count=999999,
        page_number=None,
        start_page=None,
        end_page=None,
        section_path=None,
        block_type=None,
        parent_chunk_index=0,
    )
    monkeypatch.setattr('informity.indexer.pipeline.chunk_text', lambda *_args, **_kwargs: [parent_chunk])
    monkeypatch.setattr('informity.indexer.pipeline.create_child_chunks', lambda _parents: [child_chunk])
    monkeypatch.setattr('informity.indexer.pipeline.insert_chunks_batch', AsyncMock(return_value=[11]))
    monkeypatch.setattr('informity.indexer.pipeline.embedder.get_effective_batch_size', lambda: 1)
    store_embeddings_mock = AsyncMock(return_value=0)
    monkeypatch.setattr('informity.indexer.pipeline.vector_store.store_embeddings_async', store_embeddings_mock)
    cleanup_mock = AsyncMock()
    monkeypatch.setattr('informity.indexer.pipeline._cleanup_partial_file_data', cleanup_mock)

    result = await _chunk_embed_store(
        db=AsyncMock(),
        file_id=7,
        text='parent child content',
        file_path=file_path,
        filename='example.txt',
        extension='.txt',
        category='plaintext',
        year=None,
    )

    assert result.success is False
    assert result.error_code == 'no_embeddable_chunks'
    assert result.retryable is False
    cleanup_mock.assert_awaited_once()
    store_embeddings_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_reindex_file_preserves_file_row_on_chunk_store_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    scanned = ScannedFile(
        path=Path('/tmp/example.txt'),
        filename='example.txt',
        extension='.txt',
        size_bytes=12,
        content_hash='hash-123',
        modified_at=datetime.now(UTC),
    )
    existing = IndexedFile(
        id=7,
        source_provider='filesystem',
        entity_type='file',
        source_item_id='/tmp/example.txt',
        path='/tmp/example.txt',
        filename='example.txt',
        extension='.txt',
        size_bytes=12,
        content_hash='old-hash',
        extracted_text_preview='old preview',
        category='plaintext',
        modified_at=datetime.now(UTC),
    )

    class _FakeExtractor:
        def extract(self, _path):  # type: ignore[no-untyped-def]
            return SimpleNamespace(
                error=None,
                text='hello world',
                preview_text='hello',
                metadata={},
                char_to_page_ranges={},
                char_to_block_type_ranges={},
                char_to_header_level_ranges={},
            )

    monkeypatch.setattr('informity.indexer.pipeline.get_file_by_source_identity', AsyncMock(return_value=existing))
    monkeypatch.setattr('informity.indexer.pipeline.get_file_by_path', AsyncMock(return_value=None))
    monkeypatch.setattr('informity.indexer.pipeline.get_extractor', lambda _path: _FakeExtractor())
    monkeypatch.setattr('informity.indexer.pipeline.delete_chunks_for_file', AsyncMock())
    monkeypatch.setattr('informity.indexer.pipeline.update_file', AsyncMock())
    monkeypatch.setattr('informity.indexer.pipeline.post_process_extracted_text', lambda text: text)
    monkeypatch.setattr('informity.indexer.pipeline.classify_file', lambda _path, _ext: SimpleNamespace(value='plaintext'))
    monkeypatch.setattr('informity.indexer.pipeline.extract_year', lambda _path, _text: None)
    monkeypatch.setattr('informity.indexer.pipeline.generate_tags', lambda _path: [])
    monkeypatch.setattr('informity.indexer.pipeline._build_file_metadata', lambda _path, _metadata: {
        'extractor': 'fake',
        'encoding': 'utf-8',
        'language': 'en',
        'mime_type': 'text/plain',
        'ocr_used': False,
        'page_count': None,
        'tables_count': 0,
        'form_items_count': 0,
        'key_value_items_count': 0,
        'pictures_count': 0,
        'document_hash': None,
    })
    monkeypatch.setattr('informity.indexer.pipeline.vector_store.delete_by_file_id', lambda _file_id: None)
    monkeypatch.setattr(
        'informity.indexer.pipeline._chunk_embed_store',
        AsyncMock(return_value=IndexResult(success=False, chunks_created=0, error='boom', error_code='chunk_failed', retryable=True)),
    )
    cleanup_mock = AsyncMock()
    monkeypatch.setattr('informity.indexer.pipeline._cleanup_partial_file_data', cleanup_mock)
    delete_file_mock = AsyncMock()
    monkeypatch.setattr('informity.indexer.pipeline.delete_file', delete_file_mock)

    result = await reindex_file(db=AsyncMock(), scanned=scanned)

    assert result.success is False
    assert result.error == 'boom'
    cleanup_mock.assert_awaited_once()
    delete_file_mock.assert_not_awaited()
