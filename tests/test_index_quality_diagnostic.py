from __future__ import annotations

# ruff: noqa: I001

import json
from pathlib import Path

import pytest

from informity.indexer.chunker import ChunkData
from informity.scanner.extractors.base import ExtractedDocument

from scripts.diagnostics import index_quality


class _FakeEmbedder:
    def __init__(self, vectors: list[list[float]], dimension: int = 3) -> None:
        self._vectors = vectors
        self._dimension = dimension

    def get_effective_batch_size(self) -> int:
        return 8

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        assert len(texts) == len(self._vectors)
        return self._vectors

    def get_embedding_dimension(self) -> int:
        return self._dimension


@pytest.mark.diagnostics
def test_run_quality_diagnostic_writes_success_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / 'document.txt'
    source.write_text('hello world\n' * 30, encoding='utf-8')

    monkeypatch.setattr(
        index_quality,
        '_extract_document',
        lambda file_path: ExtractedDocument(
            text='hello world\n' * 30,
            source_path=file_path,
            metadata={'converter': 'plain_text', 'ocr_used': 'false'},
            preview_text='hello world',
        ),
    )
    monkeypatch.setattr(index_quality, 'post_process_extracted_text', lambda text: text)
    monkeypatch.setattr(
        index_quality,
        'chunk_text',
        lambda text, **kwargs: [
            ChunkData(content=text, chunk_index=0, token_count=60),
        ],
    )
    monkeypatch.setattr(
        index_quality,
        'create_child_chunks',
        lambda parent_chunks, **kwargs: [
            ChunkData(
                content=parent_chunks[0].content,
                chunk_index=0,
                token_count=60,
                parent_chunk_index=parent_chunks[0].chunk_index,
            )
        ],
    )
    monkeypatch.setattr(index_quality, 'embedder', _FakeEmbedder([[0.1, 0.2, 0.3]]))

    report = index_quality.run_quality_diagnostic(source, output_dir=tmp_path)

    assert report['overall_status'] == 'OK'
    assert report['stages']['Extraction']['status'] == 'OK'
    assert report['stages']['Embedding']['status'] == 'OK'

    report_path = Path(report['report_path'])
    assert report_path.exists()

    stored = json.loads(report_path.read_text(encoding='utf-8'))
    assert stored['input_file'] == str(source.resolve())
    assert stored['overall_status'] == 'OK'


@pytest.mark.diagnostics
def test_run_quality_diagnostic_flags_warnings_and_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / 'noisy.txt'
    source.write_text('header only\n', encoding='utf-8')

    raw_text = 'B' * 160
    monkeypatch.setattr(
        index_quality,
        '_extract_document',
        lambda file_path: ExtractedDocument(
            text=raw_text,
            source_path=file_path,
            metadata={'converter': 'plain_text', 'ocr_used': 'false'},
            preview_text=raw_text[:20],
        ),
    )
    monkeypatch.setattr(index_quality, 'post_process_extracted_text', lambda text: text[:80])
    monkeypatch.setattr(
        index_quality,
        'chunk_text',
        lambda text, **kwargs: [
            ChunkData(content='page 1', chunk_index=0, token_count=2),
            ChunkData(content='#### Heading', chunk_index=1, token_count=3),
        ],
    )
    monkeypatch.setattr(
        index_quality,
        'create_child_chunks',
        lambda parent_chunks, **kwargs: [
            ChunkData(content=chunk.content, chunk_index=chunk.chunk_index, token_count=chunk.token_count, parent_chunk_index=chunk.chunk_index)
            for chunk in parent_chunks
        ],
    )
    monkeypatch.setattr(index_quality, 'embedder', _FakeEmbedder([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]))

    report = index_quality.run_quality_diagnostic(source, output_dir=tmp_path)

    assert report['overall_status'] == 'ERROR'
    assert report['stages']['Post-processing']['status'] == 'WARN'
    assert report['stages']['Chunking']['status'] == 'ERROR'
    assert Path(report['report_path']).exists()
