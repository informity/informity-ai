from __future__ import annotations

import json
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

from informity.api import routes_index, routes_search, routes_system
from informity.api.schemas import ModelActionRequest, SearchRequest, SetupStartRequest
from informity.db.models import FileCategory, IndexedFile


async def _to_thread(func, *args, **kwargs):  # type: ignore[no-untyped-def]
    return func(*args, **kwargs)


@asynccontextmanager
async def _noop_slot():
    yield


@pytest.mark.asyncio
async def test_search_documents_applies_filters_and_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(routes_search.asyncio, 'to_thread', _to_thread)
    monkeypatch.setattr(routes_search.SEARCH_GUARD, 'slot', lambda: _noop_slot())
    monkeypatch.setattr(routes_search.embedder, 'embed_query', lambda _query: [0.1] * 8)
    monkeypatch.setattr(
        routes_search.vector_store,
        'search_similar',
        lambda _vec, _limit: [
            {'file_id': 1, 'chunk_text': 'alpha chunk', 'score': 0.11},
            {'file_id': 2, 'chunk_text': 'beta chunk', 'score': 0.22},
            {'file_id': 3, 'chunk_text': 'gamma chunk', 'score': 0.33},
            {'file_id': None, 'chunk_text': 'missing file id', 'score': 0.44},
        ],
    )

    files_by_id = {
        1: IndexedFile(
            id=1,
            path='/docs/a.pdf',
            filename='a.pdf',
            extension='.pdf',
            size_bytes=10,
            content_hash='h1',
            extracted_text_preview='a',
            category=FileCategory.DOCUMENT,
            modified_at=datetime.now(UTC),
        ),
        2: IndexedFile(
            id=2,
            path='/docs/b.txt',
            filename='b.txt',
            extension='.txt',
            size_bytes=10,
            content_hash='h2',
            extracted_text_preview='b',
            category=FileCategory.PLAINTEXT,
            modified_at=datetime.now(UTC),
        ),
        3: IndexedFile(
            id=3,
            path='/docs/c.md',
            filename='c.md',
            extension='.md',
            size_bytes=10,
            content_hash='h3',
            extracted_text_preview='c',
            category=FileCategory.DOCUMENT,
            modified_at=datetime.now(UTC),
        ),
    }
    monkeypatch.setattr(routes_search, 'get_files_by_ids', AsyncMock(return_value=files_by_id))

    response = await routes_search.search_documents(
        SearchRequest(query='  find documents  ', limit=2, category='document', file_types=['.pdf']),
        db=MagicMock(),
    )

    assert response.query == 'find documents'
    assert response.total == 1
    assert len(response.results) == 1
    assert response.results[0].filename == 'a.pdf'


@pytest.mark.asyncio
async def test_search_documents_rejects_empty_query(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(routes_search.SEARCH_GUARD, 'slot', lambda: _noop_slot())
    with pytest.raises(HTTPException) as exc_info:
        await routes_search.search_documents(SearchRequest(query='   '), db=MagicMock())
    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_get_index_status_aggregates_counts_and_sizes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(routes_index.asyncio, 'to_thread', _to_thread)
    monkeypatch.setattr(routes_index, 'get_file_count', AsyncMock(return_value=7))
    monkeypatch.setattr(routes_index, 'get_chunk_count', AsyncMock(return_value=30))
    monkeypatch.setattr(routes_index, 'get_indexed_content_size_bytes', AsyncMock(return_value=4096))
    monkeypatch.setattr(routes_index, 'get_chat_count', AsyncMock(return_value=2))
    monkeypatch.setattr(
        routes_index,
        'get_index_scope_counts',
        AsyncMock(return_value=[{'source_provider': 'filesystem', 'entity_type': 'file', 'files_count': 7, 'chunks_count': 30}]),
    )
    monkeypatch.setattr(
        routes_index,
        'get_latest_completed_scan',
        AsyncMock(return_value=SimpleNamespace(completed_at=datetime(2026, 2, 1, tzinfo=UTC))),
    )
    monkeypatch.setattr(routes_index.vector_store, 'get_stats', lambda: {'total_vectors': 27, 'storage_bytes': 12345})
    monkeypatch.setattr(routes_index, '_compute_disk_sizes', lambda: (111, 222))
    monkeypatch.setattr(routes_index.op_state, 'get_reset_state_snapshot', AsyncMock(return_value=(False, {'ok': True})))

    status = await routes_index.get_index_status(db=MagicMock())

    assert status.total_files == 7
    assert status.total_chunks == 30
    assert status.total_embeddings == 27
    assert status.vectors_size_bytes == 12345
    assert status.db_size_bytes == 111
    assert status.model_size_bytes == 222
    assert status.last_reset_result == {'ok': True}
    assert status.source_scope_stats == []

    status_with_scope = await routes_index.get_index_status(
        db=MagicMock(),
        include_source_scope_stats=True,
    )
    assert status_with_scope.source_scope_stats == [
        {'source_provider': 'filesystem', 'entity_type': 'file', 'files_count': 7, 'chunks_count': 30}
    ]


@pytest.mark.asyncio
async def test_get_index_status_short_circuits_while_reset_in_progress(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _fail(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError('DB aggregate should not be called during reset polling')

    monkeypatch.setattr(routes_index.asyncio, 'to_thread', _to_thread)
    monkeypatch.setattr(routes_index.op_state, 'get_reset_state_snapshot', AsyncMock(return_value=(True, {'ok': True})))
    monkeypatch.setattr(routes_index, '_compute_disk_sizes', lambda: (111, 222))
    monkeypatch.setattr(routes_index, 'get_file_count', _fail)
    monkeypatch.setattr(routes_index, 'get_chunk_count', _fail)
    monkeypatch.setattr(routes_index, 'get_indexed_content_size_bytes', _fail)
    monkeypatch.setattr(routes_index, 'get_chat_count', _fail)

    status = await routes_index.get_index_status(db=MagicMock())

    assert status.reset_in_progress is True
    assert status.last_reset_result == {'ok': True}
    assert status.total_files == 0
    assert status.total_chunks == 0
    assert status.total_embeddings == 0
    assert status.chat_count == 0
    assert status.indexed_content_size_bytes == 0
    assert status.db_size_bytes == 111
    assert status.model_size_bytes == 222


@pytest.mark.asyncio
async def test_get_diagnostics_returns_system_and_index_stats(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    app_data_dir = tmp_path / 'app-data'
    app_data_dir.mkdir(parents=True)
    db_path = tmp_path / 'diagnostics.db'
    db_path.write_bytes(b'12345678')
    model_path = tmp_path / 'model.gguf'
    model_path.write_bytes(b'x' * 1024)

    monkeypatch.setattr(routes_system.settings, 'app_data_dir', app_data_dir)
    monkeypatch.setattr(routes_system.settings, 'db_path', db_path)
    monkeypatch.setattr(routes_system, 'llm_engine', SimpleNamespace(is_loaded=True, _get_model_path=lambda: model_path))
    monkeypatch.setattr(routes_system, 'get_file_count', AsyncMock(return_value=5))
    monkeypatch.setattr(routes_system, 'get_chunk_count', AsyncMock(return_value=12))
    monkeypatch.setattr(routes_system, 'get_indexed_content_size_bytes', AsyncMock(return_value=2048))
    monkeypatch.setattr(routes_system.vector_store, 'get_stats', lambda: {'storage_bytes': 4096})
    monkeypatch.setattr(routes_system.platform, 'python_version', lambda: '3.13.0')
    monkeypatch.setattr(routes_system.platform, 'system', lambda: 'Darwin')
    monkeypatch.setattr(routes_system.platform, 'version', lambda: '23.0')
    monkeypatch.setattr(routes_system.platform, 'machine', lambda: 'arm64')
    monkeypatch.setattr(
        routes_system.psutil,
        'virtual_memory',
        lambda: SimpleNamespace(total=16 * 1024**3, available=10 * 1024**3, used=6 * 1024**3),
    )
    monkeypatch.setattr(
        routes_system.psutil,
        'disk_usage',
        lambda _path: SimpleNamespace(total=512 * 1024**3, free=300 * 1024**3, used=212 * 1024**3),
    )

    @asynccontextmanager
    async def _fake_get_db():
        yield MagicMock()

    monkeypatch.setattr(routes_system, 'get_db', _fake_get_db)

    diagnostics = await routes_system.get_diagnostics(
        request=SimpleNamespace(client=SimpleNamespace(host='127.0.0.1')),
    )
    assert diagnostics.total_files == 5
    assert diagnostics.total_chunks == 12
    assert diagnostics.indexed_content_size_bytes == 2048
    assert diagnostics.vectors_size_bytes == 4096
    assert diagnostics.model_loaded is True
    assert diagnostics.model_filename == 'model.gguf'
    assert diagnostics.db_size_bytes == 8


@pytest.mark.asyncio
async def test_shutdown_rejects_non_localhost() -> None:
    request = SimpleNamespace(client=SimpleNamespace(host='10.0.0.8'))
    with pytest.raises(HTTPException) as exc_info:
        await routes_system.shutdown(request=request)
    assert exc_info.value.status_code == 403


@pytest.mark.asyncio
async def test_get_diagnostics_rejects_non_localhost() -> None:
    request = SimpleNamespace(client=SimpleNamespace(host='10.0.0.8'))
    with pytest.raises(HTTPException) as exc_info:
        await routes_system.get_diagnostics(request=request)
    assert exc_info.value.status_code == 403


@pytest.mark.asyncio
async def test_shutdown_allows_localhost() -> None:
    request = SimpleNamespace(client=SimpleNamespace(host='127.0.0.1'))
    result = await routes_system.shutdown(request=request)
    assert result.shutdown_initiated is True


@pytest.mark.asyncio
async def test_get_setup_status_returns_ready_when_required_models_cached(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(routes_system, '_is_setup_ready', lambda _payload=None: True)
    monkeypatch.setattr(routes_system.settings, 'app_data_dir', tmp_path)

    status = await routes_system.get_setup_status()
    assert status.state == 'ready'
    assert status.required_models_ready is True
    assert status.recommended_tier in {'small', 'balanced', 'quality'}
    assert status.tier_options


@pytest.mark.asyncio
async def test_get_setup_status_reflects_setup_state_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(routes_system, '_is_setup_ready', lambda _payload=None: False)
    monkeypatch.setattr(routes_system.settings, 'app_data_dir', tmp_path)
    (tmp_path / 'setup_state.json').write_text('{"state":"setup_in_progress"}', encoding='utf-8')

    status = await routes_system.get_setup_status()
    assert status.state == 'setup_in_progress'
    assert status.required_models_ready is False
    assert status.setup_state_file_present is True


@pytest.mark.asyncio
async def test_get_setup_status_ollama_provider_does_not_gate_on_probe_detail(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(routes_system.settings, 'app_data_dir', tmp_path)
    monkeypatch.setattr(routes_system.settings, 'llm_provider', 'ollama')
    monkeypatch.setattr(routes_system, '_is_setup_ready', lambda: False)
    monkeypatch.setattr(routes_system, '_probe_ollama_status', lambda: (True, False, 'Ollama model not found: qwen3:14b'))

    status = await routes_system.get_setup_status()
    assert status.llm_provider == 'ollama'
    assert status.ollama_reachable is True
    assert status.ollama_model_ready is False
    assert status.detail is None


@pytest.mark.asyncio
async def test_get_ollama_status_returns_probe_result(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(routes_system.settings, 'ollama_base_url', 'http://127.0.0.1:11434')
    monkeypatch.setattr(routes_system.settings, 'llm_model_id', 'qwen3:14b')
    monkeypatch.setattr(routes_system, '_probe_ollama_status', lambda: (True, True, None))

    status = await routes_system.get_ollama_status()
    assert status.reachable is True
    assert status.model_ready is True
    assert status.model == 'qwen3:14b'


@pytest.mark.asyncio
async def test_start_setup_ollama_still_requires_valid_tier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(routes_system.settings, 'llm_provider', 'ollama')
    with pytest.raises(HTTPException, match='Unknown setup tier'):
        await routes_system.start_setup(payload=routes_system.SetupStartRequest(tier='invalid', model_filename='invalid'))


@pytest.mark.asyncio
async def test_retry_setup_ollama_uses_classic_workflow(monkeypatch: pytest.MonkeyPatch) -> None:
    class _DummyTask:
        def done(self) -> bool:
            return False

    called = {'scheduled': False}

    async def _fake_setup_workflow(*, tier: str, model_filename: str) -> None:
        _ = (tier, model_filename)

    monkeypatch.setattr(routes_system.settings, 'llm_provider', 'ollama')
    monkeypatch.setattr(routes_system, '_run_setup_workflow', _fake_setup_workflow)
    monkeypatch.setattr(routes_system, '_setup_task', None)
    monkeypatch.setattr(routes_system, '_persist_setup_state_file', lambda: None)
    monkeypatch.setattr(
        routes_system.asyncio,
        'create_task',
        lambda coro: (called.__setitem__('scheduled', True), coro.close(), _DummyTask())[2],
    )
    monkeypatch.setattr(routes_system, '_load_setup_state_file', lambda _path: ({'selected_tier': 'small', 'model_filename': 'x.gguf'}, None))

    response = await routes_system.retry_setup()
    assert response.accepted is True
    assert response.state == 'setup_in_progress'
    assert called['scheduled'] is True


def test_recommend_setup_tier_prefers_small_on_16gb_class_devices() -> None:
    tier, reason = routes_system._recommend_setup_tier(ram_total_gb=16.0, free_disk_gb=200.0)
    assert tier == 'small'
    assert '24 GB' in reason


def test_recommend_setup_tier_uses_balanced_for_24gb_and_above() -> None:
    tier, reason = routes_system._recommend_setup_tier(ram_total_gb=24.0, free_disk_gb=200.0)
    assert tier == 'balanced'
    assert '24 GB' in reason


@pytest.mark.asyncio
async def test_start_setup_persists_in_progress_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(routes_system.settings, 'app_data_dir', tmp_path)

    class _DummyTask:
        def done(self) -> bool:
            return False

        def cancel(self) -> None:
            return

    def _fake_create_task(coro):  # type: ignore[no-untyped-def]
        coro.close()
        return _DummyTask()

    monkeypatch.setattr(routes_system.asyncio, 'create_task', _fake_create_task)
    routes_system._setup_task = None

    payload = SetupStartRequest(tier='balanced', model_filename='Qwen3-14B-Q5_K_M.gguf')

    response = await routes_system.start_setup(payload)
    assert response.accepted is True
    assert response.state == 'setup_in_progress'

    setup_state = (tmp_path / 'setup_state.json').read_text(encoding='utf-8')
    assert '"state": "setup_in_progress"' in setup_state
    assert '"selected_tier": "balanced"' in setup_state

    config_data = json.loads((tmp_path / 'config.json').read_text(encoding='utf-8'))
    assert config_data['llm_model_filename'] == 'Qwen3-14B-Q5_K_M.gguf'
    assert config_data['full_privacy'] is False
    assert config_data['embedding_offline'] is False
    assert config_data['llm_local_only'] is False

    assert routes_system.settings.full_privacy is False
    assert routes_system.settings.embedding_offline is False
    assert routes_system.settings.llm_local_only is False


@pytest.mark.asyncio
async def test_start_setup_rejects_mismatched_model_filename() -> None:
    with pytest.raises(HTTPException) as exc_info:
        await routes_system.start_setup(
            SetupStartRequest(tier='balanced', model_filename='Qwen3.5-9B-Q4_K_M.gguf'),
        )
    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_setup_events_reflect_runtime_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(routes_system.settings, 'app_data_dir', tmp_path)
    monkeypatch.setattr(routes_system.settings, 'models_dir', tmp_path / 'models')
    routes_system._update_setup_runtime(
        state='setup_in_progress',
        stage='downloading_model',
        overall_pct=33,
        artifact='Qwen3-14B-Q5_K_M.gguf',
        paused=False,
        error=None,
    )

    event = await routes_system.get_setup_events()
    assert event.state == 'setup_in_progress'
    assert event.paused is False


@pytest.mark.asyncio
async def test_get_models_catalog_marks_installed_and_default(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    models_dir = tmp_path / 'models'
    models_dir.mkdir(parents=True)
    (models_dir / 'Qwen_Qwen3.5-9B-Q4_K_M.gguf').write_bytes(b'x')
    monkeypatch.setattr(routes_system.settings, 'models_dir', models_dir)
    monkeypatch.setattr(routes_system.settings, 'llm_model_filename', 'Qwen_Qwen3.5-9B-Q4_K_M.gguf')

    catalog = await routes_system.get_models_catalog()
    assert catalog.default_model_filename == 'Qwen_Qwen3.5-9B-Q4_K_M.gguf'
    assert catalog.default_model_id == 'qwen-9b'
    installed = {item.model_filename: item.installed for item in catalog.models}
    assert installed['Qwen_Qwen3.5-9B-Q4_K_M.gguf'] is True
    assert installed['Qwen3-14B-Q5_K_M.gguf'] is False


@pytest.mark.asyncio
async def test_get_models_catalog_marks_quality_installed_for_legacy_35b_alias(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    models_dir = tmp_path / 'models'
    models_dir.mkdir(parents=True)
    legacy_name = 'Qwen3.5-35B-A3B-Q4_K_M.gguf'
    (models_dir / legacy_name).write_bytes(b'x')
    monkeypatch.setattr(routes_system.settings, 'models_dir', models_dir)
    monkeypatch.setattr(routes_system.settings, 'llm_model_filename', legacy_name)
    monkeypatch.setattr(routes_system.settings, 'llm_model_id', '')

    catalog = await routes_system.get_models_catalog()
    quality_entry = next(item for item in catalog.models if item.tier == 'quality')
    assert quality_entry.installed is True
    assert quality_entry.is_default is True
    assert catalog.default_model_id == 'qwen-35b-a3b'


@pytest.mark.asyncio
async def test_download_model_starts_background_operation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    models_dir = tmp_path / 'models'
    models_dir.mkdir(parents=True)
    monkeypatch.setattr(routes_system.settings, 'models_dir', models_dir)

    class _DummyTask:
        def done(self) -> bool:
            return False

        def cancel(self) -> None:
            return

    def _fake_create_task(coro):  # type: ignore[no-untyped-def]
        coro.close()
        return _DummyTask()

    monkeypatch.setattr(routes_system.asyncio, 'create_task', _fake_create_task)
    routes_system._model_task = None
    routes_system._update_model_runtime(state='idle', model_filename=None, paused=False, error=None)

    response = await routes_system.download_model(ModelActionRequest(model_filename='Qwen3-14B-Q5_K_M.gguf'))
    assert response.accepted is True
    assert response.detail == 'Download started'
    assert routes_system._model_runtime['state'] == 'in_progress'
    assert routes_system._model_runtime['model_filename'] == 'Qwen3-14B-Q5_K_M.gguf'


@pytest.mark.asyncio
async def test_set_default_model_requires_installed_file_and_updates_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    app_data_dir = tmp_path / 'app-data'
    models_dir = tmp_path / 'models'
    app_data_dir.mkdir(parents=True)
    models_dir.mkdir(parents=True)
    (models_dir / 'Qwen3-14B-Q5_K_M.gguf').write_bytes(b'x')
    monkeypatch.setattr(routes_system.settings, 'app_data_dir', app_data_dir)
    monkeypatch.setattr(routes_system.settings, 'models_dir', models_dir)

    response = await routes_system.set_default_model(ModelActionRequest(model_filename='Qwen3-14B-Q5_K_M.gguf'))
    assert response.accepted is True
    assert routes_system.settings.llm_model_id == 'qwen-14b'
    assert routes_system.settings.llm_model_filename == 'Qwen3-14B-Q5_K_M.gguf'
    config_text = (app_data_dir / 'config.json').read_text(encoding='utf-8')
    assert '"llm_model_id": "qwen-14b"' in config_text
    assert '"llm_model_filename": "Qwen3-14B-Q5_K_M.gguf"' in config_text


@pytest.mark.asyncio
async def test_model_cancel_reflect_runtime_state(monkeypatch: pytest.MonkeyPatch) -> None:
    class _DummyTask:
        def __init__(self) -> None:
            self._cancelled = False

        def done(self) -> bool:
            return False

        def cancel(self) -> None:
            self._cancelled = True

    dummy_task = _DummyTask()
    routes_system._model_task = dummy_task
    routes_system._update_model_runtime(
        state='in_progress',
        stage='downloading_model',
        model_filename='Qwen3-14B-Q5_K_M.gguf',
        paused=False,
        cancel_requested=False,
    )

    cancelled = await routes_system.cancel_model_download()
    assert cancelled.accepted is True
    assert routes_system._model_runtime['state'] == 'cancelled'

@pytest.mark.asyncio
async def test_get_diagnostics_summary_aggregates_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [
        {
            'type': 'user',
            'query_type': 'focused',
            'timeout_occurred': False,
            'has_empty_answer': False,
            'has_refusal_pattern': False,
            'generation_seconds': 2.5,
            'sources_count': 3,
            'raw_chunks_count': 8,
            'detected_issues': [],
            'created_at': datetime(2026, 2, 24, tzinfo=UTC),
        },
        {
            'type': 'user',
            'query_type': 'coverage',
            'timeout_occurred': True,
            'has_empty_answer': False,
            'has_refusal_pattern': True,
            'generation_seconds': 8.5,
            'sources_count': 5,
            'raw_chunks_count': 14,
            'detected_issues': ['timeout', 'retrieval_failure'],
            'created_at': datetime(2026, 2, 25, tzinfo=UTC),
        },
        {
            'type': 'evaluation',
            'query_type': 'metadata',
            'timeout_occurred': False,
            'has_empty_answer': True,
            'has_refusal_pattern': False,
            'generation_seconds': 1.0,
            'sources_count': 0,
            'raw_chunks_count': 0,
            'detected_issues': ['empty_answer'],
            'created_at': datetime(2026, 2, 23, tzinfo=UTC),
        },
    ]

    @asynccontextmanager
    async def _fake_get_db():
        yield MagicMock()

    monkeypatch.setattr(routes_system, 'get_db', _fake_get_db)
    monkeypatch.setattr(routes_system, 'get_diagnostics_metrics_since', AsyncMock(return_value=rows))

    summary = await routes_system.get_diagnostics_summary(days=7, type_filter=None, run_id_filter=None)

    assert summary.summary_schema == 'informity.diagnostics.summary.v2'
    assert summary.aggregation_mode == 'direct_window_scan'
    assert summary.type_taxonomy == ['user', 'evaluation']
    assert summary.query_type_taxonomy == ['simple', 'metadata', 'focused', 'coverage', 'unknown']
    assert 'timeout' in summary.issue_type_taxonomy
    assert summary.window_days == 7
    assert summary.total_responses == 3
    assert summary.by_type == {'user': 2, 'evaluation': 1}
    assert summary.by_query_type == {'focused': 1, 'coverage': 1, 'metadata': 1}
    assert summary.issue_counts == {'timeout': 1, 'retrieval_failure': 1, 'empty_answer': 1}
    assert summary.timeout_count == 1
    assert summary.empty_answer_count == 1
    assert summary.refusal_pattern_count == 1
    assert summary.timeout_rate == pytest.approx(0.3333, rel=1e-4)
    assert summary.empty_answer_rate == pytest.approx(0.3333, rel=1e-4)
    assert summary.refusal_pattern_rate == pytest.approx(0.3333, rel=1e-4)
    assert summary.avg_generation_seconds == pytest.approx(4.0)
    assert summary.p95_generation_seconds == pytest.approx(8.5)
    assert summary.avg_sources_count == pytest.approx(2.667, rel=1e-3)
    assert summary.avg_raw_chunks_count == pytest.approx(7.333, rel=1e-3)
    assert summary.created_at_oldest == datetime(2026, 2, 23, tzinfo=UTC)
    assert summary.created_at_newest == datetime(2026, 2, 25, tzinfo=UTC)


@pytest.mark.asyncio
async def test_get_diagnostics_summary_handles_empty_dataset(monkeypatch: pytest.MonkeyPatch) -> None:
    @asynccontextmanager
    async def _fake_get_db():
        yield MagicMock()

    monkeypatch.setattr(routes_system, 'get_db', _fake_get_db)
    monkeypatch.setattr(routes_system, 'get_diagnostics_metrics_since', AsyncMock(return_value=[]))

    summary = await routes_system.get_diagnostics_summary(days=30, type_filter='user', run_id_filter='run-x')

    assert summary.window_days == 30
    assert summary.type_filter == 'user'
    assert summary.run_id_filter == 'run-x'
    assert summary.total_responses == 0
    assert summary.by_type == {}
    assert summary.by_query_type == {}
    assert summary.issue_counts == {}
    assert summary.timeout_count == 0
    assert summary.empty_answer_count == 0
    assert summary.refusal_pattern_count == 0
    assert summary.timeout_rate == 0.0
    assert summary.empty_answer_rate == 0.0
    assert summary.refusal_pattern_rate == 0.0
    assert summary.avg_generation_seconds == 0.0
    assert summary.p95_generation_seconds is None
    assert summary.avg_sources_count == 0.0
    assert summary.avg_raw_chunks_count == 0.0
    assert summary.created_at_oldest is None
    assert summary.created_at_newest is None


@pytest.mark.asyncio
async def test_get_diagnostics_summary_normalizes_non_canonical_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [
        {
            'type': 'UNKNOWN_TYPE',
            'query_type': 'NON_STANDARD_QUERY',
            'timeout_occurred': False,
            'has_empty_answer': False,
            'has_refusal_pattern': False,
            'generation_seconds': 1.0,
            'sources_count': 1,
            'raw_chunks_count': 1,
            'detected_issues': ['timeout', 'not_a_real_issue'],
            'created_at': datetime(2026, 2, 25, tzinfo=UTC),
        },
    ]

    @asynccontextmanager
    async def _fake_get_db():
        yield MagicMock()

    monkeypatch.setattr(routes_system, 'get_db', _fake_get_db)
    monkeypatch.setattr(routes_system, 'get_diagnostics_metrics_since', AsyncMock(return_value=rows))

    summary = await routes_system.get_diagnostics_summary(days=7, type_filter=None, run_id_filter=None)

    assert summary.total_responses == 1
    assert summary.by_type == {}
    assert summary.by_query_type == {'unknown': 1}
    assert summary.issue_counts == {'timeout': 1}
