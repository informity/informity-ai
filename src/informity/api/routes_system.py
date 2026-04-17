# ==============================================================================
# Informity AI — System API Routes
# Endpoints for system operations: shutdown, diagnostics
# ==============================================================================

import asyncio
import json
import math
import os
import platform
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import psutil
import structlog
from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel

from informity.api.schemas import (
    DiagnosticsMetricsSummaryResponse,
    DiagnosticsResponse,
    ModelActionRequest,
    ModelActionResponse,
    ModelOperationEventResponse,
    ModelsCatalogItem,
    ModelsCatalogResponse,
    SetupActionResponse,
    SetupEventResponse,
    SetupStartRequest,
    SetupStartResponse,
    SetupStatusResponse,
    SetupTierOption,
)
from informity.api.security import is_loopback_host
from informity.api.setup_state import SetupState
from informity.config import (
    APP_DISPLAY_NAME,
    DirNames,
    are_required_models_cached,
    configure_hf_environment,
    settings,
)
from informity.db.sqlite import (
    CANONICAL_DIAGNOSTICS_QUERY_TYPES,
    CANONICAL_DIAGNOSTICS_TYPES,
    get_chunk_count,
    get_db,
    get_diagnostics_metrics_since,
    get_file_count,
    get_indexed_content_size_bytes,
)
from informity.db.vectors import vector_store
from informity.diagnostics.issue_types import IssueType
from informity.indexer.embedder import embedder
from informity.indexer.reranker import reranker
from informity.llm.engine import llm_engine
from informity.llm.model_adapter import get_model_display_name
from informity.llm.types import DiagnosticsQueryType
from informity.version import APP_VERSION

# ==============================================================================
# Logger
# ==============================================================================

log = structlog.get_logger(__name__)
_SYSTEM_DIAGNOSTICS_EXCEPTIONS = (OSError, RuntimeError, ValueError, TypeError)
_CANONICAL_DIAGNOSTICS_ISSUES = tuple(sorted(issue.value for issue in IssueType))
_SETUP_STATE_FILE = 'setup_state.json'
_SETUP_CONFIG_FILE = 'config.json'
_DECIMAL_GB = 1_000_000_000
_MODEL_SIZE_BYTES = {
    'Qwen_Qwen3.5-9B-Q4_K_M.gguf': 5_889_811_552,
    'Qwen3-14B-Q5_K_M.gguf': 10_514_569_568,
    'Qwen3.6-35B-A3B-Q4_K_M.gguf': 22_134_528_992,
}
_SETUP_TIER_OPTIONS: tuple[SetupTierOption, ...] = (
    SetupTierOption(
        tier='small',
        title='Small',
        display_name=get_model_display_name('Qwen_Qwen3.5-9B-Q4_K_M.gguf'),
        model_filename='Qwen_Qwen3.5-9B-Q4_K_M.gguf',
        model_size_bytes=_MODEL_SIZE_BYTES['Qwen_Qwen3.5-9B-Q4_K_M.gguf'],
        approx_size_gb=round(_MODEL_SIZE_BYTES['Qwen_Qwen3.5-9B-Q4_K_M.gguf'] / _DECIMAL_GB, 2),
        quality='Good',
        speed='Fast',
        ram_profile='Lower RAM',
        description='Fastest setup with lower memory footprint.',
    ),
    SetupTierOption(
        tier='balanced',
        title='Balanced',
        display_name=get_model_display_name('Qwen3-14B-Q5_K_M.gguf'),
        model_filename='Qwen3-14B-Q5_K_M.gguf',
        model_size_bytes=_MODEL_SIZE_BYTES['Qwen3-14B-Q5_K_M.gguf'],
        approx_size_gb=round(_MODEL_SIZE_BYTES['Qwen3-14B-Q5_K_M.gguf'] / _DECIMAL_GB, 2),
        quality='High',
        speed='Balanced',
        ram_profile='Medium RAM',
        description='Recommended quality and speed tradeoff.',
    ),
    SetupTierOption(
        tier='quality',
        title='Quality',
        display_name=get_model_display_name('Qwen3.6-35B-A3B-Q4_K_M.gguf'),
        model_filename='Qwen3.6-35B-A3B-Q4_K_M.gguf',
        model_size_bytes=_MODEL_SIZE_BYTES['Qwen3.6-35B-A3B-Q4_K_M.gguf'],
        approx_size_gb=round(_MODEL_SIZE_BYTES['Qwen3.6-35B-A3B-Q4_K_M.gguf'] / _DECIMAL_GB, 2),
        quality='Highest',
        speed='Slower',
        ram_profile='Higher RAM',
        description='Best answer quality with higher resource usage.',
    ),
)
_SETUP_TIER_REPOS: dict[str, str] = {
    'small': 'bartowski/Qwen_Qwen3.5-9B-GGUF',
    'balanced': 'Qwen/Qwen3-14B-GGUF',
    'quality': 'unsloth/Qwen3.6-35B-A3B-GGUF',
}
_SETUP_TIER_REVISIONS: dict[str, str] = {
    'small': 'ff13963796ee209598509a81340172bb1c3869fe',
    'balanced': '530227a7d994db8eca5ab5ced2fb692b614357fd',
}
_SETUP_MODEL_SHA256: dict[str, str] = {
    'Qwen_Qwen3.5-9B-Q4_K_M.gguf': '9437f5bf0dd0c97800caaf902f41e6a6aa00223ab232f159eda41dcbbb492645',
    'Qwen3-14B-Q5_K_M.gguf': 'e7c9aba1129ca2936be9eca01419d9f86af40e08caa01230d5574b34d08e3e31',
    'Qwen3.6-35B-A3B-Q4_K_M.gguf': 'ac0e2c1189e055faa36eff361580e79c5bd6f8e76bffb4ce547f167d53e31a61',
}
_setup_runtime: dict[str, object] = {
    'state': SetupState.REQUIRED.value,
    'stage': 'idle',
    'overall_pct': 0,
    'artifact': None,
    'artifact_pct': 0,
    'bytes_done': 0,
    'bytes_total': 0,
    'speed_bps': 0.0,
    'eta_sec': None,
    'paused': False,
    'error': None,
    'selected_tier': None,
    'model_filename': None,
    'cancel_requested': False,
    'updated_at': None,
}
_setup_task: asyncio.Task[None] | None = None
_setup_lock = asyncio.Lock()
_setup_download_cancel_event: threading.Event | None = None
_MODEL_STATE_IDLE = 'idle'
_MODEL_STATE_IN_PROGRESS = 'in_progress'
_MODEL_STATE_FAILED = 'failed'
_MODEL_STATE_COMPLETED = 'completed'
_MODEL_STATE_CANCELLED = 'cancelled'
_model_runtime: dict[str, object] = {
    'state': _MODEL_STATE_IDLE,
    'stage': 'idle',
    'model_filename': None,
    'overall_pct': 0,
    'bytes_done': 0,
    'bytes_total': 0,
    'speed_bps': 0.0,
    'eta_sec': None,
    'paused': False,
    'error': None,
    'cancel_requested': False,
    'updated_at': None,
}
_model_task: asyncio.Task[None] | None = None
_model_lock = asyncio.Lock()
_model_download_cancel_event: threading.Event | None = None

# ==============================================================================
# Router
# ==============================================================================

router = APIRouter(prefix='/api', tags=['system'])

# ==============================================================================
# Schemas
# ==============================================================================


class ShutdownResponse(BaseModel):
    """Shutdown confirmation."""
    message: str
    shutdown_initiated: bool = True


def _load_setup_state_file(path: Path) -> tuple[dict[str, object] | None, str | None]:
    if not path.exists():
        return None, None
    try:
        raw = path.read_text(encoding='utf-8')
        payload = json.loads(raw)
        if isinstance(payload, dict):
            return payload, None
        return None, 'setup_state_invalid_format'
    except (OSError, ValueError, TypeError):
        return None, 'setup_state_unreadable'


def _recommend_setup_tier(*, ram_total_gb: float, free_disk_gb: float) -> tuple[str, str]:
    if free_disk_gb < 14.0:
        return 'small', 'Low free disk detected; smaller model is safer for setup.'
    if ram_total_gb >= 32.0:
        return 'quality', 'Detected >=32 GB RAM; quality tier fits this device best.'
    if ram_total_gb >= 24.0:
        return 'balanced', 'Detected >=24 GB RAM; balanced tier is recommended.'
    return 'small', 'Detected <24 GB RAM; small tier is recommended for reliability.'


def _setup_state_path() -> Path:
    return settings.app_data_dir / _SETUP_STATE_FILE


def _setup_config_path() -> Path:
    return settings.app_data_dir / _SETUP_CONFIG_FILE


def _required_model_filename(setup_state_payload: dict[str, object] | None = None) -> str:
    selected = str((setup_state_payload or {}).get('model_filename') or '').strip()
    if selected:
        return selected
    return str(settings.llm_model_filename).strip()


def _is_model_file_ready(model_filename: str) -> bool:
    model_path = settings.models_dir / model_filename
    return model_path.exists() and model_path.is_file()


def _is_setup_ready() -> bool:
    # Setup is only complete when all required runtime assets are cached.
    # This includes the selected GGUF, embedding model, reranker, and docling artifacts.
    return are_required_models_cached()


def _update_setup_runtime(**updates: object) -> None:
    _setup_runtime.update(updates)
    _setup_runtime['updated_at'] = datetime.now(UTC).isoformat()


def _update_model_runtime(**updates: object) -> None:
    _model_runtime.update(updates)
    _model_runtime['updated_at'] = datetime.now(UTC).isoformat()


def _runtime_event_snapshot() -> SetupEventResponse:
    state = SetupState(str(_setup_runtime.get('state') or SetupState.REQUIRED.value))
    return SetupEventResponse(
        state=state,
        stage=str(_setup_runtime.get('stage') or 'idle'),
        overall_pct=int(_setup_runtime.get('overall_pct') or 0),
        artifact=str(_setup_runtime.get('artifact')) if _setup_runtime.get('artifact') else None,
        artifact_pct=int(_setup_runtime.get('artifact_pct') or 0),
        bytes_done=int(_setup_runtime.get('bytes_done') or 0),
        bytes_total=int(_setup_runtime.get('bytes_total') or 0),
        speed_bps=float(_setup_runtime.get('speed_bps') or 0.0),
        eta_sec=int(_setup_runtime.get('eta_sec')) if _setup_runtime.get('eta_sec') is not None else None,
        paused=bool(_setup_runtime.get('paused')),
        error=str(_setup_runtime.get('error')) if _setup_runtime.get('error') else None,
    )


def _model_event_snapshot() -> ModelOperationEventResponse:
    return ModelOperationEventResponse(
        state=str(_model_runtime.get('state') or 'idle'),
        stage=str(_model_runtime.get('stage') or 'idle'),
        model_filename=str(_model_runtime.get('model_filename')) if _model_runtime.get('model_filename') else None,
        overall_pct=int(_model_runtime.get('overall_pct') or 0),
        bytes_done=int(_model_runtime.get('bytes_done') or 0),
        bytes_total=int(_model_runtime.get('bytes_total') or 0),
        speed_bps=float(_model_runtime.get('speed_bps') or 0.0),
        eta_sec=int(_model_runtime.get('eta_sec')) if _model_runtime.get('eta_sec') is not None else None,
        paused=bool(_model_runtime.get('paused')),
        error=str(_model_runtime.get('error')) if _model_runtime.get('error') else None,
    )


def _persist_setup_state_file() -> None:
    path = _setup_state_path()
    payload = {
        'state': _setup_runtime.get('state'),
        'stage': _setup_runtime.get('stage'),
        'overall_pct': _setup_runtime.get('overall_pct'),
        'artifact': _setup_runtime.get('artifact'),
        'artifact_pct': _setup_runtime.get('artifact_pct'),
        'bytes_done': _setup_runtime.get('bytes_done'),
        'bytes_total': _setup_runtime.get('bytes_total'),
        'speed_bps': _setup_runtime.get('speed_bps'),
        'eta_sec': _setup_runtime.get('eta_sec'),
        'paused': _setup_runtime.get('paused'),
        'cancel_requested': _setup_runtime.get('cancel_requested'),
        'error': _setup_runtime.get('error'),
        'selected_tier': _setup_runtime.get('selected_tier'),
        'model_filename': _setup_runtime.get('model_filename'),
        'updated_at': _setup_runtime.get('updated_at'),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding='utf-8')


def _clear_setup_state_file() -> None:
    path = _setup_state_path()
    path.unlink(missing_ok=True)


def _cleanup_setup_artifacts(model_filename: str | None) -> None:
    if not model_filename:
        return
    model_name = str(model_filename).strip()
    if not model_name:
        return

    # Do not delete the final GGUF file here. Users may pre-seed models manually,
    # and download flow writes to temp artifacts before atomic replace.
    # Cleanup is intentionally limited to partial/lock artifacts.
    patterns = (
        f'{model_name}.part*',
        f'{model_name}.tmp*',
        f'{model_name}.incomplete*',
        f'*{model_name}*.incomplete*',
        f'*{model_name}*.lock',
    )

    search_roots = [
        settings.models_dir,
        settings.models_dir / '.cache' / 'huggingface' / 'download',
    ]

    for root in search_roots:
        if not root.exists():
            continue
        for pattern in patterns:
            for candidate in root.glob(pattern):
                try:
                    candidate.unlink(missing_ok=True)
                except OSError:
                    continue


def _cleanup_model_artifacts(model_filename: str | None) -> None:
    _cleanup_setup_artifacts(model_filename)


def _is_cancelled_download_error(exc: Exception) -> bool:
    message = str(exc).strip().lower()
    return 'download cancelled' in message or 'cancelled' in message


def _eta_seconds(*, bytes_done: int, bytes_total: int | None, speed_bps: float) -> int | None:
    if bytes_total is None or bytes_total <= 0:
        return None
    if speed_bps <= 0:
        return None
    remaining = max(bytes_total - bytes_done, 0)
    return int(math.ceil(remaining / speed_bps))


def _apply_setup_completion_config(model_filename: str) -> None:
    config_path = _setup_config_path()
    config_data: dict[str, object] = {}
    if config_path.exists():
        try:
            parsed = json.loads(config_path.read_text(encoding='utf-8'))
            if isinstance(parsed, dict):
                config_data = parsed
        except (OSError, ValueError, TypeError):
            config_data = {}
    config_data['llm_model_filename'] = model_filename
    config_data['full_privacy'] = True
    config_data['llm_local_only'] = True
    config_data['embedding_offline'] = True
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config_data, indent=2), encoding='utf-8')


def _apply_setup_bootstrap_config(model_filename: str) -> None:
    """
    Persist first-run bootstrap mode while setup is still in progress.

    Privacy flags stay off during setup so required dependency models can be
    downloaded and cached before switching to full privacy mode.
    """
    config_path = _setup_config_path()
    config_data: dict[str, object] = {}
    if config_path.exists():
        try:
            parsed = json.loads(config_path.read_text(encoding='utf-8'))
            if isinstance(parsed, dict):
                config_data = parsed
        except (OSError, ValueError, TypeError):
            config_data = {}
    config_data['llm_model_filename'] = model_filename
    config_data['full_privacy'] = False
    config_data['llm_local_only'] = False
    config_data['embedding_offline'] = False
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config_data, indent=2), encoding='utf-8')


def _cache_required_runtime_dependencies() -> None:
    """
    Warm/cache non-LLM runtime dependencies needed for full privacy mode.

    Runs during setup before privacy mode is switched on.
    """
    configure_hf_environment(fail_on_missing_full_privacy_models=False)

    # Embedding model cache
    embedder.embed_query('setup_warmup')

    # Reranker model cache
    reranker.rerank(
        'setup warmup',
        [{'chunk_text': 'setup warmup placeholder'}],
    )

    # Docling runtime artifacts cache
    docling_cache = settings.cache_dir / DirNames.DOCLING
    docling_cache.mkdir(parents=True, exist_ok=True)
    os.environ['DOCLING_ARTIFACTS_PATH'] = str(docling_cache)

    try:
        from docling.utils.model_downloader import download_models
    except ImportError as exc:
        raise RuntimeError('Docling model downloader is unavailable in packaged runtime') from exc

    download_models(
        output_dir=docling_cache,
        force=False,
        progress=False,
        with_layout=True,
        with_tableformer=True,
        with_code_formula=True,
        with_picture_classifier=False,
        with_smolvlm=False,
        with_granitedocling=False,
        with_granitedocling_mlx=False,
        with_smoldocling=False,
        with_smoldocling_mlx=False,
        with_granite_vision=False,
        with_granite_chart_extraction=False,
        with_rapidocr=True,
        with_easyocr=False,
    )


def _apply_model_default_config(model_filename: str) -> None:
    config_path = _setup_config_path()
    config_data: dict[str, object] = {}
    if config_path.exists():
        try:
            parsed = json.loads(config_path.read_text(encoding='utf-8'))
            if isinstance(parsed, dict):
                config_data = parsed
        except (OSError, ValueError, TypeError):
            config_data = {}
    config_data['llm_model_filename'] = model_filename
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config_data, indent=2), encoding='utf-8')


async def _run_setup_workflow(*, tier: str, model_filename: str) -> None:
    global _setup_task, _setup_download_cancel_event
    target_path = settings.models_dir / model_filename
    repo_id = _SETUP_TIER_REPOS.get(tier)
    revision = _SETUP_TIER_REVISIONS.get(tier)
    expected_sha256 = _SETUP_MODEL_SHA256.get(model_filename)
    try:
        _update_setup_runtime(
            state=SetupState.IN_PROGRESS.value,
            stage='preparing',
            overall_pct=5,
            artifact=model_filename,
            artifact_pct=0,
            bytes_done=0,
            bytes_total=0,
            speed_bps=0.0,
            eta_sec=None,
            paused=False,
            error=None,
            selected_tier=tier,
            model_filename=model_filename,
            cancel_requested=False,
        )
        _persist_setup_state_file()
        if bool(_setup_runtime.get('cancel_requested')):
            _cleanup_setup_artifacts(model_filename)
            _update_setup_runtime(
                state=SetupState.REQUIRED.value,
                stage='cancelled',
                overall_pct=0,
                artifact=None,
                paused=False,
                cancel_requested=False,
                error=None,
            )
            _persist_setup_state_file()
            return
        _update_setup_runtime(stage='downloading_model', overall_pct=20, artifact_pct=0)
        _persist_setup_state_file()
        cancel_event = threading.Event()
        _setup_download_cancel_event = cancel_event

        def _on_progress(bytes_done: int, bytes_total: int | None, speed_bps: float) -> None:
            artifact_pct = int((bytes_done / bytes_total) * 100) if bytes_total and bytes_total > 0 else 0
            overall_pct = min(84, 20 + int(artifact_pct * 0.64))
            _update_setup_runtime(
                stage='downloading_model',
                overall_pct=overall_pct,
                artifact_pct=artifact_pct,
                bytes_done=int(bytes_done),
                bytes_total=int(bytes_total or 0),
                speed_bps=float(speed_bps),
                eta_sec=_eta_seconds(bytes_done=int(bytes_done), bytes_total=bytes_total, speed_bps=float(speed_bps)),
            )

        await asyncio.to_thread(
            llm_engine._download_model,
            target_path,
            repo_id,
            model_filename,
            revision,
            expected_sha256,
            _on_progress,
            cancel_event,
        )
        if bool(_setup_runtime.get('cancel_requested')):
            _cleanup_setup_artifacts(model_filename)
            _update_setup_runtime(
                state=SetupState.REQUIRED.value,
                stage='cancelled',
                overall_pct=0,
                artifact=None,
                paused=False,
                cancel_requested=False,
                error=None,
            )
            _persist_setup_state_file()
            return
        after_size = target_path.stat().st_size if target_path.exists() else 0
        _update_setup_runtime(
            stage='downloaded',
            overall_pct=85,
            artifact_pct=100,
            bytes_done=int(after_size),
            bytes_total=int(after_size),
            speed_bps=float(_setup_runtime.get('speed_bps') or 0.0),
            eta_sec=0,
        )
        _persist_setup_state_file()
        _update_setup_runtime(stage='caching_dependencies', overall_pct=92)
        _persist_setup_state_file()
        await asyncio.to_thread(_cache_required_runtime_dependencies)
        if not are_required_models_cached():
            raise RuntimeError('Required runtime dependency cache warmup did not complete')
        _update_setup_runtime(stage='finalizing', overall_pct=97)
        _persist_setup_state_file()
        _apply_setup_completion_config(model_filename)
        settings.llm_model_filename = model_filename
        settings.full_privacy = True
        settings.llm_local_only = True
        settings.embedding_offline = True
        _update_setup_runtime(
            state=SetupState.READY.value,
            stage='completed',
            overall_pct=100,
            paused=False,
            error=None,
        )
        _clear_setup_state_file()
    except asyncio.CancelledError:
        _cleanup_setup_artifacts(model_filename)
        _update_setup_runtime(
            state=SetupState.REQUIRED.value,
            stage='cancelled',
            overall_pct=0,
            artifact=None,
            paused=False,
            cancel_requested=False,
            error=None,
        )
        _persist_setup_state_file()
        raise
    except Exception as exc:  # noqa: BLE001
        if _is_cancelled_download_error(exc):
            _cleanup_setup_artifacts(model_filename)
            _update_setup_runtime(
                state=SetupState.REQUIRED.value,
                stage='cancelled',
                overall_pct=0,
                artifact=None,
                artifact_pct=0,
                bytes_done=0,
                bytes_total=0,
                speed_bps=0.0,
                eta_sec=None,
                paused=False,
                cancel_requested=False,
                error=None,
            )
            _persist_setup_state_file()
            return
        log.error(
            'setup_workflow_failed',
            error=str(exc),
            exc_info=True,
            stage=_setup_runtime.get('stage'),
            model_filename=model_filename,
            selected_tier=tier,
        )
        _update_setup_runtime(
            state=SetupState.FAILED.value,
            stage='failed',
            paused=False,
            cancel_requested=False,
            error=str(exc),
        )
        _persist_setup_state_file()
    finally:
        _setup_download_cancel_event = None
        async with _setup_lock:
            _setup_task = None


def _resolve_tier_for_model(model_filename: str) -> tuple[str, str, str | None, str | None]:
    for option in _SETUP_TIER_OPTIONS:
        if option.model_filename == model_filename:
            tier = option.tier
            return (
                tier,
                _SETUP_TIER_REPOS[tier],
                _SETUP_TIER_REVISIONS.get(tier),
                _SETUP_MODEL_SHA256.get(model_filename),
            )
    raise HTTPException(status_code=400, detail='Unknown model filename')


async def _run_model_download_workflow(
    *,
    model_filename: str,
    repo_id: str,
    revision: str | None,
    expected_sha256: str | None,
) -> None:
    global _model_task, _model_download_cancel_event
    target_path = settings.models_dir / model_filename
    try:
        _update_model_runtime(
            state=_MODEL_STATE_IN_PROGRESS,
            stage='preparing',
            model_filename=model_filename,
            overall_pct=5,
            bytes_done=0,
            bytes_total=0,
            speed_bps=0.0,
            eta_sec=None,
            paused=False,
            error=None,
            cancel_requested=False,
        )
        if bool(_model_runtime.get('cancel_requested')):
            _update_model_runtime(state=_MODEL_STATE_CANCELLED, stage='cancelled', paused=False)
            return
        _update_model_runtime(stage='downloading_model', overall_pct=20)
        cancel_event = threading.Event()
        _model_download_cancel_event = cancel_event

        def _on_progress(bytes_done: int, bytes_total: int | None, speed_bps: float) -> None:
            pct = int((bytes_done / bytes_total) * 100) if bytes_total and bytes_total > 0 else 0
            _update_model_runtime(
                state=_MODEL_STATE_IN_PROGRESS,
                stage='downloading_model',
                overall_pct=pct,
                bytes_done=int(bytes_done),
                bytes_total=int(bytes_total or 0),
                speed_bps=float(speed_bps),
                eta_sec=_eta_seconds(bytes_done=int(bytes_done), bytes_total=bytes_total, speed_bps=float(speed_bps)),
                paused=False,
                error=None,
            )

        await asyncio.to_thread(
            llm_engine._download_model,
            target_path,
            repo_id,
            model_filename,
            revision,
            expected_sha256,
            _on_progress,
            cancel_event,
        )
        after_size = target_path.stat().st_size if target_path.exists() else 0
        if bool(_model_runtime.get('cancel_requested')):
            _cleanup_model_artifacts(model_filename)
            _update_model_runtime(
                state=_MODEL_STATE_CANCELLED,
                stage='cancelled',
                overall_pct=0,
                bytes_done=0,
                bytes_total=0,
                speed_bps=0.0,
                eta_sec=None,
                paused=False,
                error=None,
            )
            return

        _update_model_runtime(
            state=_MODEL_STATE_COMPLETED,
            stage='completed',
            overall_pct=100,
            bytes_done=int(after_size),
            bytes_total=int(after_size),
            speed_bps=float(_model_runtime.get('speed_bps') or 0.0),
            eta_sec=0,
            paused=False,
            error=None,
        )
    except Exception as exc:  # noqa: BLE001
        if _is_cancelled_download_error(exc):
            _cleanup_model_artifacts(model_filename)
            _update_model_runtime(
                state=_MODEL_STATE_CANCELLED,
                stage='cancelled',
                overall_pct=0,
                bytes_done=0,
                bytes_total=0,
                speed_bps=0.0,
                eta_sec=None,
                paused=False,
                error=None,
                cancel_requested=False,
            )
            return
        _update_model_runtime(
            state=_MODEL_STATE_FAILED,
            stage='failed',
            paused=False,
            error=str(exc),
        )
    finally:
        _model_download_cancel_event = None
        async with _model_lock:
            _model_task = None


# ==============================================================================
# Endpoints
# ==============================================================================


@router.get('/setup/status', response_model=SetupStatusResponse)
async def get_setup_status() -> SetupStatusResponse:
    """
    Return startup setup/readiness status for desktop route gating.
    """
    setup_state_path = _setup_state_path()
    setup_state_payload, read_error = _load_setup_state_file(setup_state_path)
    required_models_ready = _is_setup_ready()
    vm = psutil.virtual_memory()
    disk = psutil.disk_usage(settings.app_data_dir)
    machine_ram_gb = int(round(float(vm.total / (1024 ** 3))))
    recommended_tier, recommended_reason = _recommend_setup_tier(
        ram_total_gb=float(vm.total / (1024 ** 3)),
        free_disk_gb=float(disk.free / (1024 ** 3)),
    )
    tier_options = list(_SETUP_TIER_OPTIONS)

    if required_models_ready:
        return SetupStatusResponse(
            state=SetupState.READY,
            required_models_ready=True,
            setup_state_file_present=False,
            detail=None,
            machine_ram_gb=machine_ram_gb,
            recommended_tier=recommended_tier,
            recommended_reason=recommended_reason,
            tier_options=tier_options,
        )

    setup_state_file_present = setup_state_path.exists()
    if read_error:
        return SetupStatusResponse(
            state=SetupState.FAILED,
            required_models_ready=False,
            setup_state_file_present=setup_state_file_present,
            detail=read_error,
            machine_ram_gb=machine_ram_gb,
            recommended_tier=recommended_tier,
            recommended_reason=recommended_reason,
            tier_options=tier_options,
        )

    runtime_state = str(_setup_runtime.get('state') or '').strip().lower()
    persisted_state = str((setup_state_payload or {}).get('state') or '').strip().lower()
    if runtime_state in {SetupState.IN_PROGRESS.value, SetupState.FAILED.value}:
        state_value = runtime_state
    else:
        state_value = persisted_state
    if state_value == SetupState.IN_PROGRESS.value:
        state = SetupState.IN_PROGRESS
    elif state_value == SetupState.FAILED.value:
        state = SetupState.FAILED
    else:
        state = SetupState.REQUIRED
    detail = None
    if _setup_runtime.get('error'):
        detail = str(_setup_runtime.get('error'))
    return SetupStatusResponse(
        state=state,
        required_models_ready=False,
        setup_state_file_present=setup_state_file_present,
        detail=detail,
        machine_ram_gb=machine_ram_gb,
        recommended_tier=recommended_tier,
        recommended_reason=recommended_reason,
        tier_options=tier_options,
    )


@router.post('/setup/start', response_model=SetupStartResponse)
async def start_setup(payload: SetupStartRequest) -> SetupStartResponse:
    global _setup_task
    valid_tiers = {opt.tier: opt for opt in _SETUP_TIER_OPTIONS}
    selected_tier = str(payload.tier or '').strip().lower()
    selected_model = str(payload.model_filename or '').strip()
    option = valid_tiers.get(selected_tier)
    if option is None:
        raise HTTPException(status_code=400, detail='Unknown setup tier')
    if selected_model != option.model_filename:
        raise HTTPException(status_code=400, detail='model_filename does not match selected tier')

    async with _setup_lock:
        if _setup_task is not None and not _setup_task.done():
            return SetupStartResponse(accepted=True, state=SetupState.IN_PROGRESS)
        _apply_setup_bootstrap_config(selected_model)
        settings.llm_model_filename = selected_model
        settings.full_privacy = False
        settings.llm_local_only = False
        settings.embedding_offline = False
        _update_setup_runtime(
            state=SetupState.IN_PROGRESS.value,
            stage='queued',
            overall_pct=0,
            selected_tier=selected_tier,
            model_filename=selected_model,
            cancel_requested=False,
            paused=False,
            error=None,
        )
        _persist_setup_state_file()
        _setup_task = asyncio.create_task(_run_setup_workflow(tier=selected_tier, model_filename=selected_model))

    return SetupStartResponse(accepted=True, state=SetupState.IN_PROGRESS)


@router.post('/setup/retry', response_model=SetupActionResponse)
async def retry_setup() -> SetupActionResponse:
    global _setup_task
    _update_setup_runtime(
        state=SetupState.REQUIRED.value,
        error=None,
        paused=False,
        cancel_requested=False,
    )
    state_payload, _ = _load_setup_state_file(_setup_state_path())
    selected_tier = str((state_payload or {}).get('selected_tier') or _setup_runtime.get('selected_tier') or '').strip().lower()
    model_filename = str((state_payload or {}).get('model_filename') or _setup_runtime.get('model_filename') or '').strip()
    if not selected_tier or not model_filename:
        return SetupActionResponse(accepted=False, state=SetupState.REQUIRED, detail='No setup session to retry')

    async with _setup_lock:
        if _setup_task is not None and not _setup_task.done():
            return SetupActionResponse(accepted=True, state=SetupState.IN_PROGRESS, detail='Setup already in progress')
        _apply_setup_bootstrap_config(model_filename)
        settings.llm_model_filename = model_filename
        settings.full_privacy = False
        settings.llm_local_only = False
        settings.embedding_offline = False
        _update_setup_runtime(
            state=SetupState.IN_PROGRESS.value,
            paused=False,
            stage='queued',
            error=None,
            selected_tier=selected_tier,
            model_filename=model_filename,
        )
        _persist_setup_state_file()
        _setup_task = asyncio.create_task(_run_setup_workflow(tier=selected_tier, model_filename=model_filename))

    return SetupActionResponse(accepted=True, state=SetupState.IN_PROGRESS, detail='Retry started')


@router.post('/setup/cancel', response_model=SetupActionResponse)
async def cancel_setup() -> SetupActionResponse:
    global _setup_task
    model_filename = str(_setup_runtime.get('model_filename') or '').strip()
    if not model_filename:
        state_payload, _ = _load_setup_state_file(_setup_state_path())
        model_filename = str((state_payload or {}).get('model_filename') or '').strip()
    _update_setup_runtime(
        state=SetupState.REQUIRED.value,
        stage='cancelled',
        overall_pct=0,
        artifact=None,
        artifact_pct=0,
        bytes_done=0,
        bytes_total=0,
        speed_bps=0.0,
        eta_sec=None,
        paused=False,
        cancel_requested=True,
        error=None,
    )
    _persist_setup_state_file()
    if _setup_download_cancel_event is not None:
        _setup_download_cancel_event.set()
    _cleanup_setup_artifacts(model_filename)
    async with _setup_lock:
        if _setup_task is not None and not _setup_task.done():
            _setup_task.cancel()
    _update_setup_runtime(cancel_requested=False, stage='idle')
    _persist_setup_state_file()
    return SetupActionResponse(accepted=True, state=SetupState.REQUIRED, detail='Setup cancelled')


@router.get('/setup/events', response_model=SetupEventResponse)
async def get_setup_events() -> SetupEventResponse:
    return _runtime_event_snapshot()


@router.get('/models', response_model=ModelsCatalogResponse)
async def get_models_catalog() -> ModelsCatalogResponse:
    default_model = str(settings.llm_model_filename).strip()
    models: list[ModelsCatalogItem] = []
    for option in _SETUP_TIER_OPTIONS:
        models.append(
            ModelsCatalogItem(
                tier=option.tier,
                title=option.title,
                display_name=option.display_name,
                model_filename=option.model_filename,
                model_size_bytes=option.model_size_bytes,
                approx_size_gb=option.approx_size_gb,
                quality=option.quality,
                speed=option.speed,
                ram_profile=option.ram_profile,
                description=option.description,
                installed=_is_model_file_ready(option.model_filename),
                is_default=default_model == option.model_filename,
            ),
        )
    return ModelsCatalogResponse(
        default_model_filename=default_model,
        models=models,
    )


@router.post('/models/download', response_model=ModelActionResponse)
async def download_model(payload: ModelActionRequest) -> ModelActionResponse:
    global _model_task
    model_filename = str(payload.model_filename or '').strip()
    _, repo_id, revision, expected_sha256 = _resolve_tier_for_model(model_filename)
    if _is_model_file_ready(model_filename):
        return ModelActionResponse(accepted=False, detail='Model is already installed')

    async with _model_lock:
        active_filename = str(_model_runtime.get('model_filename') or '').strip()
        state = str(_model_runtime.get('state') or '')
        if _model_task is not None and not _model_task.done():
            if active_filename == model_filename and state == _MODEL_STATE_IN_PROGRESS:
                return ModelActionResponse(accepted=True, detail='Download already in progress')
            return ModelActionResponse(accepted=False, detail='Another model operation is already in progress')
        _update_model_runtime(
            state=_MODEL_STATE_IN_PROGRESS,
            stage='queued',
            model_filename=model_filename,
            overall_pct=0,
            paused=False,
            error=None,
            cancel_requested=False,
        )
        _model_task = asyncio.create_task(
            _run_model_download_workflow(
                model_filename=model_filename,
                repo_id=repo_id,
                revision=revision,
                expected_sha256=expected_sha256,
            )
        )
    return ModelActionResponse(accepted=True, detail='Download started')


@router.post('/models/cancel', response_model=ModelActionResponse)
async def cancel_model_download() -> ModelActionResponse:
    global _model_task
    model_filename = str(_model_runtime.get('model_filename') or '').strip()
    if not model_filename:
        return ModelActionResponse(accepted=False, detail='No model operation found')
    _update_model_runtime(
        state=_MODEL_STATE_CANCELLED,
        stage='cancelled',
        paused=False,
        cancel_requested=True,
        overall_pct=0,
        bytes_done=0,
        bytes_total=0,
        speed_bps=0.0,
        eta_sec=None,
        error=None,
    )
    if _model_download_cancel_event is not None:
        _model_download_cancel_event.set()
    _cleanup_model_artifacts(model_filename)
    async with _model_lock:
        if _model_task is not None and not _model_task.done():
            _model_task.cancel()
    _update_model_runtime(cancel_requested=False)
    return ModelActionResponse(accepted=True, detail='Cancelled')


@router.post('/models/set-default', response_model=ModelActionResponse)
async def set_default_model(payload: ModelActionRequest) -> ModelActionResponse:
    model_filename = str(payload.model_filename or '').strip()
    if not model_filename:
        raise HTTPException(status_code=400, detail='model_filename is required')
    if not model_filename.endswith('.gguf'):
        raise HTTPException(status_code=400, detail='model_filename must be a .gguf file')
    if not _is_model_file_ready(model_filename):
        raise HTTPException(status_code=400, detail='model_filename is not installed')

    _apply_model_default_config(model_filename)
    settings.llm_model_filename = model_filename
    return ModelActionResponse(accepted=True, detail='Default model updated')


@router.get('/models/events', response_model=ModelOperationEventResponse)
async def get_model_events() -> ModelOperationEventResponse:
    return _model_event_snapshot()


@router.get('/diagnostics', response_model=DiagnosticsResponse)
async def get_diagnostics(request: Request) -> DiagnosticsResponse:
    """
    Returns system diagnostics: app version, Python version, OS, RAM, disk space,
    model info, DB stats, uptime. Useful for debugging issues in packaged builds.
    """
    client_host = request.client.host if request.client else None
    if not is_loopback_host(client_host):
        raise HTTPException(
            status_code=403,
            detail='Diagnostics endpoint is only accessible from localhost',
        )

    # Get Python and platform info
    python_version = platform.python_version()
    platform_name = platform.system()
    platform_version = platform.version()
    architecture = platform.machine()

    # Get RAM info
    ram = psutil.virtual_memory()
    ram_total_gb = ram.total / (1024 ** 3)
    ram_available_gb = ram.available / (1024 ** 3)
    ram_used_gb = ram.used / (1024 ** 3)

    # Get disk info (for app data directory)
    disk = psutil.disk_usage(settings.app_data_dir)
    disk_total_gb = disk.total / (1024 ** 3)
    disk_available_gb = disk.free / (1024 ** 3)
    disk_used_gb = disk.used / (1024 ** 3)

    # Get model info
    model_loaded = llm_engine.is_loaded
    model_filename = None
    model_size_gb = None
    if model_loaded:
        try:
            model_path = llm_engine._get_model_path()
            if model_path.exists():
                model_filename = model_path.name
                model_size_gb = model_path.stat().st_size / (1024 ** 3)
        except _SYSTEM_DIAGNOSTICS_EXCEPTIONS:
            pass

    # Get DB info
    db_path = str(settings.db_path)
    db_size_bytes = 0
    if settings.db_path and settings.db_path.exists():
        db_size_bytes = settings.db_path.stat().st_size
    db_size_mb = db_size_bytes / (1024 ** 2)

    # Get vectors info
    vectors_size_bytes = 0
    vectors_size_mb = 0.0
    try:
        stats = await asyncio.to_thread(vector_store.get_stats)
        vectors_size_bytes = stats.get('storage_bytes', 0)
        vectors_size_mb = vectors_size_bytes / (1024 ** 2)
    except _SYSTEM_DIAGNOSTICS_EXCEPTIONS:
        pass

    # Get index stats
    async with get_db() as db:
        total_files = await get_file_count(db)
        total_chunks = await get_chunk_count(db)
        indexed_content_size_bytes = await get_indexed_content_size_bytes(db)

    indexed_content_size_mb = indexed_content_size_bytes / (1024 ** 2)

    # Calculate uptime (if app started timestamp available)
    # For now, we don't track this, so return None
    uptime_seconds = None

    return DiagnosticsResponse(
        app_version=APP_VERSION,
        app_display_name=APP_DISPLAY_NAME,
        python_version=python_version,
        platform=platform_name,
        platform_version=platform_version,
        architecture=architecture,
        ram_total_gb=round(ram_total_gb, 2),
        ram_available_gb=round(ram_available_gb, 2),
        ram_used_gb=round(ram_used_gb, 2),
        disk_total_gb=round(disk_total_gb, 2),
        disk_available_gb=round(disk_available_gb, 2),
        disk_used_gb=round(disk_used_gb, 2),
        model_loaded=model_loaded,
        model_filename=model_filename,
        model_size_gb=round(model_size_gb, 2) if model_size_gb else None,
        db_path=db_path,
        db_size_bytes=db_size_bytes,
        db_size_mb=round(db_size_mb, 2),
        vectors_size_bytes=vectors_size_bytes,
        vectors_size_mb=round(vectors_size_mb, 2),
        total_files=total_files,
        total_chunks=total_chunks,
        indexed_content_size_bytes=indexed_content_size_bytes,
        indexed_content_size_mb=round(indexed_content_size_mb, 2),
        uptime_seconds=uptime_seconds,
    )


@router.post('/shutdown', response_model=ShutdownResponse)
async def shutdown(request: Request) -> ShutdownResponse:
    """
    Gracefully shuts down the application. Only callable from localhost.
    Tauri will call this before killing the sidecar process.
    """
    # Security: only allow shutdown from localhost
    client_host = request.client.host if request.client else None
    if not is_loopback_host(client_host):
        raise HTTPException(
            status_code=403,
            detail='Shutdown endpoint is only accessible from localhost',
        )

    log.info('shutdown_requested', client_host=client_host)

    # Note: We can't actually shut down the FastAPI app from within a request handler.
    # The shutdown logic is handled by the lifespan context manager and signal handlers.
    # This endpoint just confirms that shutdown was requested and logs it.
    # Tauri will kill the process after calling this endpoint.

    return ShutdownResponse(
        message='Shutdown requested. Application will terminate.',
        shutdown_initiated=True,
    )


@router.get('/diagnostics/summary', response_model=DiagnosticsMetricsSummaryResponse)
async def get_diagnostics_summary(
    days: int = Query(default=30, ge=1, le=365),
    type_filter: Literal['user', 'evaluation'] | None = Query(default=None),
    run_id_filter: str | None = Query(default=None),
) -> DiagnosticsMetricsSummaryResponse:
    """
    Return aggregate runtime diagnostics metrics from response_diagnostics_metrics.
    Primarily used for operational trends and future stats dashboards.
    """
    async with get_db() as db:
        rows = await get_diagnostics_metrics_since(
            db=db,
            days=days,
            type_filter=type_filter,
            run_id_filter=run_id_filter,
        )

    total = len(rows)
    by_type: dict[str, int] = {}
    by_query_type: dict[str, int] = {}
    issue_counts: dict[str, int] = {}

    timeout_count = 0
    empty_answer_count = 0
    refusal_pattern_count = 0
    generation_seconds_values: list[float] = []
    sources_counts: list[int] = []
    raw_chunks_counts: list[int] = []
    created_at_values: list[datetime] = []

    for row in rows:
        metric_type = str(row.get('type') or '').strip().lower()
        if metric_type in CANONICAL_DIAGNOSTICS_TYPES:
            by_type[metric_type] = by_type.get(metric_type, 0) + 1
        else:
            log.warning('diagnostics_summary_unknown_type', raw_type=metric_type)

        query_type = str(row.get('query_type') or '').strip().lower()
        if query_type not in CANONICAL_DIAGNOSTICS_QUERY_TYPES:
            query_type = DiagnosticsQueryType.UNKNOWN.value
            log.warning('diagnostics_summary_unknown_query_type')
        by_query_type[query_type] = by_query_type.get(query_type, 0) + 1

        if bool(row.get('timeout_occurred')):
            timeout_count += 1
        if bool(row.get('has_empty_answer')):
            empty_answer_count += 1
        if bool(row.get('has_refusal_pattern')):
            refusal_pattern_count += 1

        generation_seconds = row.get('generation_seconds')
        if isinstance(generation_seconds, int | float):
            generation_seconds_values.append(float(generation_seconds))

        sources_count = row.get('sources_count')
        if isinstance(sources_count, int):
            sources_counts.append(sources_count)

        raw_chunks_count = row.get('raw_chunks_count')
        if isinstance(raw_chunks_count, int):
            raw_chunks_counts.append(raw_chunks_count)

        detected_issues = row.get('detected_issues') or []
        if isinstance(detected_issues, list):
            for issue in detected_issues:
                issue_name = str(issue or '').strip().lower()
                if issue_name and issue_name in _CANONICAL_DIAGNOSTICS_ISSUES:
                    issue_counts[issue_name] = issue_counts.get(issue_name, 0) + 1

        created_at = row.get('created_at')
        if isinstance(created_at, datetime):
            created_at_values.append(created_at)

    def _avg(values: list[int] | list[float]) -> float:
        if not values:
            return 0.0
        return round(float(sum(values)) / len(values), 3)

    timeout_rate = round(timeout_count / total, 4) if total else 0.0
    empty_answer_rate = round(empty_answer_count / total, 4) if total else 0.0
    refusal_pattern_rate = round(refusal_pattern_count / total, 4) if total else 0.0

    p95_generation_seconds: float | None = None
    if generation_seconds_values:
        sorted_values = sorted(generation_seconds_values)
        idx = max(0, min(len(sorted_values) - 1, math.ceil(len(sorted_values) * 0.95) - 1))
        p95_generation_seconds = round(sorted_values[idx], 3)

    created_at_oldest = min(created_at_values) if created_at_values else None
    created_at_newest = max(created_at_values) if created_at_values else None

    return DiagnosticsMetricsSummaryResponse(
        type_taxonomy=list(CANONICAL_DIAGNOSTICS_TYPES),
        query_type_taxonomy=list(CANONICAL_DIAGNOSTICS_QUERY_TYPES),
        issue_type_taxonomy=list(_CANONICAL_DIAGNOSTICS_ISSUES),
        window_days=days,
        type_filter=type_filter,
        run_id_filter=run_id_filter,
        total_responses=total,
        by_type=by_type,
        by_query_type=by_query_type,
        issue_counts=issue_counts,
        timeout_count=timeout_count,
        empty_answer_count=empty_answer_count,
        refusal_pattern_count=refusal_pattern_count,
        timeout_rate=timeout_rate,
        empty_answer_rate=empty_answer_rate,
        refusal_pattern_rate=refusal_pattern_rate,
        avg_generation_seconds=_avg(generation_seconds_values),
        p95_generation_seconds=p95_generation_seconds,
        avg_sources_count=_avg(sources_counts),
        avg_raw_chunks_count=_avg(raw_chunks_counts),
        created_at_oldest=created_at_oldest,
        created_at_newest=created_at_newest,
    )
