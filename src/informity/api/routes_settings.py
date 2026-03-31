# ==============================================================================
# Informity AI — Settings API Routes
# Endpoints for reading and updating application configuration.
# Settings are persisted to config.json under the app data directory
# (default: ~/.informity/config.json, override via INFORMITY_APP_DATA_DIR).
# ==============================================================================

import asyncio
import json
from pathlib import Path
from typing import Any

import aiosqlite
import structlog
from fastapi import APIRouter, HTTPException, Query

from informity import config
from informity.api.config_reference_metadata import get_config_reference_response
from informity.api.env_vars_metadata import get_env_vars_response
from informity.api.schemas import (
    ConfigReferenceResponse,
    CurrentChatResponse,
    CurrentChatUpdateRequest,
    EnvVarsResponse,
    FileTypeOption,
    ModelProfileInfo,
    SettingsResponse,
    SettingsUpdateRequest,
)
from informity.file_types import get_file_type_options
from informity.llm.model_adapter import discover_available_models, get_profile_for_filename
from informity.scanner.watcher import _invalidate_watcher_cache
from informity.utils.directory_utils import ensure_file_directory
from informity.utils.json_utils import serialize_config
from informity.utils.path_utils import resolve_and_check_path

# ==============================================================================
# Logger
# ==============================================================================

log = structlog.get_logger(__name__)
_SETTINGS_IO_EXCEPTIONS = (OSError, RuntimeError, ValueError, TypeError, json.JSONDecodeError)
_SETTINGS_RUNTIME_EXCEPTIONS = (aiosqlite.Error, RuntimeError, ValueError, TypeError, OSError, TimeoutError)

# ==============================================================================
# Router
# ==============================================================================

router = APIRouter(tags=['settings'])
_CONFIG_FILE_ASYNC_LOCK = asyncio.Lock()
_DIAGNOSTICS_PROFILE_PRESETS: dict[str, dict[str, object]] = {
    'standard': {
        'log_level': 'info',
        'chat_trace_logging': False,
        'chat_trace_redaction_mode': 'minimal',
        'chat_trace_user_retention_days': 30,
        'chat_trace_evaluation_retention_days': 30,
    },
    'troubleshooting': {
        'log_level': 'debug',
        'chat_trace_logging': True,
        'chat_trace_redaction_mode': 'minimal',
        'chat_trace_user_retention_days': 14,
        'chat_trace_evaluation_retention_days': 14,
    },
}


def _allowed_values_detail(field_name: str, values: tuple[str, ...]) -> str:
    return f"{field_name} must be one of: {', '.join(values)}"


_SETTINGS_RANGE_RULES: dict[str, tuple[float, float, str]] = {
    'chunk_size_tokens': (200, 1200, 'chunk_size_tokens must be between 200 and 1200'),
    'chunk_overlap_tokens': (0, 200, 'chunk_overlap_tokens must be between 0 and 200'),
    'chunk_filter_header_ratio': (0.0, 1.0, 'chunk_filter_header_ratio must be between 0.0 and 1.0'),
    'chunk_filter_min_content_chars': (
        0,
        10000,
        'chunk_filter_min_content_chars must be between 0 and 10000',
    ),
    'chunk_filter_min_content_lines': (
        0,
        100,
        'chunk_filter_min_content_lines must be between 0 and 100',
    ),
    'embedding_batch_size': (1, 256, 'embedding_batch_size must be between 1 and 256'),
    'chat_history_messages': (0, 10, 'chat_history_messages must be between 0 and 10'),
    'rag_rerank_candidates': (1, 200, 'rag_rerank_candidates must be between 1 and 200'),
    'embedding_max_threads': (
        0,
        32,
        'embedding_max_threads must be between 0 and 32 (0 = automatic)',
    ),
    'llm_cpu_threads': (0, 32, 'llm_cpu_threads must be between 0 and 32 (0 = automatic)'),
    'scan_file_timeout_seconds': (
        0,
        600,
        'scan_file_timeout_seconds must be between 0 and 600',
    ),
    'scan_hash_workers': (0, 32, 'scan_hash_workers must be between 0 and 32 (0 = automatic)'),
    'chat_trace_user_retention_days': (
        0,
        3650,
        'chat_trace_user_retention_days must be between 0 and 3650',
    ),
    'chat_trace_evaluation_retention_days': (
        0,
        3650,
        'chat_trace_evaluation_retention_days must be between 0 and 3650',
    ),
    'cpu_priority_nice': (0, 19, 'cpu_priority_nice must be between 0 and 19'),
}
_SETTINGS_ALLOWED_VALUE_RULES: dict[str, tuple[tuple[str, ...], bool, str]] = {
    'scan_hash_pool': (
        ('thread', 'process'),
        True,
        'scan_hash_pool must be one of: thread, process',
    ),
    'log_level': (
        config.LOG_LEVEL_ALLOWED_VALUES,
        True,
        _allowed_values_detail('log_level', config.LOG_LEVEL_ALLOWED_VALUES),
    ),
    'diagnostics_profile': (
        ('standard', 'troubleshooting', 'custom'),
        True,
        'diagnostics_profile must be one of: standard, troubleshooting, custom',
    ),
    'chat_trace_redaction_mode': (
        ('off', 'minimal', 'strict'),
        True,
        'chat_trace_redaction_mode must be one of: off, minimal, strict',
    ),
    'ui_theme': (
        config.UI_THEME_ALLOWED_VALUES,
        False,
        _allowed_values_detail('ui_theme', config.UI_THEME_ALLOWED_VALUES),
    ),
}


# ==============================================================================
# Helpers
# ==============================================================================

def _config_file_path() -> Path:
    # Return the path to the JSON config file.
    return config.settings.app_data_dir / 'config.json'


def _list_available_models() -> list[str]:
    # Scan the models directory for downloaded GGUF files.
    return discover_available_models()


def _build_model_profile_info(model_filename: str) -> ModelProfileInfo:
    profile = get_profile_for_filename(model_filename)
    return ModelProfileInfo(**profile.to_display_dict())


def _read_config_file() -> dict:
    # Read the existing config file, or return an empty dict.
    config_path = _config_file_path()
    if not config_path.exists():
        return {}
    try:
        return json.loads(config_path.read_text(encoding='utf-8'))
    except (json.JSONDecodeError, OSError) as exc:
        log.warning('config_file_read_error', path=str(config_path), error=str(exc))
        return {}


def _write_config_file(data: dict) -> None:
    # Write the config dict to the JSON config file.
    config_path = _config_file_path()
    ensure_file_directory(config_path)
    # Atomic write to avoid partial/truncated JSON under concurrent requests.
    temp_path = config_path.with_name(f'{config_path.name}.tmp')
    temp_path.write_text(
        serialize_config(data),
        encoding='utf-8',
    )
    temp_path.replace(config_path)
    log.info('config_file_written', path=str(config_path))


# ==============================================================================
# Updatable fields — the subset of Settings that the UI can change
# ==============================================================================

_UPDATABLE_FIELDS: set[str] = {
    'watched_directories',
    'ignore_patterns',
    'exclude_macos_system',
    'exclude_developer_data',
    'supported_extensions',
    'follow_symlinks',
    'chunk_size_tokens',
    'chunk_overlap_tokens',
    'chunk_filter_header_only',
    'chunk_filter_header_ratio',
    'chunk_filter_min_content_chars',
    'chunk_filter_min_content_lines',
    'embedding_batch_size',
    'embedding_max_threads',
    'llm_cpu_threads',
    'enable_ocr_for_images',
    'scan_file_timeout_seconds',
    'scan_hash_pool',
    'scan_hash_workers',
    'full_privacy',
    'embedding_offline',
    'llm_local_only',
    'llm_model_filename',
    # NOTE: rag_max_score and rag_context_ratio are now model-specific (in ModelProfile, not updatable)
    'adaptive_rag_tuning',
    'rag_rerank',
    'rag_rerank_coverage',
    'rag_reranker_model',
    'rag_rerank_candidates',
    'chat_history_messages',
    'diagnostics_profile',
    'chat_trace_logging',
    'chat_trace_redaction_mode',
    'chat_trace_user_retention_days',
    'chat_trace_evaluation_retention_days',
    'enable_raw_output_control',
    'log_level',
    'ui_theme',
    'enable_menu_bar_icon',
    'cpu_priority_nice',
}
# NOTE: Profile-controlled fields removed from _UPDATABLE_FIELDS:
#   llm_max_tokens, rag_coverage_top_k,
#   llm_context_length,
#   llm_temperature, rag_top_k,
#   rag_max_score, rag_context_ratio
# These are determined by the active model's ModelProfile.


# ==============================================================================
# GET /api/config/env-vars — list env variables for Configuration page
# ==============================================================================

@router.get('/api/config/env-vars', response_model=EnvVarsResponse)
async def get_env_vars() -> EnvVarsResponse:
    return get_env_vars_response(config.settings)


# ==============================================================================
# GET /api/config/reference — return application defaults and constants
# ==============================================================================

@router.get('/api/config/reference', response_model=ConfigReferenceResponse)
async def get_config_reference() -> ConfigReferenceResponse:
    return get_config_reference_response()


# ==============================================================================
# GET /api/settings — return current settings
# ==============================================================================

@router.get('/api/settings', response_model=SettingsResponse)
async def get_settings() -> SettingsResponse:
    # Access config.settings to always get the current singleton value
    s = config.settings

    profile_info = _build_model_profile_info(s.llm_model_filename)

    return SettingsResponse(
        watched_directories     = [str(p) for p in s.watched_directories],
        ignore_patterns        = list(s.ignore_patterns),
        exclude_macos_system   = s.exclude_macos_system,
        exclude_developer_data  = s.exclude_developer_data,
        supported_extensions   = list(s.supported_extensions),
        follow_symlinks        = s.follow_symlinks,
        chunk_size_tokens    = s.chunk_size_tokens,
        chunk_overlap_tokens = s.chunk_overlap_tokens,
        chunk_filter_header_only = s.chunk_filter_header_only,
        chunk_filter_header_ratio = s.chunk_filter_header_ratio,
        chunk_filter_min_content_chars = s.chunk_filter_min_content_chars,
        chunk_filter_min_content_lines = s.chunk_filter_min_content_lines,
        embedding_model         = s.embedding_model,
        embedding_batch_size    = s.embedding_batch_size,
        embedding_max_threads   = s.embedding_max_threads,
        llm_cpu_threads         = s.llm_cpu_threads,
        enable_ocr_for_images   = s.enable_ocr_for_images,
        scan_file_timeout_seconds = s.scan_file_timeout_seconds,
        scan_hash_pool          = s.scan_hash_pool,
        scan_hash_workers       = s.scan_hash_workers,
        full_privacy            = s.full_privacy,
        embedding_offline       = s.embedding_offline,
        llm_local_only          = s.llm_local_only,
        llm_model_filename   = s.llm_model_filename,
        # rag_max_score and rag_context_ratio are now in model_profile (read-only, model-specific)
        adaptive_rag_tuning   = s.adaptive_rag_tuning,
        rag_rerank            = s.rag_rerank,
        rag_rerank_coverage   = s.rag_rerank_coverage,
        rag_reranker_model    = s.rag_reranker_model.strip() or config._DEFAULT_RERANKER_MODEL,
        rag_rerank_candidates = s.rag_rerank_candidates,
        chat_history_messages = s.chat_history_messages,
        diagnostics_profile   = s.diagnostics_profile,
        log_level             = s.log_level,
        chat_trace_logging    = s.chat_trace_logging,
        chat_trace_redaction_mode = s.chat_trace_redaction_mode,
        chat_trace_user_retention_days = s.chat_trace_user_retention_days,
        chat_trace_evaluation_retention_days = s.chat_trace_evaluation_retention_days,
        enable_raw_output_control = s.enable_raw_output_control,
        available_models      = await asyncio.to_thread(_list_available_models),
        file_type_options     = [FileTypeOption(**o) for o in get_file_type_options()],
        config_file_path      = str(_config_file_path()),
        model_profile         = profile_info,
        ui_theme              = s.ui_theme,
        enable_menu_bar_icon  = s.enable_menu_bar_icon,
        cpu_priority_nice     = s.cpu_priority_nice,
    )


# ==============================================================================
# GET /api/settings/model-profile — preview profile for selected model filename
# ==============================================================================

@router.get('/api/settings/model-profile', response_model=ModelProfileInfo)
async def get_model_profile(
    model_filename: str = Query(..., min_length=1),
) -> ModelProfileInfo:
    value = str(model_filename or '').strip()
    if not value:
        raise HTTPException(status_code=400, detail='model_filename cannot be empty')
    if not value.endswith('.gguf'):
        raise HTTPException(status_code=400, detail='model_filename must be a .gguf file')
    return _build_model_profile_info(value)


# ==============================================================================
# GET /api/file-types — canonical file type options (filtering, display, settings)
# ==============================================================================

@router.get('/api/file-types', response_model=list[FileTypeOption])
async def list_file_types() -> list[FileTypeOption]:
    return [FileTypeOption(**o) for o in get_file_type_options()]


# ==============================================================================
# PUT /api/settings — update settings
# ==============================================================================

@router.put('/api/settings', response_model=SettingsResponse)
async def update_settings(request: SettingsUpdateRequest) -> SettingsResponse:
    # Validate and apply the partial update.

    # Use exclude_unset=True (not exclude_none) so that explicitly sending
    # null for nullable fields like rag_max_score is honoured rather than
    # silently dropped.
    updates = request.model_dump(exclude_unset=True)
    if not updates:
        raise HTTPException(status_code=400, detail='No fields provided to update')

    async with _CONFIG_FILE_ASYNC_LOCK:
        # Read existing config file to merge
        config_data = await asyncio.to_thread(_read_config_file)

        # Save original values for rollback if config write fails
        original_values: dict[str, Any] = {}
        for field_name in updates:
            if hasattr(config.settings, field_name):
                original_values[field_name] = getattr(config.settings, field_name)
        # full_privacy sync mutates these fields even when they are not present in
        # request payload; capture originals so rollback is lossless on write failure.
        if 'full_privacy' in updates:
            for synced_field in ('embedding_offline', 'llm_local_only'):
                if synced_field not in original_values and hasattr(config.settings, synced_field):
                    original_values[synced_field] = getattr(config.settings, synced_field)

        # Track whether full_privacy was explicitly set in this request so we can
        # apply its sync logic *after* all other fields (avoids order-dependency
        # when the same request also sends embedding_offline or llm_local_only).
        full_privacy_value: bool | None = None
        diagnostics_profile_value: str | None = None
        diagnostics_preset_originals: dict[str, object] = {}

        # Apply each update to both the live settings singleton and the config file
        for field_name, value in updates.items():
            if field_name not in _UPDATABLE_FIELDS:
                raise HTTPException(
                    status_code=400,
                    detail=f'Field "{field_name}" is not configurable via the API',
                )

            # Validate numeric ranges via centralized rules.
            range_rule = _SETTINGS_RANGE_RULES.get(field_name)
            if range_rule is not None and value is not None:
                min_value, max_value, detail = range_rule
                if not (min_value <= value <= max_value):
                    raise HTTPException(status_code=400, detail=detail)

            allowed_rule = _SETTINGS_ALLOWED_VALUE_RULES.get(field_name)
            if allowed_rule is not None and value is not None:
                allowed_values, normalize_lower, detail = allowed_rule
                normalized_value = value.strip().lower() if normalize_lower else value
                if normalized_value not in allowed_values:
                    raise HTTPException(status_code=400, detail=detail)
                value = normalized_value

            # NOTE: rag_max_score and rag_context_ratio validation removed — they're now model-specific (not updatable)
            if field_name == 'rag_reranker_model' and value is not None:
                value = (value or '').strip()
                if not value:
                    raise HTTPException(status_code=400, detail='rag_reranker_model cannot be empty')
            if field_name == 'diagnostics_profile' and value is not None:
                diagnostics_profile_value = value
                continue
            if field_name == 'llm_model_filename' and value is not None:
                value = (value or '').strip()
                if not value:
                    raise HTTPException(status_code=400, detail='llm_model_filename cannot be empty')
                if not value.endswith('.gguf'):
                    raise HTTPException(status_code=400, detail='llm_model_filename must be a .gguf file')
                available_models = await asyncio.to_thread(_list_available_models)
                if value not in available_models:
                    raise HTTPException(
                        status_code=400,
                        detail=f'llm_model_filename not found in models directory: {value}',
                    )

            # Defer full_privacy sync until after the loop so it always wins
            if field_name == 'full_privacy':
                full_privacy_value = value
                continue

            # Convert Path lists for watched_directories: resolve to absolute and validate
            if field_name == 'watched_directories':
                resolved: list[Path] = []
                for p in value:
                    path, exists = resolve_and_check_path(p)
                    if not exists:
                        raise HTTPException(
                            status_code=400,
                            detail=f'Watched directory does not exist: {path}. Use an absolute path (e.g. /Users/you/Documents).',
                        )
                    if not path.is_dir():
                        raise HTTPException(
                            status_code=400,
                            detail=f'Watched path is not a directory: {path}',
                        )
                    resolved.append(path)
                setattr(config.settings, field_name, resolved)
                config_data[field_name] = [str(p) for p in resolved]  # Persist absolute paths
            else:
                setattr(config.settings, field_name, value)
                config_data[field_name] = value

        # Apply full_privacy sync last so it always overrides embedding_offline / llm_local_only,
        # regardless of JSON key ordering or whether those fields were also in the request.
        if full_privacy_value is not None:
            config.settings.full_privacy      = full_privacy_value
            config.settings.embedding_offline  = full_privacy_value
            config.settings.llm_local_only     = full_privacy_value
            config_data['full_privacy']        = full_privacy_value
            config_data['embedding_offline']   = full_privacy_value
            config_data['llm_local_only']      = full_privacy_value

        # Apply diagnostics profile preset last so profile choice is deterministic.
        if diagnostics_profile_value is not None:
            config.settings.diagnostics_profile = diagnostics_profile_value
            config_data['diagnostics_profile'] = diagnostics_profile_value
            preset_values = _DIAGNOSTICS_PROFILE_PRESETS.get(diagnostics_profile_value)
            if preset_values:
                for preset_field, preset_value in preset_values.items():
                    diagnostics_preset_originals[preset_field] = getattr(config.settings, preset_field)
                    setattr(config.settings, preset_field, preset_value)
                    config_data[preset_field] = preset_value
        elif any(
            field_name in updates
            for field_name in (
                'log_level',
                'chat_trace_logging',
                'chat_trace_redaction_mode',
                'chat_trace_user_retention_days',
                'chat_trace_evaluation_retention_days',
            )
        ):
            config.settings.diagnostics_profile = 'custom'
            config_data['diagnostics_profile'] = 'custom'

        # Cross-field validation: ensure chunk_overlap_tokens < chunk_size_tokens
        if config.settings.chunk_overlap_tokens >= config.settings.chunk_size_tokens:
            raise HTTPException(
                status_code=400,
                detail=f'chunk_overlap_tokens ({config.settings.chunk_overlap_tokens}) must be less than chunk_size_tokens ({config.settings.chunk_size_tokens})',
            )

        # Persist to config file - if this fails, rollback singleton changes
        try:
            await asyncio.to_thread(_write_config_file, config_data)
        except _SETTINGS_IO_EXCEPTIONS as exc:
            # Rollback singleton to original values
            for field_name, original_value in original_values.items():
                setattr(config.settings, field_name, original_value)
            for field_name, original_value in diagnostics_preset_originals.items():
                setattr(config.settings, field_name, original_value)
            log.error('config_file_write_failed', error=str(exc), exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f'Failed to write config file: {exc}',
            ) from exc

    # Invalidate watcher cache if watched_directories or supported_extensions changed
    if 'watched_directories' in updates or 'supported_extensions' in updates or 'exclude_macos_system' in updates or 'exclude_developer_data' in updates or 'ignore_patterns' in updates:
        _invalidate_watcher_cache()

    # Keep adaptive top-k cache aligned with model/settings changes.
    adaptive_changed = 'adaptive_rag_tuning' in updates
    model_changed = 'llm_model_filename' in updates
    if adaptive_changed or model_changed:
        try:
            from informity.db.sqlite import get_connection
            from informity.indexer.adaptive_tuning import (
                invalidate_tuning_cache,
                update_tuning_cache,
            )

            # Always clear previous adaptive values first; if recompute fails, caller
            # falls back to model profile base values through get_retrieval_top_k().
            invalidate_tuning_cache()

            if config.settings.adaptive_rag_tuning:
                conn = await get_connection()
                try:
                    await update_tuning_cache(conn, force_recompute=True)
                finally:
                    await conn.close()
        except (ImportError, _SETTINGS_RUNTIME_EXCEPTIONS) as exc:
            log.warning('adaptive_tuning_settings_refresh_failed', error=str(exc))

    log.info('settings_updated', fields=list(updates.keys()))

    # Return the full updated settings
    return await get_settings()


# ==============================================================================
# GET /api/settings/current-chat — persisted current chat ID (Tauri-compatible)
# ==============================================================================

@router.get('/api/settings/current-chat', response_model=CurrentChatResponse)
async def get_current_chat() -> CurrentChatResponse:
    async with _CONFIG_FILE_ASYNC_LOCK:
        config_data = await asyncio.to_thread(_read_config_file)
    chat_id = config_data.get('current_chat_id')
    return CurrentChatResponse(current_chat_id=chat_id if chat_id else None)


# ==============================================================================
# PUT /api/settings/current-chat — persist current chat ID
# ==============================================================================

@router.put('/api/settings/current-chat', response_model=CurrentChatResponse)
async def update_current_chat(request: CurrentChatUpdateRequest) -> CurrentChatResponse:
    async with _CONFIG_FILE_ASYNC_LOCK:
        config_data = await asyncio.to_thread(_read_config_file)
        if request.current_chat_id is None:
            config_data.pop('current_chat_id', None)
        else:
            config_data['current_chat_id'] = request.current_chat_id
        await asyncio.to_thread(_write_config_file, config_data)
    return CurrentChatResponse(current_chat_id=request.current_chat_id)


# ==============================================================================
# POST /api/settings/reset — reset all settings to factory defaults
# ==============================================================================

@router.post('/api/settings/reset', response_model=SettingsResponse)
async def reset_settings() -> SettingsResponse:
    # Reset all user settings to factory defaults by deleting config.json
    # and rebuilding the settings singleton.
    async with _CONFIG_FILE_ASYNC_LOCK:
        await asyncio.to_thread(config.reset_to_factory_defaults)

    log.info('settings_reset_to_factory_defaults')

    # Return the factory default settings
    return await get_settings()
