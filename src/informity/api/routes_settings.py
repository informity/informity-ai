# ==============================================================================
# Informity AI — Settings API Routes
# Endpoints for reading and updating application configuration.
# Settings are persisted to config.json under the app data directory
# (default: ~/.informity/config.json, override via INFORMITY_APP_DATA_DIR).
# ==============================================================================

import asyncio
import json
import os
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
    DiagnosticsProfilePreset,
    EnvVarsResponse,
    FileTypeOption,
    McpTokenGenerateResponse,
    ModelProfileInfo,
    SettingsResponse,
    SettingsUpdateRequest,
)
from informity.api.security import is_loopback_host
from informity.file_types import get_file_type_options
from informity.llm.model_adapter import (
    discover_available_models,
    get_profile_for_filename,
    infer_model_id_from_filename,
)
from informity.llm.roles import list_role_profiles
from informity.mcp.constants import generate_mcp_access_token
from informity.mcp.lifecycle import mcp_lifecycle
from informity.scanner.watcher import invalidate_watcher_cache
from informity.utils.directory_utils import ensure_file_directory, ensure_private_file
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
_DIAG_PROFILE_STANDARD = 'standard'
_DIAG_PROFILE_TROUBLESHOOTING = 'troubleshooting'
_DIAG_PROFILE_CUSTOM = 'custom'
_DIAG_PROFILE_ALLOWED_VALUES = (
    _DIAG_PROFILE_STANDARD,
    _DIAG_PROFILE_TROUBLESHOOTING,
    _DIAG_PROFILE_CUSTOM,
)
_DIAGNOSTICS_PROFILE_PRESETS: dict[str, dict[str, object]] = {
    _DIAG_PROFILE_STANDARD: {
        'log_level': 'info',
        'chat_trace_logging': False,
        'chat_trace_redaction_mode': 'minimal',
        'chat_trace_user_retention_days': 30,
        'chat_trace_evaluation_retention_days': 30,
    },
    _DIAG_PROFILE_TROUBLESHOOTING: {
        'log_level': 'debug',
        'chat_trace_logging': True,
        'chat_trace_redaction_mode': 'minimal',
        'chat_trace_user_retention_days': 14,
        'chat_trace_evaluation_retention_days': 14,
    },
}
_SUPPORTED_MAIN_MODEL_PROFILES: set[str] = {
    'Qwen3.5 9B',
    'Qwen3 14B',
    'Qwen3.6 35B A3B',
}
_SUPPORTED_EXTENSIONS_CANONICAL_ORDER: tuple[str, ...] = tuple(
    ext
    for option in get_file_type_options()
    for ext in option.get('extensions', [])
)


def _visible_role_ids() -> set[str]:
    return {profile.id for profile in list_role_profiles(visible_only=True)}


def _normalize_enabled_chat_role_ids(values: object, *, strict: bool = True) -> list[str]:
    if not isinstance(values, list):
        if strict:
            raise HTTPException(status_code=400, detail='enabled_chat_role_ids must be a list of role IDs')
        return []
    allowed = _visible_role_ids()
    seen: set[str] = set()
    normalized: list[str] = []
    for raw in values:
        role_id = str(raw or '').strip()
        if not role_id:
            continue
        if role_id in seen:
            continue
        if role_id not in allowed:
            if strict:
                raise HTTPException(status_code=400, detail=f'Unknown role ID in enabled_chat_role_ids: {role_id}')
            continue
        seen.add(role_id)
        normalized.append(role_id)
    return normalized


def _allowed_values_detail(field_name: str, values: tuple[str, ...]) -> str:
    return f"{field_name} must be one of: {', '.join(values)}"


def _normalize_supported_extensions(value: object) -> list[str]:
    raw_items = value if isinstance(value, list) else []
    normalized: list[str] = []
    seen: set[str] = set()
    for item in raw_items:
        ext = str(item or '').strip().lower()
        if not ext or ext in seen:
            continue
        seen.add(ext)
        normalized.append(ext)

    canonical_index = {ext: idx for idx, ext in enumerate(_SUPPORTED_EXTENSIONS_CANONICAL_ORDER)}
    return sorted(
        normalized,
        key=lambda ext: (canonical_index.get(ext, 10_000), ext),
    )


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
    'chat_history_messages_assistant': (0, 20, 'chat_history_messages_assistant must be between 0 and 20'),
    'chat_history_messages_researcher': (0, 10, 'chat_history_messages_researcher must be between 0 and 10'),
    'rag_query_rewrite_max_history_messages': (
        0,
        10,
        'rag_query_rewrite_max_history_messages must be between 0 and 10',
    ),
    'rag_query_rewrite_max_chars_per_turn': (
        32,
        2000,
        'rag_query_rewrite_max_chars_per_turn must be between 32 and 2000',
    ),
    'rag_query_rewrite_max_query_chars': (
        64,
        4000,
        'rag_query_rewrite_max_query_chars must be between 64 and 4000',
    ),
    'rag_rerank_candidates': (1, 200, 'rag_rerank_candidates must be between 1 and 200'),
    'rag_minimal_answerability_threshold_focused': (
        0.0,
        1.0,
        'rag_minimal_answerability_threshold_focused must be between 0.0 and 1.0',
    ),
    'rag_minimal_answerability_threshold_coverage': (
        0.0,
        1.0,
        'rag_minimal_answerability_threshold_coverage must be between 0.0 and 1.0',
    ),
    'rag_minimal_min_chunks_focused': (
        1,
        100,
        'rag_minimal_min_chunks_focused must be between 1 and 100',
    ),
    'rag_minimal_min_chunks_coverage': (
        1,
        100,
        'rag_minimal_min_chunks_coverage must be between 1 and 100',
    ),
    'embedding_max_threads': (
        0,
        32,
        'embedding_max_threads must be between 0 and 32 (0 = automatic)',
    ),
    'llm_cpu_threads': (0, 32, 'llm_cpu_threads must be between 0 and 32 (0 = automatic)'),
    'max_indexable_file_size_mb': (
        1,
        500,
        'max_indexable_file_size_mb must be between 1 and 500',
    ),
    'scan_file_timeout_seconds': (
        1,
        600,
        'scan_file_timeout_seconds must be between 1 and 600',
    ),
    'scan_hash_workers': (0, 32, 'scan_hash_workers must be between 0 and 32 (0 = automatic)'),
    'web_search_max_results': (1, 10, 'web_search_max_results must be between 1 and 10'),
    'web_search_timeout_seconds': (
        1,
        30,
        'web_search_timeout_seconds must be between 1 and 30',
    ),
    'ollama_timeout_seconds': (
        1,
        1800,
        'ollama_timeout_seconds must be between 1 and 1800',
    ),
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
    'mcp_http_port': (1, 65535, 'mcp_http_port must be between 1 and 65535'),
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
        _DIAG_PROFILE_ALLOWED_VALUES,
        True,
        _allowed_values_detail('diagnostics_profile', _DIAG_PROFILE_ALLOWED_VALUES),
    ),
    'chat_trace_redaction_mode': (
        ('off', 'minimal', 'strict'),
        True,
        'chat_trace_redaction_mode must be one of: off, minimal, strict',
    ),
    'default_chat_mode': (
        ('assistant', 'researcher'),
        True,
        'default_chat_mode must be one of: assistant, researcher',
    ),
    'web_search_primary_provider': (
        ('tavily', 'linkup'),
        True,
        'web_search_primary_provider must be one of: tavily, linkup',
    ),
    'llm_provider': (
        ('local_gguf', 'ollama'),
        True,
        'llm_provider must be one of: local_gguf, ollama',
    ),
    'ui_theme': (
        config.UI_THEME_ALLOWED_VALUES,
        False,
        _allowed_values_detail('ui_theme', config.UI_THEME_ALLOWED_VALUES),
    ),
    'mcp_transport': (
        ('stdio', 'http'),
        True,
        'mcp_transport must be one of: stdio, http',
    ),
    'mcp_auth_mode': (
        ('token_required',),
        True,
        'mcp_auth_mode must be: token_required',
    ),
    'mcp_scope_mode': (
        ('metadata_only', 'search_snippets', 'full_chunks'),
        True,
        'mcp_scope_mode must be one of: metadata_only, search_snippets, full_chunks',
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
    discovered = discover_available_models()
    return [
        model_filename
        for model_filename in discovered
        if get_profile_for_filename(model_filename).name in _SUPPORTED_MAIN_MODEL_PROFILES
    ]


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
    ensure_private_file(temp_path)
    temp_path.replace(config_path)
    ensure_private_file(config_path)
    log.info('config_file_written', path=str(config_path))


# ==============================================================================
# Updatable fields — the subset of Settings that the UI can change
# ==============================================================================

_UPDATABLE_FIELDS: set[str] = {
    'watched_directories',
    'source_scopes_enabled',
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
    'max_indexable_file_size_mb',
    'scan_file_timeout_seconds',
    'scan_hash_pool',
    'scan_hash_workers',
    'full_privacy',
    'tavily_api_key',
    'linkup_api_key',
    'web_search_primary_provider',
    'web_search_max_results',
    'web_search_timeout_seconds',
    'embedding_offline',
    'llm_provider',
    'llm_local_only',
    'llm_model_id',
    'ollama_base_url',
    'ollama_timeout_seconds',
    'llm_model_filename',
    # NOTE: rag_max_score and rag_context_ratio are now model-specific (in ModelProfile, not updatable)
    'rag_minimal_mode',
    'rag_minimal_answerability_threshold_focused',
    'rag_minimal_answerability_threshold_coverage',
    'rag_minimal_min_chunks_focused',
    'rag_minimal_min_chunks_coverage',
    'adaptive_rag_tuning',
    'rag_rerank',
    'rag_rerank_coverage',
    'rag_reranker_model',
    'rag_rerank_candidates',
    'rag_query_rewrite_enabled',
    'rag_query_rewrite_max_history_messages',
    'rag_query_rewrite_max_chars_per_turn',
    'rag_query_rewrite_max_query_chars',
    'chat_history_messages',
    'chat_history_messages_assistant',
    'chat_history_messages_researcher',
    'default_chat_mode',
    'enable_chat_roles',
    'enabled_chat_role_ids',
    'entity_extract_acronym',
    'entity_extract_person_name',
    'entity_extract_organization',
    'entity_extract_location',
    'entity_extract_numeric_id',
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
    'mcp_enabled',
    'mcp_auto_start',
    'mcp_transport',
    'mcp_http_host',
    'mcp_http_port',
    'mcp_auth_mode',
    'mcp_scope_mode',
    'mcp_access_token',
}
# NOTE: Profile-controlled fields removed from _UPDATABLE_FIELDS:
#   llm_max_tokens, coverage_top_k,
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
    effective_llm_model_filename = str(s.llm_model_filename or '').strip()
    effective_llm_model_id = str(getattr(s, 'llm_model_id', '') or '').strip().lower()
    if not effective_llm_model_id:
        effective_llm_model_id = infer_model_id_from_filename(effective_llm_model_filename) or ''
        if effective_llm_model_id:
            s.llm_model_id = effective_llm_model_id
    if effective_llm_model_filename and effective_llm_model_filename != s.llm_model_filename:
        s.llm_model_filename = effective_llm_model_filename

    profile_info = _build_model_profile_info(effective_llm_model_filename)

    enabled_chat_role_ids = _normalize_enabled_chat_role_ids(
        getattr(s, 'enabled_chat_role_ids', []),
        strict=False,
    )
    if enabled_chat_role_ids != list(getattr(s, 'enabled_chat_role_ids', [])):
        s.enabled_chat_role_ids = enabled_chat_role_ids

    roles_enabled = len(enabled_chat_role_ids) > 0
    if s.enable_chat_roles != roles_enabled:
        s.enable_chat_roles = roles_enabled

    return SettingsResponse(
        watched_directories     = [str(p) for p in s.watched_directories],
        source_scopes_enabled   = dict(s.source_scopes_enabled),
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
        enable_ocr_for_images        = s.enable_ocr_for_images,
        max_indexable_file_size_mb   = s.max_indexable_file_size_mb,
        scan_file_timeout_seconds    = s.scan_file_timeout_seconds,
        scan_hash_pool          = s.scan_hash_pool,
        scan_hash_workers       = s.scan_hash_workers,
        full_privacy            = s.full_privacy,
        tavily_api_key_set      = bool(str(s.tavily_api_key or '').strip()),
        linkup_api_key_set      = bool(str(s.linkup_api_key or '').strip()),
        web_search_configured   = (
            bool(str(s.tavily_api_key or '').strip())
            or bool(str(s.linkup_api_key or '').strip())
        ),
        web_search_primary_provider = (
            'linkup'
            if str(s.web_search_primary_provider or '').strip().lower() == 'linkup'
            else 'tavily'
        ),
        web_search_max_results  = s.web_search_max_results,
        web_search_timeout_seconds = s.web_search_timeout_seconds,
        embedding_offline       = s.embedding_offline,
        llm_provider         = str(getattr(s, 'llm_provider', 'local_gguf') or 'local_gguf').strip().lower(),
        llm_local_only          = s.llm_local_only,
        llm_model_id         = effective_llm_model_id,
        ollama_base_url      = str(getattr(s, 'ollama_base_url', 'http://127.0.0.1:11434') or 'http://127.0.0.1:11434').strip(),
        ollama_timeout_seconds = float(getattr(s, 'ollama_timeout_seconds', 120.0) or 120.0),
        llm_model_filename   = effective_llm_model_filename,
        # rag_max_score and rag_context_ratio are now in model_profile (read-only, model-specific)
        rag_minimal_mode     = s.rag_minimal_mode,
        rag_minimal_answerability_threshold_focused = s.rag_minimal_answerability_threshold_focused,
        rag_minimal_answerability_threshold_coverage = s.rag_minimal_answerability_threshold_coverage,
        rag_minimal_min_chunks_focused = s.rag_minimal_min_chunks_focused,
        rag_minimal_min_chunks_coverage = s.rag_minimal_min_chunks_coverage,
        adaptive_rag_tuning   = s.adaptive_rag_tuning,
        rag_rerank            = s.rag_rerank,
        rag_rerank_coverage   = s.rag_rerank_coverage,
        rag_reranker_model    = s.rag_reranker_model.strip() or config.DEFAULT_RERANKER_MODEL,
        rag_rerank_candidates = s.rag_rerank_candidates,
        rag_query_rewrite_enabled = s.rag_query_rewrite_enabled,
        rag_query_rewrite_max_history_messages = s.rag_query_rewrite_max_history_messages,
        rag_query_rewrite_max_chars_per_turn = s.rag_query_rewrite_max_chars_per_turn,
        rag_query_rewrite_max_query_chars = s.rag_query_rewrite_max_query_chars,
        chat_history_messages = s.chat_history_messages,
        chat_history_messages_assistant = s.chat_history_messages_assistant,
        chat_history_messages_researcher = s.chat_history_messages_researcher,
        default_chat_mode = s.default_chat_mode,
        enable_chat_roles = roles_enabled,
        enabled_chat_role_ids = enabled_chat_role_ids,
        entity_extract_acronym = s.entity_extract_acronym,
        entity_extract_person_name = s.entity_extract_person_name,
        entity_extract_organization = s.entity_extract_organization,
        entity_extract_location = s.entity_extract_location,
        entity_extract_numeric_id = s.entity_extract_numeric_id,
        diagnostics_profile   = s.diagnostics_profile,
        diagnostics_profile_presets = {
            name: DiagnosticsProfilePreset(**values)
            for name, values in _DIAGNOSTICS_PROFILE_PRESETS.items()
        },
        log_level             = s.log_level,
        chat_trace_logging    = s.chat_trace_logging,
        chat_trace_redaction_mode = s.chat_trace_redaction_mode,
        chat_trace_user_retention_days = s.chat_trace_user_retention_days,
        chat_trace_evaluation_retention_days = s.chat_trace_evaluation_retention_days,
        mcp_enabled           = s.mcp_enabled,
        mcp_auto_start        = s.mcp_auto_start,
        mcp_transport         = s.mcp_transport,
        mcp_http_host         = s.mcp_http_host,
        mcp_http_port         = s.mcp_http_port,
        mcp_auth_mode         = s.mcp_auth_mode,
        mcp_scope_mode        = s.mcp_scope_mode,
        mcp_access_token      = str(getattr(s, 'mcp_access_token', '') or ''),
        mcp_token_configured  = bool(
            str(os.environ.get('INFORMITY_MCP_TOKEN') or '').strip()
            or str(getattr(s, 'mcp_access_token', '') or '').strip()
        ),
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
            if field_name == 'ollama_base_url' and value is not None:
                value = str(value).strip()
                if not value:
                    raise HTTPException(status_code=400, detail='ollama_base_url cannot be empty')
                if not (value.startswith('http://') or value.startswith('https://')):
                    raise HTTPException(status_code=400, detail='ollama_base_url must start with http:// or https://')
            if field_name == 'scan_file_timeout_seconds' and value is not None:
                policy = config.settings.scan_timeout_policy
                policy.default.max_seconds = int(value)
                if 'filesystem:file' in policy.overrides:
                    policy.overrides['filesystem:file'].max_seconds = int(value)
                config.settings.scan_timeout_policy = policy
                config.settings.scan_file_timeout_seconds = int(value)
                config_data[field_name] = int(value)
                continue
            if field_name == 'diagnostics_profile' and value is not None:
                diagnostics_profile_value = value
                continue
            if field_name == 'mcp_http_host' and value is not None:
                value = str(value).strip()
                if not value:
                    raise HTTPException(status_code=400, detail='mcp_http_host cannot be empty')
                if not is_loopback_host(value):
                    raise HTTPException(
                        status_code=400,
                        detail='mcp_http_host must be loopback only (127.0.0.1, localhost, or ::1)',
                    )
            if field_name == 'mcp_access_token' and value is not None:
                value = str(value).strip()
            if field_name == 'enable_chat_roles':
                # Compatibility field: canonical source of truth is enabled_chat_role_ids.
                enabled = bool(value)
                config.settings.enable_chat_roles = enabled
                config_data['enable_chat_roles'] = enabled
                if not enabled:
                    config.settings.enabled_chat_role_ids = []
                    config_data['enabled_chat_role_ids'] = []
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
                inferred_model_id = infer_model_id_from_filename(value)
                if inferred_model_id:
                    config.settings.llm_model_id = inferred_model_id
                    config_data['llm_model_id'] = inferred_model_id

            if field_name == 'tavily_api_key' and value is not None:
                value = str(value).strip()
            if field_name == 'linkup_api_key' and value is not None:
                value = str(value).strip()

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
            elif field_name == 'supported_extensions':
                normalized_extensions = _normalize_supported_extensions(value)
                setattr(config.settings, field_name, normalized_extensions)
                config_data[field_name] = normalized_extensions
            elif field_name == 'enabled_chat_role_ids':
                normalized_role_ids = _normalize_enabled_chat_role_ids(value)
                setattr(config.settings, field_name, normalized_role_ids)
                config.settings.enable_chat_roles = len(normalized_role_ids) > 0
                config_data[field_name] = normalized_role_ids
                config_data['enable_chat_roles'] = len(normalized_role_ids) > 0
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
            config.settings.diagnostics_profile = _DIAG_PROFILE_CUSTOM
            config_data['diagnostics_profile'] = _DIAG_PROFILE_CUSTOM

        # Enforce MCP token lifecycle invariants on persisted settings.
        # 1) Disabling MCP always clears any persisted token.
        # 2) STDIO transport never retains an HTTP auth token.
        mcp_enabled = bool(getattr(config.settings, 'mcp_enabled', False))
        mcp_transport = str(getattr(config.settings, 'mcp_transport', 'stdio') or 'stdio').strip().lower()
        should_clear_mcp_token = (not mcp_enabled) or mcp_transport == 'stdio'
        if should_clear_mcp_token:
            config.settings.mcp_access_token = ''
            config_data['mcp_access_token'] = ''
        # MCP auto-start is coupled to MCP enabled state in desktop UX.
        config.settings.mcp_auto_start = mcp_enabled
        config_data['mcp_auto_start'] = mcp_enabled

        # Timeout policy internals are backend-managed and derived from scalar cap.
        config_data.pop('scan_timeout_policy', None)

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
        invalidate_watcher_cache()

    # Keep adaptive top-k cache aligned with model/settings changes.
    adaptive_changed = 'adaptive_rag_tuning' in updates
    model_changed = 'llm_model_filename' in updates or 'llm_model_id' in updates
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

    mcp_fields = {
        'mcp_enabled',
        'mcp_auto_start',
        'mcp_transport',
        'mcp_http_host',
        'mcp_http_port',
        'mcp_auth_mode',
        'mcp_scope_mode',
    }
    if any(field in updates for field in mcp_fields):
        await mcp_lifecycle.restart_from_settings()

    # Return the full updated settings
    return await get_settings()


# ==============================================================================
# POST /api/settings/mcp/token/generate — generate and persist MCP access token
# ==============================================================================

@router.post('/api/settings/mcp/token/generate', response_model=McpTokenGenerateResponse)
async def generate_mcp_token() -> McpTokenGenerateResponse:
    token = generate_mcp_access_token()
    async with _CONFIG_FILE_ASYNC_LOCK:
        config_data = await asyncio.to_thread(_read_config_file)
        config.settings.mcp_access_token = token
        config_data['mcp_access_token'] = token
        await asyncio.to_thread(_write_config_file, config_data)
    return McpTokenGenerateResponse(token=token)


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
