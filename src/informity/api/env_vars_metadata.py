# ==============================================================================
# Informity AI — Environment Variables Metadata
# Defines groups, descriptions, and current/default values for all INFORMITY_*
# env vars. Used by GET /api/config/env-vars for the Configuration page.
# ==============================================================================

import json
import os
from pathlib import Path

from informity.api.schemas import EnvVarGroup, EnvVarItem, EnvVarsResponse
from informity.config import APP_SLUG, DirNames, Settings
from informity.file_types import get_file_type_options
from informity.utils.path_utils import normalize_path

# Prefix used by pydantic-settings for this app.
_ENV_PREFIX = 'INFORMITY_'


def _env_name(field: str) -> str:
    # Convert snake_case field name to INFORMITY_UPPER_SNAKE.
    return _ENV_PREFIX + field.upper()


def _path_relative_to_app(p: Path, app_dir: Path) -> str:
    # Show path relative to application directory; fall back to absolute if outside.
    try:
        normalized_p = normalize_path(p, expand_user=False)
        normalized_app_dir = normalize_path(app_dir, expand_user=False)
        return str(normalized_p.relative_to(normalized_app_dir))
    except ValueError:
        return str(normalize_path(p, expand_user=False))


def _format_value(value: object, app_dir: Path | None = None) -> str:
    # Serialize a settings value for display. Paths are shown relative to app_dir (default: cwd).
    if value is None:
        return ''
    base = normalize_path(app_dir or Path.cwd(), expand_user=True)
    if isinstance(value, Path):
        return _path_relative_to_app(value, base)
    if isinstance(value, bool):
        return 'true' if value else 'false'
    if isinstance(value, list):
        parts = [
            _path_relative_to_app(x, base) if isinstance(x, Path) else str(x)
            for x in value
        ]
        s = json.dumps(parts)
        return s if len(s) <= 120 else s[:117] + '...'
    return str(value)


# ------------------------------------------------------------------------------
# Group definitions: title, description, and list of (field_name, description).
# Current value is read from settings at runtime; variables sorted by env name.
# ------------------------------------------------------------------------------

# Section order matches Settings UI: Server → Paths → Privacy → Appearance → Data Sources → Indexing → Embeddings → LLM and RAG → Logging → Diagnostics
_GROUPS: list[tuple[str, str, list[tuple[str, str]]]] = [
    (
        'Server',
        'HTTP server binding. Change host or port to run multiple instances or expose the API.',
        [
            ('host', 'Bind address for the API server.'),
            ('port', 'Port for the API server.'),
            ('dev_reload', 'Enable auto-reload on code changes. Development only — leave off in production.'),
            ('api_docs_enabled', 'Show interactive API docs at /docs. Defaults to enabled only when dev_reload is on.'),
        ],
    ),
    (
        'MCP Server',
        'Model Context Protocol server configuration for external AI clients. Use STDIO for widest client compatibility; HTTP is loopback-only.',
        [
            ('mcp_enabled', 'Enable or disable the local MCP server integration.'),
            ('mcp_auto_start', 'When true, start MCP server automatically with the application (when MCP is enabled).'),
            ('mcp_transport', 'MCP transport mode: stdio (recommended) or http (loopback only).'),
            ('mcp_http_host', 'Loopback host for HTTP transport (127.0.0.1, localhost, or ::1).'),
            ('mcp_http_port', 'Port for HTTP MCP transport.'),
            ('mcp_auth_mode', 'HTTP authentication mode. Current value: token_required.'),
            ('mcp_scope_mode', 'Access level for exposed MCP content: metadata_only, search_snippets, or full_content.'),
            ('mcp_access_token', 'Saved MCP HTTP bearer token used when INFORMITY_MCP_TOKEN is not set.'),
            ('mcp_tool_call_timeout_seconds', 'Maximum MCP tool execution time in seconds before request timeout is returned (clamped: 5-120).'),
            ('mcp_http_max_body_bytes', 'Maximum allowed MCP HTTP request body size in bytes (clamped: 16KB-2MB).'),
        ],
    ),
    (
        'Paths and Storage',
        'Where the application stores database, model files, cache, and logs. Default: ~/.informity. Override via INFORMITY_APP_DATA_DIR.',
        [
            ('app_data_dir', 'Root directory for all application data — database, models, cache, logs, and config.'),
            ('cache_dir', f'Cache directory for downloaded models and document processing artifacts. Default: app_data_dir/{DirNames.CACHE}.'),
            ('db_path', f'SQLite database file path. Default: app_data_dir/{DirNames.DB}/{APP_SLUG}.db.'),
            ('logs_dir', 'Directory where application log files are written.'),
            (
                'models_dir',
                f'Directory for GGUF LLM model files. Default: app_data_dir/{DirNames.MODELS}/{DirNames.LLM}.',
            ),
        ],
    ),
    (
        'Privacy',
        'When full_privacy is true, the app never uses the network; embedding and LLM use cache or local files only.',
        [
            ('full_privacy', 'If true, no network access; all operations stay on this computer. Synced to embedding_offline and llm_local_only when set via UI.'),
            ('embedding_offline', 'If true, load embedding model from cache only (no network). Set automatically when full_privacy is set.'),
            ('llm_local_only', 'If true, load LLM only from models_dir (no downloads). Set automatically when full_privacy is set.'),
        ],
    ),
    (
        'Appearance',
        'Frontend UI customization settings.',
        [
            ('ui_theme', 'Color theme for the app UI. Options: canvas, ember, sage, graphite, onyx.'),
            ('enable_menu_bar_icon', 'When true, show the menu bar icon while the app is running (macOS desktop runtime).'),
        ],
    ),
    (
        'Data Sources',
        'What to scan and what to skip. List options (ignore_patterns, watched_directories) are JSON arrays in env vars.',
        [
            ('follow_symlinks', 'When true, follow symbolic links when scanning directories.'),
            ('exclude_macos_system', 'When true, exclude common macOS system and application data (.DS_Store, Library/Caches, etc.).'),
            ('exclude_developer_data', 'When true, exclude common developer data (.git, node_modules, __pycache__, etc.).'),
            ('ignore_patterns', 'Additional glob patterns for files and directories to skip (JSON array in env).'),
            ('supported_extensions', 'File extensions to index (JSON array in env).'),
            ('watched_directories', 'Directories to scan and index (JSON array of paths in env).'),
        ],
    ),
    (
        'Indexing',
        'How document text is split into chunks before embedding, and indexing performance settings. Affects quality and retrieval.',
        [
            ('chunk_child_size_tokens', 'Child chunk size in tokens used for precise search matching (typically 1–2 sentences).'),
            ('chunk_overlap_tokens', 'Token overlap between consecutive chunks for context continuity.'),
            ('chunk_size_tokens', 'Parent chunk size in tokens. Larger values give the LLM more surrounding context per retrieved passage.'),
            ('enable_ocr_for_images', 'When true, enable OCR fallback for image-only PDFs when regular text extraction fails.'),
            ('entity_extract_acronym', 'When true, extract acronyms into the term dictionary during indexing.'),
            ('entity_extract_location', 'When true, extract location names into the term dictionary during indexing.'),
            ('entity_extract_numeric_id', 'When true, extract numeric identifiers (IDs, codes) into the term dictionary during indexing.'),
            ('entity_extract_organization', 'When true, extract organization names into the term dictionary during indexing.'),
            ('entity_extract_person_name', 'When true, extract person names into the term dictionary during indexing.'),
            ('max_indexable_file_size_mb', 'Maximum file size in MB to index. Files larger than this are ignored. Hard ceiling: 500 MB.'),
            ('scan_file_timeout_seconds', 'Per-file time limit for extraction in seconds. Hard ceiling: 600 seconds.'),
        ],
    ),
    (
        'Web Search',
        'Third-party web search provider configuration. Used by the assistant when web search is enabled.',
        [
            ('linkup_api_key', 'API key for the Linkup web search provider.'),
            ('tavily_api_key', 'API key for the Tavily web search provider.'),
            ('web_search_max_results', 'Maximum number of web search results to include in the assistant context.'),
            ('web_search_primary_provider', 'Primary web search provider: tavily or linkup.'),
            ('web_search_timeout_seconds', 'Timeout in seconds for web search requests.'),
        ],
    ),
    (
        'Embeddings',
        'Embedding model selection. Task prefixes for indexing and search are applied automatically.',
        [
            ('embedding_model', 'Hugging Face model ID for sentence embeddings.'),
        ],
    ),
    (
        'LLM and RAG',
        'Local LLM model, context length, and retrieval-augmented generation tuning.',
        [
            ('adaptive_rag_tuning', 'When true, adapt the number of retrieved chunks based on index size for better answer quality.'),
            ('chat_auto_continue_enabled', 'When true, automatically continue long responses that hit the token limit.'),
            ('chat_auto_continue_default_max_rounds', 'Maximum continuation rounds per response when auto-continue is enabled.'),
            ('chat_auto_continue_hard_cap', 'Hard cap on continuation rounds regardless of other settings.'),
            ('chat_history_messages', 'Number of previous chat messages to include in prompt context. Lower values free up tokens for more document context.'),
            ('chat_history_messages_assistant', 'Assistant-mode history window. Higher values improve conversational continuity in assistant mode.'),
            ('chat_history_messages_researcher', 'Researcher-mode history window. Keep lower to preserve token budget for retrieved document context.'),
            ('default_chat_mode', 'Default chat mode shown in the Chat UI: assistant or researcher.'),
            ('fts5_candidate_limit', 'Maximum extra candidates keyword search (FTS5) can contribute on top of vector search results. Set to 0 to disable keyword search augmentation.'),
            ('llm_context_length', 'Context window size in tokens for the LLM.'),
            ('llm_cpu_threads', 'Max CPU threads for LLM generation (0 = auto; set lower to keep system responsive during chat).'),
            ('llm_hf_repo', 'Hugging Face repository for automatic LLM model downloads (e.g., "unsloth/Qwen3.6-35B-A3B-GGUF").'),
            ('llm_max_tokens', 'Maximum tokens to generate per response.'),
            ('llm_model_id', 'Canonical model identifier used for profile matching and diagnostics metadata.'),
            ('llm_model_filename', 'Filename of the GGUF model file inside the models directory.'),
            ('llm_provider', 'LLM runtime provider: local_gguf (default) or ollama.'),
            ('ollama_base_url', 'Base URL for Ollama API when llm_provider=ollama (default: http://127.0.0.1:11434).'),
            ('ollama_timeout_seconds', 'Request timeout in seconds for Ollama chat requests.'),
            ('llm_temperature', 'Sampling temperature (0 = deterministic; higher = more varied).'),
            # NOTE: rag_context_ratio, rag_max_score, rag_top_k, coverage_top_k are model-specific (ModelProfile, not configurable via env)
            ('rag_minimal_mode', 'When true, use a simplified one-pass retrieval flow instead of the full multi-route pipeline. Faster but less adaptive.'),
            ('rag_query_rewrite_enabled', 'When true, rephrase follow-up questions into self-contained search queries for better retrieval in multi-turn conversations.'),
            ('rag_rerank', 'When true, re-rank vector candidates with a cross-encoder before taking top_k.'),
            ('rag_rerank_candidates', 'Number of candidates to fetch for re-ranking when rag_rerank is true.'),
            ('rag_rerank_coverage', 'When true, also apply reranking to coverage queries (comprehensive lists/tables).'),
            ('rag_reranker_model', 'Hugging Face model ID for the cross-encoder reranker.'),
        ],
    ),
    (
        'Logging',
        'Application logging and debugging options.',
        [
            ('chat_trace_logging', 'When true, write a detailed trace file for each chat message. Useful for troubleshooting and diagnostics analysis.'),
            ('chat_trace_redaction_mode', 'Trace payload redaction level: off (full payload), minimal (truncate sensitive fields), strict (redact with metadata only).'),
            ('diagnostics_profile', 'Diagnostics profile preset: standard (low overhead), troubleshooting (richer diagnostics), custom (manual override).'),
            ('enable_raw_output_control', 'When true, show a toggle to view the raw unprocessed model output for each response. Useful for debugging reasoning models.'),
            ('log_level', 'Application log verbosity: debug, info, warning, or error.'),
        ],
    ),
    (
        'Diagnostics',
        'Diagnostics evaluation pipeline settings for quality analysis and self-improvement.',
        [
            ('diagnostics_dir', 'Directory for diagnostics data (quality evaluation runs, traces, reports).'),
        ],
    ),
]

_RUNTIME_ENV_VARS: list[tuple[str, str]] = [
    (
        'INFORMITY_TAURI_SESSION_TOKEN',
        'Desktop runtime session token for local API authorization (managed by the desktop shell).',
    ),
    (
        'INFORMITY_MCP_TOKEN',
        'Optional MCP HTTP bearer-token override. If set, this value is used instead of the saved MCP access token.',
    ),
]
_SENSITIVE_ENV_VALUE_MARKER = '(set, redacted)'
_SENSITIVE_ENV_NAME_HINTS = ('TOKEN', 'SECRET', 'PASSWORD', 'KEY')


def _should_redact_setting_field(field: str) -> bool:
    upper = str(field or '').upper()
    return any(hint in upper for hint in _SENSITIVE_ENV_NAME_HINTS)


def _describe_unmapped_field(field: str) -> str:
    # Fallback description for Settings fields not explicitly documented in _GROUPS.
    label = field.replace('_', ' ').strip()
    return f'Advanced setting: {label}.'


def _format_runtime_env_default(name: str) -> str:
    # Redact runtime secrets while still indicating presence.
    raw = str(os.environ.get(name, '')).strip()
    if not raw:
        return ''
    upper_name = str(name or '').upper()
    if any(hint in upper_name for hint in _SENSITIVE_ENV_NAME_HINTS):
        return _SENSITIVE_ENV_VALUE_MARKER
    return raw


# Prefixes used to bucket undocumented Settings fields into focused advanced groups.
_RETRIEVAL_TUNING_PREFIXES = (
    'adaptive_top_k_',
    'classification_confidence_',
    'retrieval_',
    'rag_minimal_',
)
_TERM_DICTIONARY_PREFIX    = 'term_dictionary_'
_INTERNAL_CONSTANTS_PREFIXES = (
    'extraction_',
    'scan_stale_',
    'scan_timeout_policy',
)
_SUPPORTED_EXTENSIONS_CANONICAL_ORDER: tuple[str, ...] = tuple(
    ext
    for option in get_file_type_options()
    for ext in option.get('extensions', [])
)


def _normalize_supported_extensions_display(value: object) -> object:
    if not isinstance(value, list):
        return value
    canonical_index = {ext: idx for idx, ext in enumerate(_SUPPORTED_EXTENSIONS_CANONICAL_ORDER)}
    normalized: list[str] = []
    seen: set[str] = set()
    for item in value:
        ext = str(item or '').strip().lower()
        if not ext or ext in seen:
            continue
        seen.add(ext)
        normalized.append(ext)
    return sorted(normalized, key=lambda ext: (canonical_index.get(ext, 10_000), ext))


def get_env_vars_response(settings: object) -> EnvVarsResponse:
    # Build the env vars response with actual current values from settings.
    # Paths are shown relative to the application directory (cwd).
    # Variables are sorted alphabetically by env name within each group.
    app_dir = Path.cwd()
    groups: list[EnvVarGroup] = []
    documented_fields: set[str] = set()
    for title, description, variables in _GROUPS:
        items = []
        for field, desc in sorted(variables, key=lambda x: _env_name(x[0])):
            documented_fields.add(field)
            try:
                value = getattr(settings, field)
                if field == 'supported_extensions':
                    value = _normalize_supported_extensions_display(value)
                current_value = _format_value(value, app_dir)
                if _should_redact_setting_field(field):
                    current_value = _SENSITIVE_ENV_VALUE_MARKER if str(current_value).strip() else ''
            except (AttributeError, TypeError):
                current_value = '(unset)'
            items.append(EnvVarItem(name=_env_name(field), current_value=current_value, description=desc))
        groups.append(EnvVarGroup(title=title, description=description, variables=items))

    # Fields not in _GROUPS are split into three focused advanced groups
    # rather than one catch-all bucket.

    model_fields = getattr(Settings, 'model_fields', {})
    all_settings_fields = set(model_fields.keys())
    missing_fields = sorted(all_settings_fields - documented_fields)

    retrieval_items:   list[EnvVarItem] = []
    term_dict_items:   list[EnvVarItem] = []
    internal_items:    list[EnvVarItem] = []
    leftover_items:    list[EnvVarItem] = []

    for field in missing_fields:
        try:
            value        = getattr(settings, field)
            current_display = _format_value(value, app_dir)
            if _should_redact_setting_field(field):
                current_display = _SENSITIVE_ENV_VALUE_MARKER if str(current_display).strip() else ''
        except (AttributeError, TypeError):
            current_display = '(unset)'
        item = EnvVarItem(
            name          = _env_name(field),
            current_value = current_display,
            description   = _describe_unmapped_field(field),
        )
        if any(field.startswith(p) for p in _RETRIEVAL_TUNING_PREFIXES):
            retrieval_items.append(item)
        elif field.startswith(_TERM_DICTIONARY_PREFIX):
            term_dict_items.append(item)
        elif any(field.startswith(p) for p in _INTERNAL_CONSTANTS_PREFIXES):
            internal_items.append(item)
        else:
            leftover_items.append(item)

    if retrieval_items:
        groups.append(EnvVarGroup(
            title       = 'Retrieval Tuning',
            description = (
                'Advanced constants for retrieval quality, classification confidence, '
                'adaptive top-k, and relevance thresholds. Leave at defaults unless tuning RAG behavior.'
            ),
            variables   = retrieval_items,
        ))
    if term_dict_items:
        groups.append(EnvVarGroup(
            title       = 'Term Dictionary',
            description = (
                'Controls for the term dictionary builder: entity extraction confidence, '
                'expansion limits, quality gates, and routing flags.'
            ),
            variables   = term_dict_items,
        ))
    if internal_items or leftover_items:
        groups.append(EnvVarGroup(
            title       = 'Internal Constants',
            description = (
                'Internal runtime defaults for numeric extraction guards, fit-to-budget rollout, '
                'and scan policy internals. Not intended for regular configuration.'
            ),
            variables   = internal_items + leftover_items,
        ))

    runtime_items = [
        EnvVarItem(
            name          = name,
            current_value = _format_runtime_env_default(name),
            description   = desc,
        )
        for name, desc in sorted(_RUNTIME_ENV_VARS, key=lambda x: x[0])
    ]
    groups.append(
        EnvVarGroup(
            title='Runtime Environment',
            description='Runtime-only environment variables used by launch wrappers and desktop session plumbing.',
            variables=runtime_items,
        ),
    )
    return EnvVarsResponse(groups=groups)
