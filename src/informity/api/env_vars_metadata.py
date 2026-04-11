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
            ('dev_reload', 'When true, uvicorn runs with --reload (dev only). Leave false for production.'),
            ('api_docs_enabled', 'Override API docs exposure. If unset, docs are enabled only when dev_reload is true; set true/false to force behavior.'),
            ('cpu_priority_nice', 'Lower process priority at startup (0 = off; 1-19 on POSIX, higher means lower priority).'),
        ],
    ),
    (
        'Paths and Storage',
        'Where the application stores database, model files, cache, and logs. Default: ~/.informity. Override via INFORMITY_APP_DATA_DIR.',
        [
            ('app_data_dir', 'Root directory for all app data (DB, models, cache, logs, config). Default: ~/.informity.'),
            ('cache_dir', f'Unified cache root for Hugging Face/docling artifacts. Default: app_data_dir/{DirNames.CACHE}.'),
            ('db_path', f'SQLite database file path (default: app_data_dir/{DirNames.DB}/{APP_SLUG}.db).'),
            ('logs_dir', 'Directory for log files.'),
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
            ('ui_theme', 'Color theme for the app UI: gray, purple, blue, green, orange, mono. Applied via data-theme on <html>.'),
            ('enable_menu_bar_icon', 'When true, show the menu bar icon while the app is running (macOS desktop runtime).'),
        ],
    ),
    (
        'Data Sources',
        'What to scan and what to skip. List options (ignore_patterns, watched_directories) are JSON arrays in env vars.',
        [
            ('follow_symlinks', 'Whether to follow symbolic links when scanning directories.'),
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
            ('chunk_child_size_tokens', 'Child chunk size in tokens (for precise search matching, typically 1-2 sentences). Smaller chunks improve retrieval precision.'),
            ('chunk_filter_header_only', 'When true, filter out chunks that contain only headers/separators without meaningful content. Quality heuristic to avoid indexing empty table headers.'),
            ('chunk_filter_header_ratio', 'Threshold ratio (0.0-1.0) for header-only detection. Chunks with header/separator lines exceeding this ratio are filtered.'),
            ('chunk_filter_min_content_chars', 'Minimum content length (characters) to avoid filtering. Chunks shorter than this are never filtered regardless of header ratio.'),
            ('chunk_filter_min_content_lines', 'Minimum content lines to avoid filtering. Chunks with fewer lines than this are never filtered regardless of header ratio.'),
            ('chunk_overlap_tokens', 'Token overlap between consecutive chunks for context continuity.'),
            ('chunk_size_tokens', 'Parent chunk size in tokens (for context windows). Larger chunks provide more context for LLM generation.'),
            ('embedding_batch_size', 'Number of texts to embed in one batch; higher uses more memory. Trade off indexing speed against keeping your Mac responsive.'),
            ('embedding_max_threads', 'Max CPU threads for embedding model (0 = auto; set lower to keep system responsive).'),
            ('scan_hash_pool', 'Hash executor for scan crawling: thread (default) or process.'),
            ('scan_hash_workers', 'Hash worker count for scan crawling (0 = auto).'),
            ('scan_hash_max_file_size_bytes', 'Maximum file size (bytes) eligible for scan-time SHA-256 hashing. Oversized files are skipped.'),
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
            ('chat_history_messages', 'Number of previous chat messages to include in prompt context. Lower values free up tokens for more document context.'),
            ('chat_history_messages_assistant', 'Assistant-mode history window. Higher values improve conversational continuity in assistant mode.'),
            ('chat_history_messages_researcher', 'Researcher-mode history window. Keep lower to preserve token budget for retrieved document context.'),
            ('default_chat_mode', 'Default chat mode shown in the Chat UI: assistant or researcher.'),
            ('llm_context_length', 'Context window size in tokens for the LLM.'),
            ('llm_cpu_threads', 'Max CPU threads for llama-cpp generation (0 = auto; set lower to keep system responsive during chat).'),
            ('llm_hf_repo', 'Hugging Face repository for automatic LLM model downloads (e.g., "Qwen/Qwen3.5-35B-A3B-GGUF").'),
            ('llm_max_tokens', 'Maximum tokens to generate per response.'),
            ('llm_model_filename', 'GGUF filename in models_dir.'),
            ('llm_temperature', 'Sampling temperature (0 = deterministic; higher = more varied).'),
            # NOTE: rag_context_ratio, rag_max_score, rag_top_k, rag_coverage_top_k are model-specific (ModelProfile, not configurable via env)
            ('adaptive_rag_tuning', 'When true, adapt retrieval top-k based on corpus size (file count, parent chunk count). Default true.'),
            ('rag_query_rewrite_enabled', 'Enable researcher follow-up retrieval query rewriting for referential questions.'),
            ('rag_query_rewrite_max_history_messages', 'Maximum recent history messages considered when building retrieval rewrite context.'),
            ('rag_query_rewrite_max_chars_per_turn', 'Maximum characters taken from each history turn for retrieval rewrite context.'),
            ('rag_query_rewrite_max_query_chars', 'Maximum characters allowed in the final rewritten retrieval query.'),
            ('rag_rerank', 'When true, re-rank vector candidates with a cross-encoder before taking top_k.'),
            ('rag_rerank_coverage', 'When true, also apply reranking to coverage queries (comprehensive lists/tables).'),
            ('rag_reranker_model', 'Hugging Face model ID for the cross-encoder reranker.'),
            ('rag_rerank_candidates', 'Number of candidates to fetch for re-ranking when rag_rerank is true.'),
        ],
    ),
    (
        'Logging',
        'Application logging and debugging options.',
        [
            ('log_level', 'Application log level: debug, info, warning, error. Default info to reduce noise.'),
            ('chat_trace_logging', 'When true, write a per-chat trace log (chat_{chat_id}.json) for each chat message. Used for troubleshooting and LLM-assisted analysis.'),
        ],
    ),
    (
        'Diagnostics',
        'Diagnostics evaluation pipeline settings for quality analysis and self-improvement.',
        [
            ('diagnostics_dir', 'Directory for diagnostics data (quality evaluation runs, traces, reports).'),
            ('diagnostics_llm_analysis_enabled', 'When true, use local LLM to enhance root cause analysis in diagnostics pipeline. Default false (opt-in feature).'),
            ('diagnostics_llm_max_issues_per_run', 'Maximum number of issues to analyze per run (limits analysis scope to prevent excessive processing time).'),
            ('diagnostics_llm_model_filename', 'GGUF filename in diagnostics_models_dir for LLM-powered analysis (default: DeepSeek R1 optimized for analysis tasks).'),
            ('diagnostics_llm_timeout_seconds', 'Maximum seconds for LLM inference during diagnostics analysis. Generous default so analysis can produce full results.'),
            (
                'diagnostics_models_dir',
                f'Directory for diagnostics LLM model files '
                f'(default: {{app_data_dir}}/{DirNames.MODELS}/{DirNames.DIAGNOSTICS_LLM}). '
                f'Separate from chat and classifier models.',
            ),
        ],
    ),
]

_RUNTIME_ENV_VARS: list[tuple[str, str]] = [
    (
        'INFORMITY_REPO_ROOT',
        'Repository root override for diagnostics paths and local tooling resolution.',
    ),
    (
        'INFORMITY_SUPPRESS_CONSOLE_LOGS',
        'When set to 1, suppress startup/config console logs in CLI contexts.',
    ),
    (
        'INFORMITY_TAURI_SESSION_TOKEN',
        'Desktop runtime session token for local API authorization (managed by the desktop shell).',
    ),
]
_SENSITIVE_ENV_VALUE_MARKER = '***set***'
_SENSITIVE_ENV_NAME_HINTS = ('TOKEN', 'SECRET', 'PASSWORD', 'KEY')


def _describe_unmapped_field(field: str) -> str:
    # Fallback description for Settings fields not explicitly documented in _GROUPS.
    label = field.replace('_', ' ').strip()
    return (
        f'Advanced setting: {label}. '
        f'Configurable via {_env_name(field)}.'
    )


def _format_runtime_env_default(name: str) -> str:
    # Redact runtime secrets while still indicating presence.
    raw = str(os.environ.get(name, '')).strip()
    if not raw:
        return ''
    upper_name = str(name or '').upper()
    if any(hint in upper_name for hint in _SENSITIVE_ENV_NAME_HINTS):
        return _SENSITIVE_ENV_VALUE_MARKER
    return raw


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
                default_display = _format_value(value, app_dir)
            except (AttributeError, TypeError):
                default_display = '(unset)'
            items.append(EnvVarItem(name=_env_name(field), default=default_display, description=desc))
        groups.append(EnvVarGroup(title=title, description=description, variables=items))

    # Ensure env-vars metadata stays in sync with Settings by auto-including any
    # Settings fields not explicitly curated above.
    model_fields = getattr(Settings, 'model_fields', {})
    all_settings_fields = set(model_fields.keys())
    missing_fields = sorted(all_settings_fields - documented_fields)
    if missing_fields:
        advanced_items: list[EnvVarItem] = []
        for field in missing_fields:
            try:
                value = getattr(settings, field)
                current_display = _format_value(value, app_dir)
            except (AttributeError, TypeError):
                current_display = '(unset)'
            advanced_items.append(
                EnvVarItem(
                    name=_env_name(field),
                    default=current_display,
                    description=_describe_unmapped_field(field),
                ),
            )
        groups.append(
            EnvVarGroup(
                title='Advanced and Internal',
                description='Additional Settings fields that are env-configurable but not shown in primary categories.',
                variables=advanced_items,
            ),
        )

    runtime_items = [
        EnvVarItem(
            name=name,
            default=_format_runtime_env_default(name),
            description=desc,
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
