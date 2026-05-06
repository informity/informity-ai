# ==============================================================================
# Informity AI — Configuration Module
# Loads settings with the following priority (highest wins):
#   1. Persisted config.json (written by the Settings API) — for keys present
#   2. Environment variables (INFORMITY_*)
#   3. Hard-coded defaults below
# Config.json wins over env vars so UI state survives restarts.
#
# Default app data is stored in ~/.informity.
# Override with INFORMITY_APP_DATA_DIR to use a custom location.
# ==============================================================================

import json
import os
from pathlib import Path
from typing import Literal

import structlog
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings

from informity.timeout_policy import ScopedTimeoutPolicy, default_scoped_timeout_policy
from informity.utils.directory_utils import ensure_directories, ensure_private_file
from informity.utils.json_utils import serialize_config
from informity.utils.path_utils import normalize_path, normalize_paths

# ==============================================================================
# Logger
# ==============================================================================

log = structlog.get_logger(__name__)

# ==============================================================================
# Application identity (single source of truth for branding and file names)
# ==============================================================================

APP_SLUG        = 'informity'      # Used for db filename (e.g. informity.db), log filename (e.g. informity.log)
APP_DISPLAY_NAME = 'Informity AI'  # User-facing product name (UI, prompts, API docs)
APP_DATA_DIRNAME = '.informity'    # Default app data directory name under user home

# ==============================================================================
# Defaults
# ==============================================================================

# All platforms default to ~/.informity.
_DEFAULT_APP_DATA_DIR = Path.home() / APP_DATA_DIRNAME

# Default model for reset-to-factory and first load: Qwen3.6 35B A3B.
_DEFAULT_LLM_MODEL_FILENAME = 'Qwen3.6-35B-A3B-UD-Q4_K_M.gguf'

# Default embedding model (sentence-transformers)
_DEFAULT_EMBEDDING_MODEL = 'nomic-ai/nomic-embed-text-v1.5'

# Default reranker model (sentence-transformers cross-encoder)
_DEFAULT_RERANKER_MODEL = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
# Public alias for schema/default consumers.
DEFAULT_RERANKER_MODEL = _DEFAULT_RERANKER_MODEL

# Default Hugging Face repository for LLM model downloads
_DEFAULT_LLM_HF_REPO = 'unsloth/Qwen3.6-35B-A3B-GGUF'

# Default auto-continuation policy for long responses.
_DEFAULT_CHAT_AUTO_CONTINUE_PROMPT = (
    'Continue with the remaining sections from your last answer. '
    'Keep the same structure and avoid repeating completed sections.'
)
LOG_LEVEL_ALLOWED_VALUES: tuple[str, ...] = ('debug', 'info', 'warning', 'error')
UI_THEME_ALLOWED_VALUES: tuple[str, ...] = ('gray', 'purple', 'blue', 'green', 'orange', 'mono')
_DEFAULT_LOG_LEVEL = 'info'
_DEFAULT_UI_THEME = 'mono'


# ==============================================================================
# Directory Name Constants
# ==============================================================================

class DirNames:
    """
    Directory name constants for application directory structure.
    Single source of truth for all directory names used throughout the application.
    """
    # User data directories (under app_data_dir)
    TOOLS = 'tools'
    DB = 'db'
    LOGS = 'logs'
    DIAGNOSTICS = 'diagnostics'
    MODELS = 'models'
    CHAT_LOGS = 'chats'  # Per-message trace logs: app_data_dir/chats/{chat_id}/{message_id}.json

    # Unified cache directory (under app_data_dir, not committed to repo)
    CACHE = 'cache'

    # Model/cache subdirectories
    LLM = 'llm'  # LLM models (*.gguf files)
    HUGGINGFACE = 'huggingface'  # HuggingFace cache under cache/huggingface/
    HUB = 'hub'  # HuggingFace hub cache under cache/huggingface/hub/
    DOCLING = 'docling'  # Docling models under cache/docling/ (flat, docling creates its own structure inside)

    # Diagnostics subdirectories (under diagnostics_dir)
    RUNS = 'runs'
    TRACES = 'traces'
    QUERIES = 'queries'
    RESULTS = 'results'
    CHATS = 'chats'
    REPORTS = 'reports'
    EVALUATIONS = 'evaluations'


# ==============================================================================
# Diagnostics Pipeline Constants
# ==============================================================================

class DiagnosticsConstants:
    """
    Constants for diagnostics evaluation pipeline (run IDs, chat IDs, query IDs).
    Single source of truth for all diagnostics pipeline naming patterns.
    """
    # Run ID prefix (e.g., "run-20260214-2009")
    RUN_ID_PREFIX = 'run-'

    # Chat ID prefix for evaluation/trace chats (e.g., "trace-1-doc-totals-ModelName")
    EVAL_CHAT_ID_PREFIX = 'trace-'


# ==============================================================================
# Default Supported Extensions
# ==============================================================================

def _get_default_supported_extensions() -> list[str]:
    """
    Derive default supported extensions from extractor registry.
    This ensures that the default list always matches what extractors are available.

    Note: .json, .yaml, .yml, .toml are excluded by default even if extractors exist;
    user can enable them in Settings. PDF (.pdf) is now included by default since
    docling provides reliable extraction.

    Returns:
        List of extensions that have extractors, excluding data files
        (e.g., ['.pdf', '.csv', '.docx', '.html', ...])
    """
    # Extensions to exclude from defaults (even if extractors exist)
    # PDF is now included by default since docling provides reliable extraction
    excluded_by_default = {'.json', '.yaml', '.yml', '.toml'}

    # Import here to avoid circular imports (extractors may import config).
    # Keep fallback data-driven (file_types), never hardcoded extension lists.
    try:
        from informity.scanner.extractors.base import get_all_extractable_extensions
        all_extensions = get_all_extractable_extensions()
        # Filter out extensions that should be excluded by default
        return [ext for ext in all_extensions if ext not in excluded_by_default]
    except ImportError:
        from informity.file_types import get_file_type_options

        options = get_file_type_options()
        derived: list[str] = []
        seen: set[str] = set()
        for option in options:
            for raw_ext in option.get('extensions', []):
                ext = str(raw_ext).strip().lower()
                if not ext or ext in excluded_by_default or ext in seen:
                    continue
                seen.add(ext)
                derived.append(ext)
        return derived

# Preset pattern lists for "Exclude common macOS system and application data" and
# "Exclude common developer data". Used when the corresponding settings are enabled.
# Matching is by path component (e.g. "Library" skips any path with that segment).
EXCLUDE_MACOS_SYSTEM_PATTERNS: tuple[str, ...] = (
    '*.app',
    '*.dmg',
    '*.icloud',
    '*.ipa',
    '*.pkg',
    '*.webloc',
    '.DocumentRevisions-V100',
    '.DS_Store',
    '.Spotlight-V100',
    '.TemporaryItems',
    '.VolumeIcon.icns',
    '.apdisk',
    '.fseventsd',
    '.localized',
    '.Trash',
    'Library',
)
EXCLUDE_DEVELOPER_PATTERNS: tuple[str, ...] = (
    '.env',
    '.git',
    '.mypy_cache',
    '.next',
    '.pytest_cache',
    '.tox',
    '.venv',
    '__pycache__',
    'build',
    'dist',
    'node_modules',
    'venv',
)


# ==============================================================================
# Config file loader
# ==============================================================================

def _config_path_for_loader() -> Path:
    # Config file path using same resolution as _load_config_file_values.
    raw_dir = os.environ.get('INFORMITY_APP_DATA_DIR', '')
    app_data_dir = Path(raw_dir) if raw_dir else _DEFAULT_APP_DATA_DIR
    app_data_dir = normalize_path(app_data_dir, expand_user=True)
    return app_data_dir / 'config.json'


def _load_config_file_values() -> dict:
    # Determine the app data dir (env var takes priority over default)
    # so we know where to find config.json before Settings is instantiated.
    config_path = _config_path_for_loader()
    if not config_path.exists():
        return {}

    try:
        data = json.loads(config_path.read_text(encoding='utf-8'))
        if not isinstance(data, dict):
            return {}
        # Resolve watched_directories to absolute paths so scans use consistent paths
        if 'watched_directories' in data and isinstance(data['watched_directories'], list):
            normalized_paths = normalize_paths(
                [p for p in data['watched_directories'] if isinstance(p, str)],
                expand_user=True
            )
            data['watched_directories'] = [str(p) for p in normalized_paths]
        # Guard against invalid theme values in persisted config.json.
        raw_theme = data.get('ui_theme')
        if raw_theme is not None:
            if isinstance(raw_theme, str) and raw_theme in UI_THEME_ALLOWED_VALUES:
                pass
            else:
                data['ui_theme'] = _DEFAULT_UI_THEME
        return data
    except (json.JSONDecodeError, OSError):
        return {}


# ==============================================================================
# Settings
# ==============================================================================

class Settings(BaseSettings):
    # -- Paths ----------------------------------------------------------------
    app_data_dir:  Path       = _DEFAULT_APP_DATA_DIR
    cache_dir:     Path | None = Field(default=None)   # Unified cache root; default app_data_dir/DirNames.CACHE. Override via INFORMITY_CACHE_DIR.
    db_path:       Path | None = Field(default=None)   # Computed: app_data_dir / DirNames.DB / f'{APP_SLUG}.db'
    models_dir:    Path | None = Field(default=None)   # Computed: desktop -> app_data_dir/DirNames.MODELS/DirNames.LLM; otherwise cache_dir/DirNames.LLM
    logs_dir:      Path | None = Field(default=None)   # Computed: app_data_dir / DirNames.LOGS
    diagnostics_dir: Path | None = Field(default=None)  # Computed: app_data_dir / DirNames.DIAGNOSTICS

    # -- Scanner --------------------------------------------------------------
    watched_directories:       list[Path] = Field(default_factory=list)
    ignore_patterns:           list[str]  = Field(default_factory=list)  # Custom only; presets from checkboxes
    exclude_macos_system:      bool       = True   # When True, apply EXCLUDE_MACOS_SYSTEM_PATTERNS
    exclude_developer_data:    bool       = True   # When True, apply EXCLUDE_DEVELOPER_PATTERNS
    supported_extensions: list[str] = Field(default_factory=_get_default_supported_extensions)
    # Default derives from extractor registry (all extensions that have extractors).
    # User can customize via Settings UI to enable/disable specific file types.
    # Note: .json, .yaml, .yml, .toml may be excluded by default even if extractors exist;
    # user can enable them in Settings. PDF (.pdf) is included by default since docling
    # provides reliable extraction.
    follow_symlinks:  bool = False
    source_scopes_enabled: dict[str, bool] = Field(
        default_factory=lambda: {
            'filesystem:file': True,
            'mail.apple:mail': False,
            'mail.outlook:mail': False,
        }
    )
    # User-facing timeout cap (seconds) for single-item processing.
    # Runtime timeout policy uses this value as the default/override hard cap.
    scan_file_timeout_seconds: int = 600
    # Shared per-item timeout policy by source scope.
    scan_timeout_policy: ScopedTimeoutPolicy = Field(default_factory=default_scoped_timeout_policy)
    # Running-scan stale detection threshold (seconds) used when a new scan/rebuild
    # request checks for already-running operations.
    scan_stale_threshold_seconds: int = 300
    # Hash executor mode for scan crawling: thread (default) or process.
    # Thread mode avoids process spawn overhead and is usually better for mixed I/O+CPU hashing.
    scan_hash_pool: Literal['thread', 'process'] = 'thread'
    # Hash worker count for crawl hashing. 0 = auto (min(4, max(2, cpu_count // 3))).
    scan_hash_workers: int = 0
    # Max file size (bytes) for scan-time SHA-256 hashing. Oversized files are skipped.
    # This prevents long scan stalls on very large binaries and media blobs.
    scan_hash_max_file_size_bytes: int = 500 * 1024 * 1024  # 500 MB

    # -- Indexer --------------------------------------------------------------
    chunk_size_tokens:    int = 512  # Parent chunk size (for context windows)
    chunk_overlap_tokens: int = 60
    chunk_child_size_tokens: int = 150  # Child chunk size (for precise search matching, 1-2 sentences)
    # Header-only chunk filter: quality heuristic to prevent indexing chunks that contain
    # only table/form headers without body content. Some documents (e.g., form templates,
    # empty tables) genuinely contain header-only structures that provide little value for RAG.
    chunk_filter_header_only: bool = True  # Enable/disable header-only chunk filtering
    chunk_filter_header_ratio: float = 0.7  # Threshold: chunks with >70% header/separator lines are considered header-only
    chunk_filter_min_content_chars: int = 300  # Minimum content length (chars) to avoid filtering
    chunk_filter_min_content_lines: int = 3  # Minimum content lines to avoid filtering
    embedding_model:      str = _DEFAULT_EMBEDDING_MODEL
    embedding_batch_size: int = 32
    # When True, load embedding model from local cache only (no Hugging Face requests).
    # Synced from full_privacy when that setting is updated via the UI.
    embedding_offline:    bool = True
    # Maximum CPU threads for the embedding model (PyTorch + tokenizers).
    # Set to 0 for automatic (uses all cores — will peg CPU at 100%).
    # Default: 6 (~50% of a 12-core M3 Pro) for a balance between
    # indexing speed and keeping the system responsive during scans.
    embedding_max_threads: int = 6
    # CPU threads for xllamacpp (separate from embedding threads —
    # xllamacpp uses its own threading and is not controlled by OMP_NUM_THREADS).
    # Set to 0 for automatic. Default: 8 to reduce high-TTFT regressions
    # observed in long diagnostics suites while keeping CPU headroom.
    llm_cpu_threads: int = 8
    # When True, enable OCR (Optical Character Recognition) for image-only PDFs
    # when regular text extraction fails. OCR is slower but can extract text from
    # scanned documents, photographed pages, and PDFs with embedded images.
    # Default: True so scanned/image PDFs work out of the box via fallback OCR.
    enable_ocr_for_images: bool = True
    # Maximum file size (MB) eligible for indexing. Files larger than this are ignored.
    # Hard ceiling: 500 MB. Increase if you have large documents and sufficient RAM.
    max_indexable_file_size_mb: int = 100

    # -- Privacy ------------------------------------------------------------------
    # When True, no network access: embedding and LLM use cache/local only (fully local).
    # When False, network is allowed (e.g. for model downloads). Synced to embedding_offline and llm_local_only.
    full_privacy:         bool = True
    tavily_api_key: str = ''
    linkup_api_key: str = ''
    web_search_primary_provider: Literal['tavily', 'linkup'] = 'tavily'
    web_search_max_results: int = 5
    web_search_timeout_seconds: float = 8.0

    # -- LLM ------------------------------------------------------------------
    # When True, load LLM only from models_dir; never download from the network.
    # Synced from full_privacy when that setting is updated via the UI.
    llm_local_only:       bool = True
    llm_model_filename:   str  = _DEFAULT_LLM_MODEL_FILENAME  # Default: Qwen3.6 35B A3B
    llm_hf_repo:          str  = _DEFAULT_LLM_HF_REPO  # Hugging Face repo for automatic model downloads
    llm_context_length:   int  = 16384  # 16K is ample (10K chunks + 4K prompt/history + 2K gen); prevents over-assembly
    llm_max_tokens:     int   = 2048
    llm_temperature:    float = 0.2     # Low for factual extraction; avoids determinism-induced loops
    # Retrieval top-k: model-profile-only (ModelProfile.rag_top_k, coverage_top_k).
    # Use model_adapter.get_retrieval_top_k(query_type). No config/env.
    # NOTE: rag_max_score and rag_context_ratio are now model-specific (moved to ModelProfile).
    # Each model has optimal values tuned for its capabilities.
    # Minimal RAG mode (single-path runtime): when enabled, bypasses route/profile
    # fallback lattice and uses one retrieval call + answerability decision.
    rag_minimal_mode: bool = True
    # When True, adapt retrieval top-k based on corpus size (file count, parent chunk count).
    # When False, always use model profile base values.
    adaptive_rag_tuning:  bool  = True
    # When True, re-rank vector search candidates with a cross-encoder (query, chunk) before taking top_k.
    # Disable to trade result quality for speed (skips 100–300ms cross-encoder pass).
    rag_rerank:          bool  = True
    # When True, also apply reranking to coverage queries (comprehensive lists/tables).
    # Only evaluated when rag_rerank is True. Enabled by default: reranking
    # prevents irrelevant files from polluting coverage answers.
    rag_rerank_coverage: bool  = True
    # sentence-transformers model ID for the cross-encoder reranker (default: cross-encoder/ms-marco-MiniLM-L-6-v2).
    rag_reranker_model:   str   = _DEFAULT_RERANKER_MODEL
    # Number of candidates to fetch for re-ranking when rag_rerank is True.
    # Reduced from 35 to 25 for speed (saves ~30-50ms); reranker is most effective on top 20-30.
    rag_rerank_candidates: int = 25
    # Adaptive top-k formula constants (app compliance: no magic numbers). See adaptive-tuning.md.
    adaptive_top_k_focused_small_threshold:  int   = 500   # Parent chunks below this: use small-corpus formula
    adaptive_top_k_focused_small_cap:        int   = 12    # Max top-k for small corpus (focused)
    adaptive_top_k_focused_base:             int   = 8     # Log curve base
    adaptive_top_k_focused_scale:            int   = 3     # Log curve scale
    adaptive_top_k_focused_max:              int   = 25    # Max top-k for focused (large corpus)
    adaptive_top_k_coverage_ratio:           float = 0.25   # Target file coverage (20–25%)
    adaptive_top_k_coverage_max:             int   = 30    # Max top-k for coverage (timeout prevention)
    adaptive_top_k_staleness_hours:          int   = 24    # Recompute if cache older than this
    adaptive_top_k_staleness_delta:          float = 0.2    # Recompute if chunk delta > 20%
    # Classification confidence band thresholds.
    # confidence_band='high' when confidence >= classification_confidence_high_threshold.
    # confidence_band='medium' when confidence >= classification_confidence_medium_threshold.
    # confidence_band='low' otherwise. These gate route selection and multi-step retrieval.
    # Last calibration: 2026-03-18 (provenance flags item; values derived from original three-tier constants).
    classification_confidence_high_threshold: float = 0.80
    classification_confidence_medium_threshold: float = 0.55
    # Retrieval quality gates (runtime policy; avoid hardcoded thresholds in handlers).
    retrieval_relevance_threshold_focused:  float = 0.03
    retrieval_relevance_threshold_coverage: float = 0.02
    # Minimal RAG answerability thresholds. Applied only when rag_minimal_mode=true.
    rag_minimal_answerability_threshold_focused: float = 0.0
    rag_minimal_answerability_threshold_coverage: float = 0.0
    rag_minimal_min_chunks_focused: int = 1
    rag_minimal_min_chunks_coverage: int = 1
    # Coverage fallback hard floor (controls whether evidence-floor override
    # requires score to clear `retrieval_coverage_evidence_floor_min_score`):
    # - Disabled by default until thresholds are calibrated.
    retrieval_coverage_evidence_floor_hard_floor_enabled: bool = False
    retrieval_coverage_evidence_floor_min_score: float = 0.05
    # Researcher follow-up retrieval query rewrite controls.
    # When enabled, referential follow-up questions are expanded with recent
    # chat context before vector retrieval so retrieval remains self-contained.
    rag_query_rewrite_enabled: bool = True
    rag_query_rewrite_max_history_messages: int = 3
    rag_query_rewrite_max_chars_per_turn: int = 260
    rag_query_rewrite_max_query_chars: int = 900
    # Maximum net-new chunk candidates FTS5 keyword search may add to the vector search pool
    # before reranking (focused queries only). FTS5 augmentation recovers exact-match pool
    # misses (e.g., document title queries not semantically close to any indexed vector).
    # Cap prevents high-frequency terms from swamping the reranker. Set to 0 to disable.
    fts5_candidate_limit: int = 10
    # Retention window for continuation pass artifacts.
    # Artifacts older than this are pruned on insert and on startup.
    # Set to 0 to disable pruning.
    continuation_artifact_retention_days: int = 30
    # Term dictionary runtime expansion controls (corpus-agnostic, deterministic).
    term_dictionary_enabled: bool = True
    term_dictionary_routing_enabled: bool = False
    term_dictionary_high_confidence: float = 0.85
    term_dictionary_medium_confidence: float = 0.65
    term_dictionary_max_embed_expansions: int = 6
    term_dictionary_max_fts_expansions: int = 10
    term_dictionary_max_fuzzy_expansions: int = 2
    term_dictionary_max_fuzzy_per_canonical: int = 1
    term_dictionary_max_routing_expansions: int = 4
    # Term dictionary post-index builder controls.
    term_dictionary_build_enabled: bool = True
    term_dictionary_build_batch_size: int = 500
    entity_extract_acronym: bool = True
    entity_extract_person_name: bool = False
    entity_extract_organization: bool = False
    entity_extract_location: bool = False
    entity_extract_numeric_id: bool = False
    term_dictionary_quality_gate_enabled: bool = True
    term_dictionary_quality_noise_rate_threshold: float = 0.55
    term_dictionary_quality_min_candidates_for_gate: int = 20
    # Number of previous messages to include in prompt context.
    # Lower values free up tokens for more document context, improving answer quality.
    # Higher values maintain better conversation continuity for follow-up questions.
    chat_history_messages:   int   = 5
    # Mode-specific history window overrides.
    # Assistant mode can keep a larger conversational window because it does not
    # include corpus retrieval chunks in prompt context.
    # Researcher mode should remain conservative to preserve retrieval context budget.
    chat_history_messages_assistant: int = 12
    chat_history_messages_researcher: int = 5
    # Chat-summary mode controls (chat-history recap intent).
    # Direct mode includes all turns when chat length is within this message count.
    # Above the threshold, runtime switches to hierarchical summarize-then-merge.
    chat_summary_direct_max_messages: int = 40
    chat_summary_chunk_messages: int = 20
    chat_summary_max_chunks: int = 12
    chat_summary_max_chars_per_message: int = 900
    # Default chat mode shown in the Chat UI.
    default_chat_mode: Literal['assistant', 'researcher'] = 'researcher'
    # Chat auto-continuation policy for long/strict outputs.
    chat_auto_continue_enabled: bool = True
    chat_auto_continue_default_max_rounds: int = 2
    chat_auto_continue_hard_cap: int = 3
    chat_auto_continue_prompt: str = _DEFAULT_CHAT_AUTO_CONTINUE_PROMPT
    # -- Server ---------------------------------------------------------------
    host: str = '127.0.0.1'
    port: int = 8420
    # When True, uvicorn runs with --reload (dev only). Leave False for production.
    dev_reload: bool = False
    # API docs exposure toggle:
    # - None (default): enabled in dev_reload sessions, disabled otherwise
    # - True: always expose /docs, /redoc, /openapi.json
    # - False: always disable docs and OpenAPI routes
    api_docs_enabled: bool | None = None
    # Process priority lowering at startup. 0 disables priority changes.
    # On POSIX, applied via os.nice(cpu_priority_nice). On Windows, any value > 0
    # applies BELOW_NORMAL_PRIORITY_CLASS.
    cpu_priority_nice: int = 10

    # -- Logging --------------------------------------------------------------
    # Diagnostics profile presets:
    # - standard: privacy-safe defaults and low overhead for daily use
    # - troubleshooting: richer diagnostics for incident analysis
    # - custom: manual override mode (advanced users)
    diagnostics_profile: Literal['standard', 'troubleshooting', 'custom'] = 'standard'
    # Application log level: debug, info, warning, error. Default info to reduce noise.
    # Third-party loggers (e.g. aiosqlite) are always set to WARNING in logging_config.
    log_level: str = _DEFAULT_LOG_LEVEL
    # When True, write a per-message trace log (chats/{chat_id}/{message_id}.json) for each
    # chat message. Used for troubleshooting and diagnostics analysis of relevance/accuracy.
    chat_trace_logging: bool = False
    # Trace payload redaction level:
    # - off: full trace payload (max debugging, least privacy)
    # - minimal: keep structure, truncate sensitive text fields
    # - strict: redact sensitive text fields with metadata only
    chat_trace_redaction_mode: Literal['off', 'minimal', 'strict'] = 'minimal'
    # Retention window (days) for user chat trace files under app_data_dir/chats/.
    # <= 0 disables retention pruning.
    chat_trace_user_retention_days: int = 30
    # Retention window (days) for diagnostics evaluation trace files under diagnostics/runs/*/traces/.
    # <= 0 disables retention pruning.
    chat_trace_evaluation_retention_days: int = 30
    # When True, show a control to fetch and display raw model output (with <think> blocks)
    # for each assistant message. Useful for debugging. Disabled by default.
    enable_raw_output_control: bool = False

    # Diagnostics performance/resource alert budgets for run artifacts.
    diagnostics_alert_max_elapsed_seconds: float = 120.0
    diagnostics_alert_analysis_max_elapsed_seconds: float = 150.0
    diagnostics_alert_max_first_token_seconds: float = 45.0
    diagnostics_alert_max_rss_delta_mb: float = 1024.0

    # -- UI (frontend-only; persisted so theme survives restarts) -------------
    # Color theme for the app UI: gray, purple, blue, green, orange, mono.
    ui_theme: Literal['gray', 'purple', 'blue', 'green', 'orange', 'mono'] = _DEFAULT_UI_THEME
    # When true, show the macOS menu bar icon while the app is running.
    enable_menu_bar_icon: bool = False
    # -- Pydantic Settings Config ---------------------------------------------
    model_config = {
        'env_prefix': 'INFORMITY_',
    }

    # -- Computed Defaults ----------------------------------------------------
    @model_validator(mode='after')
    def _compute_derived_paths(self) -> 'Settings':
        # Resolve relative paths (e.g. ./data) to absolute
        self.app_data_dir = normalize_path(self.app_data_dir, expand_user=True)

        # Cache directory: defaults to app_data_dir/cache (same root as models, DB, logs).
        # Override via INFORMITY_CACHE_DIR env var.
        if self.cache_dir is None:
            self.cache_dir = self.app_data_dir / DirNames.CACHE
        else:
            self.cache_dir = normalize_path(self.cache_dir, expand_user=True)

        # LLM models directory: always under app_data_dir/models/llm.
        # This is the same location used by the desktop .app bundle, so dev tooling
        # (diagnostics, tests) shares the installed models without duplication.
        if self.models_dir is None:
            self.models_dir = self.app_data_dir / DirNames.MODELS / DirNames.LLM
        else:
            self.models_dir = normalize_path(self.models_dir, expand_user=True)

        # User data paths derive from app_data_dir
        if self.db_path is None:
            self.db_path = self.app_data_dir / DirNames.DB / f'{APP_SLUG}.db'
        else:
            self.db_path = normalize_path(self.db_path, expand_user=True)

        if self.logs_dir is None:
            self.logs_dir = self.app_data_dir / DirNames.LOGS
        if self.diagnostics_dir is None:
            self.diagnostics_dir = self.app_data_dir / DirNames.DIAGNOSTICS

        self.scan_timeout_policy = default_scoped_timeout_policy()
        timeout_cap = int(self.scan_file_timeout_seconds)
        timeout_cap = max(1, min(600, timeout_cap))
        self.scan_file_timeout_seconds = timeout_cap
        self.scan_timeout_policy.default.max_seconds = timeout_cap
        if 'filesystem:file' in self.scan_timeout_policy.overrides:
            self.scan_timeout_policy.overrides['filesystem:file'].max_seconds = timeout_cap

        return self

    # -- Directory Creation ---------------------------------------------------
    def ensure_directories(self) -> None:
        # Create all required directories if they don't exist.
        # Runtime directory structure:
        # - app_data_dir/models/llm/ - Chat/RAG LLM models (*.gguf files)
        # - app_data_dir/cache/huggingface/hub/ - HuggingFace cache
        # - app_data_dir/cache/docling/ - Docling models
        cache_root = self.cache_dir
        llm_dir = self.models_dir
        hf_cache      = cache_root / DirNames.HUGGINGFACE
        hf_hub        = hf_cache / DirNames.HUB
        docling_cache = cache_root / DirNames.DOCLING
        db_dir        = self.app_data_dir / DirNames.DB
        chats_dir     = self.app_data_dir / DirNames.CHAT_LOGS
        # NOTE: diagnostics_chats_dir, diagnostics_reports_dir, and diagnostics_evaluations_dir
        # are NOT created here. The normal pipeline does not use those directories.
        # The normal pipeline uses runs/{run_id}/traces/ for evaluation traces and
        # app_data_dir/chats/ for user chat traces.
        dirs = [
            # Cache structure (non-model artifacts)
            cache_root,
            llm_dir,
            hf_cache,
            hf_hub,
            docling_cache,
            # User data (in app_data_dir, cleared by install_uninstall_app.sh)
            self.app_data_dir,
            db_dir,
            self.logs_dir,
            self.diagnostics_dir,
            chats_dir,
            # diagnostics_chats_dir, diagnostics_reports_dir, and diagnostics_evaluations_dir removed - no longer needed
        ]
        # Filter out None values and ensure directories exist
        directories_to_create = [d for d in dirs if d is not None]
        ensure_directories(directories_to_create)
        for directory in directories_to_create:
            log.info('created_directory', path=str(directory))

def get_chat_trace_logging() -> bool:
    """
    Return whether chat trace logging is enabled.
    Uses the persisted config file as source of truth so the checkbox state
    is respected immediately after Save, even if the in-memory singleton
    was not updated (e.g. race or different code path).
    """
    vals = _load_config_file_values()
    if 'chat_trace_logging' in vals:
        return bool(vals['chat_trace_logging'])
    return bool(settings.chat_trace_logging)


def get_effective_ignore_patterns(s: Settings) -> list[str]:
    # Combine preset patterns (when enabled) with custom ignore_patterns.
    result: list[str] = []
    if s.exclude_macos_system:
        result.extend(EXCLUDE_MACOS_SYSTEM_PATTERNS)
    if s.exclude_developer_data:
        result.extend(EXCLUDE_DEVELOPER_PATTERNS)
    result.extend(s.ignore_patterns)
    return result


def get_supported_extensions_for_scan() -> list[str]:
    """
    Return supported_extensions to use for a scan. Re-reads the persisted
    config file so the crawl uses the latest saved file types (e.g. if the user
    unchecked PDF and saved, we use that even when the in-memory singleton was
    built earlier with defaults). Falls back to settings.supported_extensions
    if the key is missing from the file.
    """
    vals = _load_config_file_values()
    raw = vals.get('supported_extensions')
    if isinstance(raw, list) and raw:
        return [str(x).strip().lower() for x in raw if x]
    return list(settings.supported_extensions)


# ==============================================================================
# Singleton instance
# ==============================================================================

def _build_settings() -> Settings:
    # Build the Settings instance from config.json with env var overrides.
    # Persisted config (saved from the UI) should win over env so that the
    # checkbox state survives restarts. We do this by temporarily unsetting
    # env vars for any key present in the config file before building Settings.
    config_values = _load_config_file_values()

    # Only log if console logging is not suppressed (for CLI tools)
    suppress_console = os.environ.get('INFORMITY_SUPPRESS_CONSOLE_LOGS') == '1'
    if config_values and not suppress_console:
        log.info(
            'loaded_config_file',
            path=str(_config_path_for_loader()),
            fields=list(config_values.keys()),
        )

    # Only pass keys that exist on Settings (config.json might have extras).
    settings_field_names = set(Settings.model_fields)
    init_kwargs = {k: v for k, v in config_values.items() if k in settings_field_names}

    # Temporarily unset env vars for keys that we have in config, so that
    # pydantic uses our config values (init_kwargs) instead of env.
    # Otherwise pydantic-settings would let env override init_kwargs.
    saved_env: dict[str, str] = {}
    for key in init_kwargs:
        env_key = f'INFORMITY_{key.upper()}'
        if env_key in os.environ:
            saved_env[env_key] = os.environ.pop(env_key)

    try:
        return Settings(**init_kwargs)
    finally:
        for env_key, value in saved_env.items():
            os.environ[env_key] = value


settings = _build_settings()


def reset_to_factory_defaults() -> Settings:
    # Reset settings to factory defaults by deleting config.json and rebuilding.
    # Writes a minimal config with Qwen3.6 35B A3B model so reset always returns
    # to the default large profile.
    # Returns the new Settings instance with factory defaults.
    config_path = _config_path_for_loader()
    if config_path.exists():
        try:
            config_path.unlink()
            log.info('config_file_deleted_for_reset', path=str(config_path))
        except OSError as exc:
            log.warning('config_file_delete_failed', path=str(config_path), error=str(exc))

    # Write minimal config with default models and theme so env vars cannot override the reset result
    config_path.parent.mkdir(parents=True, exist_ok=True)
    default_config = {
        'llm_model_filename':      _DEFAULT_LLM_MODEL_FILENAME,
        'diagnostics_profile':     'standard',
        'chat_trace_logging':      False,
        'enable_raw_output_control': False,
        'default_chat_mode':      'researcher',
        'rag_minimal_mode':        True,
        'adaptive_rag_tuning':     True,  # Enabled by default
        'ui_theme':                 _DEFAULT_UI_THEME,
    }
    config_path.write_text(
        serialize_config(default_config),
        encoding='utf-8',
    )
    ensure_private_file(config_path)
    log.info('config_file_written_factory_defaults', path=str(config_path))

    # Rebuild settings (will load default config; profile supplies all other defaults)
    new_settings = _build_settings()

    # Update the module-level singleton
    global settings
    settings = new_settings

    # Reapply thread limits with new settings
    _apply_thread_limits_early()

    return settings


# ==============================================================================
# Apply CPU thread limits IMMEDIATELY at import time
# ==============================================================================
# These env vars MUST be set before PyTorch/torch is imported by any module.
# Since config.py is the first module imported by everything, this is the
# only reliable place to set them.

def _apply_thread_limits_early() -> None:
    # Set CPU thread limits from config before any heavy libraries load.
    max_threads = settings.embedding_max_threads

    # Always disable tokenizers multiprocessing to prevent zombie processes
    # that survive Ctrl+C and cannot be killed normally.
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    thread_env_vars = (
        'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
        'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS',
    )

    # Only log if console logging is not suppressed (for CLI tools)
    suppress_console = os.environ.get('INFORMITY_SUPPRESS_CONSOLE_LOGS') == '1'

    if max_threads > 0:
        thread_str = str(max_threads)
        for var in thread_env_vars:
            os.environ[var] = thread_str
        if not suppress_console:
            log.info('cpu_thread_limits_applied', max_threads=max_threads)
    else:
        # 0 means "automatic" — remove any thread-limit env vars so the
        # process truly uses automatic threading (clearing the main.py
        # setdefault of 6).
        for var in thread_env_vars:
            os.environ.pop(var, None)
        if not suppress_console:
            log.info('cpu_thread_limits_auto')


_apply_thread_limits_early()


# ==============================================================================
# Hugging Face environment setup — shared by main.py (LLM downloads), docling, llm/engine
# ==============================================================================

def configure_hf_environment(*, fail_on_missing_full_privacy_models: bool = True) -> bool:
    # Set Hugging Face cache paths and offline flags based on the current settings.
    # Called during LLM model downloads (main.py, llm/engine.py), docling model loading,
    # and sentence-transformers (embedding/reranker) model loading.
    # Uses unified cache_dir (under app_data_dir) so all HF artefacts are in one place.
    # Returns True when offline mode is active.

    project_hf_home = settings.cache_dir / DirNames.HUGGINGFACE
    project_hf_hub = project_hf_home / DirNames.HUB

    # Always use project cache (never fall back to default ~/.cache location)
    # This ensures all application data stays in the project root directory
    hf_home = str(project_hf_home)
    hf_hub = str(project_hf_hub)

    os.environ['HF_HOME']      = hf_home
    os.environ['HF_HUB_CACHE'] = hf_hub

    # When full_privacy is on, models must be cached (from install/bootstrap).
    # If Full Privacy is enabled but models aren't cached, fail fast with clear error.
    # This enforces that install/bootstrap must complete before Full Privacy can be used.
    use_offline = settings.full_privacy or settings.embedding_offline
    if use_offline:
        # Check if models are cached before enabling offline mode
        if are_required_models_cached():
            os.environ['HF_HUB_OFFLINE']      = '1'
            os.environ['TRANSFORMERS_OFFLINE'] = '1'
            log.debug('offline_mode_enabled', reason='models_cached')
        else:
            # Models not cached - if Full Privacy is enabled, fail fast
            if settings.full_privacy and fail_on_missing_full_privacy_models:
                from informity.exceptions import ConfigurationError
                raise ConfigurationError(
                    'Full Privacy Mode is enabled but required models are not cached. '
                    'Please run the install script to download models: ./scripts/install_app.sh or make install\n\n'
                    'Required models:\n'
                    f'  - Embedding: {settings.embedding_model}\n'
                    f'  - Reranker: {settings.rag_reranker_model}\n'
                    '  - Docling models (for document extraction)\n'
                    f'  - LLM: {settings.llm_model_filename or "not configured"}\n'
                    + '\n'
                    'After install completes, models will be cached and Full Privacy will work without network access.'
                )
            # Allow setup/bootstrap flows to proceed even when full_privacy is configured
            # but required caches are not present yet.
            os.environ.pop('HF_HUB_OFFLINE', None)
            os.environ.pop('TRANSFORMERS_OFFLINE', None)
            log.info(
                'offline_mode_deferred',
                reason='models_not_cached',
                full_privacy=settings.full_privacy,
                fail_on_missing_full_privacy_models=fail_on_missing_full_privacy_models,
                embedding_offline=settings.embedding_offline,
                message='Models not cached; allowing download. Offline mode will be enabled after models are cached.',
            )
            return False
    else:
        os.environ.pop('HF_HUB_OFFLINE', None)
        os.environ.pop('TRANSFORMERS_OFFLINE', None)

    return use_offline


# ==============================================================================
# Model cache verification — check if models are cached before enabling offline mode
# ==============================================================================

_WEIGHT_EXTENSIONS = frozenset({'.bin', '.safetensors', '.onnx'})


def _is_hf_model_cached(model_name: str, hf_hub_cache: Path) -> bool:
    """
    Check if a HuggingFace model is cached.

    HuggingFace stores models in cache_dir with structure:
    - cache_dir/models--{org}--{model_name}/snapshots/{hash}/... (model files, configs, etc.)
    """
    if not hf_hub_cache.exists():
        return False

    model_dir_pattern = f'models--{model_name.replace("/", "--")}'
    model_dir = hf_hub_cache / model_dir_pattern

    if not model_dir.exists():
        return False

    try:
        snapshots_dir = model_dir / 'snapshots'
        if not snapshots_dir.exists():
            return False

        for snapshot_dir in snapshots_dir.iterdir():
            if not snapshot_dir.is_dir():
                continue
            has_config = (snapshot_dir / 'config.json').exists()
            has_weights = any(
                f.suffix in _WEIGHT_EXTENSIONS for f in snapshot_dir.rglob('*') if f.is_file()
            )
            if has_config and has_weights:
                return True
        return False
    except OSError:
        return False


def _is_gguf_model_cached(model_filename: str, models_dir: Path | None) -> bool:
    """Check if a GGUF model file exists in the provided model directory."""
    if not models_dir:
        return False
    model_path = models_dir / model_filename
    return model_path.exists() and model_path.is_file()


_DOCLING_EXTENSIONS = frozenset({'.bin', '.safetensors', '.onnx', '.pt', '.pth'})


def _is_docling_cached(cache_dir: Path | None = None) -> bool:
    """Check if docling runtime artifacts are cached under cache/docling.

    Important: runtime sets DOCLING_ARTIFACTS_PATH to cache/docling, so Hugging Face
    hub snapshots alone are not sufficient to guarantee extraction works offline.

    Args:
        cache_dir: Optional override for the cache root. Defaults to settings.cache_dir.
    """
    docling_cache = (cache_dir if cache_dir is not None else settings.cache_dir) / DirNames.DOCLING
    try:
        if docling_cache.exists():
            return any(f.suffix in _DOCLING_EXTENSIONS for f in docling_cache.rglob('*') if f.is_file())
    except OSError:
        pass
    return False


def are_required_models_cached() -> bool:
    """
    Check if all required models are cached.

    Required models:
    - Embedding model (sentence-transformers, HuggingFace cache)
    - Reranker model (sentence-transformers, HuggingFace cache)
    - Docling models (for document extraction)
    - LLM model (optional, but checked if configured)
    - Classifier LLM model (always used for query intent classification)

    Returns:
        True if all required models are cached, False otherwise.
    """

    hf_hub_cache = settings.cache_dir / DirNames.HUGGINGFACE / DirNames.HUB

    # Check embedding model (project cache only)
    embedding_model = settings.embedding_model
    if not _is_hf_model_cached(embedding_model, hf_hub_cache):
        log.debug('embedding_model_not_cached', model=embedding_model, cache=str(hf_hub_cache))
        return False

    # Check reranker model (project cache only)
    reranker_model = settings.rag_reranker_model
    if not _is_hf_model_cached(reranker_model, hf_hub_cache):
        log.debug('reranker_model_not_cached', model=reranker_model, cache=str(hf_hub_cache))
        return False

    # Check docling models
    if not _is_docling_cached():
        log.debug('docling_models_not_cached')
        return False

    # Check LLM model (if configured)
    llm_filename = settings.llm_model_filename
    if llm_filename and not _is_gguf_model_cached(llm_filename, settings.models_dir):
        log.debug('llm_model_not_cached', model=llm_filename)
        return False

    return True
