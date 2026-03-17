# ==============================================================================
# Informity AI — Configuration Module
# Loads settings with the following priority (highest wins):
#   1. Persisted config.json (written by the Settings API) — for keys present
#   2. Environment variables (INFORMITY_*)
#   3. Hard-coded defaults below
# Config.json wins over env vars so UI state survives restarts.
#
# Default app data is stored in a "data" directory relative to the process
# working directory (e.g. project root). Override with INFORMITY_APP_DATA_DIR
# to use e.g. ~/Library/Application Support/Informity AI for production.
# ==============================================================================

import json
import os
from pathlib import Path
from typing import Literal

import structlog
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings

from informity.utils.directory_utils import ensure_directories
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

# ==============================================================================
# Repo root detection
# ==============================================================================

def _get_repo_root() -> Path:
    """
    Find project root (directory containing pyproject.toml).
    Starts from config.py location and walks up to find pyproject.toml.
    Falls back to current working directory if not found.
    """
    env_repo_root = os.environ.get('INFORMITY_REPO_ROOT', '').strip()
    if env_repo_root:
        return normalize_path(Path(env_repo_root), expand_user=True)

    # config.py is at: src/informity/config.py
    # So repo root is: config.py -> informity -> src -> repo_root
    current = Path(__file__).resolve().parent.parent.parent
    while current != current.parent:
        if (current / 'pyproject.toml').exists():
            return current
        current = current.parent
    # Fallback: use current working directory
    return Path.cwd()

# ==============================================================================
# Defaults
# ==============================================================================

# Local to application: "data" relative to process cwd (resolved in validator / loader).
_DEFAULT_APP_DATA_DIR = Path('data')

# Default model for reset-to-factory and first load: Qwen 14B (Q5_K_M).
_DEFAULT_LLM_MODEL_FILENAME = 'Qwen3-14B-Q5_K_M.gguf'

# Default diagnostics analysis model filename (DeepSeek R1 optimized for analysis tasks).
_DEFAULT_DIAGNOSTICS_LLM_MODEL_FILENAME = 'DeepSeek-R1-Distill-Qwen-14B-Q4_K_M.gguf'

# Default query classification model filename (Qwen2.5-3B optimized for classification accuracy).
_DEFAULT_CLASSIFIER_LLM_MODEL_FILENAME = 'Qwen2.5-3B-Instruct-Q4_K_M.gguf'

# Default embedding model (sentence-transformers)
_DEFAULT_EMBEDDING_MODEL = 'nomic-ai/nomic-embed-text-v1.5'

# Default reranker model (sentence-transformers cross-encoder)
_DEFAULT_RERANKER_MODEL = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
# Public aliases for schema/default consumers.
DEFAULT_CLASSIFIER_LLM_MODEL_FILENAME = _DEFAULT_CLASSIFIER_LLM_MODEL_FILENAME
DEFAULT_RERANKER_MODEL = _DEFAULT_RERANKER_MODEL

# Default Hugging Face repository for LLM model downloads
_DEFAULT_LLM_HF_REPO = 'Qwen/Qwen3-14B-GGUF'

# Default auto-continuation policy for long responses.
_DEFAULT_CHAT_AUTO_CONTINUE_PROMPT = (
    'Continue with the remaining sections from your last answer. '
    'Keep the same structure and avoid repeating completed sections.'
)
LOG_LEVEL_ALLOWED_VALUES: tuple[str, ...] = ('debug', 'info', 'warning', 'warn', 'error')
UI_THEME_ALLOWED_VALUES: tuple[str, ...] = ('gray', 'purple', 'blue', 'green', 'orange', 'mono')
RESPONSE_MODE_ALLOWED_VALUES: tuple[str, ...] = ('balanced', 'analysis', 'research')
_DEFAULT_LOG_LEVEL = 'info'
_DEFAULT_UI_THEME = 'mono'
_DEFAULT_RESPONSE_MODE = 'balanced'


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

    # Unified cache directory (at repo root, not committed to repo)
    CACHE = '.cache'  # Dot prefix since not committed to repo

    # Model/cache subdirectories
    LLM = 'chat-llm'  # Chat/RAG LLM models (*.gguf files)
    QUERY_CLASSIFIER_MODELS = 'query-classifier-llm'  # Query classifier model (*.gguf)
    DIAGNOSTICS_MODELS = 'models'  # Diagnostics LLM models (*.gguf files) under tools/diagnostics/models/
    HUGGINGFACE = 'huggingface'  # HuggingFace cache under .cache/huggingface/
    HUB = 'hub'  # HuggingFace hub cache under .cache/huggingface/hub/
    DOCLING = 'docling'  # Docling models under .cache/docling/ (flat, docling creates its own structure inside)

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
        # Guard against stale/invalid theme values in persisted config.json.
        # This prevents startup failures when theme enums change.
        raw_theme = data.get('ui_theme')
        if raw_theme is not None:
            if raw_theme == 'informity':
                data['ui_theme'] = 'mono'
            elif isinstance(raw_theme, str) and raw_theme in UI_THEME_ALLOWED_VALUES:
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
    cache_dir:     Path | None = Field(default=None)   # Unified cache root; default {repo_root}/DirNames.CACHE. Override via INFORMITY_CACHE_DIR.
    db_path:       Path | None = Field(default=None)   # Computed: app_data_dir / DirNames.DB / f'{APP_SLUG}.db'
    # vectors_dir removed - vectors now stored in SQLite via sqlite-vec
    models_dir:    Path | None = Field(default=None)   # Computed: desktop -> app_data_dir/DirNames.MODELS/DirNames.LLM; otherwise cache_dir/DirNames.LLM
    query_classifier_models_dir: Path | None = Field(default=None)  # Computed: desktop -> app_data_dir/DirNames.MODELS/DirNames.QUERY_CLASSIFIER_MODELS; otherwise cache_dir/DirNames.QUERY_CLASSIFIER_MODELS
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
    # Per-file processing timeout during scan/index (seconds). Prevents a single
    # broken or corrupted file from stalling the entire scan indefinitely.
    # Set to 0 to disable (not recommended — a hung file will block the scan forever).
    # Default 300s (5 min) handles large PDFs; increase for very large/complex documents.
    scan_file_timeout_seconds: int = 300
    # Running-scan stale detection threshold (seconds) used when a new scan/rebuild
    # request checks for already-running operations.
    scan_stale_threshold_seconds: int = 300
    # Hash executor mode for scan crawling: thread (default) or process.
    # Thread mode avoids process spawn overhead and is usually better for mixed I/O+CPU hashing.
    scan_hash_pool: Literal['thread', 'process'] = 'thread'
    # Hash worker count for crawl hashing. 0 = auto (min(4, max(2, cpu_count // 3))).
    scan_hash_workers: int = 0

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
    # CPU threads for llama-cpp-python (separate from embedding threads —
    # llama-cpp uses its own threading and is not controlled by OMP_NUM_THREADS).
    # Set to 0 for automatic. Default: 4 to leave cores for embedder/reranker/OS.
    llm_cpu_threads: int = 4
    # When True, enable OCR (Optical Character Recognition) for image-only PDFs
    # when regular text extraction fails. OCR is slower but can extract text from
    # scanned documents, photographed pages, and PDFs with embedded images.
    # Default: True so scanned/image PDFs work out of the box via fallback OCR.
    enable_ocr_for_images: bool = True

    # -- Privacy ------------------------------------------------------------------
    # When True, no network access: embedding and LLM use cache/local only (fully local).
    # When False, network is allowed (e.g. for model downloads). Synced to embedding_offline and llm_local_only.
    full_privacy:         bool = True

    # -- LLM ------------------------------------------------------------------
    # When True, load LLM only from models_dir; never download from the network.
    # Synced from full_privacy when that setting is updated via the UI.
    llm_local_only:       bool = True
    llm_model_filename:   str  = _DEFAULT_LLM_MODEL_FILENAME  # Default: Qwen3 14B
    llm_hf_repo:          str  = _DEFAULT_LLM_HF_REPO  # Hugging Face repo for automatic model downloads
    llm_context_length:   int  = 16384  # 16K is ample (10K chunks + 4K prompt/history + 2K gen); prevents over-assembly
    llm_max_tokens:     int   = 2048
    llm_temperature:    float = 0.2     # Low for factual extraction; avoids determinism-induced loops
    # Retrieval top-k: model-profile-only (ModelProfile.rag_top_k, coverage_top_k).
    # Use model_adapter.get_retrieval_top_k(query_type). No config/env.
    # NOTE: rag_max_score and rag_context_ratio are now model-specific (moved to ModelProfile).
    # Each model has optimal values tuned for its capabilities.
    # When True, adapt retrieval top-k based on corpus size (file count, parent chunk count).
    # When False, always use model profile base values. See .internal/features/adaptive-tuning.md.
    adaptive_rag_tuning:  bool  = True
    # When True, re-rank vector search candidates with a cross-encoder (query, chunk) before taking top_k.
    rag_rerank:          bool  = True
    # When True, also apply reranking to coverage queries (comprehensive lists/tables).
    # Enabled: the 100-300ms cost is trivial vs total query time, and reranking
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
    # Retrieval quality gates (runtime policy; avoid hardcoded thresholds in handlers).
    retrieval_relevance_threshold_focused:    float = 0.03
    retrieval_relevance_threshold_coverage:   float = 0.02
    retrieval_relevance_threshold_structured: float = 0.02
    # Coverage fallback hard floor (EH-11 rollback control):
    # - When enabled, evidence-floor override may only bypass relevance gate
    #   when score clears `retrieval_coverage_evidence_floor_min_score`.
    # - Disabled by default until thresholds are calibrated.
    retrieval_coverage_evidence_floor_hard_floor_enabled: bool = False
    retrieval_coverage_evidence_floor_min_score: float = 0.05
    retrieval_precloseout_min_relevance_score: float = 0.62
    # Grounding repair gate for contract-heavy answers. Values are intentionally
    # centralized here to keep continuation behavior policy-driven.
    chat_grounding_repair_min_coverage_rate: float = 0.4
    chat_grounding_repair_max_unsupported_claims: int = 0
    chat_grounding_repair_max_not_found_count: int = 12
    # Deterministic retrieval widening before terminal unresolved closeout.
    retrieval_widening_retry_multiplier: float = 1.5
    retrieval_widening_retry_extra_k: int = 4
    retrieval_widening_retry_cap: int = 40
    # Structured numeric extraction plausibility guards (runtime policy; avoid hardcoded thresholds).
    extraction_numeric_max_abs_value: float = 100000000.0
    extraction_numeric_max_unformatted_digits: int = 9
    extraction_numeric_noise_small_value_threshold: float = 2.0
    extraction_numeric_noise_large_value_threshold: float = 100.0
    extraction_finance_conflict_require_same_category: bool = True
    extraction_finance_conflict_min_evidence_overlap_tokens: int = 2
    # Number of previous messages to include in prompt context.
    # Lower values free up tokens for more document context, improving answer quality.
    # Higher values maintain better conversation continuity for follow-up questions.
    chat_history_messages:   int   = 5
    # Chat auto-continuation policy for long/strict outputs.
    chat_auto_continue_enabled: bool = True
    chat_auto_continue_default_max_rounds: int = 2
    chat_auto_continue_hard_cap: int = 3
    chat_auto_continue_prompt: str = _DEFAULT_CHAT_AUTO_CONTINUE_PROMPT
    # Fit-to-budget rollout controls (Phase 5):
    # - dev: enabled only in dev_reload sessions
    # - power_users: deterministic subset rollout (~35%)
    # - default_on: enabled for everyone
    fit_to_budget_enabled: bool = True
    fit_to_budget_rollout_stage: Literal['dev', 'power_users', 'default_on'] = 'default_on'
    fit_to_budget_tuning_days: int = 14
    fit_to_budget_tuning_min_samples: int = 20

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
    # Diagnostics strict contract gates (EH-09 rollback control):
    # - False (default): do not hard-fail on strict schema/grounding gates.
    # - True: enforce strict schema/grounding gates as test failures.
    diagnostics_strict_contract_gates_enforced: bool = False
    # Application log level: debug, info, warning, error. Default info to reduce noise.
    # Third-party loggers (e.g. aiosqlite) are always set to WARNING in logging_config.
    log_level: str = _DEFAULT_LOG_LEVEL
    # When True, write a per-chat trace log (chat_{chat_id}.json) for each
    # chat message. Used for troubleshooting and LLM-assisted analysis of relevance/accuracy.
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

    # -- Diagnostics Pipeline LLM Enhancement -------------------------------------
    # When True, use local LLM to enhance root cause analysis in diagnostics pipeline.
    # Default: False (opt-in feature). Set to True to enable LLM-powered analysis.
    diagnostics_llm_analysis_enabled: bool = False
    # Directory for diagnostics analysis models.
    # Default: {repo_root}/tools/diagnostics/models (decoupled from .cache/).
    diagnostics_models_dir: Path | None = Field(default=None)
    # Model filename to use for diagnostics analysis (default: DeepSeek R1 for analysis tasks).
    # User can override via config.json or INFORMITY_DIAGNOSTICS_LLM_MODEL_FILENAME env var.
    diagnostics_llm_model_filename: str = _DEFAULT_DIAGNOSTICS_LLM_MODEL_FILENAME
    # Maximum seconds for LLM inference. Generous default so diagnostics analysis can
    # produce full results (14B models need several minutes for long JSON output).
    diagnostics_llm_timeout_seconds: int = 600
    # Maximum number of issues to analyze per run (limit analysis scope)
    diagnostics_llm_max_issues_per_run: int = 10
    # Diagnostics performance/resource alert budgets for run artifacts.
    diagnostics_alert_max_elapsed_seconds: float = 120.0
    diagnostics_alert_analysis_max_elapsed_seconds: float = 150.0
    diagnostics_alert_max_first_token_seconds: float = 45.0
    diagnostics_alert_max_rss_delta_mb: float = 1024.0

    # -- Query Classification ------------------------------------------------------
    # LLM-based query classification is always enabled (no user toggle).
    # Model filename for query classification (default: Qwen2.5-3B-Instruct-Q4_K_M).
    # Must be in query_classifier_models_dir (.cache/query-classifier-llm/). User can override via config.json or env var.
    classifier_llm_model: str = _DEFAULT_CLASSIFIER_LLM_MODEL_FILENAME

    # -- UI (frontend-only; persisted so theme survives restarts) -------------
    # Color theme for the app UI: gray, purple, blue, green, orange, mono.
    ui_theme: Literal['gray', 'purple', 'blue', 'green', 'orange', 'mono'] = _DEFAULT_UI_THEME
    # When true, show the macOS menu bar icon while the app is running.
    enable_menu_bar_icon: bool = False
    # Default chat response mode for new messages when request does not specify response_mode.
    default_response_mode: Literal['balanced', 'analysis', 'research'] = _DEFAULT_RESPONSE_MODE

    # -- Pydantic Settings Config ---------------------------------------------
    model_config = {
        'env_prefix': 'INFORMITY_',
    }

    # -- Computed Defaults ----------------------------------------------------
    @model_validator(mode='after')
    def _compute_derived_paths(self) -> 'Settings':
        # Resolve relative paths (e.g. ./data) to absolute
        self.app_data_dir = normalize_path(self.app_data_dir, expand_user=True)

        # Get repo root for unified cache (application assets, not user data)
        repo_root = _get_repo_root()

        # Unified cache directory at repo root (all models, HuggingFace cache, docling, etc.)
        if self.cache_dir is None:
            self.cache_dir = repo_root / DirNames.CACHE
        else:
            self.cache_dir = normalize_path(self.cache_dir, expand_user=True)

        # Desktop runtime stores persistent model files under app_data_dir so they
        # survive cache eviction and align with installed-app semantics.
        desktop_session_mode = bool(os.environ.get('INFORMITY_TAURI_SESSION_TOKEN', '').strip())

        # LLM models directory.
        if self.models_dir is None:
            if desktop_session_mode:
                self.models_dir = self.app_data_dir / DirNames.MODELS / DirNames.LLM
            else:
                self.models_dir = self.cache_dir / DirNames.LLM
        else:
            self.models_dir = normalize_path(self.models_dir, expand_user=True)

        # Query classifier model directory.
        if self.query_classifier_models_dir is None:
            if desktop_session_mode:
                self.query_classifier_models_dir = (
                    self.app_data_dir / DirNames.MODELS / DirNames.QUERY_CLASSIFIER_MODELS
                )
            else:
                self.query_classifier_models_dir = self.cache_dir / DirNames.QUERY_CLASSIFIER_MODELS
        else:
            self.query_classifier_models_dir = normalize_path(self.query_classifier_models_dir, expand_user=True)

        # Diagnostics models directory: tools/diagnostics/models (decoupled from .cache/).
        if self.diagnostics_models_dir is None:
            self.diagnostics_models_dir = (
                repo_root / DirNames.TOOLS / DirNames.DIAGNOSTICS / DirNames.DIAGNOSTICS_MODELS
            )
        else:
            self.diagnostics_models_dir = normalize_path(self.diagnostics_models_dir, expand_user=True)

        # User data paths derive from app_data_dir
        if self.db_path is None:
            self.db_path = self.app_data_dir / DirNames.DB / f'{APP_SLUG}.db'
        else:
            self.db_path = normalize_path(self.db_path, expand_user=True)

        # vectors_dir removed - vectors now stored in SQLite via sqlite-vec
        if self.logs_dir is None:
            self.logs_dir = self.app_data_dir / DirNames.LOGS
        if self.diagnostics_dir is None:
            self.diagnostics_dir = self.app_data_dir / DirNames.DIAGNOSTICS

        return self

    # -- Directory Creation ---------------------------------------------------
    def ensure_directories(self) -> None:
        # Create all required directories if they don't exist.
        # Runtime directory structure:
        # - models_dir - Chat/RAG LLM models (*.gguf files)
        # - query_classifier_models_dir - Query classifier model (*.gguf file)
        # - tools/diagnostics/models/ - Diagnostics LLM models (*.gguf files)
        # - .cache/huggingface/hub/ - HuggingFace cache (LLM downloads, docling models)
        # - .cache/docling/ - Docling models (docling creates its own structure inside)
        cache_root = self.cache_dir
        llm_dir = self.models_dir
        query_classifier_dir = self.query_classifier_models_dir
        diagnostics_models_dir = self.diagnostics_models_dir
        hf_cache   = cache_root / DirNames.HUGGINGFACE if cache_root else None
        hf_hub     = hf_cache / DirNames.HUB if hf_cache else None
        docling_cache = cache_root / DirNames.DOCLING if cache_root else None
        db_dir     = self.app_data_dir / DirNames.DB
        # NOTE: diagnostics_chats_dir, diagnostics_reports_dir, and diagnostics_evaluations_dir
        # are NOT created here. The normal pipeline does not use those directories.
        # The normal pipeline uses runs/{run_id}/traces/ for evaluation traces and
        # app_data_dir/chats/ for user chat traces.
        dirs = [
            # Cache structure (non-model artifacts)
            cache_root,  # .cache/
            llm_dir,  # models_dir (desktop default: app_data_dir/models/chat-llm)
            query_classifier_dir,  # query_classifier_models_dir (desktop default: app_data_dir/models/query-classifier-llm)
            diagnostics_models_dir,  # tools/diagnostics/models/
            hf_cache,  # .cache/huggingface/
            hf_hub,  # .cache/huggingface/hub/
            docling_cache,  # .cache/docling/
            # User data (in app_data_dir, cleared by reset.sh)
            self.app_data_dir,
            db_dir,
            # vectors_dir removed - vectors now stored in SQLite via sqlite-vec
            self.logs_dir,
            self.diagnostics_dir,
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
    # Writes a minimal config with Qwen3 14B model so reset always returns to
    # Qwen3 14B profile.
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
        'diagnostics_llm_model_filename': _DEFAULT_DIAGNOSTICS_LLM_MODEL_FILENAME,
        'diagnostics_llm_analysis_enabled': False,  # Disabled by default (opt-in feature)
        'diagnostics_profile':     'standard',
        'chat_trace_logging':      False,
        'enable_raw_output_control': False,
        'adaptive_rag_tuning':     True,  # Enabled by default
        'ui_theme':                 _DEFAULT_UI_THEME,
        'default_response_mode':    _DEFAULT_RESPONSE_MODE,
    }
    config_path.write_text(
        serialize_config(default_config),
        encoding='utf-8',
    )
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

def configure_hf_environment() -> bool:
    # Set Hugging Face cache paths and offline flags based on the current settings.
    # Called during LLM model downloads (main.py, llm/engine.py), docling model loading,
    # and sentence-transformers (embedding/reranker) model loading.
    # Uses unified cache_dir (at repo root) so all HF artefacts are in one place.
    # Returns True when offline mode is active.

    cache_dir = settings.cache_dir
    if cache_dir is None:
        # Fallback: use repo root / cache (shouldn't happen after _compute_derived_paths)
        cache_dir = _get_repo_root() / DirNames.CACHE

    project_hf_home = cache_dir / DirNames.HUGGINGFACE
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
            if settings.full_privacy:
                from informity.exceptions import ConfigurationError
                raise ConfigurationError(
                    'Full Privacy Mode is enabled but required models are not cached. '
                    'Please run the install script to download models: ./scripts/install.sh or make install\n\n'
                    'Required models:\n'
                    f'  - Embedding: {settings.embedding_model}\n'
                    f'  - Reranker: {settings.rag_reranker_model}\n'
                    '  - Docling models (for document extraction)\n'
                    f'  - LLM: {settings.llm_model_filename or "not configured"}\n'
                    + f'  - Classifier LLM: {settings.classifier_llm_model}\n'
                    + '\n'
                    'After install completes, models will be cached and Full Privacy will work without network access.'
                )
            # If only embedding_offline is True (not full_privacy), allow downloads
            # This provides flexibility for development/testing
            os.environ.pop('HF_HUB_OFFLINE', None)
            os.environ.pop('TRANSFORMERS_OFFLINE', None)
            log.info(
                'offline_mode_deferred',
                reason='models_not_cached',
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

def _is_hf_model_cached(model_name: str, hf_hub_cache: Path) -> bool:
    """
    Check if a HuggingFace model is cached.

    HuggingFace stores models in cache_dir with structure:
    - cache_dir/models--{org}--{model_name}/snapshots/{hash}/... (model files, configs, etc.)
    """
    if not hf_hub_cache.exists():
        return False

    # HuggingFace creates a directory with structure: models--{org}--{model_name}
    # e.g., "nomic-ai/nomic-embed-text-v1.5" -> "models--nomic-ai--nomic-embed-text-v1.5"
    model_dir_pattern = f'models--{model_name.replace("/", "--")}'
    model_dir = hf_hub_cache / model_dir_pattern

    if not model_dir.exists():
        return False

    # Check for HuggingFace model files in snapshots subdirectory
    # Models are stored in snapshots/{hash}/ subdirectories
    try:
        snapshots_dir = model_dir / 'snapshots'
        if not snapshots_dir.exists():
            return False

        # Check each snapshot directory
        for snapshot_dir in snapshots_dir.iterdir():
            if not snapshot_dir.is_dir():
                continue
            # HuggingFace models have config.json and model weights (.bin, .safetensors, or .onnx)
            has_config = (snapshot_dir / 'config.json').exists()
            has_weights = (
                any(snapshot_dir.rglob('*.bin')) or
                any(snapshot_dir.rglob('*.safetensors')) or
                any(snapshot_dir.rglob('*.onnx'))
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


def _is_docling_cached() -> bool:
    """Check if docling runtime artifacts are cached under cache/docling.

    Important: runtime sets DOCLING_ARTIFACTS_PATH to cache/docling, so Hugging Face
    hub snapshots alone are not sufficient to guarantee extraction works offline.
    """
    cache_dir = settings.cache_dir
    if not cache_dir:
        cache_dir = _get_repo_root() / DirNames.CACHE
    docling_cache = cache_dir / DirNames.DOCLING
    # Native docling artifact cache
    try:
        if docling_cache.exists():
            for item in docling_cache.iterdir():
                if item.is_dir():
                    if any(item.rglob('*.bin')) or any(item.rglob('*.safetensors')) or any(item.rglob('*.onnx')):
                        return True
                elif item.suffix in ('.bin', '.safetensors', '.onnx', '.pt', '.pth'):
                    return True
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

    # Get HF hub cache directory (project cache only - no fallback to default)
    cache_dir = settings.cache_dir
    if cache_dir is None:
        cache_dir = _get_repo_root() / DirNames.CACHE
    hf_hub_cache = cache_dir / DirNames.HUGGINGFACE / DirNames.HUB

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

    # Check classifier LLM model (always used for query intent classification)
    classifier_filename = settings.classifier_llm_model
    if classifier_filename and not _is_gguf_model_cached(classifier_filename, settings.query_classifier_models_dir):
        log.debug('classifier_model_not_cached', model=classifier_filename)
        return False

    return True
