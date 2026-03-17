# ==============================================================================
# Informity AI — API Schemas
# Pydantic models for all API request and response bodies.
# These are the contracts between the frontend and backend.
# ==============================================================================

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from informity.config import (
    APP_DISPLAY_NAME,
    DEFAULT_CLASSIFIER_LLM_MODEL_FILENAME,
    DEFAULT_RERANKER_MODEL,
)
from informity.db.sqlite import CANONICAL_DIAGNOSTICS_QUERY_TYPES, CANONICAL_DIAGNOSTICS_TYPES
from informity.diagnostics.issue_types import IssueType
from informity.version import APP_VERSION

# ==============================================================================
# Scan
# ==============================================================================

class ScanRequest(BaseModel):
    # Request to trigger a file scan.
    directories: list[str] | None = None   # Override watched_directories
    force:       bool             = False   # Re-scan even unchanged files


class ScanErrorItem(BaseModel):
    path: str
    filename: str
    extension: str
    operation: str
    error_code: str | None = None
    error_message: str
    is_timeout: bool = False
    created_at: datetime | None = None


class ScanStatusResponse(BaseModel):
    # Current status of a scan operation.
    status:          str        # running, completed, failed
    files_scanned:   int
    files_indexed:   int
    errors:          int
    timeout_errors:  int = 0
    recent_errors:   list[ScanErrorItem] = Field(default_factory=list)
    started_at:      datetime
    elapsed_seconds: float


# ==============================================================================
# Search
# ==============================================================================

class SearchRequest(BaseModel):
    # Semantic search request across indexed documents.
    query:      str
    limit:      int              = Field(default=20, ge=1, le=200)
    category:   str | None       = None
    file_types: list[str] | None = None


class SearchResult(BaseModel):
    # A single search result with file info and relevant chunk.
    file_id:  int
    filename: str
    path:     str
    preview:  str       # Relevant chunk or excerpt
    score:    float     # Similarity score (lower = more similar)
    category: str


class SearchResponse(BaseModel):
    # Response containing search results.
    results: list[SearchResult]
    total:   int
    query:   str


# ==============================================================================
# Chat
# ==============================================================================

class ChatRequest(BaseModel):
    # Request to send a message in a chat.
    message:  str
    chat_id:  str | None = None   # None = start new chat
    response_mode: Literal['balanced', 'analysis', 'research'] | None = None
    run_id: str | None = None      # Optional diagnostics run correlation ID


class ChatStopRequest(BaseModel):
    # Request to stop an in-flight chat stream.
    stream_id: str
    chat_id: str | None = None


class ChatSourceReference(BaseModel):
    # A source document cited in a chat response.
    filename:          str
    path:              str
    chunk_preview:     str      # The chunk text that was used
    relevance_score:   float
    extraction_quality: str | None = None  # 'high', 'medium', 'low' (derived from confidence)
    extraction_note:    str | None = None  # e.g. "Text extracted from PDF" or "OCR used"


# ==============================================================================
# Files
# ==============================================================================

class OpenFileRequest(BaseModel):
    # Request to open a file in the system default application (e.g. Finder double-click).
    path: str


class FileListResponse(BaseModel):
    # Paginated list of indexed files.
    files:  list[dict]   # IndexedFile as dict
    total:  int
    offset: int
    limit:  int


# ==============================================================================
# Index
# ==============================================================================

class RebuildRequest(BaseModel):
    # Request to trigger a full index rebuild.
    force: bool = False   # If True, cancel any running scan/rebuild and start rebuild


class IndexStatusResponse(BaseModel):
    # Statistics about the current index state.
    total_files:                int
    total_chunks:               int
    total_embeddings:           int
    chat_count:                 int  = 0
    last_scan_at:               datetime | None
    db_size_bytes:              int
    vectors_size_bytes:         int
    model_size_bytes:           int
    indexed_content_size_bytes: int  = 0
    reset_in_progress:          bool  = False
    last_reset_result:         dict | None = None   # Set when reset completes: files_deleted, etc.


# ==============================================================================
# Settings
# ==============================================================================

class FileTypeOption(BaseModel):
    """One file type option (id, label, extensions). Canonical source: file_types.get_file_type_options()."""
    id:         str
    label:      str
    extensions: list[str]


class ModelProfileInfo(BaseModel):
    # Read-only model profile information for the Settings UI.
    # All values are determined by the model profile — not user-editable.
    name:                    str       # "Qwen 30B", "Qwen3 14B", etc.
    family:                  str       # "chatml", "llama", etc.
    supported_modes:         list[str] # Modes supported by this profile (balanced/analysis/research)
    supports_reasoning:      bool      # Can use <think> blocks
    reasoning_mode:          str       # "Focused queries only", "Off", etc.
    max_tokens_simple:       int       # Max tokens for simple queries
    max_tokens_focused:      int       # Max tokens for focused queries
    max_tokens_coverage:     int       # Max tokens for coverage queries
    max_tokens_analysis:     int       # Max tokens for analysis mode
    max_tokens_research:     int       # Max tokens for research mode
    coverage_top_k:          int       # Chunks retrieved for coverage queries
    top_k_analysis:          int       # Retrieval top-k for analysis mode
    top_k_research:          int       # Retrieval top-k for research mode
    min_tokens_coverage:     int       # Min tokens before EOS for coverage
    prompt_format:           str       # "Native (GGUF template)", "ChatML"
    coverage_prompt_format:  str       # Prompt format for coverage queries
    context_length:          int       # Max context window (tokens)
    temperature:             float     # Sampling temperature
    top_p:                   float     # Nucleus sampling (1.0 = disabled)
    rag_top_k:               int       # Chunks to retrieve before filtering
    rag_max_score:           float     # Max L2 distance for relevant chunk (lower = stricter)
    rag_context_ratio:       float     # Share of prompt budget for context (rest for history)
    rag_context_ratio_analysis: float  # Context ratio for analysis mode
    rag_context_ratio_research: float  # Context ratio for research mode
    timeout_seconds_analysis: int      # Timeout for analysis mode
    timeout_seconds_research: int      # Timeout for research mode


class SettingsResponse(BaseModel):
    # Current application settings exposed to the frontend.
    watched_directories:       list[str]
    ignore_patterns:           list[str]   # Custom exclude patterns only
    exclude_macos_system:      bool         = True
    exclude_developer_data:   bool         = True
    supported_extensions: list[str]
    follow_symlinks:      bool
    chunk_size_tokens:    int
    chunk_overlap_tokens: int
    chunk_filter_header_only: bool = True  # Enable/disable header-only chunk filtering
    chunk_filter_header_ratio: float = 0.7  # Threshold: chunks with >70% header/separator lines are considered header-only
    chunk_filter_min_content_chars: int = 300  # Minimum content length (chars) to avoid filtering
    chunk_filter_min_content_lines: int = 3  # Minimum content lines to avoid filtering
    embedding_model:         str
    embedding_batch_size:    int
    embedding_max_threads:   int   = 6
    llm_cpu_threads:         int   = 4
    enable_ocr_for_images:   bool  = True  # Enable OCR fallback for image-only PDFs by default
    scan_file_timeout_seconds: int = 300   # Per-file processing timeout (seconds); 0 = no timeout, max 600
    scan_hash_pool:          Literal['thread', 'process'] = 'thread'
    scan_hash_workers:       int = 0  # 0 = auto
    full_privacy:            bool  = True
    embedding_offline:       bool
    llm_local_only:          bool
    llm_model_filename:   str
    # NOTE: rag_max_score and rag_context_ratio are now model-specific (in ModelProfile, read-only)
    adaptive_rag_tuning:    bool        = True   # Adapt retrieval top-k based on corpus size
    rag_rerank:            bool        = True
    rag_rerank_coverage:   bool        = False
    rag_reranker_model:    str         = DEFAULT_RERANKER_MODEL
    rag_rerank_candidates: int        = 25
    chat_history_messages: int       = 5  # Number of previous messages to include in context
    diagnostics_profile: str = 'standard'  # standard, troubleshooting, custom
    log_level:             str        = 'info'  # debug, info, warning, error
    chat_trace_logging:     bool       = False   # Per-chat trace file for debugging
    chat_trace_redaction_mode: str = 'minimal'  # off, minimal, strict
    chat_trace_user_retention_days: int = 30
    chat_trace_evaluation_retention_days: int = 30
    enable_raw_output_control: bool = False   # Show control to fetch raw model output per assistant message
    classifier_llm_model:   str        = DEFAULT_CLASSIFIER_LLM_MODEL_FILENAME  # Model filename for query classification (read-only, always LLM-based)
    available_models:       list[str]        = Field(default_factory=list)
    file_type_options:      list[FileTypeOption] = Field(default_factory=list)  # Canonical list for UI
    config_file_path:       str               = ''
    model_profile:          ModelProfileInfo | None = None  # Main model profile (read-only)
    classifier_model_profile: ModelProfileInfo | None = None  # Query classification model profile (read-only)
    ui_theme:               str               = 'mono'     # Color theme: gray, purple, blue, green, orange, mono
    enable_menu_bar_icon:   bool              = False      # Show menu bar icon while app is running (macOS desktop runtime)
    default_response_mode: Literal['balanced', 'analysis', 'research'] = 'balanced'  # Default chat response mode
    cpu_priority_nice:      int = 10  # 0 = off, >0 lowers process priority at startup


class SettingsUpdateRequest(BaseModel):
    # Partial update of application settings. Only include fields to change.
    # NOTE: Profile-controlled fields are NOT updatable — they are determined
    # by the selected model's profile: llm_max_tokens, rag_coverage_top_k,
    # llm_context_length,
    # llm_temperature, rag_top_k.
    watched_directories:       list[str] | None = None
    ignore_patterns:           list[str] | None = None
    exclude_macos_system:      bool | None      = None
    exclude_developer_data:    bool | None       = None
    supported_extensions: list[str] | None  = None
    follow_symlinks:      bool | None       = None
    chunk_size_tokens:    int | None        = None
    chunk_overlap_tokens: int | None        = None
    chunk_filter_header_only: bool | None = None  # Enable/disable header-only chunk filtering
    chunk_filter_header_ratio: float | None = None  # Threshold: chunks with >N% header/separator lines are considered header-only
    chunk_filter_min_content_chars: int | None = None  # Minimum content length (chars) to avoid filtering
    chunk_filter_min_content_lines: int | None = None  # Minimum content lines to avoid filtering
    embedding_batch_size:   int | None = None
    embedding_max_threads:  int | None = None
    llm_cpu_threads:        int | None = None
    enable_ocr_for_images:  bool | None = None  # Enable OCR for image-only PDFs when regular extraction fails
    scan_file_timeout_seconds: int | None = None  # Per-file processing timeout (seconds); 0 = no timeout, max 600
    scan_hash_pool:         Literal['thread', 'process'] | None = None
    scan_hash_workers:      int | None = None
    full_privacy:           bool | None = None
    embedding_offline:      bool | None = None
    llm_local_only:        bool | None = None
    llm_model_filename:  str | None        = None
    # NOTE: rag_max_score and rag_context_ratio are now model-specific (in ModelProfile, not updatable)
    adaptive_rag_tuning:    bool | None = None   # Adapt retrieval top-k based on corpus size
    rag_rerank:            bool | None = None
    rag_rerank_coverage:   bool | None = None
    rag_reranker_model:    str | None  = None
    rag_rerank_candidates: int | None  = None
    chat_history_messages: int | None  = None  # Number of previous messages to include in context
    diagnostics_profile: str | None = None  # standard, troubleshooting, custom
    log_level:             str | None  = None  # debug, info, warning, error
    chat_trace_logging:    bool | None = None   # Per-chat trace file for debugging
    chat_trace_redaction_mode: str | None = None  # off, minimal, strict
    chat_trace_user_retention_days: int | None = None
    chat_trace_evaluation_retention_days: int | None = None
    enable_raw_output_control: bool | None = None   # Show control to fetch raw model output per assistant message
    classifier_llm_model:  str | None  = None  # Model filename for query classification (advanced, rarely changed)
    ui_theme:             str | None  = None  # Color theme: gray, purple, blue, green, orange, mono
    enable_menu_bar_icon: bool | None = None  # Show menu bar icon while app is running (macOS desktop runtime)
    default_response_mode: Literal['balanced', 'analysis', 'research'] | None = None  # Default chat response mode
    cpu_priority_nice:    int | None = None


class CurrentChatResponse(BaseModel):
    """Current chat ID persisted in config (Tauri-compatible, survives reload/navigation)."""
    current_chat_id: str | None = None


class CurrentChatUpdateRequest(BaseModel):
    """Request to update the persisted current chat ID."""
    current_chat_id: str | None = None


# ==============================================================================
# System
# ==============================================================================

_DIAGNOSTICS_SUMMARY_SCHEMA = 'informity.diagnostics.summary.v2'
_DIAGNOSTICS_SUMMARY_AGGREGATION_MODE = 'direct_window_scan'
_CANONICAL_DIAGNOSTICS_ISSUES = tuple(sorted(issue.value for issue in IssueType))


class DiagnosticsResponse(BaseModel):
    """System diagnostics information."""
    app_version: str = APP_VERSION
    app_display_name: str = APP_DISPLAY_NAME
    python_version: str
    platform: str
    platform_version: str
    architecture: str
    ram_total_gb: float
    ram_available_gb: float
    ram_used_gb: float
    disk_total_gb: float
    disk_available_gb: float
    disk_used_gb: float
    model_loaded: bool
    model_filename: str | None = None
    model_size_gb: float | None = None
    db_path: str
    db_size_bytes: int
    db_size_mb: float
    vectors_size_bytes: int
    vectors_size_mb: float
    total_files: int
    total_chunks: int
    indexed_content_size_bytes: int
    indexed_content_size_mb: float
    uptime_seconds: float | None = None


class DiagnosticsMetricsSummaryResponse(BaseModel):
    """Aggregated response diagnostics metrics for runtime observability."""
    summary_schema: str = Field(default=_DIAGNOSTICS_SUMMARY_SCHEMA, serialization_alias='schema')
    aggregation_mode: str = _DIAGNOSTICS_SUMMARY_AGGREGATION_MODE
    type_taxonomy: list[str] = list(CANONICAL_DIAGNOSTICS_TYPES)
    query_type_taxonomy: list[str] = list(CANONICAL_DIAGNOSTICS_QUERY_TYPES)
    issue_type_taxonomy: list[str] = list(_CANONICAL_DIAGNOSTICS_ISSUES)
    window_days: int
    type_filter: Literal['user', 'evaluation'] | None = None
    run_id_filter: str | None = None
    total_responses: int
    by_type: dict[str, int]
    by_query_type: dict[str, int]
    issue_counts: dict[str, int]
    timeout_count: int
    empty_answer_count: int
    refusal_pattern_count: int
    timeout_rate: float
    empty_answer_rate: float
    refusal_pattern_rate: float
    avg_generation_seconds: float
    p95_generation_seconds: float | None = None
    avg_sources_count: float
    avg_raw_chunks_count: float
    created_at_oldest: datetime | None = None
    created_at_newest: datetime | None = None


# ==============================================================================
# Configuration (env vars reference)
# ==============================================================================

class EnvVarItem(BaseModel):
    # Single environment variable: name, default value string, description.
    name:        str   # e.g. INFORMITY_APP_DATA_DIR
    default:     str   # Display string for default value
    description: str


class EnvVarGroup(BaseModel):
    # Logical group of env vars with title and description.
    title:       str
    description: str
    variables:   list[EnvVarItem]


class EnvVarsResponse(BaseModel):
    # Full list of env variable groups for the Configuration page.
    groups: list[EnvVarGroup]


# ==============================================================================
# Application Defaults and Constants Reference
# ==============================================================================

class ConstantItem(BaseModel):
    # Single constant: name, default value string, description.
    name:        str   # e.g. "STALE_SCAN_THRESHOLD_SECONDS"
    default:     str   # Display string for default value
    description: str


class ConstantGroup(BaseModel):
    # Logical group of constants with title and description.
    title:       str
    description: str
    constants:   list[ConstantItem]


class ConfigReferenceResponse(BaseModel):
    # Full list of constant groups for the Configuration page reference section.
    groups: list[ConstantGroup]


# ==============================================================================
# Health
# ==============================================================================

class HealthResponse(BaseModel):
    # Health check response.
    status:           str = 'ok'
    version:          str = APP_VERSION
    app_display_name: str   # Product name for UI (from config.APP_DISPLAY_NAME)
