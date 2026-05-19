# ==============================================================================
# Informity AI — API Schemas
# Pydantic models for all API request and response bodies.
# These are the contracts between the frontend and backend.
# ==============================================================================

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from informity.api.setup_state import SetupState
from informity.config import (
    APP_DISPLAY_NAME,
    DEFAULT_PDF_EXTRACTION_STRATEGY_ORDER,
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


class ScanErrorsResponse(BaseModel):
    scan_id: int
    total: int
    offset: int
    limit: int
    errors: list[ScanErrorItem] = Field(default_factory=list)


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
    model_config = ConfigDict(extra='forbid')

    message:  str
    chat_id:  str | None = None   # None = start new chat
    scoped_file_ids: list[int] | None = Field(default=None, min_length=1)  # Optional one-or-more file scope for researcher retrieval
    scoped_upload_ids: list[str] | None = Field(default=None, min_length=1)  # Optional one-or-more upload IDs for chat-scoped attachments
    request_id: str | None = None  # Optional client-generated request ID for deterministic stop
    run_id: str | None = None      # Optional diagnostics run correlation ID
    mode: str | None = None        # Optional chat mode: assistant | researcher (invalid -> researcher)
    role_id: str | None = None     # Optional domain role overlay ID
    chat_web_search_enabled: bool | None = None  # Optional chat-scoped assistant web-search toggle
    chat_web_search_privacy_override: bool | None = None  # Optional chat-scoped privacy override for web search

    @model_validator(mode='after')
    def _normalize_scoped_file_ids(self) -> 'ChatRequest':
        if self.scoped_file_ids is None:
            return self
        normalized: list[int] = []
        seen: set[int] = set()
        for raw_file_id in self.scoped_file_ids:
            file_id = int(raw_file_id)
            if file_id < 1:
                raise ValueError('scoped_file_ids must contain positive integer IDs')
            if file_id in seen:
                continue
            seen.add(file_id)
            normalized.append(file_id)
        if not normalized:
            raise ValueError('scoped_file_ids must contain at least one file ID')
        self.scoped_file_ids = normalized
        return self

    @model_validator(mode='after')
    def _normalize_scoped_upload_ids(self) -> 'ChatRequest':
        if self.scoped_upload_ids is None:
            return self
        normalized: list[str] = []
        seen: set[str] = set()
        for raw_upload_id in self.scoped_upload_ids:
            upload_id = str(raw_upload_id or '').strip()
            if not upload_id:
                raise ValueError('scoped_upload_ids must contain non-empty upload IDs')
            if upload_id in seen:
                continue
            seen.add(upload_id)
            normalized.append(upload_id)
        if not normalized:
            raise ValueError('scoped_upload_ids must contain at least one upload ID')
        self.scoped_upload_ids = normalized
        return self


class ChatPreferencesUpdateRequest(BaseModel):
    # Request to update chat-scoped UX preferences.
    chat_web_search_enabled: bool | None = None
    chat_web_search_privacy_override: bool | None = None

    @model_validator(mode='after')
    def _validate_non_empty(self) -> 'ChatPreferencesUpdateRequest':
        if self.chat_web_search_enabled is None and self.chat_web_search_privacy_override is None:
            raise ValueError('At least one chat preference field is required')
        return self


class ChatRoleDefinition(BaseModel):
    id: str
    name: str
    description: str
    icon: str | None = None
    disclaimer: str | None = None


class ChatStopRequest(BaseModel):
    # Request to stop an in-flight chat stream.
    stream_id: str | None = None
    request_id: str | None = None
    chat_id: str | None = None

    @model_validator(mode='after')
    def _validate_keys(self) -> 'ChatStopRequest':
        if not self.stream_id and not self.request_id:
            raise ValueError('Either stream_id or request_id is required')
        return self


class ChatSourceReference(BaseModel):
    # A source document cited in a chat response.
    filename:        str
    path:            str
    chunk_preview:   str      # The chunk text that was used
    relevance_score: float
    file_id: int | None = None


class ChatUploadAttachmentResponse(BaseModel):
    upload_id: str
    chat_id: str
    file_id: int | None = None
    filename_at_upload: str
    size_bytes: int = 0
    content_hash: str | None = None
    state: str
    referenced_message_ids: list[int] = Field(default_factory=list)
    uploaded_at: datetime | None = None
    updated_at: datetime | None = None
    removed_at: datetime | None = None


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
    source_scope_stats:        list[dict[str, object]] = Field(default_factory=list)


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
    name:                    str       # "Qwen3.6 35B A3B", "Qwen3 14B", etc.
    family:                  str       # "chatml", "llama", etc.
    supports_reasoning:      bool      # Can use <think> blocks
    reasoning_mode:          str       # "Focused queries only", "Off", etc.
    max_tokens:              int       # Max tokens
    coverage_top_k:          int       # Chunks retrieved for coverage queries
    min_tokens_coverage:     int       # Min tokens target for coverage (pipeline-enforced)
    prompt_format:           str       # "Native (GGUF template)", "ChatML"
    coverage_prompt_format:  str       # Prompt format for coverage queries
    context_length:          int       # Max context window (tokens)
    temperature:             float     # Sampling temperature
    top_p:                   float     # Nucleus sampling (1.0 = disabled)
    rag_top_k:               int       # Chunks to retrieve before filtering
    rag_max_score:           float     # Max L2 distance for relevant chunk (lower = stricter)
    rag_context_ratio:       float     # Share of prompt budget for context (rest for history)
    timeout_seconds:         int       # Timeout seconds


class DiagnosticsProfilePreset(BaseModel):
    # Backend-defined diagnostics preset values used by Settings UI.
    log_level: str
    chat_trace_logging: bool
    chat_trace_redaction_mode: str
    chat_trace_user_retention_days: int
    chat_trace_evaluation_retention_days: int


class SettingsResponse(BaseModel):
    # Current application settings exposed to the frontend.
    watched_directories:       list[str]
    source_scopes_enabled:     dict[str, bool] = Field(default_factory=dict)
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
    llm_cpu_threads:         int   = 8
    enable_ocr_for_images:        bool  = True  # Enable OCR fallback for image-only PDFs by default
    max_indexable_file_size_mb:   int   = 100
    scan_file_timeout_seconds:    int   = 600
    pdf_extraction_strategy_order: list[str] = Field(default_factory=lambda: list(DEFAULT_PDF_EXTRACTION_STRATEGY_ORDER))
    scan_hash_pool:          Literal['thread', 'process'] = 'thread'
    scan_hash_workers:       int = 0  # 0 = auto
    full_privacy:            bool  = True
    tavily_api_key_set:      bool = False
    linkup_api_key_set:      bool = False
    web_search_configured:   bool = False
    web_search_primary_provider: Literal['tavily', 'linkup'] = 'tavily'
    web_search_max_results:  int = 5
    web_search_timeout_seconds: float = 8.0
    embedding_offline:       bool
    llm_provider:         Literal['local_gguf', 'ollama'] = 'local_gguf'
    llm_local_only:          bool
    llm_model_id:         str
    ollama_base_url:      str = 'http://127.0.0.1:11434'
    ollama_timeout_seconds: float = 120.0
    llm_model_filename:   str
    # NOTE: rag_max_score and rag_context_ratio are now model-specific (in ModelProfile, read-only)
    rag_minimal_mode:      bool        = True
    rag_minimal_answerability_threshold_focused: float = 0.0
    rag_minimal_answerability_threshold_coverage: float = 0.0
    rag_minimal_min_chunks_focused: int = 1
    rag_minimal_min_chunks_coverage: int = 1
    adaptive_rag_tuning:    bool        = True   # Adapt retrieval top-k based on corpus size
    rag_rerank:            bool        = True
    rag_rerank_coverage:   bool        = True
    rag_reranker_model:    str         = DEFAULT_RERANKER_MODEL
    rag_rerank_candidates: int        = 25
    rag_query_rewrite_enabled: bool = True
    rag_query_rewrite_max_history_messages: int = 3
    rag_query_rewrite_max_chars_per_turn: int = 260
    rag_query_rewrite_max_query_chars: int = 900
    chat_history_messages: int       = 5  # Default history window when mode is unresolved
    chat_history_messages_assistant: int = 12  # Assistant mode history window
    chat_history_messages_researcher: int = 5  # Researcher mode history window
    default_chat_mode: Literal['assistant', 'researcher'] = 'researcher'
    enable_chat_roles: bool = False
    enabled_chat_role_ids: list[str] = Field(default_factory=list)
    entity_extract_acronym: bool = True
    entity_extract_person_name: bool = False
    entity_extract_organization: bool = False
    entity_extract_location: bool = False
    entity_extract_numeric_id: bool = False
    diagnostics_profile: str = 'standard'  # standard, troubleshooting, custom
    diagnostics_profile_presets: dict[str, DiagnosticsProfilePreset] = Field(default_factory=dict)
    log_level:             str        = 'info'  # debug, info, warning, error
    chat_trace_logging:     bool       = False   # Per-chat trace file for debugging
    chat_trace_redaction_mode: str = 'minimal'  # off, minimal, strict
    chat_trace_user_retention_days: int = 30
    chat_trace_evaluation_retention_days: int = 30
    mcp_enabled: bool = False
    mcp_auto_start: bool = False
    mcp_transport: Literal['stdio', 'http'] = 'stdio'
    mcp_http_host: str = '127.0.0.1'
    mcp_http_port: int = 8431
    mcp_auth_mode: Literal['token_required'] = 'token_required'
    mcp_scope_mode: Literal['metadata_only', 'search_snippets', 'full_content'] = 'metadata_only'
    mcp_access_token: str = ''
    mcp_token_configured: bool = False
    enable_raw_output_control: bool = False   # Show control to fetch raw model output per assistant message
    available_models:       list[str]        = Field(default_factory=list)
    file_type_options:      list[FileTypeOption] = Field(default_factory=list)  # Canonical list for UI
    config_file_path:       str               = ''
    model_profile:          ModelProfileInfo | None = None  # Main model profile (read-only)
    ui_theme:               str               = 'onyx'     # Color theme: canvas, ember, sage, graphite, onyx
    enable_menu_bar_icon:   bool              = False      # Show menu bar icon while app is running (macOS desktop runtime)
    cpu_priority_nice:      int = 10  # 0 = off, >0 lowers process priority at startup


class SettingsUpdateRequest(BaseModel):
    # Partial update of application settings. Only include fields to change.
    # NOTE: Profile-controlled fields are NOT updatable — they are determined
    # by the selected model's profile: llm_max_tokens, coverage_top_k,
    # llm_context_length,
    # llm_temperature, rag_top_k.
    watched_directories:       list[str] | None = None
    source_scopes_enabled:     dict[str, bool] | None = None
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
    enable_ocr_for_images:        bool | None = None  # Enable OCR for image-only PDFs when regular extraction fails
    max_indexable_file_size_mb:   int | None  = None
    scan_file_timeout_seconds:    int | None  = None
    pdf_extraction_strategy_order: list[str] | None = None
    scan_hash_pool:         Literal['thread', 'process'] | None = None
    scan_hash_workers:      int | None = None
    full_privacy:           bool | None = None
    tavily_api_key:         str | None = None
    linkup_api_key:         str | None = None
    web_search_primary_provider: Literal['tavily', 'linkup'] | None = None
    web_search_max_results: int | None = None
    web_search_timeout_seconds: float | None = None
    embedding_offline:      bool | None = None
    llm_provider:        Literal['local_gguf', 'ollama'] | None = None
    llm_local_only:        bool | None = None
    llm_model_id:        str | None        = None
    ollama_base_url:     str | None        = None
    ollama_timeout_seconds: float | None = None
    llm_model_filename:  str | None        = None
    # NOTE: rag_max_score and rag_context_ratio are now model-specific (in ModelProfile, not updatable)
    rag_minimal_mode:      bool | None = None
    rag_minimal_answerability_threshold_focused: float | None = None
    rag_minimal_answerability_threshold_coverage: float | None = None
    rag_minimal_min_chunks_focused: int | None = None
    rag_minimal_min_chunks_coverage: int | None = None
    adaptive_rag_tuning:    bool | None = None   # Adapt retrieval top-k based on corpus size
    rag_rerank:            bool | None = None
    rag_rerank_coverage:   bool | None = None
    rag_reranker_model:    str | None  = None
    rag_rerank_candidates: int | None  = None
    rag_query_rewrite_enabled: bool | None = None
    rag_query_rewrite_max_history_messages: int | None = None
    rag_query_rewrite_max_chars_per_turn: int | None = None
    rag_query_rewrite_max_query_chars: int | None = None
    chat_history_messages: int | None  = None  # Default history window when mode is unresolved
    chat_history_messages_assistant: int | None = None  # Assistant mode history window
    chat_history_messages_researcher: int | None = None  # Researcher mode history window
    default_chat_mode: Literal['assistant', 'researcher'] | None = None
    enable_chat_roles: bool | None = None
    enabled_chat_role_ids: list[str] | None = None
    entity_extract_acronym: bool | None = None
    entity_extract_person_name: bool | None = None
    entity_extract_organization: bool | None = None
    entity_extract_location: bool | None = None
    entity_extract_numeric_id: bool | None = None
    diagnostics_profile: str | None = None  # standard, troubleshooting, custom
    log_level:             str | None  = None  # debug, info, warning, error
    chat_trace_logging:    bool | None = None   # Per-chat trace file for debugging
    chat_trace_redaction_mode: str | None = None  # off, minimal, strict
    chat_trace_user_retention_days: int | None = None
    chat_trace_evaluation_retention_days: int | None = None
    mcp_enabled: bool | None = None
    mcp_auto_start: bool | None = None
    mcp_transport: Literal['stdio', 'http'] | None = None
    mcp_http_host: str | None = None
    mcp_http_port: int | None = None
    mcp_auth_mode: Literal['token_required'] | None = None
    mcp_scope_mode: Literal['metadata_only', 'search_snippets', 'full_content'] | None = None
    mcp_access_token: str | None = None
    enable_raw_output_control: bool | None = None   # Show control to fetch raw model output per assistant message
    ui_theme:             str | None  = None  # Color theme: canvas, ember, sage, graphite, onyx
    enable_menu_bar_icon: bool | None = None  # Show menu bar icon while app is running (macOS desktop runtime)
    cpu_priority_nice:    int | None = None


class McpTokenGenerateResponse(BaseModel):
    token: str


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
    llm_provider: Literal['local_gguf', 'ollama'] = 'local_gguf'
    llm_model_id: str | None = None
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
    # Single environment variable: name, current runtime value string, description.
    name:          str   # e.g. INFORMITY_APP_DATA_DIR
    current_value: str   # Display string for current active value
    description:   str


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


class SetupTierOption(BaseModel):
    tier: str
    model_id: str | None = None
    title: str
    display_name: str
    model_filename: str
    model_size_bytes: int
    approx_size_gb: float
    quality: str
    speed: str
    ram_profile: str
    description: str


class SetupStatusResponse(BaseModel):
    # Setup readiness state used by desktop startup gating.
    state: SetupState
    required_models_ready: bool
    setup_state_file_present: bool = False
    detail: str | None = None
    machine_ram_gb: int | None = None
    recommended_tier: str | None = None
    recommended_reason: str | None = None
    llm_provider: Literal['local_gguf', 'ollama'] = 'local_gguf'
    ollama_reachable: bool | None = None
    ollama_model_ready: bool | None = None
    tier_options: list[SetupTierOption] = Field(default_factory=list)


class SetupStartRequest(BaseModel):
    tier: str
    model_filename: str


class SetupStartResponse(BaseModel):
    accepted: bool = True
    state: SetupState


class SetupActionResponse(BaseModel):
    accepted: bool = True
    state: SetupState
    detail: str | None = None


class SetupEventResponse(BaseModel):
    state: SetupState
    stage: str
    overall_pct: int = 0
    artifact: str | None = None
    artifact_pct: int = 0
    bytes_done: int = 0
    bytes_total: int = 0
    speed_bps: float = 0.0
    eta_sec: int | None = None
    paused: bool = False
    error: str | None = None


class OllamaStatusResponse(BaseModel):
    reachable: bool
    model_ready: bool
    model: str
    base_url: str
    detail: str | None = None


class ModelsCatalogItem(BaseModel):
    tier: str
    model_id: str | None = None
    title: str
    display_name: str
    model_filename: str
    model_size_bytes: int
    approx_size_gb: float
    quality: str
    speed: str
    ram_profile: str
    description: str
    installed: bool
    is_default: bool


class ModelsCatalogResponse(BaseModel):
    default_model_id: str | None = None
    default_model_filename: str
    models: list[ModelsCatalogItem] = Field(default_factory=list)


class ModelActionRequest(BaseModel):
    model_filename: str


class ModelActionResponse(BaseModel):
    accepted: bool = True
    detail: str | None = None


class ModelOperationEventResponse(BaseModel):
    state: str
    stage: str
    model_filename: str | None = None
    overall_pct: int = 0
    bytes_done: int = 0
    bytes_total: int = 0
    speed_bps: float = 0.0
    eta_sec: int | None = None
    paused: bool = False
    error: str | None = None
