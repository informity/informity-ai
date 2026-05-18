# ARCHITECTURE.md — Informity AI Architecture Reference

This file is the **single source of truth** for types, interfaces, and module responsibilities. When generating code for any module, consult this file first.

**Project structure:** `src/informity/` holds all backend code: `main.py`, `config.py`, `logging_config.py`, `chat_trace.py`, `file_types.py`, `file_patterns.py`, `upload_policy.py`, `exceptions.py`, `category_patterns.py`; `api/` (routes_scan, routes_index, routes_search, routes_chat, routes_settings, routes_system, schemas, env_vars_metadata, config_reference_metadata, operation_state, setup_state, security, chat_completion_policy, chat_out_of_corpus, chat_sources, error_messages, chat_orchestrator, chat_continuation, chat_sse, chat_closeout, chat_stream_registry, context_scope_manager); `db/` (sqlite, vectors, models, utils); `utils/` (path_utils, json_utils, directory_utils, file_utils, number_utils); `sources/` (base, filesystem_adapter, registry, orchestrator); `scanner/` (crawler, watcher, extractors — docling unified extractor + EPUB extractor + text extractor); `indexer/` (chunker, embedder, classifier, reranker, pipeline, post_process, adaptive_tuning, term_dictionary_builder); `llm/` (engine, model_adapter, rag, query_classifier, query_patterns, rag_patterns, nlp_heuristics, roles, promptcue_signals, types, retrieval, prompt_builder, streaming, metadata_filters, intent_router, classification_policy, promptcue_adapter, term_dictionary, chat_mode, contract_gate, contract_prompt_parser, metrics_payload, system_prompts, timeout_policy, user_messages, web_search, rag_runtime/, handlers/ — metadata, rag, simple). Diagnostics runtime modules: `src/informity/diagnostics/` (issue_types, observer, resource_snapshot). Frontend: `src/frontend/` (React + Vite; build output `dist/` served by FastAPI; context/: ChatContext, ToastContext, ConfirmContext). Vanilla backup archived at `.archive/frontend-bak/`. Tests: `tests/`. Scripts: `scripts/`.

---

## General Rules

**File Creation:** Do NOT create markdown (.md) files unless the user explicitly requests them. Documentation files should only be created when specifically asked for.

---

## Core Data Types

These are the canonical type definitions. All code must use these exactly.

### Configuration

```python
# src/informity/config.py

from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field, model_validator

# Application identity (single source of truth)
APP_SLUG         = 'informity'       # db filename (informity.db), log filename
APP_DISPLAY_NAME = 'Informity AI'   # User-facing product name (UI, prompts, API docs)

# Preset pattern lists: EXCLUDE_MACOS_SYSTEM_PATTERNS, EXCLUDE_DEVELOPER_PATTERNS
# Applied when exclude_macos_system / exclude_developer_data are True (Settings).

class Settings(BaseSettings):
    # Paths (default: ~/.informity; override via INFORMITY_APP_DATA_DIR)
    app_data_dir: Path = ...            # default: ~/.informity
    cache_dir:    Path | None = Field(default=None)  # Unified cache root; default app_data_dir/cache. Override via INFORMITY_CACHE_DIR.
    db_path:       Path | None = Field(default=None)  # Computed: app_data_dir / 'db' / f'{APP_SLUG}.db'
    # Note: vectors_dir removed - vectors now stored in SQLite database (vec_chunks table) via sqlite-vec extension
    models_dir:    Path | None = Field(default=None)  # Computed: app_data_dir/models/llm (shared between desktop and dev)
    logs_dir:      Path | None = Field(default=None)   # Computed: app_data_dir / 'logs'

    # Scanner
    watched_directories:     list[Path] = Field(default_factory=list)
    ignore_patterns:         list[str]  = Field(default_factory=list)  # Custom only; presets from checkboxes
    exclude_macos_system:    bool       = True   # When True, apply EXCLUDE_MACOS_SYSTEM_PATTERNS
    exclude_developer_data:   bool       = True   # When True, apply EXCLUDE_DEVELOPER_PATTERNS
    supported_extensions:    list[str]  = Field(default_factory=lambda: [
        '.pdf',  # Included by default (docling provides reliable extraction)
        '.txt', '.md', '.rst', '.log',
        '.docx', '.xlsx', '.csv', '.pptx',
        '.html', '.htm',
        # .json, .yaml, .yml, .toml excluded by default; user can enable in Settings
    ])
    follow_symlinks: bool = False

    # Indexer
    chunk_size_tokens:       int   = 512   # Parent chunk size (for context windows)
    chunk_overlap_tokens:    int   = 60
    chunk_child_size_tokens: int  = 150   # Child chunk size (for precise search matching)
    chunk_filter_header_only: bool = True  # Header-only chunk filter (quality heuristic)
    chunk_filter_header_ratio: float = 0.7
    chunk_filter_min_content_chars:  int = 300
    chunk_filter_min_content_lines:  int = 3
    embedding_model:         str = 'nomic-ai/nomic-embed-text-v1.5'  # 768-dim, 8192-token context
    embedding_batch_size:     int = 32
    embedding_offline:        bool = True   # Synced from full_privacy when set via UI
    embedding_max_threads:    int = 6       # CPU threads for embedding (0 = automatic)
    enable_ocr_for_images:   bool = True  # OCR for image-only PDFs when text extraction fails

    # Privacy — when full_privacy=True, no network; embedding and LLM use cache/local only
    full_privacy:   bool = True
    llm_local_only: bool = True   # Synced from full_privacy when set via UI

    # LLM — model configurable via env / config.json
    # Current default: Qwen3.6 35B A3B (Q4_K_M quantization)
    llm_model_filename:   str   = 'Qwen3.6-35B-A3B-Q4_K_M.gguf'
    llm_context_length:   int   = 16384  # 16K is ample; profile may override for other models
    llm_max_tokens:      int   = 2048
    llm_temperature:      float = 0.2
    # NOTE: rag_top_k and rag_coverage_top_k are NOT in config — retrieval top-k is model-profile-only.
    # NOTE: rag_max_score and rag_context_ratio are model-profile-owned (not global settings).
    # Use model_adapter profile defaults/overrides for retrieval thresholds and context budgeting.
    rag_rerank:           bool  = True
    rag_reranker_model:    str   = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
    rag_rerank_candidates: int  = 25    # Reduced from 35 for speed (reranker most effective on top 20-30)
    rag_rerank_coverage:   bool  = True  # Also rerank coverage queries (lists, comparisons)

    # Server
    host:       str  = '127.0.0.1'
    port:       int  = 8420
    dev_reload: bool = False   # uvicorn --reload (dev only)

    # Logging
    log_level:         str  = 'info'   # debug, info, warning, error
    chat_trace_logging: bool = False    # Per-chat trace file for debugging

    # Diagnostics Evaluation (optional)
    diagnostics_metrics_enabled:      bool = False   # Enable diagnostics metrics collection during chat
    diagnostics_dir:               Path | None = Field(default=None)  # Computed: app_data_dir / 'diagnostics'

    model_config = {'env_prefix': 'INFORMITY_'}

    @model_validator(mode='after')
    def _compute_derived_paths(self) -> 'Settings': ...
    def ensure_directories(self) -> None: ...
```

**Helper:** `get_effective_ignore_patterns(settings)` — combines preset patterns (when enabled) with custom `ignore_patterns`. Used by crawler.

### Extractor Types

```python
# src/informity/scanner/extractors/base.py

from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
from typing import Protocol, runtime_checkable

@dataclass(frozen=True)
class ExtractedDocument:
    """Output of any extractor. Immutable."""
    text: str                              # Full extracted text
    source_path: Path                      # Absolute path to source file
    metadata: dict[str, str] = field(default_factory=dict)  # Format-specific metadata
    page_count: int | None = None          # For PDFs, PPTX
    word_count: int = 0                    # Computed from text
    extraction_time_ms: float = 0.0        # How long extraction took
    error: str | None = None               # Non-fatal extraction warnings
    preview_text: str = ''                 # Clean preview text (first 500 chars, no markdown noise)
    # Per-chunk metadata mappings (for docling formats with provenance, primarily PDFs/PPTX)
    # Range-based storage: list of (start, end, value) tuples for memory efficiency
    char_to_page_ranges: list[tuple[int, int, int]] | None = None  # (start, end, page_no) ranges
    char_to_block_type_ranges: list[tuple[int, int, str]] | None = None  # (start, end, block_type) ranges
    char_to_header_level_ranges: list[tuple[int, int, int]] | None = None  # (start, end, header_level) ranges

@runtime_checkable
class BaseExtractor(Protocol):
    """Protocol that all extractors must implement."""
    supported_extensions: list[str]

    def extract(self, path: Path) -> ExtractedDocument:
        """Extract text content from a file. Must not raise — return error in ExtractedDocument."""
        ...

    def can_handle(self, path: Path) -> bool:
        """Check if this extractor supports the given file."""
        ...
```

### Database Models (SQLite rows ↔ Pydantic)

```python
# src/informity/db/models.py

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field

class FileCategory(StrEnum):
    DOCUMENT  = 'document'    # .pdf, .docx, .pptx
    PLAINTEXT = 'plaintext'   # .txt, .md, .rst, .log
    DATA      = 'data'        # .csv, .xlsx, .json, .yaml
    WEB       = 'web'         # .html, .htm
    CODE      = 'code'        # .py, .js, .ts (future)
    OTHER     = 'other'

class ScanStatus(StrEnum):
    RUNNING   = 'running'
    COMPLETED = 'completed'
    FAILED    = 'failed'
    CANCELLED = 'cancelled'

class IndexedFile(BaseModel):
    """Represents a file in the SQLite `files` table."""
    id:                     int | None = None
    path:                   str              # Absolute POSIX path
    filename:               str
    extension:              str
    size_bytes:             int
    content_hash:           str              # SHA-256
    extracted_text_preview:  str              # First ~500 chars
    category:               FileCategory
    tags:                   list[str] = Field(default_factory=list)  # Stored as JSON
    year:                   int | None = None   # Extracted at index time (filename/path/text)
    indexed_at:             datetime | None = None
    modified_at:            datetime
    created_at:             datetime | None = None

class Chunk(BaseModel):
    """Represents a text chunk in the SQLite `chunks` table."""
    id:           int | None = None
    file_id:      int
    chunk_index:  int
    content:      str
    token_count:  int
    parent_id:    int | None = None     # v2: Link to parent window chunk (for parent document retrieval)
    page_number:  int | None = None     # v2: Page number in source document (PDF)
    section_path: str | None = None     # v2: Section hierarchy path (e.g., "Introduction/Overview")
    block_type:   str | None = None     # v2: Block type ('table', 'form', 'narrative') from docling provenance
    created_at:   datetime | None = None

class ScanRecord(BaseModel):
    """Represents a scan run in the SQLite `scan_history` table."""
    id:            int | None = None
    started_at:    datetime
    completed_at:  datetime | None = None
    files_scanned: int = 0
    files_indexed: int = 0
    errors:        int = 0
    status:        ScanStatus = ScanStatus.RUNNING

class ChatMessage(BaseModel):
    """A single message in a chat."""
    id:                 int | None = None
    chat_id:            str         # UUID
    role:              str         # 'user' or 'assistant'
    content:            str
    sources:           list[dict] = Field(default_factory=list)  # Full source reference objects
    generation_seconds: float | None = None   # Time to generate answer (assistant only)
    completion_mode:    str | None = None
    stopped_by_user:    bool = False
    has_remaining_scope: bool = False
    next_action:        str | None = None
    next_action_reason: str | None = None
    chat_mode:          str | None = None
    retrieval_scope_kind: str | None = None   # assistant_mode | indexed_corpus | indexed_files | chat_uploads
    retrieval_scope_key:  str | None = None   # scope-specific key used for history partitioning
    model_filename:     str | None = None
    is_internal:        bool = False
    created_at:        datetime | None = None

class ChatUploadAttachment(BaseModel):
    """Chat-scoped uploaded file attachment lifecycle state."""
    id:                    int | None = None
    upload_id:             str
    chat_id:               str
    file_id:               int | None = None
    filename_at_upload:    str
    size_bytes:            int = 0
    content_hash:          str | None = None
    state:                 str = 'uploading'  # uploading | indexing | ready | deleting | deleted | failed
    referenced_message_ids: list[int] = Field(default_factory=list)
    uploaded_at:           datetime | None = None
    updated_at:            datetime | None = None
    removed_at:            datetime | None = None
```

### API Schemas (Request/Response)

```python
# src/informity/api/schemas.py

from pydantic import BaseModel, Field, model_validator
from datetime import datetime

# --- Scan ---
class ScanRequest(BaseModel):
    directories: list[str] | None = None  # Override watched_directories
    force: bool = False                   # Re-scan even unchanged files

class ScanStatusResponse(BaseModel):
    status: str                    # running, completed, failed
    files_scanned: int
    files_indexed: int
    errors: int
    started_at: datetime
    elapsed_seconds: float

# --- Search ---
class SearchRequest(BaseModel):
    query: str
    limit: int = 20
    category: str | None = None
    file_types: list[str] | None = None

class SearchResult(BaseModel):
    file_id: int
    filename: str
    path: str
    preview: str                   # Relevant chunk or excerpt
    score: float                   # Similarity score
    category: str

class SearchResponse(BaseModel):
    results: list[SearchResult]
    total: int
    query: str

# --- Chat ---
class ChatRequest(BaseModel):
    message: str
    chat_id: str | None = None  # None = new chat
    scoped_file_ids: list[int] | None = Field(default=None, min_length=1)  # Optional one-or-more file scope
    scoped_upload_ids: list[str] | None = Field(default=None, min_length=1)  # Optional one-or-more upload_id scope
    request_id: str | None = None
    run_id: str | None = None
    mode: str | None = None
    chat_web_search_enabled: bool | None = None
    chat_web_search_privacy_override: bool | None = None

    @model_validator(mode='before')
    @classmethod
    def _reject_legacy_file_id(cls, values):
        if isinstance(values, dict) and values.get('file_id') is not None:
            raise ValueError('file_id is no longer supported. Use scoped_file_ids (or scoped_upload_ids).')
        return values

class ChatSourceReference(BaseModel):
    filename: str
    path: str
    chunk_preview: str             # The chunk that was used
    relevance_score: float

# --- Settings ---
class FileTypeOption(BaseModel):
    """One file type option (id, label, extensions). Canonical source: file_types.get_file_type_options()."""
    id:         str
    label:      str
    extensions: list[str]

class ModelProfileInfo(BaseModel):
    # Read-only model profile information for the Settings UI.
    # All values are determined by the model profile — not user-editable.
    name:                    str       # e.g. "Qwen3.6 35B A3B"
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

class SettingsResponse(BaseModel):
    # Current application settings exposed to the frontend.
    # Profile-controlled fields (llm_context_length, llm_max_tokens, llm_temperature,
    # rag_top_k, coverage_top_k) are in model_profile, not here.
    watched_directories:       list[str]
    ignore_patterns:           list[str]   # Custom exclude patterns only
    exclude_macos_system:      bool         = True
    exclude_developer_data:    bool         = True
    supported_extensions:      list[str]
    follow_symlinks:           bool
    chunk_size_tokens:         int
    chunk_overlap_tokens:      int
    embedding_model:           str
    embedding_batch_size:      int
    embedding_max_threads:     int   = 6
    full_privacy:              bool  = True
    embedding_offline:         bool
    llm_local_only:            bool
    llm_model_filename:        str
    rag_max_score:             float | None = None
    rag_context_ratio:         float        = 0.75
    rag_rerank:                bool         = True
    rag_rerank_coverage:       bool         = False
    rag_reranker_model:        str          = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
    rag_rerank_candidates:     int          = 25
    log_level:                 str          = 'info'
    chat_trace_logging:        bool         = False
    diagnostics_metrics_enabled: bool      = False   # Collect response diagnostics metrics for self-improvement
    rag_minimal_mode:          bool        = True
    available_models:          list[str]        = Field(default_factory=list)
    file_type_options:         list[FileTypeOption] = Field(default_factory=list)
    config_file_path:          str               = ''
    model_profile:             ModelProfileInfo | None = None  # Active model profile (read-only)

class SettingsUpdateRequest(BaseModel):
    # Partial update. Profile-controlled fields are NOT updatable — they are
    # determined by the selected model's profile.
    # All fields optional; only listed fields are updatable via API (_UPDATABLE_FIELDS in routes_settings)

class FileListResponse(BaseModel):
    files: list[dict]              # IndexedFile as dict
    total: int
    offset: int
    limit: int

# --- Files (open) ---
class OpenFileRequest(BaseModel):
    path: str   # Absolute path; opens in system default app (macOS: open, Windows: os.startfile, Linux: xdg-open)

# --- Index ---
class RebuildRequest(BaseModel):
    force: bool = False   # If True, cancel any running scan/rebuild and start rebuild

class IndexStatusResponse(BaseModel):
    total_files:                int
    total_chunks:               int
    total_embeddings:           int
    last_scan_at:               datetime | None
    db_size_bytes:              int
    vectors_size_bytes:         int
    model_size_bytes:           int
    indexed_content_size_bytes: int = 0
    reset_in_progress:          bool = False
    last_reset_result:          dict | None = None   # Set when reset completes

# --- Config (env vars reference) ---
class EnvVarItem(BaseModel):
    name:        str   # e.g. INFORMITY_APP_DATA_DIR
    default:     str
    description: str

class EnvVarGroup(BaseModel):
    title:       str
    description: str
    variables:   list[EnvVarItem]

class EnvVarsResponse(BaseModel):
    groups: list[EnvVarGroup]

# --- Health ---
class HealthResponse(BaseModel):
    status:           str = 'ok'
    version:          str = APP_VERSION  # from informity.version
    app_display_name: str   # From config.APP_DISPLAY_NAME
```

---

## Module Responsibilities

### `config.py`
- Loads settings with priority (highest wins): (1) persisted `config.json` (written by Settings API) for keys present, (2) env vars `INFORMITY_*`, (3) hard-coded defaults. Config file wins over env so UI state survives restarts.
- Retrieval top-k (`rag_top_k`, `coverage_top_k`) is NOT in config — model-profile-only via `model_adapter.get_retrieval_top_k()`. When `adaptive_rag_tuning` is enabled, corpus-aware values override from `indexer.adaptive_tuning` cache.
- Exports `APP_SLUG`, `APP_DISPLAY_NAME`; preset pattern lists `EXCLUDE_MACOS_SYSTEM_PATTERNS`, `EXCLUDE_DEVELOPER_PATTERNS`; `DiagnosticsConstants` (run ID prefixes, chat ID prefixes); `DirNames` class (directory name constants).
- Provides singleton `settings`; `ensure_directories()`; `get_effective_ignore_patterns(settings)` to combine preset (when enabled) + custom ignore patterns.
- Computes derived paths: `cache_dir` (default: `app_data_dir/cache`), `models_dir` (default: `app_data_dir/models/llm`), user data paths from `app_data_dir`.
- Uses `utils.path_utils.normalize_path()` and `normalize_paths()` for path resolution, `utils.json_utils.serialize_config()` for config file serialization, `utils.directory_utils.ensure_directories()` for directory creation.
- Applies CPU thread limits at import time (`embedding_max_threads` → OMP_NUM_THREADS etc., TOKENIZERS_PARALLELISM=false) before any heavy libs load.
- Provides `configure_hf_environment()` to set HF cache paths and offline flags based on settings.
- **Imports:** pydantic-settings, structlog, utils.path_utils, utils.json_utils, utils.directory_utils
- **Imported by:** everything

### `db/sqlite.py`
- Manages async SQLite via aiosqlite; `init_db()` creates/repairs schema and tracks `SCHEMA_VERSION` via `schema_version` table. Includes forward startup migrations (currently through v3) plus index/FTS repair checks; fails closed on unsupported future schema versions.
- Tables: `files`, `chunks` (with `parent_id`, `page_number`, `section_path`, `block_type`), `config`, `scan_history`, `chat_messages` (chat_id, role, content, sources, generation_seconds), `chats` (chat_id, title, created_at, updated_at), `response_diagnostics_metrics` (diagnostics metrics for evaluation runs and user chats when enabled).
- Provides: `get_connection` (opens new async connection), `get_db` (FastAPI `Depends` helper), `init_db`; `insert_file`, `get_file_by_path`, `get_file_by_source_identity`, `get_file_by_id`, `update_file`, `delete_file`, `get_files`, `get_files_by_ids`, `get_all_files_for_scan`, `get_distinct_years`, `get_distinct_categories`, `get_distinct_extensions`, `get_distinct_tags`, `get_file_ids_matching_filters`, `get_file_ids_ordered_for_coverage`; `insert_chunks_batch`, `get_chunks_by_parent_ids`, `delete_chunks_for_file`; `insert_scan_record`, `update_scan_record`, `get_latest_scan`, `get_latest_completed_scan`, `clear_stale_running_scans`; `insert_chat_message`, `get_chat`, `get_chats`, `get_chat_count`, `set_chat_title`, `ensure_chat_exists`, `delete_chat`; `insert_diagnostics_metrics`, `get_diagnostics_metrics_since`; `reset_all_data`, `get_file_count`, `get_chunk_count`, `get_corpus_stats`, `get_indexed_content_size_bytes`.
- **Imports:** aiosqlite, config, db.models
- **Imported by:** api routes, indexer pipeline, llm.rag, llm.retrieval, llm.handlers.*, diagnostics modules

### `db/vectors.py`
- Manages SQLite vector storage via sqlite-vec extension (VectorStore class); vectors stored in `vec_chunks` table in same SQLite database as metadata. Schema includes `chunk_id`, `file_id`, `file_path`, `filename`, `chunk_text`, `vector`, `year` (nullable, for predicate filtering), `category`, `extension`.
- Data type: `ChunkEmbedding` (chunk_id, file_id, file_path, chunk_text, vector, year, filename, extension, category).
- Provides: `store_embeddings(list[ChunkEmbedding])`, `search_similar`, `delete_by_file_id`, `get_stats`, `drop_all`.
- Uses synchronous sqlite3 connections (called via `asyncio.to_thread` from async contexts).
- **Imports:** sqlite3, sqlite_vec, config
- **Imported by:** indexer.pipeline, llm.rag

### `db/utils.py`
- Shared utilities for database operations: timestamp parsing, row conversion helpers.
- Provides: `parse_timestamp()` (handles multiple SQLite timestamp formats, returns datetime | None), `parse_file_category()` (converts string to FileCategory enum), `parse_json_tags()` (parses JSON tags array), `parse_json_sources()` (parses JSON sources array).
- Used by `db/sqlite.py` for consistent row-to-model conversions.
- **Imports:** json, datetime, structlog, db.models
- **Imported by:** db.sqlite

### `scanner/crawler.py`
- Walks directories using `get_effective_ignore_patterns(settings)` and extension whitelist
- Computes SHA-256 hash per file; returns list of `ScannedFile` (path, filename, extension, size_bytes, content_hash, modified_at)
- Compares against DB via `compare_with_db()` to determine new, changed, unchanged, deleted
- **Imports:** pathlib, hashlib, config, utils.path_utils
- **Imported by:** api.routes_scan, api.routes_index

### `scanner/watcher.py`
- Uses watchdog to monitor watched directories for changes
- Emits events that trigger incremental re-indexing
- **Imports:** watchdog, config
- **Imported by:** main.py (started in lifespan)

### `scanner/extractors/*.py`
- Extractors: TextExtractor (.txt, .md, .rst, .log), DoclingExtractor (unified: .pdf, .docx, .pptx, .xlsx, .html, .csv), EpubExtractor (.epub). Registry in `base.py`: `register_extractors()`, `get_extractor(path)`.
- **DoclingExtractor** uses docling's `DocumentConverter` to convert documents to markdown (runtime provided by the `docling-slim` dependency). Docling provides superior structure preservation including tables, formulas, reading order detection, and built-in OCR support. The converter handles all document types (text-based, scanned, image-only) automatically without requiring external OCR tools.
- **text_utils.py:** shared utilities: `elapsed_ms()`, `decode_bytes()` (UTF-8 then chardet), `repair_hyphenation()` (rejoin hyphenated line breaks); used by extractors and by indexer/post_process.
- All extractors implement `BaseExtractor` protocol; must never raise — return errors in `ExtractedDocument.error`.
- **Imports:** docling, base, text_utils where needed
- **Imported by:** indexer.pipeline (via get_extractor), indexer.post_process (repair_hyphenation)

### `indexer/chunker.py`
- Splits text into overlapping chunks using parent-child chunking (v2): child chunks (~150 tokens) for precise search, parent windows (~512 tokens) for LLM context. Respects paragraph/sentence boundaries; uses pysbd for sentence segmentation, tiktoken (cl100k_base) for token counts.
- `chunk_text(text, chunk_size=None, overlap=None, char_to_page_ranges=None, char_to_block_type_ranges=None, char_to_header_level_ranges=None) -> list[ChunkData]`. Returns `list[ChunkData]` (content, chunk_index, token_count, page_number, section_path, block_type, parent_chunk_index) — IDs assigned on DB insert. Parent chunks inserted first, then children with `parent_id`.
- **Imports:** pysbd, tiktoken, config
- **Imported by:** indexer.pipeline

### `indexer/embedder.py`
- Loads embedding model (lazy, on first use); default `nomic-ai/nomic-embed-text-v1.5` (768-dim, 8192-token context); `trust_remote_code=True` when loading.
- Task prefixes: `search_document: ` for indexing, `search_query: ` for queries.
- `embed_texts(texts: list[str]) -> list[list[float]]`; batching internally. `unload()` for shutdown (releases joblib/loky resources).
- **Imports:** sentence-transformers, config
- **Imported by:** indexer.pipeline, llm.rag (query embedding)

### `indexer/classifier.py`
- `classify_file(path, extension) -> FileCategory`; `generate_tags(path) -> list[str]` (directory-based); `extract_year(path, extracted_text) -> int | None` (filename/path/text, for temporal filtering).
- Uses `category_patterns.get_category_for_extension()` for category; `file_patterns.YEAR_PATTERN` for year extraction.
- **Imports:** category_patterns, db.models, file_patterns
- **Imported by:** indexer.pipeline

### `indexer/post_process.py`
- Quality refinements on extracted text before chunking: `post_process_extracted_text(text) -> str` applies hyphenation repair. Used at index time only (v2: no query-time cleaning to avoid semantic drift).
- **Imports:** scanner.extractors.text_utils (repair_hyphenation)
- **Imported by:** indexer.pipeline

### `indexer/pipeline.py`
- Orchestrates indexing for a single file: extract → `post_process_extracted_text(doc.text)` → classify/tag/year → insert file → chunk (parent-child: parents first, then children with `parent_id`) → insert chunks → embed (clean content-only, no metadata prefix) → store in SQLite `vec_chunks` table via sqlite-vec with structured metadata fields (filename, category, extension, year).
- `index_file(db, scanned) -> IndexResult`, `reindex_file(db, scanned) -> IndexResult`, `remove_file(db, file) -> bool`. `IndexResult`: path, success, chunks_created, error.
- Deduplicates embedding cache-missing errors per scan; `reset_repeated_embedding_errors()` called at scan start.
- **Imports:** db.sqlite, db.vectors (ChunkEmbedding), chunker, classifier, embedder, extractors.base, indexer.post_process
- **Imported by:** api.routes_scan, api.routes_index

### `indexer/reranker.py`
- Lazy-loads cross-encoder for (query, chunk) re-ranking; model via `rag_reranker_model`; same HF cache as embedding; respects `embedding_offline`. Exposes `unload()` for shutdown.
- Reranking is enabled by default and applied according to runtime settings (`rag_rerank`, `rag_rerank_coverage`).
- **Imports:** sentence-transformers (CrossEncoder), config
- **Imported by:** llm.retrieval

### `indexer/adaptive_tuning.py`
- Corpus-aware retrieval top-k tuning. When `adaptive_rag_tuning` is enabled, computes top-k from corpus stats (file count, parent chunk count) using config formula constants.
- In-memory cache populated at startup, scan completion, rebuild completion; invalidated on index reset.
- `update_tuning_cache(db)` (async), `invalidate_tuning_cache()`, `get_effective_top_k(query_type)` (sync), `calculate_adaptive_top_k(...)`.
- **Imports:** config, db.sqlite (get_corpus_stats), llm.model_adapter (get_profile)
- **Imported by:** main (lifespan), api.routes_scan, api.routes_index, llm.model_adapter (get_retrieval_top_k)

### `llm/engine.py`
- Loads GGUF via xllamacpp (CommonParams + Server, in-process); default `llm_model_filename` = `Qwen3.6-35B-A3B-Q4_K_M.gguf`; Apple Metal by default.
- Chat template extracted from GGUF metadata via `gguf.GGUFReader` at load time. Token counting via tiktoken cl100k_base (±15% approximation).
- Provides `generate_stream`; `count_tokens(text)` for RAG prompt budget. Handles model download when not local-only.
- Uses `utils.directory_utils.ensure_file_directory()` for model directory creation.
- **Imports:** xllamacpp, gguf, tiktoken, huggingface_hub, config, utils.directory_utils
- **Imported by:** llm.rag

### `llm/model_adapter.py`
- Per-model configuration via frozen `ModelProfile` dataclass. Each supported model has its own profile with prompt format, reasoning behavior, token limits, stop sequences, and post-processing rules.
- Profiles: `QWEN3_14B_PROFILE`, `QWEN3_5_9B_PROFILE`, `QWEN3_5_35B_A3B_PROFILE`, `DEFAULT_PROFILE`.
- Enums: `ModelFamily` (CHATML, LLAMA, MISTRAL), `PromptFormat` (NATIVE_GGUF, CHATML), `ReasoningMode` (ALWAYS, FOCUSED_ONLY, NEVER).
- Profile methods: `get_stop_sequences(reasoning_enabled)`, `get_max_tokens(query_type)`, `get_reasoning_enabled(query_type)`, `get_prompt_format(query_type)`, `prepare_messages(messages, query_type)`.
- `get_profile()` / `get_profile_for_filename(filename)` for profile resolution.
- `get_retrieval_top_k(query_type)` — single source for retrieval top-k. When `adaptive_rag_tuning` enabled, returns corpus-aware value from cache; else profile base.
- **Imports:** config
- **Imported by:** llm.rag, llm.handlers.rag, api.routes_chat

### `llm/rag.py`
- `_resolve_handler_for_classification` (v2): classifies query via `query_classifier.classify_query()` → dispatches to handler (MetadataHandler, SimpleHandler, or RAGHandler) based on intent.
- **Imports:** query_classifier, handlers (metadata, simple, rag), db.models (ChatMessage), chat_trace (TraceWriter)
- **Imported by:** api.routes_chat

### `llm/query_classifier.py`
- Deterministic slot extraction and intent routing (v2). Classifies query using NLP heuristics + promptcue intent router (no separate LLM call). Extracts year, category, file_type, filename filters; detects intent and assigns IntentProfileId. Applies term dictionary expansion via `term_dictionary.expand_query_for_routing()`.
- Consumes PromptCue `prompt_signals` (when available) for continuation and requested output-format handling, while keeping app-specific routing policy local.
- Returns `QueryClassification` dataclass with intent, filters, intent profile, output shape, group-by, block type, and routing reason codes.
- **Imports:** structlog, query_patterns, promptcue_signals, intent_router, term_dictionary, llm.types
- **Imported by:** llm.rag, llm.handlers.*

### `llm/query_patterns.py`
- Standardized patterns for query intent classification: count, file-listing, coverage, aggregation, continuation, referential follow-up, entity inventory, structured output, and more.
- Provides building-block functions: `build_count_pattern()`, `build_file_list_pattern()`, `build_coverage_pattern()`, `build_aggregation_pattern()`, `build_referential_followup_pattern()`, `build_global_entity_listing_pattern()`, `build_exhaustive_entity_inventory_scope_pattern()`, etc.
- Single source of truth for query pattern regexes used across the codebase.
- **Imports:** re
- **Imported by:** llm.query_classifier

### `llm/rag_patterns.py`
- Shared RAG intent and topic-shift cue patterns that coordinate classifier/runtime behaviors.
- Keeps RAG-specific pattern ownership centralized (separate from generic query-pattern inventory).
- **Imports:** llm.promptcue_signals (plus stdlib helpers)
- **Imported by:** llm handlers/runtime modules and context-scope logic

### `llm/promptcue_signals.py`
- App-side adapter for prompt-shape cues. Uses precomputed PromptCue outputs when available, otherwise evaluates centralized PromptCue pattern constants directly (no extra model/classification pass).
- Exposes `extract_prompt_signals()` returning normalized cue snapshot (`has_topic_shift_cue`, `has_referential_followup`, `requests_continuation`, output-format cues, etc.).
- Keeps policy decisions in app modules (`context_scope_manager`, `query_classifier`, handlers) while avoiding duplicated generic cue regex ownership.
- **Imports:** re (+ optional `promptcue.patterns`)
- **Imported by:** llm.query_classifier, llm.rag_patterns, api.context_scope_manager

### `llm/roles.py`
- Defines chat role profiles and role registry helpers used for role-scoped behavior and settings exposure.
- Provides visibility filtering and stable role lookup by ID.
- **Imports:** dataclasses/typing utilities
- **Imported by:** API routes and chat/runtime role-selection paths

### `llm/retrieval.py`
- Unified retrieval pipeline (v2): embed query → vector search with WHERE clauses (year, category, extension filters, upload-source exclusion for unscoped corpus turns) → rerank (when enabled by settings) → top-k. For coverage queries, uses file-anchored retrieval (one chunk per file, exhaustive). Supports summary-oriented substantive-section preference to de-prioritize structural sections (appendix/contents/etc.) when synthesis intent is detected.
- Records metrics in trace writer: `raw_chunks_count`, `children_reranked`, `children_after_structural_filter`, `children_returned`, `parents_returned`.
- **Imports:** embedder, reranker, db.vectors, db.sqlite (get_chunks_by_parent_ids, get_file_ids_matching_filters), metadata_filters, upload_policy (`UPLOAD_PROVIDER`, `UPLOAD_ENTITY_TYPE`), chat_trace (TraceWriter)
- **Imported by:** llm.handlers.rag

### `llm/prompt_builder.py`
- Prompt construction: builds messages with system prompt (3 rules), context chunks, chat history. Token budget managed by `rag_context_ratio`.
- **Imports:** llm.model_adapter (get_profile), config
- **Imported by:** llm.handlers.rag

### `llm/streaming.py`
- Minimal streaming: yields tokens from LLM engine, no post-processing bandaids.
- **Imports:** llm.engine
- **Imported by:** llm.handlers.rag

### `llm/metadata_filters.py`
- Unified metadata filter extraction and WHERE clause building for sqlite-vec queries.
- Extracts year, category, file_type, and filename filters from queries using standardized patterns from `file_patterns`.
- `extract_metadata_filters()` returns list of `MetadataFilter` objects; `build_where_clause()` converts to SQLite WHERE clause string for `vec_chunks` table queries.
- Filename filters are only applied to metadata queries (file listing/counting), not content queries (focused/coverage) where filename is semantic context.
- **Imports:** re, dataclasses, file_patterns
- **Imported by:** llm.retrieval, llm.query_classifier

### `llm/handlers/query_handler.py`
- QueryHandler protocol: `matches(classification) -> bool`, `handle(...) -> AsyncGenerator[str | list[ChatSourceReference], None]`.
- **Imports:** typing (Protocol), query_classifier (QueryClassification)
- **Imported by:** llm.handlers.metadata, llm.handlers.rag, llm.handlers.simple

### `llm/handlers/metadata.py`
- MetadataHandler: handles metadata queries (count, enumeration, aggregation, file listing) using SQLite only, no vector search.
- Supports aggregation queries: date range (min/max years), per year counts (grouped by year with optional filters).
- **Imports:** query_handler, db.sqlite (get_distinct_years), query_classifier, query_patterns
- **Imported by:** llm.rag

### `llm/handlers/rag.py`
- RAGHandler: handles focused and coverage queries using vector search → rerank → LLM pipeline.
- **Imports:** query_handler, retrieval, prompt_builder, streaming, model_adapter, query_classifier, db.sqlite, db.models
- **Imported by:** llm.rag

### `llm/handlers/simple.py`
- SimpleHandler: handles simple/conversational queries (greetings, clarifications, off-topic) with direct LLM, no retrieval.
- **Imports:** query_handler, streaming, model_adapter, query_classifier
- **Imported by:** llm.rag

### `api/operation_state.py`
- Module-level flags for long-running operations: `reset_in_progress`, `last_reset_result`; `STALE_SCAN_THRESHOLD_SECONDS`. `resolve_running_scan(db, force, operation)` — checks for running scan, cancels if force or marks failed if stale; raises HTTP 409 if recent scan running and force=False.
- **Imports:** db.sqlite (get_latest_scan, update_scan_record), db.models
- **Imported by:** api.routes_scan, api.routes_index

### `api/routes_scan.py`
- `POST /api/scan` — trigger scan (background); supports force=true to cancel running/stale scan (uses operation_state.resolve_running_scan)
- `GET /api/scan/status` — current scan status
- `GET /api/scan/errors` — list scan errors for the latest scan
- `GET /api/files` — list indexed files (paginated, filterable)
- `GET /api/files/{id}` — single file details + extracted text preview
- `POST /api/files/{id}/reindex` — re-index a single file
- `DELETE /api/files/{id}` — remove file from index
- `POST /api/files/open` — open file in system default app (body: OpenFileRequest with path)
- **Imports:** scanner.crawler, scanner.extractors.base (register_extractors), indexer.pipeline (index_file, reindex_file, remove_file, reset_repeated_embedding_errors), db.sqlite, config (get_effective_ignore_patterns), api.operation_state

### `api/routes_index.py`
- `POST /api/index/rebuild` — full re-index of all indexed files (background); body `RebuildRequest` with `force` to cancel running scan/rebuild
- `GET /api/index/status` — index statistics (IndexStatusResponse: total_files, chunks, embeddings, sizes, reset_in_progress, last_reset_result)
- `POST /api/index/reset` — delete all indexed data (vectors, SQLite tables, chat messages), clear watched_directories in config
- `GET /api/index/term-dictionary/status` — term dictionary build status and statistics
- `POST /api/index/term-dictionary/rebuild` — trigger a term dictionary rebuild from current corpus
- `POST /api/index/term-dictionary/purge` — delete all term dictionary data
- **Imports:** indexer.pipeline (reindex_file), indexer.term_dictionary_builder, db.sqlite, db.vectors, scanner.crawler, scanner.extractors.base, api.operation_state

### `api/routes_search.py`
- `POST /api/search` — semantic search across documents
- **Imports:** indexer.embedder, db.vectors, db.sqlite

### `api/routes_chat.py`
- `POST /api/chat` — send message, stream response (SSE); returns chat_id in first event
- Request supports optional scoped retrieval via:
  - `scoped_file_ids` (one or more indexed file IDs)
  - `scoped_upload_ids` (one or more chat upload IDs; Researcher mode only)
- Upload lifecycle endpoints:
  - `GET /api/chat/chats/{chat_id}/uploads` — list chat-scoped uploads
  - `POST /api/chat/uploads` — upload + index a temporary chat attachment
- `DELETE /api/chat/uploads/{upload_id}` — delete one chat attachment (bytes + index artifacts)

### `api/context_scope_manager.py`
- Resolves retrieval/generation scope continuity across turns (topic shift vs referential follow-up cues).
- Tracks scoped pass progression and “remaining scope” semantics used by continuation flows.
- **Imports:** llm.promptcue_signals, llm.query_classifier, config/runtime helpers
- **Imported by:** api.routes_chat and chat orchestration flows
- `POST /api/chat/stop` — stop active stream by stream/request/chat identifiers
- `GET /api/chat/chats` — list chats (chat_id, last_message_preview, title, etc.)
- `GET /api/chat/chats/{chat_id}` — chat messages for one chat
- `PUT /api/chat/chats/{chat_id}/title` — set chat title
- `DELETE /api/chat/chats/{chat_id}` — delete chat and its messages
- Optionally collects diagnostics metrics when `diagnostics_metrics_enabled=True` (lazy import from `diagnostics.observer`, stores in `response_diagnostics_metrics` table via `insert_diagnostics_metrics()`)
- Uses `utils.json_utils.serialize_api_response()` for SSE event data serialization.
- **Imports:** llm.rag, db.sqlite, chat_trace (get_trace_writer, flush_trace_writer), config (settings), utils.json_utils
- **Lazy import (conditional):** `diagnostics.observer` (EvalMetrics, detect_issues, populate_signals) — only when `settings.diagnostics_metrics_enabled=True`

### `api/routes_settings.py`
- `GET /api/settings` — current settings (SettingsResponse, includes file_type_options, available_models, config_file_path)
- `PUT /api/settings` — partial update (SettingsUpdateRequest); validates and persists to config.json; syncs full_privacy → embedding_offline, llm_local_only
- `GET /api/config/env-vars` — env variable groups for Configuration page (EnvVarsResponse)
- `GET /api/config/reference` — application constants reference (ConfigReferenceResponse)
- `GET /api/file-types` — canonical file type options (list[FileTypeOption])
- Uses `utils.json_utils.serialize_config()` for config file writes, `utils.directory_utils.ensure_file_directory()` for directory creation, `utils.path_utils.resolve_and_check_path()` for path validation.
- **Imports:** schemas, config, env_vars_metadata (get_env_vars_response), config_reference_metadata (get_config_reference_response), file_types (get_file_type_options), utils.json_utils, utils.directory_utils, utils.path_utils

### `api/env_vars_metadata.py`
- Defines groups and descriptions for INFORMITY_* env vars; `get_env_vars_response(settings) -> EnvVarsResponse` for GET /api/config/env-vars.
- **Imports:** schemas, config (APP_SLUG)
- **Imported by:** api.routes_settings

### `api/routes_system.py`
- `GET /api/diagnostics` — returns system diagnostics (app version, Python version, OS, RAM, disk space, model info, DB stats, uptime)
- `POST /api/shutdown` — gracefully shutdown application (localhost-only, for Tauri sidecar lifecycle)
- **Imports:** platform, psutil, structlog, fastapi, pydantic, config, db.sqlite, db.vectors, llm.engine
- **Imported by:** main.py

### `file_types.py`
- Canonical list `FILE_TYPE_OPTIONS` (id, label, extensions); `get_file_type_options()` for Settings UI and file filtering.
- **Imported by:** api.routes_settings, file_patterns

### `file_patterns.py`
- Standardized patterns for file metadata extraction and matching: extension lists, filename patterns, year extraction.
- Provides: `get_all_supported_extensions()`, `get_extensions_without_dot()`, `YEAR_PATTERN`, `extract_year_from_text()`, `build_filename_detection_patterns()`, `build_extension_query_patterns()`, etc.
- Single source of truth for file-related regex patterns used across the codebase.
- **Imports:** re, file_types
- **Imported by:** llm.metadata_filters, indexer.classifier

### `upload_policy.py`
- Upload-scope ingestion policy and limits for chat attachments.
- Provides scope IDs: `UPLOAD_PROVIDER`, `UPLOAD_ENTITY_TYPE`.
- Provides limits: `MAX_UPLOAD_FILE_SIZE_MB`, `MAX_UPLOAD_FILES_PER_CHAT`, `MAX_UPLOAD_TOTAL_SIZE_MB`.
- Provides helpers: `upload_root_dir()`, `max_upload_file_size_bytes()`, `max_upload_total_size_bytes()`, `is_allowed_extension(filename)`, `is_allowed_mime(content_type)`.
- **Imports:** pathlib, config
- **Imported by:** api.routes_chat, api.routes_index, api.routes_scan

### `utils/path_utils.py`
- Standardized path resolution and normalization utilities.
- Provides: `normalize_path()` (resolves and expands user home), `normalize_paths()` (batch normalization), `resolve_and_check_path()` (validates path existence).
- **Imports:** pathlib
- **Imported by:** config, scanner.crawler, scanner.watcher, indexer.pipeline, api.env_vars_metadata, api.routes_settings

### `utils/json_utils.py`
- Standardized JSON serialization patterns for different contexts.
- Provides: `serialize_config()` (config file format with trailing newline), `serialize_trace()` (trace file format), `serialize_api_response()` (SSE event data), `parse_json_safe()` (safe JSON parsing with defaults).
- **Imports:** json
- **Imported by:** config, api.routes_settings, api.routes_chat, chat_trace

### `utils/directory_utils.py`
- Standardized directory creation and management utilities.
- Provides: `ensure_directory()` (creates directory if missing), `ensure_directories()` (batch creation), `ensure_file_directory()` (ensures parent directory of file path).
- **Imports:** pathlib
- **Imported by:** config, logging_config, chat_trace, llm.engine, api.routes_settings

### `utils/file_utils.py`
- File-name/extension normalization helpers shared across indexing and retrieval.
- Provides: `normalize_extension(extension: str | None) -> str` (canonical lowercase extension with leading `.` or empty string).
- **Imports:** none
- **Imported by:** llm.retrieval, llm.handlers.metadata, indexer.pipeline

### `utils/number_utils.py`
- Safe numeric coercion helpers for API/runtime payload parsing.
- Provides: `safe_int(value, default=0) -> int`, `safe_float(value, default=0.0) -> float`.
- **Imports:** none
- **Imported by:** api.routes_chat

### `logging_config.py`
- Configures structlog: console + general.log (app log_level) + errors.log (ERROR only); daily rotation, 7-day retention; third-party loggers set to WARNING.
- Uses `utils.directory_utils.ensure_directory()` for log directory creation.
- **Imports:** structlog, config, utils.directory_utils
- **Imported by:** main (before creating loggers)

### `chat_trace.py`
- Per-chat trace logging when `chat_trace_logging` is True; `TraceWriter` protocol; used by llm.rag, llm.retrieval for debugging/analysis. Records trace steps (intent, retrieval, prompt, llm, sources) to JSON files.
- Uses `utils.directory_utils.ensure_directory()` and `ensure_file_directory()` for trace directory creation, `utils.json_utils.serialize_trace()` for trace file serialization.
- **Imports:** pathlib, json, structlog, config, utils.directory_utils, utils.json_utils
- **Imported by:** llm.rag, llm.retrieval, llm.handlers.rag

---

## Diagnostics

Core types live in `src/informity/diagnostics/` (issue_types, observer, resource_snapshot). Strict one-way imports: diagnostics → informity ✅ | informity → diagnostics ❌ (except one lazy conditional import in `routes_chat.py`).

### `informity/diagnostics/issue_types.py`
- `IssueType` enum (6 types for v2): `retrieval_failure`, `insufficient_retrieval`, `empty_answer`, `refusal_bias`, `timeout`, `very_short_answer`.
- **Imported by:** diagnostics.observer

### `informity/diagnostics/observer.py`
- `EvalMetrics` dataclass (OTel-named fields via `openinference-semantic-conventions`): chat_id, question, model_filename, query_type, raw_chunks_count, sources_count, generation_seconds, answer_length, timeout_occurred, has_empty_answer, has_refusal_pattern.
- `detect_issues(answer: str, metrics: EvalMetrics) -> list[IssueType]` — heuristic issue detection.
- `populate_signals(answer: str, metrics: EvalMetrics) -> dict` — quality signal extraction.
- **Imports:** diagnostics.issue_types, openinference.semconv.trace (SpanAttributes, DocumentAttributes)
- **Imported by:** api.routes_chat (lazy conditional)

### `informity/diagnostics/resource_snapshot.py`
- System resource snapshot at trace time: CPU, memory, disk. Used by observer to attach resource context to diagnostics metrics.
- **Imported by:** diagnostics.observer

### `main.py`
- FastAPI app; health `GET /api/health` (HealthResponse); mounts static frontend from `src/frontend/dist/` when built (Vite output). Vanilla backup archived at `.archive/frontend-bak/`.
- Sets HF env (HF_HOME, HF_HUB_CACHE, HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE from full_privacy/embedding_offline) before importing embedder/reranker.
- Lifespan: ensure_directories, remove_models_dir_cache, init_db, clear_stale_running_scans, start_watcher; shutdown: stop_watcher, _cleanup_models (embedder.unload, reranker.unload), _kill_child_processes.
- Signal/atexit: _cleanup_models; SIGTERM/SIGINT (when not dev_reload) call _signal_cleanup then _exit.
- **Imports:** routers (scan, index, chat, search, settings), config, db.sqlite (init_db, clear_stale_running_scans), embedder, reranker, watcher, logging_config, llm.engine (remove_models_dir_cache)

---


---

## Dependency Graph (Import Direction)

```
config.py ← (everything imports config)
     ↑
db/models.py ← (shared types, no other internal imports)
     ↑
db/sqlite.py    db/vectors.py
     ↑                ↑
scanner/           indexer/
  crawler.py         chunker.py
  watcher.py         classifier.py
  extractors/*       embedder.py ←─────┐
     ↑               pipeline.py      │
     │               reranker.py ─────┤
     └── api/routes_scan.py            │
     └── api/routes_index.py           │
                                       │
file_types.py  env_vars_metadata.py    │
     ↑                ↑                │
api/routes_settings.py                 │
                                       │
llm/engine.py   llm/model_adapter.py      │
     ↑                 ↑                  │
llm/query_classifier   │                  │
     ↑                 │                  │
llm/retrieval.py       │                  │
llm/prompt_builder.py  │                  │
llm/streaming.py       │                  │
llm/metadata_filters.py│                  │
llm/handlers/*         │                  │
     ↑                 │                  │
     └── llm/rag.py ←──┘ ← chat_trace ─┘
     ↑
api/routes_chat.py  api/routes_search.py
     ↑
     │ (lazy conditional import only)
     ↓
diagnostics/ (sibling package)
  issue_types.py
  observer.py ──────┐
  resource_snapshot.py │
     ↑
main.py ← logging_config (configure_logging before loggers)
```

**Rule: imports only flow upward in this diagram. No circular dependencies.**
**Diagnostics rule:** `diagnostics` → `informity` ✅ | `informity` → `diagnostics` ❌ (except one lazy conditional import in `routes_chat.py`).

---

## Prompts

### RAG System Prompt (llm/rag.py)
- Header: answer generator (not conversational); answer using ONLY provided context; output ONLY final answer — no intros, sources, or meta-commentary; start with answer content; never start with "Based on", "According to", etc.
- Reasoning behavior is profile-controlled (`model_adapter.ReasoningMode`) rather than a global `rag_enable_reasoning` switch.
- Footer: if list, start with first item; if statement, start with key fact; if insufficient context, say exactly "The available documents do not contain enough information to answer this question."; no speculation; use markdown (**bold**, bullets, tables); concise; stop after answer; no sources in answer.

### RAG Context Template (per chunk)
```
[Source: {filename} | Path: {path}]
{chunk_text}
```
(Page number omitted in current template; year is in embedding prefix, not in context template.)

---

## File Path Conventions

| What | Path |
|---|---|
| App data root | `~/.informity` (default); override via `INFORMITY_APP_DATA_DIR` |
| SQLite database | `{app_data_dir}/db/informity.db` |
| SQLite vectors | Stored in `vec_chunks` table within SQLite database (sqlite-vec extension) |
| LLM models (RAG) | `{app_data_dir}/models/llm/` (shared between desktop .app and development workflows) |
| Config file | `{app_data_dir}/config.json` |
| Logs | `{app_data_dir}/logs/` |
| Unified cache root | `{app_data_dir}/cache/` (default) or override via `INFORMITY_CACHE_DIR` |
| Hugging Face cache (embedding + reranker) | `{cache_dir}/huggingface/hub/` (hub/ + modules/) |
| Docling models | `{cache_dir}/docling/` (docling creates its own structure inside) |
| Diagnostics directory | `{app_data_dir}/diagnostics/` (default) or `{diagnostics_dir}/` if set |
| Diagnostics evaluation runs | `{diagnostics_dir}/runs/{run_id}/` (queries/ with golden_queries.json, queries.json; traces/; results/ with run.json, report.md, report.json, llm_insights.json, tasks.json, tasks.md) |
| Trace files (evaluation) | `{diagnostics_dir}/runs/{run_id}/traces/{chat_id}--{message_id}.json` |
| Trace files (user chats) | `{app_data_dir}/chats/{chat_id}/{message_id}.json` (when chat_trace_logging enabled) |

---

## API Endpoints Summary

| Method | Path | Description | Background? |
|--------|------|-------------|-------------|
| `POST` | `/api/scan` | Trigger file scan + index (force=true to cancel running) | Yes |
| `GET` | `/api/scan/status` | Current scan progress | No |
| `GET` | `/api/scan/errors` | Scan errors for the latest scan | No |
| `GET` | `/api/files` | List indexed files (paginated, filterable) | No |
| `GET` | `/api/files/{id}` | Single file detail | No |
| `POST` | `/api/files/{id}/reindex` | Re-index a single file | No |
| `DELETE` | `/api/files/{id}` | Remove file from index | No |
| `POST` | `/api/files/open` | Open file in system default app (body: `{ path }`) | No |
| `POST` | `/api/search` | Semantic search | No |
| `POST` | `/api/index/rebuild` | Force full re-index (body: RebuildRequest.force to cancel running) | Yes |
| `GET` | `/api/index/status` | Index statistics (incl. reset_in_progress, last_reset_result) | No |
| `POST` | `/api/index/reset` | Delete all indexed data, clear watched_directories | No |
| `GET` | `/api/index/term-dictionary/status` | Term dictionary build status | No |
| `POST` | `/api/index/term-dictionary/rebuild` | Trigger term dictionary rebuild | Yes |
| `POST` | `/api/index/term-dictionary/purge` | Delete all term dictionary data | No |
| `POST` | `/api/chat` | Send message, stream response (SSE) | No (streaming) |
| `POST` | `/api/chat/stop` | Stop active chat stream | No |
| `GET` | `/api/chat/chats` | List chats | No |
| `GET` | `/api/chat/chats/{chat_id}` | Get chat messages | No |
| `GET` | `/api/chat/chats/{chat_id}/uploads` | List chat-scoped uploaded attachments | No |
| `POST` | `/api/chat/uploads` | Upload + index temporary chat attachment | No |
| `DELETE` | `/api/chat/uploads/{upload_id}` | Delete one chat-scoped uploaded attachment | No |
| `PUT` | `/api/chat/chats/{chat_id}/title` | Set chat title | No |
| `DELETE` | `/api/chat/chats/{chat_id}` | Delete chat and messages | No |
| `GET` | `/api/settings` | Get current settings | No |
| `PUT` | `/api/settings` | Update settings (partial) | No |
| `POST` | `/api/settings/reset` | Reset all settings to factory defaults (Qwen3.6 35B A3B) | No |
| `GET` | `/api/config/env-vars` | Env variable groups for Configuration page | No |
| `GET` | `/api/file-types` | Canonical file type options | No |
| `GET` | `/api/health` | Health check (HealthResponse) | No |
| `GET` | `/api/diagnostics` | System diagnostics (app version, Python version, OS, RAM, disk, model info, DB stats) | No |
| `POST` | `/api/shutdown` | Gracefully shutdown application (localhost-only, for Tauri) | No |

---

## Common Patterns

### Background task pattern (scan/index)
```python
@router.post("/api/scan")
async def trigger_scan(
    request: ScanRequest,
    background_tasks: BackgroundTasks,
    db = Depends(get_db),
):
    scan_record = await create_scan_record(db)
    background_tasks.add_task(run_scan, scan_record.id, request)
    return {"scan_id": scan_record.id, "status": "started"}
```

### SSE streaming pattern (chat)
```python
from sse_starlette.sse import EventSourceResponse

@router.post("/api/chat")
async def chat(request: ChatRequest):
    async def event_generator():
        async for token in rag.answer_question(
            request.message,
            request.chat_id,
            file_ids=request.scoped_file_ids,
        ):
            yield {"event": "token", "data": token}
        yield {"event": "done", "data": ""}
    return EventSourceResponse(event_generator())
```

### Extractor registry pattern (scanner/extractors/base.py)
```python
EXTRACTOR_REGISTRY: dict[str, BaseExtractor] = {}
_registry_initialized = False

def register_extractors() -> None:
    # Instantiate all extractors and register them by supported extension.
    # Imports inside to avoid circular imports.
    global _registry_initialized
    if _registry_initialized:
        return
    
    from informity.scanner.extractors.docling import DoclingExtractor
    from informity.scanner.extractors.epub import EpubExtractor
    from informity.scanner.extractors.text import TextExtractor
    
    extractor_classes = [
        DoclingExtractor,  # Unified extractor for PDF, DOCX, PPTX, XLSX, HTML, CSV
        EpubExtractor,     # EPUB ebooks
        TextExtractor,      # Plain text files (.txt, .md, .rst, .log)
    ]
    
    for extractor_class in extractor_classes:
        extractor = extractor_class()
        for ext in extractor.supported_extensions:
            EXTRACTOR_REGISTRY[ext] = extractor
    
    _registry_initialized = True

def get_extractor(path: Path) -> BaseExtractor | None:
    # Look up the appropriate extractor for a file by its extension.
    if not _registry_initialized:
        register_extractors()
    return EXTRACTOR_REGISTRY.get(path.suffix.lower())
```
