# ==============================================================================
# Informity AI — Database Models
# Pydantic models that map to SQLite rows. These are the canonical data
# structures shared across the application.
# ==============================================================================

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field

from informity.llm.types import ChatRole, CompletionMode, NextAction

# ==============================================================================
# Enums
# ==============================================================================

class FileCategory(StrEnum):
    # Categories for indexed files, based on extension group
    DOCUMENT  = 'document'    # .pdf, .docx, .pptx
    PLAINTEXT = 'plaintext'   # .txt, .md, .rst, .log
    DATA      = 'data'        # .csv, .xlsx (tabular data; config formats map to PLAINTEXT)
    WEB       = 'web'         # .html, .htm
    CODE      = 'code'        # .py, .js, .ts (future)
    OTHER     = 'other'


class ScanStatus(StrEnum):
    # Status of a scan run
    RUNNING   = 'running'
    COMPLETED = 'completed'
    FAILED    = 'failed'
    CANCELLED = 'cancelled'


# ==============================================================================
# IndexedFile — maps to the `files` table
# ==============================================================================

class IndexedFile(BaseModel):
    # Represents a file in the SQLite `files` table.
    id:                     int | None       = None
    path:                   str                         # Absolute POSIX path
    filename:               str
    extension:              str
    size_bytes:             int
    content_hash:           str                         # SHA-256
    extracted_text_preview: str                         # First ~500 chars
    category:               FileCategory
    tags:                   list[str]        = Field(default_factory=list)  # Stored as JSON
    year:                   int | None       = None     # Extracted at index time (filename/path/text)
    extractor:              str | None       = None
    encoding:               str | None       = None
    language:               str | None       = None
    mime_type:              str | None       = None
    ocr_used:               bool             = False
    page_count:             int | None       = None
    tables_count:           int | None       = None
    form_items_count:       int | None       = None
    key_value_items_count:  int | None       = None
    pictures_count:         int | None       = None
    document_hash:          str | None       = None
    indexed_at:             datetime | None  = None
    modified_at:            datetime
    created_at:             datetime | None  = None


# ==============================================================================
# Chunk — maps to the `chunks` table
# ==============================================================================

class Chunk(BaseModel):
    # Represents a text chunk in the SQLite `chunks` table.
    # v2 additions: parent_id, page_number, section_path, block_type for parent document retrieval and filtering
    id:           int | None      = None
    file_id:      int
    chunk_index:  int
    content:      str
    token_count:  int
    parent_id:    int | None      = None     # v2: Link to parent window chunk (for parent document retrieval)
    page_number:  int | None      = None     # v2: Page number in source document (PDF)
    start_page:   int | None      = None     # v2: Start page for chunks spanning multiple pages
    end_page:     int | None      = None     # v2: End page for chunks spanning multiple pages
    section_path: str | None      = None     # v2: Section hierarchy path (e.g., "Introduction/Overview")
    block_type:   str | None      = None     # v2: Block type ('table', 'form', 'narrative') from docling provenance
    created_at:   datetime | None = None


# ==============================================================================
# ScanRecord — maps to the `scan_history` table
# ==============================================================================

class ScanRecord(BaseModel):
    # Represents a scan run in the SQLite `scan_history` table.
    id:            int | None      = None
    started_at:    datetime
    completed_at:  datetime | None = None
    files_scanned: int             = 0
    files_indexed: int             = 0
    errors:        int             = 0
    status:        ScanStatus      = ScanStatus.RUNNING


class ScanErrorRecord(BaseModel):
    # Represents a per-file scan error in `scan_errors`.
    id:            int | None      = None
    scan_id:       int
    path:          str
    filename:      str
    extension:     str
    operation:     str
    error_code:    str | None      = None
    error_message: str
    is_timeout:    bool            = False
    created_at:    datetime | None = None


# ==============================================================================
# ChatMessage — maps to the `chat_messages` table
# ==============================================================================

class ChatMessage(BaseModel):
    # A single message in a chat.
    id:                int | None = None
    chat_id:           str         # UUID
    role:              ChatRole
    content:          str
    sources:          list[dict] = Field(default_factory=list)  # Full source reference objects
    generation_seconds: float | None = None  # Time taken to generate answer (assistant messages only)
    completion_mode: CompletionMode | str | None = None
    stopped_by_user: bool = False
    has_remaining_scope: bool = False
    next_action: NextAction | str | None = None
    next_action_reason: str | None = None
    chat_mode: str | None = None
    is_internal: bool = False
    created_at:       datetime | None = None


class ContinuationPassArtifact(BaseModel):
    # Durable per-pass continuation artifact for diagnostics/compaction input.
    # Non-authoritative: chat_messages.content remains canonical assistant raw output.
    id: int | None = None
    chat_id: str
    request_id: str
    pass_index: int
    stitch_mode: str  # 'append' | 'overwrite'
    raw_answer: str = ''
    cleaned_answer: str = ''
    has_remaining_scope: bool = False
    completion_mode: CompletionMode | str | None = None
    next_action_reason: str | None = None
    sources: list[dict] = Field(default_factory=list)
    pass_details: dict = Field(default_factory=dict)
    status_transitions: list[dict] = Field(default_factory=list)
    payload_hash: str = ''
    created_at: datetime | None = None
