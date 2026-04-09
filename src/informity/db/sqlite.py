# ==============================================================================
# Informity AI — SQLite Database Module (v2)
# Async connection management via aiosqlite. All SQL queries live here.
# v2: sqlite-vec for vector search, FTS5 for candidate augmentation.
# ==============================================================================

import asyncio
import hashlib
import json
from collections.abc import AsyncGenerator
from contextlib import suppress
from datetime import UTC, datetime

import aiosqlite
import structlog

from informity.config import settings
from informity.db.models import (
    ChatMessage,
    Chunk,
    ContinuationPassArtifact,
    IndexedFile,
    ScanErrorRecord,
    ScanRecord,
    ScanStatus,
)
from informity.db.utils import (
    parse_file_category,
    parse_json_sources,
    parse_json_tags,
    parse_timestamp,
)
from informity.diagnostics.issue_types import IssueType
from informity.llm.types import ChatRole, DiagnosticsQueryType

# ==============================================================================
# Logger
# ==============================================================================

log = structlog.get_logger(__name__)
_SQLITE_EXTENSION_LOAD_EXCEPTIONS = (
    ImportError,
    AttributeError,
    TypeError,
    ValueError,
    RuntimeError,
    OSError,
    aiosqlite.Error,
)
_SQLITE_BUSY_TIMEOUT_MS = 5000
_CHAT_PREVIEW_TRUNCATE_LENGTH = 100
_RESET_SCHEMA_RETRY_ATTEMPTS = 15
_RESET_SCHEMA_RETRY_BASE_DELAY_SECONDS = 0.2
_RESET_COMPACTION_RETRY_ATTEMPTS = 10

# ==============================================================================
# Schema — DDL statements for all tables
# ==============================================================================

SCHEMA_VERSION = 2

DIAGNOSTICS_TYPE_USER = 'user'
DIAGNOSTICS_TYPE_EVALUATION = 'evaluation'
CANONICAL_DIAGNOSTICS_TYPES = (DIAGNOSTICS_TYPE_USER, DIAGNOSTICS_TYPE_EVALUATION)
CANONICAL_DIAGNOSTICS_QUERY_TYPES = tuple(item.value for item in DiagnosticsQueryType)
CANONICAL_DIAGNOSTICS_ISSUE_TYPES = tuple(sorted(issue.value for issue in IssueType))

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY
);

CREATE TABLE IF NOT EXISTS files (
    id                     INTEGER PRIMARY KEY AUTOINCREMENT,
    source_provider        TEXT NOT NULL DEFAULT 'filesystem',
    source_item_id         TEXT NOT NULL DEFAULT '',
    path                   TEXT UNIQUE NOT NULL,
    filename               TEXT NOT NULL,
    extension              TEXT,
    size_bytes             INTEGER,
    content_hash           TEXT,
    extracted_text_preview TEXT,
    category               TEXT,
    tags                   TEXT,
    year                   INTEGER,
    extractor              TEXT,
    encoding               TEXT,
    language               TEXT,
    mime_type              TEXT,
    ocr_used               INTEGER DEFAULT 0,
    page_count             INTEGER,
    tables_count           INTEGER,
    form_items_count       INTEGER,
    key_value_items_count  INTEGER,
    pictures_count         INTEGER,
    document_hash          TEXT,
    indexed_at             TIMESTAMP,
    modified_at            TIMESTAMP,
    created_at             TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_files_source_provider_item_id
    ON files(source_provider, source_item_id)
    WHERE source_item_id != '';
CREATE INDEX IF NOT EXISTS idx_files_content_hash  ON files(content_hash);
CREATE INDEX IF NOT EXISTS idx_files_category      ON files(category);
CREATE INDEX IF NOT EXISTS idx_files_extension     ON files(extension);
CREATE INDEX IF NOT EXISTS idx_files_year          ON files(year);
CREATE INDEX IF NOT EXISTS idx_files_filters_composite ON files(year, category, extension);

CREATE TABLE IF NOT EXISTS file_failures (
    path          TEXT PRIMARY KEY,
    content_hash  TEXT NOT NULL,
    error_code    TEXT,
    error_message TEXT,
    retryable     INTEGER DEFAULT 1,
    failure_count INTEGER DEFAULT 1,
    first_seen_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_seen_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS chunks (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id      INTEGER REFERENCES files(id) ON DELETE CASCADE,
    chunk_index  INTEGER,
    content      TEXT NOT NULL,
    token_count  INTEGER,
    parent_id    INTEGER,                    -- v2: Link to parent window chunk
    page_number  INTEGER,                    -- v2: Page number in source document
    start_page   INTEGER,                    -- v2: Start page for multi-page chunks
    end_page     INTEGER,                    -- v2: End page for multi-page chunks
    section_path TEXT,                       -- v2: Section hierarchy path
    block_type   TEXT,                       -- v2: Block type ('table', 'form', 'narrative') from docling provenance
    created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_chunks_file_id ON chunks(file_id);
CREATE INDEX IF NOT EXISTS idx_chunks_file_index ON chunks(file_id, chunk_index);
-- Note: idx_chunks_parent_id is created in init_db().

CREATE TABLE IF NOT EXISTS config (
    key   TEXT PRIMARY KEY,
    value TEXT
);

CREATE TABLE IF NOT EXISTS scan_history (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at    TIMESTAMP,
    completed_at  TIMESTAMP,
    files_scanned INTEGER DEFAULT 0,
    files_indexed INTEGER DEFAULT 0,
    errors        INTEGER DEFAULT 0,
    status        TEXT DEFAULT 'running'
);

CREATE TABLE IF NOT EXISTS scan_errors (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    scan_id       INTEGER NOT NULL REFERENCES scan_history(id) ON DELETE CASCADE,
    path          TEXT NOT NULL,
    filename      TEXT NOT NULL,
    extension     TEXT NOT NULL,
    operation     TEXT NOT NULL,
    error_code    TEXT,
    error_message TEXT NOT NULL,
    is_timeout    INTEGER DEFAULT 0,
    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_scan_errors_scan_id ON scan_errors(scan_id);
CREATE INDEX IF NOT EXISTS idx_scan_errors_created_at ON scan_errors(created_at);

CREATE TABLE IF NOT EXISTS chat_messages (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id            TEXT NOT NULL,
    role               TEXT NOT NULL,
    content            TEXT NOT NULL,
    sources            TEXT,
    generation_seconds REAL,
    completion_mode    TEXT,
    stopped_by_user    INTEGER DEFAULT 0,
    has_remaining_scope INTEGER DEFAULT 0,
    next_action        TEXT,
    next_action_reason TEXT,
    chat_mode          TEXT,
    is_internal        INTEGER DEFAULT 0,
    created_at         TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_chat_chat_id    ON chat_messages(chat_id);
CREATE INDEX IF NOT EXISTS idx_chat_created_at ON chat_messages(created_at);

CREATE TABLE IF NOT EXISTS chats (
    chat_id    TEXT PRIMARY KEY,
    title      TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_chats_updated_at ON chats(updated_at);

CREATE TABLE IF NOT EXISTS chat_preferences (
    chat_id                           TEXT PRIMARY KEY REFERENCES chats(chat_id) ON DELETE CASCADE,
    chat_web_search_enabled           INTEGER DEFAULT 0,
    chat_web_search_privacy_override  INTEGER DEFAULT 0,
    updated_at                        TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS response_diagnostics_metrics (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id             TEXT NOT NULL,
    question            TEXT,
    type                TEXT NOT NULL CHECK(type IN ('evaluation', 'user')),
    model_filename      TEXT,
    run_id              TEXT,        -- NULL for user chats
    query_type          TEXT NOT NULL CHECK(query_type IN ('simple', 'metadata', 'focused', 'coverage', 'unknown')),
    raw_chunks_count    INTEGER,
    sources_count       INTEGER,
    generation_seconds  REAL,
    answer_length       INTEGER,
    timeout_occurred    INTEGER,
    has_empty_answer    INTEGER,
    has_refusal_pattern INTEGER,
    unsupported_claim_count INTEGER,
    evidence_coverage_rate REAL,
    not_found_count INTEGER,
    detected_issues     TEXT,        -- JSON list of IssueType strings
    created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_diagnostics_chat_id ON response_diagnostics_metrics(chat_id);
CREATE INDEX IF NOT EXISTS idx_diagnostics_type ON response_diagnostics_metrics(type);
CREATE INDEX IF NOT EXISTS idx_diagnostics_run_id ON response_diagnostics_metrics(run_id);
CREATE INDEX IF NOT EXISTS idx_diagnostics_created_at ON response_diagnostics_metrics(created_at);

CREATE TABLE IF NOT EXISTS continuation_pass_artifacts (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id             TEXT NOT NULL REFERENCES chats(chat_id) ON DELETE CASCADE,
    request_id          TEXT NOT NULL,
    pass_index          INTEGER NOT NULL,
    stitch_mode         TEXT NOT NULL,
    raw_answer          TEXT NOT NULL,
    cleaned_answer      TEXT NOT NULL,
    has_remaining_scope INTEGER DEFAULT 0,
    completion_mode     TEXT,
    next_action_reason  TEXT,
    sources             TEXT,
    pass_details        TEXT,
    status_transitions  TEXT,
    payload_hash        TEXT NOT NULL,
    created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_cont_pass_unique
    ON continuation_pass_artifacts(chat_id, request_id, pass_index);
CREATE INDEX IF NOT EXISTS idx_cont_pass_chat_id
    ON continuation_pass_artifacts(chat_id);
CREATE INDEX IF NOT EXISTS idx_cont_pass_created_at
    ON continuation_pass_artifacts(created_at);

-- Vector storage table (sqlite-vec)
CREATE TABLE IF NOT EXISTS vec_chunks (
    chunk_id    INTEGER PRIMARY KEY,
    file_id     INTEGER NOT NULL,
    file_path   TEXT NOT NULL,
    chunk_text  TEXT NOT NULL,
    vector      BLOB NOT NULL,  -- Serialized float32 vector (768-dim for nomic-embed-text-v1.5)
    year        INTEGER,
    filename    TEXT NOT NULL,
    extension   TEXT NOT NULL,
    category    TEXT NOT NULL,
    FOREIGN KEY (chunk_id) REFERENCES chunks(id) ON DELETE CASCADE,
    FOREIGN KEY (file_id) REFERENCES files(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_vec_chunks_file_id ON vec_chunks(file_id);
CREATE INDEX IF NOT EXISTS idx_vec_chunks_year ON vec_chunks(year);
CREATE INDEX IF NOT EXISTS idx_vec_chunks_category ON vec_chunks(category);
CREATE INDEX IF NOT EXISTS idx_vec_chunks_extension ON vec_chunks(extension);
CREATE INDEX IF NOT EXISTS idx_vec_chunks_filename ON vec_chunks(filename);
CREATE INDEX IF NOT EXISTS idx_vec_chunks_filters_composite ON vec_chunks(year, category, extension);

-- FTS5 full-text index for candidate augmentation in focused retrieval.
-- FTS5 provides additional candidate chunk IDs to the vector search pool before
-- reranking. The reranker remains the sole scorer — FTS5 contributes recall only.
-- Metadata columns (year, filename, extension, category) are UNINDEXED so the
-- same WHERE clause used for vec_chunks applies directly to FTS5 results.
CREATE VIRTUAL TABLE IF NOT EXISTS fts_chunks USING fts5(
    chunk_text,
    chunk_id  UNINDEXED,
    file_id   UNINDEXED,
    file_path UNINDEXED,
    year      UNINDEXED,
    filename  UNINDEXED,
    extension UNINDEXED,
    category  UNINDEXED,
    tokenize='porter unicode61'
);

CREATE TRIGGER IF NOT EXISTS fts_chunks_ai AFTER INSERT ON vec_chunks BEGIN
    INSERT INTO fts_chunks(rowid, chunk_text, chunk_id, file_id, file_path, year, filename, extension, category)
    VALUES (new.chunk_id, new.chunk_text, new.chunk_id, new.file_id, new.file_path, new.year, new.filename, new.extension, new.category);
END;

CREATE TRIGGER IF NOT EXISTS fts_chunks_ad AFTER DELETE ON vec_chunks BEGIN
    INSERT INTO fts_chunks(fts_chunks, rowid, chunk_text, chunk_id, file_id, file_path, year, filename, extension, category)
    VALUES ('delete', old.chunk_id, old.chunk_text, old.chunk_id, old.file_id, old.file_path, old.year, old.filename, old.extension, old.category);
END;

CREATE TRIGGER IF NOT EXISTS fts_chunks_au AFTER UPDATE ON vec_chunks BEGIN
    INSERT INTO fts_chunks(fts_chunks, rowid, chunk_text, chunk_id, file_id, file_path, year, filename, extension, category)
    VALUES ('delete', old.chunk_id, old.chunk_text, old.chunk_id, old.file_id, old.file_path, old.year, old.filename, old.extension, old.category);
    INSERT INTO fts_chunks(rowid, chunk_text, chunk_id, file_id, file_path, year, filename, extension, category)
    VALUES (new.chunk_id, new.chunk_text, new.chunk_id, new.file_id, new.file_path, new.year, new.filename, new.extension, new.category);
END;

CREATE TABLE IF NOT EXISTS term_dictionary_state (
    singleton_id INTEGER PRIMARY KEY CHECK(singleton_id = 1),
    current_version INTEGER NOT NULL DEFAULT 0,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS term_dictionary_build_runs (
    run_id TEXT PRIMARY KEY,
    target_version INTEGER NOT NULL,
    status TEXT NOT NULL,
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP,
    last_processed_chunk_id INTEGER NOT NULL DEFAULT 0,
    processed_chunks INTEGER NOT NULL DEFAULT 0,
    terms_inserted INTEGER NOT NULL DEFAULT 0,
    aliases_inserted INTEGER NOT NULL DEFAULT 0,
    error_summary TEXT
);

CREATE INDEX IF NOT EXISTS idx_term_dictionary_build_runs_started_at
    ON term_dictionary_build_runs(started_at);

CREATE TABLE IF NOT EXISTS term_entries (
    term_id INTEGER PRIMARY KEY AUTOINCREMENT,
    canonical_term TEXT NOT NULL,
    normalized_term TEXT NOT NULL,
    type TEXT NOT NULL,
    confidence REAL NOT NULL,
    status TEXT NOT NULL,
    dict_version INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_term_entries_version_type_norm
    ON term_entries(dict_version, type, normalized_term);
CREATE INDEX IF NOT EXISTS idx_term_entries_status_version
    ON term_entries(status, dict_version);

CREATE TABLE IF NOT EXISTS term_aliases (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    term_id INTEGER NOT NULL REFERENCES term_entries(term_id) ON DELETE CASCADE,
    alias TEXT NOT NULL,
    normalized_alias TEXT NOT NULL,
    alias_type TEXT NOT NULL,
    confidence REAL NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_term_aliases_term_alias
    ON term_aliases(term_id, normalized_alias);
CREATE INDEX IF NOT EXISTS idx_term_aliases_norm
    ON term_aliases(normalized_alias);

CREATE TABLE IF NOT EXISTS term_evidence (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    term_id INTEGER NOT NULL REFERENCES term_entries(term_id) ON DELETE CASCADE,
    file_id INTEGER REFERENCES files(id) ON DELETE SET NULL,
    chunk_id INTEGER REFERENCES chunks(id) ON DELETE SET NULL,
    evidence_snippet TEXT,
    extraction_method TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_term_evidence_term_id
    ON term_evidence(term_id);
CREATE INDEX IF NOT EXISTS idx_term_evidence_chunk_id
    ON term_evidence(chunk_id);
"""

_RESET_DROP_SQL = '''
DROP TRIGGER IF EXISTS fts_chunks_ai;
DROP TRIGGER IF EXISTS fts_chunks_ad;
DROP TRIGGER IF EXISTS fts_chunks_au;
DROP TABLE IF EXISTS fts_chunks;
DROP TABLE IF EXISTS vec_chunks;
DROP TABLE IF EXISTS term_evidence;
DROP TABLE IF EXISTS term_aliases;
DROP TABLE IF EXISTS term_entries;
DROP TABLE IF EXISTS term_dictionary_build_runs;
DROP TABLE IF EXISTS term_dictionary_state;
DROP TABLE IF EXISTS response_diagnostics_metrics;
DROP TABLE IF EXISTS continuation_pass_artifacts;
DROP TABLE IF EXISTS chat_messages;
DROP TABLE IF EXISTS chat_preferences;
DROP TABLE IF EXISTS chunks;
DROP TABLE IF EXISTS files;
DROP TABLE IF EXISTS chats;
DROP TABLE IF EXISTS scan_errors;
DROP TABLE IF EXISTS file_failures;
DROP TABLE IF EXISTS scan_history;
DROP TABLE IF EXISTS config;
DROP TABLE IF EXISTS schema_version;
'''

# ==============================================================================
# Connection Management
# ==============================================================================

async def get_connection() -> aiosqlite.Connection:
    # Open a new connection to the SQLite database.
    db_path = str(settings.db_path)
    conn    = await aiosqlite.connect(db_path)
    conn.row_factory = aiosqlite.Row
    await conn.execute('PRAGMA journal_mode=WAL')
    await conn.execute('PRAGMA foreign_keys=ON')
    await conn.execute(f'PRAGMA busy_timeout={_SQLITE_BUSY_TIMEOUT_MS}')
    return conn


async def _load_sqlite_vec_extension(conn: aiosqlite.Connection) -> bool:
    """
    Attempt to load sqlite-vec extension into the database connection.

    Note: Python 3.13's sqlite3 module may not support extension loading if Python
    was not compiled with --enable-loadable-sqlite-extensions. This function
    gracefully handles that case.

    Returns:
        True if extension was loaded successfully, False otherwise
    """
    try:
        import sqlite_vec

        # Access the underlying sqlite3.Connection (aiosqlite wraps it)
        # This is safe because we're already in aiosqlite's worker thread context
        underlying_conn = conn._conn

        # Check if extension loading is supported
        if not hasattr(underlying_conn, 'enable_load_extension'):
            log.debug('sqlite_extension_loading_not_supported', msg='Python was not compiled with extension support')
            return False

        # Enable extension loading (required before load_extension())
        underlying_conn.enable_load_extension(True)

        # Load the extension using sqlite_vec's load() function
        sqlite_vec.load(underlying_conn)

        # Disable extension loading for security (best practice)
        underlying_conn.enable_load_extension(False)

        return True
    except _SQLITE_EXTENSION_LOAD_EXCEPTIONS as exc:
        # Extension loading not available or already loaded - that's OK
        # sqlite-vec operations will fail later if extension is truly needed
        log.debug(
            'sqlite_vec_extension_load_failed',
            error=str(exc),
            error_type=type(exc).__name__,
            msg='Extension loading failed (may already be loaded or not supported)'
        )
        return False


async def init_db() -> None:
    # Initialize database from current schema.
    log.info('initializing_database', db_path=str(settings.db_path))
    conn = await get_connection()
    try:
        # Attempt to load sqlite-vec extension (may fail if Python wasn't compiled with extension support)
        extension_loaded = await _load_sqlite_vec_extension(conn)
        if extension_loaded:
            log.info('sqlite_vec_extension_loaded')
        else:
            log.debug('sqlite_vec_extension_not_loaded', msg='Extension loading not available or already loaded')

        await conn.executescript(_SCHEMA_SQL)

        # Term dictionary uniqueness is typed by design:
        # allow same normalized term across different entity types.
        await conn.execute('DROP INDEX IF EXISTS idx_term_entries_version_norm')
        await conn.execute('DROP INDEX IF EXISTS idx_term_entries_version_type_norm')
        await conn.execute(
            '''
            CREATE UNIQUE INDEX IF NOT EXISTS idx_term_entries_version_type_norm
            ON term_entries(dict_version, type, normalized_term)
            '''
        )

        # Ensure index exists
        await conn.execute('CREATE INDEX IF NOT EXISTS idx_chunks_parent_id ON chunks(parent_id)')

        await conn.execute('DELETE FROM schema_version')
        await conn.execute(
            'INSERT INTO schema_version (version) VALUES (?)',
            (SCHEMA_VERSION,),
        )
        await conn.execute(
            '''
            INSERT INTO term_dictionary_state (singleton_id, current_version)
            VALUES (1, 0)
            ON CONFLICT(singleton_id) DO NOTHING
            '''
        )
        await conn.commit()
        await _compact_empty_db_if_bloated(conn)
        log.info('database_initialized', schema_version=SCHEMA_VERSION)
    finally:
        await conn.close()


async def _compact_empty_db_if_bloated(conn: aiosqlite.Connection) -> None:
    # Self-heal path: if DB is logically empty but file still retains many free
    # pages (e.g., reset vacuum was blocked by concurrent readers), compact now.
    try:
        cursor = await conn.execute(
            '''
            SELECT
              (SELECT COUNT(*) FROM files) AS files_count,
              (SELECT COUNT(*) FROM chunks) AS chunks_count,
              (SELECT COUNT(*) FROM vec_chunks) AS vectors_count,
              (SELECT COUNT(*) FROM chat_messages) AS chats_count,
              (SELECT COUNT(*) FROM continuation_pass_artifacts) AS continuation_count,
              (SELECT COUNT(*) FROM term_entries) AS term_entries_count
            '''
        )
        row = await cursor.fetchone()
        if row is None:
            return

        is_empty = (
            int(row['files_count']) == 0
            and int(row['chunks_count']) == 0
            and int(row['vectors_count']) == 0
            and int(row['chats_count']) == 0
            and int(row['continuation_count']) == 0
            and int(row['term_entries_count']) == 0
        )
        if not is_empty:
            return

        page_count_row = await (await conn.execute('PRAGMA page_count')).fetchone()
        freelist_row = await (await conn.execute('PRAGMA freelist_count')).fetchone()
        page_size_row = await (await conn.execute('PRAGMA page_size')).fetchone()
        if page_count_row is None or freelist_row is None or page_size_row is None:
            return

        page_count = int(page_count_row[0])
        freelist_count = int(freelist_row[0])
        page_size = int(page_size_row[0])
        file_size_bytes = page_count * page_size
        free_ratio = (freelist_count / page_count) if page_count > 0 else 0.0

        # Compact only when bloat is meaningful; avoid startup overhead otherwise.
        if file_size_bytes < 16 * 1024 * 1024 or free_ratio < 0.5:
            return

        await conn.execute('PRAGMA wal_checkpoint(TRUNCATE)')
        await conn.execute('VACUUM')
        await conn.commit()
        log.info(
            'database_compacted_on_startup',
            file_size_bytes=file_size_bytes,
            freelist_count=freelist_count,
            page_count=page_count,
            free_ratio=round(free_ratio, 4),
        )
    except (aiosqlite.Error, RuntimeError, OSError, ValueError, TypeError) as exc:
        log.warning('startup_database_compaction_skipped', error=str(exc))


async def get_db() -> AsyncGenerator[aiosqlite.Connection]:
    # FastAPI dependency that yields an aiosqlite connection.
    conn = await get_connection()
    try:
        yield conn
    finally:
        await conn.close()


# ==============================================================================
# Helper — Row to Model Converters
# ==============================================================================

def row_to_indexed_file(row: aiosqlite.Row) -> IndexedFile:
    # Convert a SQLite row to an IndexedFile model.
    return IndexedFile(
        id                     = row['id'],
        source_provider        = row['source_provider'] or 'filesystem',
        source_item_id         = row['source_item_id'] or row['path'] or '',
        path                   = row['path'],
        filename               = row['filename'],
        extension              = row['extension'] or '',
        size_bytes             = row['size_bytes'] or 0,
        content_hash           = row['content_hash'] or '',
        extracted_text_preview = row['extracted_text_preview'] or '',
        category               = parse_file_category(row['category']),
        tags                   = parse_json_tags(row['tags']),
        year                   = row['year'],
        extractor              = row['extractor'],
        encoding               = row['encoding'],
        language               = row['language'],
        mime_type              = row['mime_type'],
        ocr_used               = bool(row['ocr_used']),
        page_count             = row['page_count'],
        tables_count           = row['tables_count'],
        form_items_count       = row['form_items_count'],
        key_value_items_count  = row['key_value_items_count'],
        pictures_count         = row['pictures_count'],
        document_hash          = row['document_hash'],
        indexed_at             = parse_timestamp(row['indexed_at']),
        modified_at            = parse_timestamp(row['modified_at']) or datetime.now(UTC),
        created_at             = parse_timestamp(row['created_at']),
    )


def _row_to_chunk(row: aiosqlite.Row) -> Chunk:
    # Convert a SQLite row to a Chunk model.
    return Chunk(
        id           = row['id'],
        file_id      = row['file_id'],
        chunk_index  = row['chunk_index'],
        content      = row['content'],
        token_count  = row['token_count'] or 0,
        parent_id    = row['parent_id'],
        page_number  = row['page_number'],
        start_page   = row['start_page'],
        end_page     = row['end_page'],
        section_path = row['section_path'],
        block_type   = row['block_type'],
        created_at   = parse_timestamp(row['created_at']),
    )


def _row_to_scan_record(row: aiosqlite.Row) -> ScanRecord:
    # Convert a SQLite row to a ScanRecord model.
    status = ScanStatus.RUNNING
    if row['status']:
        try:
            status = ScanStatus(row['status'])
        except (ValueError, KeyError):
            status = ScanStatus.RUNNING

    return ScanRecord(
        id            = row['id'],
        started_at    = parse_timestamp(row['started_at']) or datetime.now(UTC),
        completed_at  = parse_timestamp(row['completed_at']),
        files_scanned = row['files_scanned'] or 0,
        files_indexed = row['files_indexed'] or 0,
        errors        = row['errors'] or 0,
        status        = status,
    )


def _row_to_scan_error_record(row: aiosqlite.Row) -> ScanErrorRecord:
    # Convert a SQLite row to a ScanErrorRecord model.
    return ScanErrorRecord(
        id=row['id'],
        scan_id=row['scan_id'],
        path=row['path'] or '',
        filename=row['filename'] or '',
        extension=row['extension'] or '',
        operation=row['operation'] or '',
        error_code=row['error_code'],
        error_message=row['error_message'] or '',
        is_timeout=bool(row['is_timeout']),
        created_at=parse_timestamp(row['created_at']),
    )


def _row_to_chat_message(row: aiosqlite.Row) -> ChatMessage:
    # Convert a SQLite row to a ChatMessage model.
    return ChatMessage(
        id                 = row['id'],
        chat_id            = row['chat_id'],
        role               = row['role'],
        content            = row['content'],
        sources            = parse_json_sources(row['sources']),
        generation_seconds = row['generation_seconds'],
        completion_mode    = row['completion_mode'],
        stopped_by_user    = bool(row['stopped_by_user']),
        has_remaining_scope = bool(row['has_remaining_scope']),
        next_action        = row['next_action'],
        next_action_reason = row['next_action_reason'],
        chat_mode          = row['chat_mode'],
        is_internal        = bool(row['is_internal']),
        created_at         = parse_timestamp(row['created_at']),
    )


# ==============================================================================
# Files — CRUD
# ==============================================================================

async def insert_file(db: aiosqlite.Connection, file: IndexedFile) -> IndexedFile:
    # Insert a new file record. Returns the file with its assigned id.
    cursor = await db.execute(
        """
        INSERT INTO files (
            source_provider, source_item_id,
            path, filename, extension, size_bytes, content_hash,
            extracted_text_preview, category, tags, year,
            extractor, encoding, language, mime_type, ocr_used,
            page_count, tables_count, form_items_count, key_value_items_count, pictures_count, document_hash,
            indexed_at, modified_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            file.source_provider,
            file.source_item_id,
            file.path,
            file.filename,
            file.extension,
            file.size_bytes,
            file.content_hash,
            file.extracted_text_preview,
            file.category.value,
            json.dumps(file.tags),
            file.year,
            file.extractor,
            file.encoding,
            file.language,
            file.mime_type,
            1 if file.ocr_used else 0,
            file.page_count,
            file.tables_count,
            file.form_items_count,
            file.key_value_items_count,
            file.pictures_count,
            file.document_hash,
            file.indexed_at.isoformat() if file.indexed_at else None,
            file.modified_at.isoformat(),
        ),
    )
    await db.commit()
    file.id = cursor.lastrowid
    return file


async def get_file_by_path(db: aiosqlite.Connection, path: str) -> IndexedFile | None:
    # Look up a file by its absolute path.
    cursor = await db.execute('SELECT * FROM files WHERE path = ?', (path,))
    row    = await cursor.fetchone()
    if row is None:
        return None
    return row_to_indexed_file(row)


async def get_file_by_source_identity(
    db: aiosqlite.Connection,
    *,
    source_provider: str,
    source_item_id: str,
) -> IndexedFile | None:
    # Look up a file by provider-safe source identity.
    cursor = await db.execute(
        '''
        SELECT * FROM files
        WHERE source_provider = ? AND source_item_id = ?
        ''',
        (source_provider, source_item_id),
    )
    row = await cursor.fetchone()
    if row is None:
        return None
    return row_to_indexed_file(row)


async def should_skip_file_retry(
    db: aiosqlite.Connection,
    path: str,
    content_hash: str,
) -> tuple[bool, str | None]:
    # Return True when a file has a non-retryable failure for the same content hash.
    cursor = await db.execute(
        """
        SELECT retryable, content_hash, error_code
        FROM file_failures
        WHERE path = ?
        """,
        (path,),
    )
    row = await cursor.fetchone()
    if row is None:
        return False, None
    if row['retryable'] == 0 and row['content_hash'] == content_hash:
        return True, row['error_code']
    return False, None


async def record_file_failure(
    db: aiosqlite.Connection,
    *,
    path: str,
    content_hash: str,
    error_code: str | None,
    error_message: str | None,
    retryable: bool,
) -> None:
    # Upsert extraction/indexing failure state for retry suppression.
    now = datetime.now(UTC).isoformat()
    cursor = await db.execute(
        """
        INSERT INTO file_failures (
            path, content_hash, error_code, error_message, retryable,
            failure_count, first_seen_at, last_seen_at
        ) VALUES (?, ?, ?, ?, ?, 1, ?, ?)
        ON CONFLICT(path) DO UPDATE SET
            content_hash  = excluded.content_hash,
            error_code    = excluded.error_code,
            error_message = excluded.error_message,
            retryable     = excluded.retryable,
            failure_count = CASE
                WHEN file_failures.content_hash = excluded.content_hash
                 AND file_failures.error_code IS excluded.error_code
                THEN file_failures.failure_count + 1
                ELSE 1
            END,
            last_seen_at  = excluded.last_seen_at
        """,
        (
            path,
            content_hash,
            error_code,
            error_message,
            1 if retryable else 0,
            now,
            now,
        ),
    )
    await db.commit()
    await cursor.close()


async def clear_file_failure(db: aiosqlite.Connection, path: str) -> None:
    # Remove failure state after a successful index/reindex.
    await db.execute('DELETE FROM file_failures WHERE path = ?', (path,))
    await db.commit()


async def get_file_by_id(db: aiosqlite.Connection, file_id: int) -> IndexedFile | None:
    # Look up a file by its id.
    cursor = await db.execute('SELECT * FROM files WHERE id = ?', (file_id,))
    row    = await cursor.fetchone()
    if row is None:
        return None
    return row_to_indexed_file(row)


async def get_files_by_ids(db: aiosqlite.Connection, file_ids: list[int]) -> dict[int, IndexedFile]:
    # Batch-fetch files by a list of IDs.
    if not file_ids:
        return {}
    placeholders = ', '.join('?' * len(file_ids))
    cursor = await db.execute(
        f'SELECT * FROM files WHERE id IN ({placeholders})',
        file_ids,
    )
    rows = await cursor.fetchall()
    return {row['id']: row_to_indexed_file(row) for row in rows}


async def get_all_files_for_scan(db: aiosqlite.Connection) -> list[IndexedFile]:
    # Return all indexed files (no pagination). Used by scan task for change
    # detection so every file on disk can be matched; avoids pagination limits.
    cursor = await db.execute('SELECT * FROM files ORDER BY path ASC')
    rows   = await cursor.fetchall()
    return [row_to_indexed_file(row) for row in rows]


async def get_files(
    db: aiosqlite.Connection,
    category:    str | None = None,
    extensions:  list[str] | None = None,
    search:      str | None = None,
    tag:         str | None = None,
    sort_by:     str        = 'indexed_at',
    order:       str        = 'desc',
    offset:      int        = 0,
    limit:       int        = 50,
) -> tuple[list[IndexedFile], int]:
    # List files with optional filtering, sorting, and pagination.
    conditions: list[str] = []
    params:     list[str | int] = []

    if category is not None:
        conditions.append('category = ?')
        params.append(category)

    if extensions is not None and len(extensions) > 0:
        placeholders = ', '.join('?' * len(extensions))
        conditions.append(f'extension IN ({placeholders})')
        params.extend(extensions)

    if tag is not None and tag.strip():
        conditions.append(
            "EXISTS (SELECT 1 FROM json_each(files.tags) WHERE json_each.value = ?)"
        )
        params.append(tag.strip())

    if search is not None:
        escaped = search.replace('\\', '\\\\').replace('%', '\\%').replace('_', '\\_')
        conditions.append("(filename LIKE ? ESCAPE '\\' OR path LIKE ? ESCAPE '\\')")
        search_pattern = f'%{escaped}%'
        params.extend([search_pattern, search_pattern])

    where_clause = ''
    if conditions:
        where_clause = 'WHERE ' + ' AND '.join(conditions)

    sort_col = sort_by if sort_by in {'filename', 'category', 'extension', 'size_bytes', 'indexed_at', 'modified_at'} else 'indexed_at'
    order_val = 'ASC' if order.lower() == 'asc' else 'DESC'

    count_cursor = await db.execute(
        f'SELECT COUNT(*) as cnt FROM files {where_clause}',
        params,
    )
    count_row   = await count_cursor.fetchone()
    total_count = count_row['cnt'] if count_row else 0

    query_params = params + [limit, offset]
    cursor = await db.execute(
        f'SELECT * FROM files {where_clause} ORDER BY {sort_col} {order_val} LIMIT ? OFFSET ?',
        query_params,
    )
    rows  = await cursor.fetchall()
    files = [row_to_indexed_file(row) for row in rows]

    return files, total_count


async def get_file_ids_matching_filters(
    db: aiosqlite.Connection,
    year_filter:      int | None = None,
    category_filter:  str | None = None,
    extension_filter: str | None = None,
) -> list[int]:
    """
    Get file IDs matching metadata filters.

    Used for file-anchored retrieval in coverage queries to ensure
    all matching files are represented (one chunk per file).

    Args:
        db: Database connection
        year_filter: Filter by year (exact match)
        category_filter: Filter by category (exact match)
        extension_filter: Filter by extension (exact match, should include dot prefix)

    Returns:
        List of file IDs matching the filters
    """
    conditions: list[str] = []
    params:     list[int | str] = []

    if year_filter is not None:
        conditions.append('year = ?')
        params.append(year_filter)

    if category_filter is not None:
        # Sanitize category filter (only alphanumeric, underscore, hyphen)
        safe_category = ''.join(c for c in category_filter if c.isalnum() or c in '_-')
        if safe_category:
            conditions.append('category = ?')
            params.append(safe_category)

    if extension_filter is not None:
        # Ensure extension has dot prefix
        safe_extension = extension_filter if extension_filter.startswith('.') else f'.{extension_filter}'
        conditions.append('extension = ?')
        params.append(safe_extension)

    where_clause = ''
    if conditions:
        where_clause = 'WHERE ' + ' AND '.join(conditions)

    cursor = await db.execute(
        f'SELECT id FROM files {where_clause}',
        params,
    )
    rows = await cursor.fetchall()
    return [row['id'] for row in rows]


async def update_file(db: aiosqlite.Connection, file: IndexedFile) -> IndexedFile:
    # Update an existing file record by id.
    await db.execute(
        """
        UPDATE files SET
            source_provider = ?, source_item_id = ?,
            path = ?, filename = ?, extension = ?, size_bytes = ?,
            content_hash = ?, extracted_text_preview = ?, category = ?,
            tags = ?, year = ?, extractor = ?, encoding = ?, language = ?, mime_type = ?,
            ocr_used = ?, page_count = ?, tables_count = ?, form_items_count = ?,
            key_value_items_count = ?, pictures_count = ?, document_hash = ?,
            indexed_at = ?, modified_at = ?
        WHERE id = ?
        """,
        (
            file.source_provider,
            file.source_item_id,
            file.path,
            file.filename,
            file.extension,
            file.size_bytes,
            file.content_hash,
            file.extracted_text_preview,
            file.category.value,
            json.dumps(file.tags),
            file.year,
            file.extractor,
            file.encoding,
            file.language,
            file.mime_type,
            1 if file.ocr_used else 0,
            file.page_count,
            file.tables_count,
            file.form_items_count,
            file.key_value_items_count,
            file.pictures_count,
            file.document_hash,
            file.indexed_at.isoformat() if file.indexed_at else None,
            file.modified_at.isoformat(),
            file.id,
        ),
    )
    await db.commit()
    return file


async def delete_file(db: aiosqlite.Connection, file_id: int) -> bool:
    # Delete a file and its chunks (CASCADE).
    cursor = await db.execute('DELETE FROM files WHERE id = ?', (file_id,))
    await db.commit()
    return cursor.rowcount > 0


# ==============================================================================
# Chunks
# ==============================================================================

async def insert_chunks_batch(db: aiosqlite.Connection, file_id: int, chunks: list[Chunk]) -> list[int]:
    # Insert multiple chunks in a single transaction. Returns list of chunk IDs.
    if not chunks:
        return []

    await db.executemany(
        """
        INSERT INTO chunks (
            file_id, chunk_index, content, token_count, parent_id,
            page_number, start_page, end_page, section_path, block_type
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                file_id,
                c.chunk_index,
                c.content,
                c.token_count,
                c.parent_id,
                c.page_number,
                c.start_page,
                c.end_page,
                c.section_path,
                c.block_type,
            )
            for c in chunks
        ],
    )
    await db.commit()

    # Fetch ONLY the IDs of the chunks we just inserted (match by chunk_index AND parent_id)
    # This is critical: we must return only the newly inserted chunks, not all chunks for the file.
    # Parent and child chunks can share chunk_index values, so we must also match by parent_id
    # to distinguish them. Otherwise, when inserting children after parents, we'd return parent IDs.
    chunk_indices = [c.chunk_index for c in chunks]
    # Determine parent_id filter: if all chunks have parent_id=None, filter for parents;
    # if all have parent_id set, filter for children; if mixed, we need per-chunk matching
    parent_id_values = [c.parent_id for c in chunks]
    all_parents = all(pid is None for pid in parent_id_values)
    all_children = all(pid is not None for pid in parent_id_values)

    if all_parents:
        # All chunks are parents (parent_id IS NULL)
        placeholders = ','.join('?' * len(chunk_indices))
        cursor = await db.execute(
            f'SELECT id FROM chunks WHERE file_id = ? AND chunk_index IN ({placeholders}) AND parent_id IS NULL ORDER BY chunk_index',
            (file_id, *chunk_indices),
        )
    elif all_children:
        # All chunks are children (parent_id IS NOT NULL)
        placeholders = ','.join('?' * len(chunk_indices))
        cursor = await db.execute(
            f'SELECT id FROM chunks WHERE file_id = ? AND chunk_index IN ({placeholders}) AND parent_id IS NOT NULL ORDER BY chunk_index',
            (file_id, *chunk_indices),
        )
    else:
        # Mixed: need to match each chunk individually by (chunk_index, parent_id)
        # Build a query that matches each (chunk_index, parent_id) pair
        conditions = []
        params = [file_id]
        for c in chunks:
            if c.parent_id is None:
                conditions.append('(chunk_index = ? AND parent_id IS NULL)')
            else:
                conditions.append('(chunk_index = ? AND parent_id = ?)')
            params.append(c.chunk_index)
            if c.parent_id is not None:
                params.append(c.parent_id)
        where_clause = ' OR '.join(conditions)
        cursor = await db.execute(
            f'SELECT id FROM chunks WHERE file_id = ? AND ({where_clause}) ORDER BY chunk_index',
            params,
        )

    rows = await cursor.fetchall()
    return [row['id'] for row in rows]


async def get_chunks_by_ids(db: aiosqlite.Connection, chunk_ids: list[int]) -> list[dict]:
    # Return chunk_id, file_id, file_path, chunk_text for the given chunk IDs.
    if not chunk_ids:
        return []
    chunk_ids = list(dict.fromkeys(chunk_ids))
    placeholders = ','.join('?' * len(chunk_ids))
    cursor = await db.execute(
        f"""
        SELECT c.id AS chunk_id, c.file_id, f.path AS file_path, f.filename, c.content AS chunk_text,
               c.page_number, c.start_page, c.end_page, c.section_path, c.block_type
        FROM chunks c
        JOIN files f ON c.file_id = f.id
        WHERE c.id IN ({placeholders})
        """,
        chunk_ids,
    )
    rows = await cursor.fetchall()
    result = []
    for row in rows:
        chunk_dict = {
            'chunk_id':   row['chunk_id'],
            'file_id':    row['file_id'],
            'file_path':  row['file_path'] or '',
            'filename':   row['filename'] or '',
            'chunk_text': row['chunk_text'] or '',
            'page_number': row['page_number'],
            'start_page': row['start_page'],
            'end_page': row['end_page'],
            'section_path': row['section_path'],
            'block_type': row['block_type'],
        }
        result.append(chunk_dict)
    return result


async def get_chunk_by_id(db: aiosqlite.Connection, chunk_id: int) -> Chunk | None:
    # Get a single chunk by ID.
    cursor = await db.execute('SELECT * FROM chunks WHERE id = ?', (chunk_id,))
    row = await cursor.fetchone()
    if row is None:
        return None
    return _row_to_chunk(row)


async def get_chunks_by_parent_ids(db: aiosqlite.Connection, parent_ids: list[int]) -> list[dict]:
    # Return parent chunks for Parent Document Retrieval.
    # Given a list of parent IDs, fetch the parent chunk content for LLM context.
    # Deduplicates: if same parent_id appears multiple times, returns it once.
    if not parent_ids:
        return []

    # Deduplicate parent_ids
    unique_parent_ids = list(dict.fromkeys(parent_ids))
    placeholders = ','.join('?' * len(unique_parent_ids))

    cursor = await db.execute(
        f"""
        SELECT c.id AS chunk_id, c.file_id, f.path AS file_path, f.filename, c.content AS chunk_text,
               c.page_number, c.start_page, c.end_page, c.section_path, c.block_type
        FROM chunks c
        JOIN files f ON c.file_id = f.id
        WHERE c.id IN ({placeholders})
        """,
        unique_parent_ids,
    )
    rows = await cursor.fetchall()
    result = []
    for row in rows:
        chunk_dict = {
            'chunk_id':   row['chunk_id'],
            'file_id':    row['file_id'],
            'file_path':  row['file_path'] or '',
            'filename':   row['filename'] or '',
            'chunk_text': row['chunk_text'] or '',
            'page_number': row['page_number'],
            'start_page': row['start_page'],
            'end_page': row['end_page'],
            'section_path': row['section_path'],
            'block_type': row['block_type'],
        }
        result.append(chunk_dict)
    return result


async def delete_chunks_for_file(db: aiosqlite.Connection, file_id: int) -> int:
    # Delete all chunks for a file.
    cursor = await db.execute('DELETE FROM chunks WHERE file_id = ?', (file_id,))
    await db.commit()
    return cursor.rowcount


async def get_chunk_count_for_file(db: aiosqlite.Connection, file_id: int) -> int:
    # Return the number of chunks for a file (all chunks, for display).
    cursor = await db.execute('SELECT COUNT(*) as cnt FROM chunks WHERE file_id = ?', (file_id,))
    row = await cursor.fetchone()
    return row['cnt'] if row else 0


# ==============================================================================
# Scan History
# ==============================================================================

async def insert_scan_record(db: aiosqlite.Connection, record: ScanRecord) -> ScanRecord:
    # Insert a new scan history record.
    cursor = await db.execute(
        """
        INSERT INTO scan_history (started_at, completed_at, files_scanned, files_indexed, errors, status)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            record.started_at.isoformat(),
            record.completed_at.isoformat() if record.completed_at else None,
            record.files_scanned,
            record.files_indexed,
            record.errors,
            record.status.value,
        ),
    )
    await db.commit()
    record.id = cursor.lastrowid
    return record


async def update_scan_record(db: aiosqlite.Connection, record: ScanRecord) -> ScanRecord:
    # Update an existing scan history record.
    await db.execute(
        """
        UPDATE scan_history SET
            completed_at = ?, files_scanned = ?, files_indexed = ?,
            errors = ?, status = ?
        WHERE id = ?
        """,
        (
            record.completed_at.isoformat() if record.completed_at else None,
            record.files_scanned,
            record.files_indexed,
            record.errors,
            record.status.value,
            record.id,
        ),
    )
    await db.commit()
    return record


async def get_latest_scan(db: aiosqlite.Connection) -> ScanRecord | None:
    # Get the most recent scan record.
    cursor = await db.execute(
        'SELECT * FROM scan_history ORDER BY started_at DESC LIMIT 1'
    )
    row = await cursor.fetchone()
    if row is None:
        return None
    return _row_to_scan_record(row)


async def insert_scan_error_record(db: aiosqlite.Connection, record: ScanErrorRecord) -> ScanErrorRecord:
    # Insert a per-file scan error row.
    cursor = await db.execute(
        """
        INSERT INTO scan_errors (
            scan_id, path, filename, extension, operation, error_code, error_message, is_timeout
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            record.scan_id,
            record.path,
            record.filename,
            record.extension,
            record.operation,
            record.error_code,
            record.error_message,
            1 if record.is_timeout else 0,
        ),
    )
    await db.commit()
    record.id = cursor.lastrowid
    return record


async def get_scan_error_records(
    db: aiosqlite.Connection,
    scan_id: int,
    limit: int = 10,
) -> list[ScanErrorRecord]:
    # Return most recent per-file scan errors for a scan.
    safe_limit = max(1, min(int(limit), 100))
    cursor = await db.execute(
        """
        SELECT *
        FROM scan_errors
        WHERE scan_id = ?
        ORDER BY created_at DESC, id DESC
        LIMIT ?
        """,
        (scan_id, safe_limit),
    )
    rows = await cursor.fetchall()
    return [_row_to_scan_error_record(row) for row in rows]


async def get_scan_timeout_error_count(
    db: aiosqlite.Connection,
    scan_id: int,
) -> int:
    # Return count of timeout errors recorded for a scan.
    cursor = await db.execute(
        """
        SELECT COUNT(*) AS cnt
        FROM scan_errors
        WHERE scan_id = ? AND is_timeout = 1
        """,
        (scan_id,),
    )
    row = await cursor.fetchone()
    return int(row['cnt']) if row else 0


async def get_latest_completed_scan(db: aiosqlite.Connection) -> ScanRecord | None:
    # Get the most recent scan that has completed (for "last scan" display).
    cursor = await db.execute(
        '''
        SELECT * FROM scan_history
        WHERE completed_at IS NOT NULL
        ORDER BY completed_at DESC
        LIMIT 1
        '''
    )
    row = await cursor.fetchone()
    if row is None:
        return None
    return _row_to_scan_record(row)




async def clear_stale_running_scans() -> None:
    # Mark any scan_history rows still 'running' as 'failed'.
    conn = await get_connection()
    try:
        now = datetime.now(UTC).isoformat()
        cursor = await conn.execute(
            """
            UPDATE scan_history SET status = ?, completed_at = ?
            WHERE status = ?
            """,
            (ScanStatus.FAILED.value, now, ScanStatus.RUNNING.value),
        )
        await conn.commit()
        if cursor.rowcount > 0:
            log.info('stale_running_scans_cleared', count=cursor.rowcount)
    finally:
        await conn.close()


# ==============================================================================
# Chat Messages
# ==============================================================================

async def ensure_chat_exists(db: aiosqlite.Connection, chat_id: str, first_user_message: str | None = None) -> None:
    # Ensure a chat record exists. Generate title from first user message if provided.
    cursor = await db.execute('SELECT chat_id FROM chats WHERE chat_id = ?', (chat_id,))
    if await cursor.fetchone():
        return

    title = None
    if first_user_message:
        # Simple title: first 50 chars of message
        title = first_user_message[:50].strip()
        if len(first_user_message) > 50:
            title += '...'

    await db.execute(
        'INSERT INTO chats (chat_id, title) VALUES (?, ?)',
        (chat_id, title),
    )
    await db.commit()


async def insert_chat_message(db: aiosqlite.Connection, message: ChatMessage) -> ChatMessage:
    # Insert a new chat message and update chat's updated_at timestamp.
    first_user_message = message.content if message.role == ChatRole.USER else None
    await ensure_chat_exists(db, message.chat_id, first_user_message=first_user_message)

    cursor = await db.execute(
        """
        INSERT INTO chat_messages (
            chat_id, role, content, sources, generation_seconds,
            completion_mode, stopped_by_user, has_remaining_scope, next_action, next_action_reason,
            chat_mode, is_internal
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            message.chat_id,
            message.role,
            message.content,
            json.dumps(message.sources),
            message.generation_seconds,
            message.completion_mode,
            1 if message.stopped_by_user else 0,
            1 if message.has_remaining_scope else 0,
            message.next_action,
            message.next_action_reason,
            message.chat_mode,
            1 if message.is_internal else 0,
        ),
    )
    # Update chat's updated_at timestamp when a message is added
    await db.execute(
        'UPDATE chats SET updated_at = CURRENT_TIMESTAMP WHERE chat_id = ?',
        (message.chat_id,),
    )
    await db.commit()
    message.id = cursor.lastrowid
    return message


async def get_chat_preferences(
    db: aiosqlite.Connection,
    chat_id: str,
) -> dict[str, bool]:
    # Read chat-scoped UI/runtime preferences used by the chat surface.
    cursor = await db.execute(
        """
        SELECT chat_web_search_enabled, chat_web_search_privacy_override
        FROM chat_preferences
        WHERE chat_id = ?
        """,
        (chat_id,),
    )
    row = await cursor.fetchone()
    if row is None:
        return {
            'chat_web_search_enabled': False,
            'chat_web_search_privacy_override': False,
        }
    return {
        'chat_web_search_enabled': bool(row['chat_web_search_enabled']),
        'chat_web_search_privacy_override': bool(row['chat_web_search_privacy_override']),
    }


async def upsert_chat_preferences(
    db: aiosqlite.Connection,
    chat_id: str,
    *,
    chat_web_search_enabled: bool | None = None,
    chat_web_search_privacy_override: bool | None = None,
) -> dict[str, bool]:
    # Upsert chat-scoped preferences. Unset fields preserve prior values.
    await ensure_chat_exists(db, chat_id)
    current = await get_chat_preferences(db, chat_id)
    resolved_web_search_enabled = (
        current['chat_web_search_enabled']
        if chat_web_search_enabled is None
        else bool(chat_web_search_enabled)
    )
    resolved_privacy_override = (
        current['chat_web_search_privacy_override']
        if chat_web_search_privacy_override is None
        else bool(chat_web_search_privacy_override)
    )
    await db.execute(
        """
        INSERT INTO chat_preferences (
            chat_id,
            chat_web_search_enabled,
            chat_web_search_privacy_override,
            updated_at
        )
        VALUES (?, ?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(chat_id) DO UPDATE SET
            chat_web_search_enabled = excluded.chat_web_search_enabled,
            chat_web_search_privacy_override = excluded.chat_web_search_privacy_override,
            updated_at = CURRENT_TIMESTAMP
        """,
        (
            chat_id,
            1 if resolved_web_search_enabled else 0,
            1 if resolved_privacy_override else 0,
        ),
    )
    await db.commit()
    return {
        'chat_web_search_enabled': resolved_web_search_enabled,
        'chat_web_search_privacy_override': resolved_privacy_override,
    }


def _build_continuation_artifact_payload_hash(artifact: ContinuationPassArtifact) -> str:
    payload = {
        'chat_id': artifact.chat_id,
        'request_id': artifact.request_id,
        'pass_index': artifact.pass_index,
        'stitch_mode': artifact.stitch_mode,
        'raw_answer': artifact.raw_answer,
        'cleaned_answer': artifact.cleaned_answer,
        'has_remaining_scope': bool(artifact.has_remaining_scope),
        'completion_mode': artifact.completion_mode,
        'next_action_reason': artifact.next_action_reason,
        'sources': artifact.sources,
        'pass_details': artifact.pass_details,
        'status_transitions': artifact.status_transitions,
    }
    payload_json = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(',', ':'))
    return hashlib.sha256(payload_json.encode('utf-8')).hexdigest()


async def _prune_old_continuation_artifacts(db: aiosqlite.Connection) -> int:
    retention_days = int(settings.continuation_artifact_retention_days)
    if retention_days <= 0:
        return 0
    cursor = await db.execute(
        """
        DELETE FROM continuation_pass_artifacts
        WHERE created_at < datetime('now', ?)
        """,
        (f'-{retention_days} days',),
    )
    return int(cursor.rowcount or 0)


async def prune_continuation_artifacts(db: aiosqlite.Connection) -> int:
    """Public entry point for startup-time artifact pruning. Returns pruned row count."""
    pruned = await _prune_old_continuation_artifacts(db)
    if pruned > 0:
        log.info('continuation_artifacts_pruned', pruned_count=pruned, retention_days=int(settings.continuation_artifact_retention_days))
    return pruned


async def insert_continuation_pass_artifact(
    db: aiosqlite.Connection,
    artifact: ContinuationPassArtifact,
) -> ContinuationPassArtifact:
    # Ensure parent chat exists before writing continuation artifacts.
    await ensure_chat_exists(db, artifact.chat_id)

    payload_hash = _build_continuation_artifact_payload_hash(artifact)
    artifact.payload_hash = payload_hash
    cursor = await db.execute(
        """
        SELECT id, payload_hash
        FROM continuation_pass_artifacts
        WHERE chat_id = ? AND request_id = ? AND pass_index = ?
        LIMIT 1
        """,
        (artifact.chat_id, artifact.request_id, artifact.pass_index),
    )
    existing = await cursor.fetchone()
    if existing is not None:
        existing_hash = str(existing['payload_hash'] or '')
        if existing_hash == payload_hash:
            artifact.id = int(existing['id'])
            return artifact
        raise RuntimeError(
            'continuation_pass_artifact_conflict: existing row has different payload hash '
            f'for key ({artifact.chat_id}, {artifact.request_id}, {artifact.pass_index})'
        )

    insert_cursor = await db.execute(
        """
        INSERT INTO continuation_pass_artifacts (
            chat_id, request_id, pass_index, stitch_mode,
            raw_answer, cleaned_answer, has_remaining_scope, completion_mode,
            next_action_reason, sources, pass_details, status_transitions,
            payload_hash
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            artifact.chat_id,
            artifact.request_id,
            artifact.pass_index,
            artifact.stitch_mode,
            artifact.raw_answer,
            artifact.cleaned_answer,
            1 if artifact.has_remaining_scope else 0,
            artifact.completion_mode,
            artifact.next_action_reason,
            json.dumps(artifact.sources, ensure_ascii=False),
            json.dumps(artifact.pass_details, ensure_ascii=False),
            json.dumps(artifact.status_transitions, ensure_ascii=False),
            artifact.payload_hash,
        ),
    )
    artifact.id = int(insert_cursor.lastrowid or 0) or None
    await _prune_old_continuation_artifacts(db)
    await db.commit()
    return artifact


async def get_chat(db: aiosqlite.Connection, chat_id: str) -> list[ChatMessage]:
    # Get all messages for a chat.
    cursor = await db.execute(
        'SELECT * FROM chat_messages WHERE chat_id = ? ORDER BY created_at ASC',
        (chat_id,),
    )
    rows = await cursor.fetchall()
    return [_row_to_chat_message(row) for row in rows]


async def get_chat_message_by_id(db: aiosqlite.Connection, message_id: int) -> ChatMessage | None:
    # Get a single message by id. Returns None if not found.
    cursor = await db.execute(
        'SELECT * FROM chat_messages WHERE id = ?',
        (message_id,),
    )
    row = await cursor.fetchone()
    if row is None:
        return None
    return _row_to_chat_message(row)


async def get_chats(
    db: aiosqlite.Connection,
    limit: int = 50,
    offset: int = 0,
    search: str | None = None,
) -> list[dict]:
    # List all chats with last message preview, message count, first user message, last activity date, and generation time.
    # Optimized: Uses window functions and JOINs instead of correlated subqueries for better performance.
    # When search is provided, filters by title, last message, or first user message (case-insensitive).
    search_trimmed = search.strip() if search else ''
    search_pattern = f'%{search_trimmed.lower()}%' if search_trimmed else None
    where_clause = ''
    params: tuple = (limit, offset)
    if search_pattern:
        where_clause = """
        WHERE (
            (c.title IS NOT NULL AND LOWER(c.title) LIKE ?)
            OR (lm.last_message IS NOT NULL AND LOWER(lm.last_message) LIKE ?)
            OR (fum.first_user_message IS NOT NULL AND LOWER(fum.first_user_message) LIKE ?)
        )
        """
        params = (search_pattern, search_pattern, search_pattern, limit, offset)

    cursor = await db.execute(
        f"""
        WITH message_stats AS (
            SELECT
                chat_id,
                COUNT(*) as message_count,
                MAX(CASE WHEN role = 'user' THEN created_at END) as last_user_message_at,
                MIN(CASE WHEN role = 'user' THEN created_at END) as first_user_message_at
            FROM chat_messages
            GROUP BY chat_id
        ),
        last_messages AS (
            SELECT
                chat_id,
                content as last_message,
                created_at as last_message_at,
                ROW_NUMBER() OVER (PARTITION BY chat_id ORDER BY created_at DESC) as rn
            FROM chat_messages
        ),
        first_user_messages AS (
            SELECT
                chat_id,
                content as first_user_message,
                ROW_NUMBER() OVER (PARTITION BY chat_id ORDER BY created_at ASC) as rn
            FROM chat_messages
            WHERE role = 'user'
        ),
        last_assistant_gen AS (
            SELECT
                chat_id,
                generation_seconds as last_generation_seconds,
                ROW_NUMBER() OVER (PARTITION BY chat_id ORDER BY created_at DESC) as rn
            FROM chat_messages
            WHERE role = 'assistant' AND generation_seconds IS NOT NULL
        )
        SELECT
            c.chat_id,
            c.title,
            c.created_at,
            c.updated_at,
            lm.last_message,
            lm.last_message_at,
            fum.first_user_message,
            ms.message_count,
            lag.last_generation_seconds
        FROM chats c
        LEFT JOIN message_stats ms ON c.chat_id = ms.chat_id
        LEFT JOIN last_messages lm ON c.chat_id = lm.chat_id AND lm.rn = 1
        LEFT JOIN first_user_messages fum ON c.chat_id = fum.chat_id AND fum.rn = 1
        LEFT JOIN last_assistant_gen lag ON c.chat_id = lag.chat_id AND lag.rn = 1
        {where_clause}
        ORDER BY c.updated_at DESC
        LIMIT ? OFFSET ?
        """,
        params,
    )
    rows = await cursor.fetchall()
    return [
        {
            'chat_id': row['chat_id'],
            'title': row['title'],
            'created_at': parse_timestamp(row['created_at']),
            'updated_at': parse_timestamp(row['updated_at']),
            'last_message_preview': row['last_message'][:_CHAT_PREVIEW_TRUNCATE_LENGTH] if row['last_message'] else None,
            'first_user_message': row['first_user_message'][:_CHAT_PREVIEW_TRUNCATE_LENGTH] if row['first_user_message'] else None,
            'message_count': row['message_count'] or 0,
            'last_message_at': parse_timestamp(row['last_message_at']) if row['last_message_at'] else None,
            'last_generation_seconds': row['last_generation_seconds'],
        }
        for row in rows
    ]


async def get_chat_count(db: aiosqlite.Connection, search: str | None = None) -> int:
    # Return the total number of chats. Use chats table (canonical source) so count matches History.
    if not search or not search.strip():
        cursor = await db.execute('SELECT COUNT(*) AS cnt FROM chats')
        row = await cursor.fetchone()
        return row['cnt'] if row else 0

    search_trimmed = search.strip()
    search_pattern = f'%{search_trimmed.lower()}%'
    cursor = await db.execute(
        """
        WITH chat_search AS (
            SELECT DISTINCT c.chat_id
            FROM chats c
            LEFT JOIN (
                SELECT chat_id, content as last_message, ROW_NUMBER() OVER (PARTITION BY chat_id ORDER BY created_at DESC) as rn
                FROM chat_messages
            ) lm ON c.chat_id = lm.chat_id AND lm.rn = 1
            LEFT JOIN (
                SELECT chat_id, content as first_user_message, ROW_NUMBER() OVER (PARTITION BY chat_id ORDER BY created_at ASC) as rn
                FROM chat_messages WHERE role = 'user'
            ) fum ON c.chat_id = fum.chat_id AND fum.rn = 1
            WHERE (c.title IS NOT NULL AND LOWER(c.title) LIKE ?)
               OR (lm.last_message IS NOT NULL AND LOWER(lm.last_message) LIKE ?)
               OR (fum.first_user_message IS NOT NULL AND LOWER(fum.first_user_message) LIKE ?)
        )
        SELECT COUNT(*) AS cnt FROM chat_search
        """,
        (search_pattern, search_pattern, search_pattern),
    )
    row = await cursor.fetchone()
    return row['cnt'] if row else 0


async def set_chat_title(db: aiosqlite.Connection, chat_id: str, title: str) -> None:
    # Set the title for a chat.
    await db.execute(
        'UPDATE chats SET title = ?, updated_at = CURRENT_TIMESTAMP WHERE chat_id = ?',
        (title, chat_id),
    )
    await db.commit()


async def delete_chat(db: aiosqlite.Connection, chat_id: str) -> bool:
    # Delete a chat and all its messages. Must delete chat_messages first since
    # get_chat_count counts from chat_messages; otherwise the dashboard shows stale counts.
    await db.execute('DELETE FROM chat_messages WHERE chat_id = ?', (chat_id,))
    await db.execute('DELETE FROM response_diagnostics_metrics WHERE chat_id = ?', (chat_id,))
    cursor = await db.execute('DELETE FROM chats WHERE chat_id = ?', (chat_id,))
    await db.commit()
    return cursor.rowcount > 0


# ==============================================================================
# Diagnostics Metrics
# ==============================================================================

async def insert_diagnostics_metrics(
    db: aiosqlite.Connection,
    metrics: object,  # EvalMetrics dataclass (lazy import to avoid circular dependency)
    detected_issues: list[str],  # List of IssueType enum values (as strings)
    run_id: str | None = None,
) -> None:
    """
    Insert diagnostics metrics into response_diagnostics_metrics table.

    Args:
        db: Database connection
        metrics: EvalMetrics dataclass instance
        detected_issues: List of IssueType enum values (as strings)
        run_id: Optional run ID for evaluation runs (None for user chats)
    """
    # Access EvalMetrics attributes dynamically to avoid import dependency.
    raw_query_type = str(getattr(metrics, 'query_type', '') or '').strip().lower()
    if raw_query_type in CANONICAL_DIAGNOSTICS_QUERY_TYPES:
        query_type = raw_query_type
    else:
        log.warning('diagnostics_query_type_unknown', raw_query_type=raw_query_type)
        query_type = DiagnosticsQueryType.UNKNOWN.value

    normalized_detected_issues: list[str] = []
    seen_issues: set[str] = set()
    for issue in detected_issues:
        normalized_issue = str(issue or '').strip().lower()
        if not normalized_issue:
            continue
        if normalized_issue not in CANONICAL_DIAGNOSTICS_ISSUE_TYPES:
            log.warning('diagnostics_issue_type_unknown', issue=normalized_issue)
            continue
        if normalized_issue in seen_issues:
            continue
        seen_issues.add(normalized_issue)
        normalized_detected_issues.append(normalized_issue)

    await db.execute(
        """
        INSERT INTO response_diagnostics_metrics (
            chat_id, question, type, model_filename, run_id, query_type,
            raw_chunks_count, sources_count, generation_seconds, answer_length,
            timeout_occurred, has_empty_answer, has_refusal_pattern,
            unsupported_claim_count, evidence_coverage_rate, not_found_count,
            detected_issues
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            metrics.chat_id,
            metrics.question,
            'evaluation' if run_id else 'user',
            metrics.model_filename,
            run_id,
            query_type,
            metrics.raw_chunks_count,
            metrics.sources_count,
            metrics.generation_seconds,
            metrics.answer_length,
            1 if metrics.timeout_occurred else 0,
            1 if metrics.has_empty_answer else 0,
            1 if metrics.has_refusal_pattern else 0,
            int(getattr(metrics, 'unsupported_claim_count', 0) or 0),
            float(getattr(metrics, 'evidence_coverage_rate', 0.0) or 0.0),
            int(getattr(metrics, 'not_found_count', 0) or 0),
            json.dumps(normalized_detected_issues),
        ),
    )
    await db.commit()


async def get_diagnostics_metrics_since(
    db: aiosqlite.Connection,
    days: int = 30,
    type_filter: str | None = None,  # 'evaluation' or 'user'
    run_id_filter: str | None = None,
) -> list[dict]:
    """
    Get diagnostics metrics since N days ago, optionally filtered by type and run_id.

    Args:
        db: Database connection
        days: Number of days to look back (default: 30)
        type_filter: Filter by type ('evaluation' or 'user'), None for all
        run_id_filter: Filter by run_id, None for all

    Returns:
        List of dictionaries with metrics data
    """
    conditions: list[str] = []
    params: list[str | int] = []

    # Date filter
    conditions.append("created_at >= datetime('now', '-' || ? || ' days')")
    params.append(days)

    # Type filter
    if type_filter:
        conditions.append('type = ?')
        params.append(type_filter)

    # Run ID filter
    if run_id_filter:
        conditions.append('run_id = ?')
        params.append(run_id_filter)

    where_clause = 'WHERE ' + ' AND '.join(conditions) if conditions else ''

    cursor = await db.execute(
        f"""
        SELECT * FROM response_diagnostics_metrics
        {where_clause}
        ORDER BY created_at DESC
        """,
        params,
    )
    rows = await cursor.fetchall()

    # Convert rows to dictionaries
    result = []
    for row in rows:
        # Parse detected_issues JSON
        detected_issues = []
        if row['detected_issues']:
            try:
                detected_issues = json.loads(row['detected_issues'])
            except (json.JSONDecodeError, TypeError):
                detected_issues = []

        query_type = str(row['query_type'] or '').strip().lower()
        if query_type not in CANONICAL_DIAGNOSTICS_QUERY_TYPES:
            query_type = DiagnosticsQueryType.UNKNOWN.value

        normalized_detected_issues = []
        for issue in detected_issues:
            normalized_issue = str(issue or '').strip().lower()
            if normalized_issue in CANONICAL_DIAGNOSTICS_ISSUE_TYPES:
                normalized_detected_issues.append(normalized_issue)

        result.append({
            'id': row['id'],
            'chat_id': row['chat_id'],
            'question': row['question'],
            'type': row['type'],
            'model_filename': row['model_filename'],
            'run_id': row['run_id'],
            'query_type': query_type,
            'raw_chunks_count': row['raw_chunks_count'],
            'sources_count': row['sources_count'],
            'generation_seconds': row['generation_seconds'],
            'answer_length': row['answer_length'],
            'timeout_occurred': bool(row['timeout_occurred']),
            'has_empty_answer': bool(row['has_empty_answer']),
            'has_refusal_pattern': bool(row['has_refusal_pattern']),
            'unsupported_claim_count': int(row['unsupported_claim_count'] or 0),
            'evidence_coverage_rate': float(row['evidence_coverage_rate'] or 0.0),
            'not_found_count': int(row['not_found_count'] or 0),
            'detected_issues': normalized_detected_issues,
            'created_at': parse_timestamp(row['created_at']),
        })

    return result


# ==============================================================================
# Utility Functions
# ==============================================================================

async def get_file_count(db: aiosqlite.Connection) -> int:
    # Total count of indexed files.
    cursor = await db.execute('SELECT COUNT(*) as count FROM files')
    row = await cursor.fetchone()
    return int(row['count']) if row else 0


async def get_chunk_count(db: aiosqlite.Connection) -> int:
    # Total count of chunks.
    cursor = await db.execute('SELECT COUNT(*) as count FROM chunks')
    row = await cursor.fetchone()
    return int(row['count']) if row else 0


async def get_corpus_stats(db: aiosqlite.Connection) -> dict:
    """
    Return corpus statistics for adaptive top-k tuning.

    Returns:
        dict with total_files, total_parent_chunks, total_child_chunks, last_scan_at
    """
    cursor = await db.execute(
        '''
        SELECT
            (SELECT COUNT(*) FROM files) AS total_files,
            (SELECT COUNT(*) FROM chunks WHERE parent_id IS NULL) AS total_parent_chunks,
            (SELECT COUNT(*) FROM chunks WHERE parent_id IS NOT NULL) AS total_child_chunks
        '''
    )
    row = await cursor.fetchone()
    total_files        = int(row['total_files']) if row else 0
    total_parent_chunks = int(row['total_parent_chunks']) if row else 0
    total_child_chunks  = int(row['total_child_chunks']) if row else 0

    latest = await get_latest_completed_scan(db)
    last_scan_at = latest.completed_at if latest else None

    return {
        'total_files':         total_files,
        'total_parent_chunks': total_parent_chunks,
        'total_child_chunks':  total_child_chunks,
        'last_scan_at':        last_scan_at,
    }


async def get_indexed_content_size_bytes(db: aiosqlite.Connection) -> int:
    # Sum of size_bytes of all indexed files (logical size of content we index).
    cursor = await db.execute('SELECT COALESCE(SUM(size_bytes), 0) as total FROM files')
    row = await cursor.fetchone()
    return int(row['total']) if row else 0


async def get_distinct_years(
    db: aiosqlite.Connection,
    filename_pattern: str | None = None,
) -> list[int]:
    """Distinct years from indexed files, optionally filtered by filename substring."""
    if filename_pattern:
        cursor = await db.execute(
            'SELECT DISTINCT year FROM files WHERE year IS NOT NULL AND filename LIKE ? ORDER BY year ASC',
            (f'%{filename_pattern}%',),
        )
    else:
        cursor = await db.execute(
            'SELECT DISTINCT year FROM files WHERE year IS NOT NULL ORDER BY year ASC',
        )
    rows = await cursor.fetchall()
    return [int(r['year']) for r in rows]


async def get_distinct_categories(db: aiosqlite.Connection) -> list[str]:
    # Distinct file categories.
    cursor = await db.execute(
        'SELECT DISTINCT category FROM files WHERE category IS NOT NULL ORDER BY category ASC',
    )
    rows = await cursor.fetchall()
    return [str(r['category']) for r in rows]


# ==============================================================================
# Term Dictionary
# ==============================================================================

async def get_term_dictionary_current_version(db: aiosqlite.Connection) -> int:
    cursor = await db.execute(
        'SELECT current_version FROM term_dictionary_state WHERE singleton_id = 1'
    )
    row = await cursor.fetchone()
    return int(row['current_version']) if row and row['current_version'] is not None else 0


async def set_term_dictionary_current_version(db: aiosqlite.Connection, version: int) -> None:
    await db.execute(
        '''
        INSERT INTO term_dictionary_state (singleton_id, current_version, updated_at)
        VALUES (1, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(singleton_id) DO UPDATE SET
            current_version = excluded.current_version,
            updated_at = CURRENT_TIMESTAMP
        ''',
        (max(0, int(version)),),
    )
    await db.commit()


async def start_term_dictionary_build_run(
    db: aiosqlite.Connection,
    *,
    run_id: str,
    target_version: int,
) -> None:
    await db.execute(
        '''
        INSERT INTO term_dictionary_build_runs (
            run_id, target_version, status, started_at, last_processed_chunk_id,
            processed_chunks, terms_inserted, aliases_inserted
        ) VALUES (?, ?, 'running', CURRENT_TIMESTAMP, 0, 0, 0, 0)
        ''',
        (run_id, int(target_version)),
    )
    await db.commit()


async def update_term_dictionary_build_run_progress(
    db: aiosqlite.Connection,
    *,
    run_id: str,
    last_processed_chunk_id: int,
    processed_chunks: int,
) -> None:
    await db.execute(
        '''
        UPDATE term_dictionary_build_runs
        SET last_processed_chunk_id = ?,
            processed_chunks = ?
        WHERE run_id = ?
        ''',
        (int(last_processed_chunk_id), int(processed_chunks), run_id),
    )
    await db.commit()


async def finalize_term_dictionary_build_run(
    db: aiosqlite.Connection,
    *,
    run_id: str,
    status: str,
    terms_inserted: int = 0,
    aliases_inserted: int = 0,
    error_summary: str | None = None,
) -> None:
    await db.execute(
        '''
        UPDATE term_dictionary_build_runs
        SET status = ?,
            completed_at = CURRENT_TIMESTAMP,
            terms_inserted = ?,
            aliases_inserted = ?,
            error_summary = ?
        WHERE run_id = ?
        ''',
        (status, int(terms_inserted), int(aliases_inserted), error_summary, run_id),
    )
    await db.commit()


async def get_latest_term_dictionary_build_run(db: aiosqlite.Connection) -> dict | None:
    cursor = await db.execute(
        '''
        SELECT run_id, target_version, status, started_at, completed_at,
               last_processed_chunk_id, processed_chunks, terms_inserted, aliases_inserted, error_summary
        FROM term_dictionary_build_runs
        ORDER BY started_at DESC
        LIMIT 1
        '''
    )
    row = await cursor.fetchone()
    if row is None:
        return None
    return {
        'run_id': row['run_id'],
        'target_version': row['target_version'],
        'status': row['status'],
        'started_at': row['started_at'],
        'completed_at': row['completed_at'],
        'last_processed_chunk_id': row['last_processed_chunk_id'],
        'processed_chunks': row['processed_chunks'],
        'terms_inserted': row['terms_inserted'],
        'aliases_inserted': row['aliases_inserted'],
        'error_summary': row['error_summary'],
    }


async def delete_term_dictionary_version(db: aiosqlite.Connection, *, dict_version: int) -> None:
    await db.execute(
        'DELETE FROM term_entries WHERE dict_version = ?',
        (int(dict_version),),
    )
    await db.commit()


async def insert_term_entry(
    db: aiosqlite.Connection,
    *,
    canonical_term: str,
    normalized_term: str,
    term_type: str,
    confidence: float,
    status: str,
    dict_version: int,
) -> int:
    cursor = await db.execute(
        '''
        INSERT INTO term_entries (
            canonical_term, normalized_term, type, confidence, status, dict_version, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
        ''',
        (
            canonical_term,
            normalized_term,
            term_type,
            float(confidence),
            status,
            int(dict_version),
        ),
    )
    await db.commit()
    return int(cursor.lastrowid)


async def insert_term_alias(
    db: aiosqlite.Connection,
    *,
    term_id: int,
    alias: str,
    normalized_alias: str,
    alias_type: str,
    confidence: float,
) -> None:
    await db.execute(
        '''
        INSERT INTO term_aliases (
            term_id, alias, normalized_alias, alias_type, confidence, created_at
        ) VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        ''',
        (
            int(term_id),
            alias,
            normalized_alias,
            alias_type,
            float(confidence),
        ),
    )
    await db.commit()


async def insert_term_evidence(
    db: aiosqlite.Connection,
    *,
    term_id: int,
    file_id: int | None,
    chunk_id: int | None,
    evidence_snippet: str,
    extraction_method: str,
) -> None:
    await db.execute(
        '''
        INSERT INTO term_evidence (
            term_id, file_id, chunk_id, evidence_snippet, extraction_method, created_at
        ) VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        ''',
        (
            int(term_id),
            file_id,
            chunk_id,
            evidence_snippet,
            extraction_method,
        ),
    )
    await db.commit()


async def get_active_term_alias_rows(db: aiosqlite.Connection) -> list[dict]:
    cursor = await db.execute(
        '''
        SELECT
            ta.alias,
            ta.normalized_alias,
            ta.alias_type,
            ta.confidence AS alias_confidence,
            te.canonical_term,
            te.normalized_term,
            te.type AS term_type,
            te.confidence AS term_confidence
        FROM term_aliases ta
        JOIN term_entries te ON te.term_id = ta.term_id
        JOIN term_dictionary_state tds ON tds.singleton_id = 1
        WHERE te.dict_version = tds.current_version
          AND te.status = 'active'
        ORDER BY LENGTH(ta.normalized_alias) DESC, ta.normalized_alias ASC
        '''
    )
    rows = await cursor.fetchall()
    return [
        {
            'alias': row['alias'] or '',
            'normalized_alias': row['normalized_alias'] or '',
            'alias_type': row['alias_type'] or '',
            'alias_confidence': float(row['alias_confidence'] or 0.0),
            'canonical_term': row['canonical_term'] or '',
            'normalized_term': row['normalized_term'] or '',
            'term_type': row['term_type'] or '',
            'term_confidence': float(row['term_confidence'] or 0.0),
        }
        for row in rows
    ]


async def get_term_dictionary_source_rows(
    db: aiosqlite.Connection,
    *,
    after_chunk_id: int = 0,
    limit: int = 500,
) -> list[dict]:
    cursor = await db.execute(
        '''
        SELECT
            c.id AS chunk_id,
            c.file_id AS file_id,
            c.content AS content
        FROM chunks c
        WHERE c.id > ?
        ORDER BY c.id ASC
        LIMIT ?
        ''',
        (int(after_chunk_id), int(limit)),
    )
    rows = await cursor.fetchall()
    return [
        {
            'chunk_id': int(row['chunk_id']),
            'file_id': int(row['file_id']) if row['file_id'] is not None else None,
            'content': row['content'] or '',
        }
        for row in rows
    ]


async def purge_term_dictionary(db: aiosqlite.Connection) -> None:
    await db.execute('DELETE FROM term_evidence')
    await db.execute('DELETE FROM term_aliases')
    await db.execute('DELETE FROM term_entries')
    await db.execute('DELETE FROM term_dictionary_build_runs')
    await db.execute(
        '''
        INSERT INTO term_dictionary_state (singleton_id, current_version, updated_at)
        VALUES (1, 0, CURRENT_TIMESTAMP)
        ON CONFLICT(singleton_id) DO UPDATE SET
            current_version = 0,
            updated_at = CURRENT_TIMESTAMP
        '''
    )
    await db.commit()


async def get_index_integrity_issues(db: aiosqlite.Connection) -> dict[str, int]:
    # Detect index consistency issues across files/chunks/vec_chunks tables.
    checks: dict[str, str] = {
        'orphan_chunks_missing_file': '''
            SELECT COUNT(*) AS cnt
            FROM chunks c
            LEFT JOIN files f ON f.id = c.file_id
            WHERE f.id IS NULL
        ''',
        'orphan_vectors_missing_chunk': '''
            SELECT COUNT(*) AS cnt
            FROM vec_chunks v
            LEFT JOIN chunks c ON c.id = v.chunk_id
            WHERE c.id IS NULL
        ''',
        'orphan_vectors_missing_file': '''
            SELECT COUNT(*) AS cnt
            FROM vec_chunks v
            LEFT JOIN files f ON f.id = v.file_id
            WHERE f.id IS NULL
        ''',
        'child_chunks_missing_parent': '''
            SELECT COUNT(*) AS cnt
            FROM chunks child
            LEFT JOIN chunks parent ON parent.id = child.parent_id
            WHERE child.parent_id IS NOT NULL AND parent.id IS NULL
        ''',
        'files_without_chunks': '''
            SELECT COUNT(*) AS cnt
            FROM files f
            LEFT JOIN chunks c ON c.file_id = f.id
            WHERE c.id IS NULL
        ''',
        'child_chunks_without_vector': '''
            SELECT COUNT(*) AS cnt
            FROM chunks c
            LEFT JOIN vec_chunks v ON v.chunk_id = c.id
            WHERE c.parent_id IS NOT NULL AND v.chunk_id IS NULL
        ''',
        'vec_file_path_mismatch': '''
            SELECT COUNT(*) AS cnt
            FROM vec_chunks v
            JOIN files f ON f.id = v.file_id
            WHERE v.file_path != f.path
        ''',
        'vec_filename_mismatch': '''
            SELECT COUNT(*) AS cnt
            FROM vec_chunks v
            JOIN files f ON f.id = v.file_id
            WHERE v.filename != f.filename
        ''',
        'vec_extension_mismatch': '''
            SELECT COUNT(*) AS cnt
            FROM vec_chunks v
            JOIN files f ON f.id = v.file_id
            WHERE v.extension != f.extension
        ''',
        'vec_category_mismatch': '''
            SELECT COUNT(*) AS cnt
            FROM vec_chunks v
            JOIN files f ON f.id = v.file_id
            WHERE v.category != f.category
        ''',
        'vec_year_mismatch': '''
            SELECT COUNT(*) AS cnt
            FROM vec_chunks v
            JOIN files f ON f.id = v.file_id
            WHERE COALESCE(v.year, -1) != COALESCE(f.year, -1)
        ''',
        'vec_chunk_text_mismatch': '''
            SELECT COUNT(*) AS cnt
            FROM vec_chunks v
            JOIN chunks c ON c.id = v.chunk_id
            WHERE v.chunk_text != c.content
        ''',
    }
    issues: dict[str, int] = {}
    for key, sql in checks.items():
        cursor = await db.execute(sql)
        row = await cursor.fetchone()
        issues[key] = int(row['cnt']) if row else 0
    return issues


async def repair_index_integrity_issues(db: aiosqlite.Connection) -> dict[str, int]:
    # Repair known integrity issues by removing orphaned/incomplete rows.
    repairs: dict[str, tuple[str, tuple[object, ...]]] = {
        'orphan_chunks_deleted': (
            '''
            DELETE FROM chunks
            WHERE file_id NOT IN (SELECT id FROM files)
            ''',
            (),
        ),
        'orphan_vectors_missing_chunk_deleted': (
            '''
            DELETE FROM vec_chunks
            WHERE chunk_id NOT IN (SELECT id FROM chunks)
            ''',
            (),
        ),
        'orphan_vectors_missing_file_deleted': (
            '''
            DELETE FROM vec_chunks
            WHERE file_id NOT IN (SELECT id FROM files)
            ''',
            (),
        ),
        'child_chunks_missing_parent_deleted': (
            '''
            DELETE FROM chunks
            WHERE parent_id IS NOT NULL
              AND parent_id NOT IN (SELECT id FROM chunks)
            ''',
            (),
        ),
        'files_without_chunks_deleted': (
            '''
            DELETE FROM files
            WHERE id NOT IN (SELECT DISTINCT file_id FROM chunks)
            ''',
            (),
        ),
        'child_chunks_without_vector_deleted': (
            '''
            DELETE FROM chunks
            WHERE parent_id IS NOT NULL
              AND id NOT IN (SELECT chunk_id FROM vec_chunks)
            ''',
            (),
        ),
        'vec_file_fields_synced': (
            '''
            UPDATE vec_chunks
            SET
                file_path = (SELECT f.path FROM files f WHERE f.id = vec_chunks.file_id),
                filename = (SELECT f.filename FROM files f WHERE f.id = vec_chunks.file_id),
                extension = (SELECT f.extension FROM files f WHERE f.id = vec_chunks.file_id),
                category = (SELECT f.category FROM files f WHERE f.id = vec_chunks.file_id),
                year = (SELECT f.year FROM files f WHERE f.id = vec_chunks.file_id)
            WHERE file_id IN (SELECT id FROM files)
            ''',
            (),
        ),
        'vec_chunk_text_synced': (
            '''
            UPDATE vec_chunks
            SET chunk_text = (SELECT c.content FROM chunks c WHERE c.id = vec_chunks.chunk_id)
            WHERE chunk_id IN (SELECT id FROM chunks)
            ''',
            (),
        ),
    }
    results: dict[str, int] = {}
    for key, (sql, params) in repairs.items():
        cursor = await db.execute(sql, params)
        results[key] = cursor.rowcount if cursor.rowcount is not None else 0
    await db.commit()
    return results


async def reset_all_data(db: aiosqlite.Connection) -> dict[str, object]:
    # Drop all tables and recreate from current schema so the database has the
    # latest structure (e.g. new columns). Returns a dict with table names and
    # 0 counts (tables are recreated empty).
    reset_error: Exception | None = None
    for attempt in range(1, _RESET_SCHEMA_RETRY_ATTEMPTS + 1):
        try:
            await db.executescript(_RESET_DROP_SQL)
            await db.executescript(_SCHEMA_SQL)
            await db.execute('DELETE FROM schema_version')
            await db.execute('INSERT INTO schema_version (version) VALUES (?)', (SCHEMA_VERSION,))
            await db.commit()
            reset_error = None
            break
        except (aiosqlite.Error, RuntimeError, OSError, ValueError, TypeError) as exc:
            reset_error = exc
            message = str(exc).lower()
            is_lock_error = 'locked' in message or 'busy' in message
            if is_lock_error and attempt < _RESET_SCHEMA_RETRY_ATTEMPTS:
                with suppress(aiosqlite.Error, RuntimeError, OSError, ValueError, TypeError):
                    await db.rollback()
                await asyncio.sleep(min(1.0, _RESET_SCHEMA_RETRY_BASE_DELAY_SECONDS * attempt))
                continue
            raise

    if reset_error is not None:
        raise reset_error

    # Reclaim disk space after destructive reset. VACUUM needs an exclusive lock,
    # so retry briefly to tolerate transient read traffic.
    storage_compacted = False
    compaction_error: str | None = None
    for attempt in range(1, _RESET_COMPACTION_RETRY_ATTEMPTS + 1):
        try:
            await db.execute('PRAGMA wal_checkpoint(TRUNCATE)')
            await db.execute('VACUUM')
            await db.commit()
            storage_compacted = True
            compaction_error = None
            break
        except (aiosqlite.Error, RuntimeError, OSError, ValueError, TypeError) as exc:
            compaction_error = str(exc)
            if attempt < _RESET_COMPACTION_RETRY_ATTEMPTS:
                await asyncio.sleep(0.2 * attempt)
            continue

    counts: dict[str, object] = {
        t: 0
        for t in (
            'response_diagnostics_metrics',
            'continuation_pass_artifacts',
            'chat_messages',
            'chunks',
            'chats',
            'scan_errors',
            'file_failures',
            'scan_history',
            'files',
            'vec_chunks',
            'term_entries',
            'term_aliases',
            'term_evidence',
            'term_dictionary_build_runs',
        )
    }
    counts['storage_compacted'] = storage_compacted
    counts['compaction_error'] = compaction_error
    log.info(
        'all_data_reset',
        schema_recreated=True,
        schema_version=SCHEMA_VERSION,
        storage_compacted=storage_compacted,
        compaction_error=compaction_error,
    )
    return counts
