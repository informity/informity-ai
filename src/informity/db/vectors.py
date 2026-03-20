# ==============================================================================
# Informity AI — SQLite Vector Storage Module (v2)
# Manages vector storage using sqlite-vec extension (vectors stored in SQLite).
# v2: Unified storage (vectors in same SQLite file as metadata).
# ==============================================================================

import re
import sqlite3
import threading
from dataclasses import dataclass, field

import structlog

from informity.config import settings
from informity.indexer.embedder import get_effective_embedding_dimension

# ==============================================================================
# Logger
# ==============================================================================

log = structlog.get_logger(__name__)

_FTS5_BOOLEAN_OPS = re.compile(r'\b(AND|OR|NOT)\b', re.IGNORECASE)
_FTS5_NON_WORD_CHARS = re.compile(r'[^\w\s]')


def _sanitize_fts5_query(query: str) -> str:
    """Normalize user text into conservative plain-token MATCH input."""
    sanitized = _FTS5_NON_WORD_CHARS.sub(' ', query)
    sanitized = _FTS5_BOOLEAN_OPS.sub(' ', sanitized)
    return ' '.join(sanitized.split())


_SQLITE_VEC_LOAD_EXCEPTIONS = (
    ImportError,
    AttributeError,
    TypeError,
    ValueError,
    RuntimeError,
    sqlite3.Error,
    OSError,
)

# ==============================================================================
# Constants
# ==============================================================================

VECTOR_DIMENSION = get_effective_embedding_dimension()


def _get_expected_vector_dimension() -> int:
    return get_effective_embedding_dimension()

# ==============================================================================
# Data types
# ==============================================================================

@dataclass
class ChunkEmbedding:
    # A chunk paired with its embedding vector, ready for storage.
    # Storage Contract (intentional denormalization):
    # - vec_chunks is a read-optimized index table for hot retrieval paths.
    # - It intentionally duplicates selected file/chunk fields to avoid JOINs
    #   in vector-search critical paths.
    chunk_id:     int
    file_id:      int
    file_path:    str
    chunk_text:   str
    vector:       list[float] = field(default_factory=list)
    year:         int | None  = None   # File year for temporal filtering
    filename:     str         = ''     # v2: filename for exact filename filtering
    extension:    str         = ''     # v2: extension for file type filtering
    category:     str         = ''     # v2: category for filtering


# ==============================================================================
# Helper: Load sqlite-vec extension on a connection
# ==============================================================================

def _load_sqlite_vec_extension_sync(conn: sqlite3.Connection) -> bool:
    """
    Load sqlite-vec extension on a synchronous sqlite3 connection.

    Returns:
        True if extension was loaded successfully, False otherwise
    """
    try:
        import sqlite_vec

        # Enable extension loading (required before load_extension())
        conn.enable_load_extension(True)

        # Load the extension
        sqlite_vec.load(conn)

        # Disable extension loading for security (best practice)
        conn.enable_load_extension(False)

        return True
    except _SQLITE_VEC_LOAD_EXCEPTIONS as exc:
        log.warning(
            'sqlite_vec_extension_load_failed',
            error=str(exc),
            error_type=type(exc).__name__,
            msg='Extension loading failed; vector search will not work'
        )
        return False


# ==============================================================================
# VectorStore — manages SQLite vector storage via sqlite-vec
# ==============================================================================

class VectorStore:
    # Encapsulates all vector storage operations using sqlite-vec extension.
    # Vectors are stored in vec_chunks table in the same SQLite database as metadata.
    # Storage contract is intentionally denormalized for read speed.
    # All methods use synchronous sqlite3 connections since they're called from
    # thread pool workers (asyncio.to_thread) or async contexts that can wait.

    def __init__(self) -> None:
        self._thread_local = threading.local()

    def _get_thread_connection(self) -> sqlite3.Connection:
        # Reuse one sqlite connection per worker thread to avoid repeated
        # sqlite-vec extension load/unload overhead on every operation.
        conn = getattr(self._thread_local, 'conn', None)
        if conn is not None:
            return conn

        db_path = str(settings.db_path)
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        conn.execute('PRAGMA journal_mode=WAL')
        conn.execute('PRAGMA foreign_keys=ON')
        conn.execute('PRAGMA busy_timeout=5000')

        if not _load_sqlite_vec_extension_sync(conn):
            log.error('sqlite_vec_extension_not_loaded', action='vector_store_connection_init_failed')
            raise RuntimeError('sqlite-vec extension could not be loaded')

        self._thread_local.conn = conn
        return conn

    def _get_fts_chunks_columns(self, conn: sqlite3.Connection) -> set[str]:
        columns = getattr(self._thread_local, 'fts_chunks_columns', None)
        if isinstance(columns, set) and columns:
            return columns

        try:
            rows = conn.execute('PRAGMA table_info(fts_chunks)').fetchall()
        except sqlite3.Error:
            rows = []

        resolved = {
            str(row['name']).strip().casefold()
            for row in rows
            if isinstance(row, sqlite3.Row) and row['name']
        }
        self._thread_local.fts_chunks_columns = resolved
        return resolved

    def store_embeddings(self, embeddings: list[ChunkEmbedding]) -> int:
        if not embeddings:
            return 0

        try:
            import sqlite_vec
            serialize_float32 = sqlite_vec.serialize_float32
        except ImportError:
            log.error('sqlite_vec_not_available', action='cannot_store_embeddings')
            raise RuntimeError('sqlite-vec extension not available') from None

        records = []
        expected_dimension = _get_expected_vector_dimension()
        for e in embeddings:
            if len(e.vector) != expected_dimension:
                log.warning(
                    'invalid_vector_dimension',
                    chunk_id=e.chunk_id,
                    expected=expected_dimension,
                    actual=len(e.vector),
                    embedding_model=settings.embedding_model,
                    action='skipping_chunk',
                )
                continue

            records.append(
                (
                    e.chunk_id,
                    e.file_id,
                    e.file_path,
                    e.chunk_text,
                    serialize_float32(e.vector),
                    e.year,
                    e.filename,
                    e.extension,
                    e.category,
                )
            )

        if not records:
            return 0

        conn = self._get_thread_connection()
        try:
            conn.execute('BEGIN')
            conn.executemany(
                '''
                INSERT OR REPLACE INTO vec_chunks
                (chunk_id, file_id, file_path, chunk_text, vector, year, filename, extension, category)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''',
                records,
            )
            conn.commit()
            return len(records)
        except sqlite3.Error:
            conn.rollback()
            raise

    async def store_embeddings_async(self, embeddings: list[ChunkEmbedding]) -> int:
        import asyncio
        return await asyncio.to_thread(self.store_embeddings, embeddings)

    def search_similar(
        self,
        query_vector: list[float],
        top_k: int = 5,
        where_clause: str | None = None,
        where_params: list[int | str] | None = None,
    ) -> list[dict]:
        try:
            import sqlite_vec
            serialize_float32 = sqlite_vec.serialize_float32
        except ImportError:
            log.error('sqlite_vec_not_available', action='cannot_search_vectors')
            return []

        expected_dimension = _get_expected_vector_dimension()
        if len(query_vector) != expected_dimension:
            log.warning(
                'invalid_query_vector_dimension',
                expected=expected_dimension,
                actual=len(query_vector),
                embedding_model=settings.embedding_model,
                action='returning_no_results',
            )
            return []
        query_blob = serialize_float32(query_vector)

        where_sql = f'WHERE {where_clause}' if where_clause else ''
        query = f'''
            SELECT chunk_id, file_id, file_path, filename, chunk_text, distance
            FROM (
                SELECT
                    chunk_id,
                    file_id,
                    file_path,
                    filename,
                    chunk_text,
                    vec_distance_cosine(vector, ?) as distance
                FROM vec_chunks
                {where_sql}
            ) ranked
            ORDER BY distance ASC
            LIMIT ?
        '''

        params: list[object] = [query_blob]
        if where_clause and where_params:
            params.extend(where_params)
        params.append(top_k)

        conn = self._get_thread_connection()
        rows = conn.execute(query, params).fetchall()
        return [
            {
                'chunk_id': row['chunk_id'],
                'file_id': row['file_id'],
                'file_path': row['file_path'],
                'filename': row['filename'],
                'chunk_text': row['chunk_text'],
                'score': float(row['distance']),
            }
            for row in rows
        ]

    def search_top1_per_file(
        self,
        query_vector: list[float],
        file_ids: list[int],
        where_clause: str | None = None,
        where_params: list[int | str] | None = None,
    ) -> list[dict]:
        if not file_ids:
            return []

        try:
            import sqlite_vec
            serialize_float32 = sqlite_vec.serialize_float32
        except ImportError:
            log.error('sqlite_vec_not_available', action='cannot_search_top1_per_file')
            return []

        expected_dimension = _get_expected_vector_dimension()
        if len(query_vector) != expected_dimension:
            log.warning(
                'invalid_query_vector_dimension',
                expected=expected_dimension,
                actual=len(query_vector),
                embedding_model=settings.embedding_model,
                action='returning_no_results',
            )
            return []
        query_blob = serialize_float32(query_vector)

        file_ids_unique = list(dict.fromkeys(file_ids))
        file_placeholders = ', '.join('?' * len(file_ids_unique))
        where_parts: list[str] = [f'file_id IN ({file_placeholders})']
        if where_clause:
            where_parts.append(f'({where_clause})')
        where_sql = ' AND '.join(where_parts)

        query = f'''
            WITH ranked AS (
                SELECT
                    chunk_id,
                    file_id,
                    file_path,
                    filename,
                    chunk_text,
                    vec_distance_cosine(vector, ?) AS distance,
                    ROW_NUMBER() OVER (
                        PARTITION BY file_id
                        ORDER BY vec_distance_cosine(vector, ?) ASC
                    ) AS rn
                FROM vec_chunks
                WHERE {where_sql}
            )
            SELECT chunk_id, file_id, file_path, filename, chunk_text, distance
            FROM ranked
            WHERE rn = 1
            ORDER BY distance ASC
        '''
        params: list[object] = [query_blob, query_blob, *file_ids_unique]
        if where_clause and where_params:
            params.extend(where_params)

        conn = self._get_thread_connection()
        rows = conn.execute(query, params).fetchall()
        return [
            {
                'chunk_id': row['chunk_id'],
                'file_id': row['file_id'],
                'file_path': row['file_path'],
                'filename': row['filename'],
                'chunk_text': row['chunk_text'],
                'score': float(row['distance']),
            }
            for row in rows
        ]

    def fts5_augment_candidates(
        self,
        query: str,
        limit: int,
        existing_ids: set[int],
        where_clause: str | None = None,
        where_params: list[int | str] | None = None,
        neutral_score: float = 1.0,
    ) -> list[dict]:
        """
        Run FTS5 keyword search and return candidate dicts for chunk IDs not
        already in existing_ids. Returns at most `limit` net-new candidates.

        FTS5 provides candidate recall only — all returned chunks receive
        neutral_score so the reranker remains the sole scorer. The same
        WHERE clause used for vec_chunks (year, extension, filename filters)
        applies directly to FTS5 UNINDEXED columns.
        """
        sanitized = _sanitize_fts5_query(query)
        if not sanitized:
            return []

        conn = self._get_thread_connection()
        fts_columns = self._get_fts_chunks_columns(conn)
        file_path_select = (
            'file_path AS file_path'
            if 'file_path' in fts_columns
            else "'' AS file_path"
        )
        filename_select = (
            'filename AS filename'
            if 'filename' in fts_columns
            else "'' AS filename"
        )
        select_fields = (
            f'chunk_id, file_id, {file_path_select}, {filename_select}, chunk_text'
        )

        fts_where_parts: list[str] = ['fts_chunks MATCH ?']
        fts_params: list[object] = [sanitized]

        if where_clause:
            fts_where_parts.append(where_clause)
            if where_params:
                fts_params.extend(where_params)

        fts_where = ' AND '.join(fts_where_parts)
        # Over-fetch to account for dedup against existing_ids
        fetch_limit = limit * 2

        fts_query = f"""
            SELECT {select_fields}
            FROM fts_chunks
            WHERE {fts_where}
            ORDER BY rank
            LIMIT ?
        """
        fts_params.append(fetch_limit)

        try:
            rows = conn.execute(fts_query, fts_params).fetchall()
        except sqlite3.Error as exc:
            error_text = str(exc)
            # Compatibility fallback for older/mismatched local schemas:
            # retry once with MATCH-only clause to preserve keyword augmentation.
            if where_clause and ('no such column' in error_text.casefold() or 'fts_chunks' in error_text.casefold()):
                fallback_query = f"""
                    SELECT {select_fields}
                    FROM fts_chunks
                    WHERE fts_chunks MATCH ?
                    ORDER BY rank
                    LIMIT ?
                """
                try:
                    rows = conn.execute(fallback_query, [sanitized, fetch_limit]).fetchall()
                    log.warning(
                        'fts5_search_filter_compat_fallback',
                        error=error_text,
                        query_preview=sanitized[:80],
                    )
                except sqlite3.Error as fallback_exc:
                    log.warning('fts5_search_failed', error=str(fallback_exc), query_preview=sanitized[:80])
                    return []
            else:
                log.warning('fts5_search_failed', error=error_text, query_preview=sanitized[:80])
                return []

        results: list[dict] = []
        for row in rows:
            chunk_id = int(row['chunk_id'])
            if chunk_id in existing_ids:
                continue
            results.append({
                'chunk_id':   chunk_id,
                'file_id':    row['file_id'],
                'file_path':  row['file_path'] or '',
                'filename':   row['filename'] or '',
                'chunk_text': row['chunk_text'] or '',
                'score':      neutral_score,
            })
            if len(results) >= limit:
                break

        return results

    def delete_by_file_id(self, file_id: int) -> None:
        conn = self._get_thread_connection()
        conn.execute('DELETE FROM vec_chunks WHERE file_id = ?', (file_id,))
        conn.commit()

    def drop_all(self) -> int:
        conn = self._get_thread_connection()
        cursor = conn.execute('SELECT COUNT(*) as count FROM vec_chunks')
        row = cursor.fetchone()
        count = int(row['count']) if row else 0
        conn.execute('DELETE FROM vec_chunks')
        conn.commit()
        log.info('sqlite_vec_drop_all_complete', rows_deleted=count)
        return count

    def build_index(self) -> bool:
        log.info('vector_index_build_skipped', reason='exact_search_mode')
        return True

    def get_stats(self) -> dict:
        conn = self._get_thread_connection()
        row = conn.execute('SELECT COUNT(*) as count FROM vec_chunks').fetchone()
        total_vectors = int(row['count']) if row else 0
        storage_bytes = settings.db_path.stat().st_size if settings.db_path and settings.db_path.exists() else 0
        return {'total_vectors': total_vectors, 'storage_bytes': storage_bytes}


# ==============================================================================
# Module-level singleton
# ==============================================================================

vector_store = VectorStore()
