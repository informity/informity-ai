# ==============================================================================
# Informity AI — SQLite Vector Storage Module (v2)
# Manages vector storage using sqlite-vec extension (vectors stored in SQLite).
# v2: Unified storage (vectors in same SQLite file as metadata).
# ==============================================================================

import sqlite3
from dataclasses import dataclass, field

import structlog

from informity.config import settings
from informity.indexer.embedder import get_effective_embedding_dimension

# ==============================================================================
# Logger
# ==============================================================================

log = structlog.get_logger(__name__)
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

VECTOR_DIMENSION = get_effective_embedding_dimension()  # Backward-compat alias for tests; use runtime resolver below.


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

    def store_embeddings(self, embeddings: list[ChunkEmbedding]) -> int:
        """
        Batch-insert chunk embeddings into SQLite vec_chunks table.

        Synchronous version for use in thread pool workers.
        For async contexts, use store_embeddings_async() instead.
        """
        if not embeddings:
            return 0

        try:
            import sqlite_vec
            serialize_float32 = sqlite_vec.serialize_float32
        except ImportError:
            log.error('sqlite_vec_not_available', action='cannot_store_embeddings')
            raise RuntimeError('sqlite-vec extension not available') from None

        # Serialize vectors to BLOB format
        records = []
        expected_dimension = _get_expected_vector_dimension()
        for e in embeddings:
            # Validate vector dimension
            if len(e.vector) != expected_dimension:
                log.warning(
                    'invalid_vector_dimension',
                    chunk_id=e.chunk_id,
                    expected=expected_dimension,
                    actual=len(e.vector),
                    embedding_model=settings.embedding_model,
                    action='skipping_chunk'
                )
                continue

            # Serialize vector to BLOB
            vector_blob = serialize_float32(e.vector)

            records.append((
                e.chunk_id,
                e.file_id,
                e.file_path,
                e.chunk_text,
                vector_blob,
                e.year,
                e.filename,
                e.extension,
                e.category,
            ))

        if not records:
            return 0

        # Use synchronous sqlite3 connection (we're in a thread worker)
        db_path = str(settings.db_path)
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row

        try:
            # Load sqlite-vec extension
            if not _load_sqlite_vec_extension_sync(conn):
                log.error('sqlite_vec_extension_not_loaded', action='cannot_store_embeddings')
                raise RuntimeError('sqlite-vec extension could not be loaded')
            conn.execute('BEGIN')
            conn.executemany(
                '''
                INSERT OR REPLACE INTO vec_chunks
                (chunk_id, file_id, file_path, chunk_text, vector, year, filename, extension, category)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''',
                records
            )
            conn.commit()
            return len(records)
        except sqlite3.Error:
            conn.rollback()
            raise
        finally:
            conn.close()

    async def store_embeddings_async(self, embeddings: list[ChunkEmbedding]) -> int:
        """
        Async version of store_embeddings for use in async contexts.

        This wraps the synchronous store_embeddings() in a thread pool worker
        since sqlite-vec requires synchronous sqlite3 connections.
        """
        import asyncio
        return await asyncio.to_thread(self.store_embeddings, embeddings)

    def search_similar(
        self,
        query_vector: list[float],
        top_k:        int = 5,
        where_clause: str | None = None,
        where_params: list[int | str] | None = None,
    ) -> list[dict]:
        """
        Search for the most similar chunks to a query vector.

        Supports WHERE clause filtering (e.g., "year = 2023", "category = 'document'").
        Returns list of dicts with: chunk_id, file_id, file_path, filename,
        chunk_text, score (lower = more similar for cosine distance).

        This is a synchronous method called from thread pool workers.
        """
        try:
            import sqlite_vec
            serialize_float32 = sqlite_vec.serialize_float32
        except ImportError:
            log.error('sqlite_vec_not_available', action='cannot_search_vectors')
            return []

        # Serialize query vector to BLOB
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

        # Use synchronous sqlite3 connection (we're in a thread worker)
        db_path = str(settings.db_path)
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row

        try:
            # Load sqlite-vec extension
            if not _load_sqlite_vec_extension_sync(conn):
                log.error('sqlite_vec_extension_not_loaded', action='cannot_search_vectors')
                return []

            # Build WHERE clause for metadata filtering
            where_sql = ''
            if where_clause:
                where_sql = f'WHERE {where_clause}'

            # Compute vec_distance_cosine() once in an inner query, then sort by alias.
            # Lower distance = more similar (cosine distance: 0 = identical, 2 = opposite).
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

            cursor = conn.execute(query, params)
            rows = cursor.fetchall()

            return [
                {
                    'chunk_id':     row['chunk_id'],
                    'file_id':      row['file_id'],
                    'file_path':    row['file_path'],
                    'filename':     row['filename'],
                    'chunk_text':   row['chunk_text'],
                    'score':        float(row['distance']),  # Distance (lower = more similar)
                }
                for row in rows
            ]
        finally:
            conn.close()

    def search_top1_per_file(
        self,
        query_vector: list[float],
        file_ids: list[int],
        where_clause: str | None = None,
        where_params: list[int | str] | None = None,
    ) -> list[dict]:
        # Return top-1 most similar chunk per file for the provided file IDs.
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
        db_path = str(settings.db_path)
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row

        try:
            if not _load_sqlite_vec_extension_sync(conn):
                log.error('sqlite_vec_extension_not_loaded', action='cannot_search_top1_per_file')
                return []

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
                SELECT
                    chunk_id,
                    file_id,
                    file_path,
                    filename,
                    chunk_text,
                    distance
                FROM ranked
                WHERE rn = 1
                ORDER BY distance ASC
            '''
            params: list[object] = [query_blob, query_blob, *file_ids_unique]
            if where_clause and where_params:
                params.extend(where_params)
            cursor = conn.execute(query, params)
            rows = cursor.fetchall()
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
        finally:
            conn.close()

    def delete_by_file_id(self, file_id: int) -> None:
        """Remove all vectors for a given file."""
        db_path = str(settings.db_path)
        conn = sqlite3.connect(db_path)

        try:
            # Load sqlite-vec extension (needed for any vec_chunks operations)
            if not _load_sqlite_vec_extension_sync(conn):
                log.warning('sqlite_vec_extension_not_loaded', action='delete_may_fail')

            conn.execute('DELETE FROM vec_chunks WHERE file_id = ?', (file_id,))
            conn.commit()
        finally:
            conn.close()

    def drop_all(self) -> int:
        """Delete all vectors from vec_chunks table."""
        db_path = str(settings.db_path)
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row

        try:
            # Load sqlite-vec extension
            if not _load_sqlite_vec_extension_sync(conn):
                log.warning('sqlite_vec_extension_not_loaded', action='drop_all_may_fail')

            # Get count before deletion
            cursor = conn.execute('SELECT COUNT(*) as count FROM vec_chunks')
            row = cursor.fetchone()
            count = int(row['count']) if row else 0

            # Delete all rows
            conn.execute('DELETE FROM vec_chunks')
            conn.commit()

            log.info('sqlite_vec_drop_all_complete', rows_deleted=count)
            return count
        finally:
            conn.close()

    def build_index(self) -> bool:
        """
        Placeholder for ANN index build.

        Current implementation intentionally keeps exact cosine search semantics.
        Metadata B-tree indexes are managed in SQLite schema migration/init.

        Returns:
            True to indicate no-op completed.
        """
        log.info('vector_index_build_skipped', reason='exact_search_mode')
        return True

    def get_stats(self) -> dict:
        """Return statistics about the vector store."""
        db_path = str(settings.db_path)
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row

        try:
            # Load sqlite-vec extension (needed for vec_chunks queries)
            if not _load_sqlite_vec_extension_sync(conn):
                log.warning('sqlite_vec_extension_not_loaded', action='stats_may_be_incomplete')

            cursor = conn.execute('SELECT COUNT(*) as count FROM vec_chunks')
            row = cursor.fetchone()
            total_vectors = int(row['count']) if row else 0

            # Storage size is part of SQLite database, not separate directory
            # Get database file size
            storage_bytes = 0
            if settings.db_path and settings.db_path.exists():
                storage_bytes = settings.db_path.stat().st_size

            return {
                'total_vectors': total_vectors,
                'storage_bytes': storage_bytes,
            }
        finally:
            conn.close()


# ==============================================================================
# Module-level singleton
# ==============================================================================

vector_store = VectorStore()
