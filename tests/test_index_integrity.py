import sqlite3
from pathlib import Path

import pytest

from informity.config import settings
from informity.db.sqlite import (
    get_connection,
    get_index_integrity_issues,
    init_db,
    repair_index_integrity_issues,
)


@pytest.mark.asyncio
async def test_index_integrity_detect_and_repair(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    db_path = tmp_path / 'integrity-test.db'
    monkeypatch.setattr(settings, 'db_path', db_path)

    await init_db()
    db = await get_connection()
    try:
        # Temporarily disable FK checks so we can seed broken rows for validation.
        await db.execute('PRAGMA foreign_keys=OFF')
        await db.execute(
            '''
            INSERT INTO files (id, path, filename, extension, size_bytes, content_hash, extracted_text_preview, category, tags, indexed_at, modified_at)
            VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'), datetime('now'))
            ''',
            ('/tmp/a.txt', 'a.txt', '.txt', 10, 'hash-a', 'preview', 'plaintext', '[]'),
        )
        await db.execute(
            '''
            INSERT INTO files (id, path, filename, extension, size_bytes, content_hash, extracted_text_preview, category, tags, indexed_at, modified_at)
            VALUES (2, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'), datetime('now'))
            ''',
            ('/tmp/b.txt', 'b.txt', '.txt', 10, 'hash-b', 'preview', 'plaintext', '[]'),
        )
        await db.execute(
            '''
            INSERT INTO chunks (id, file_id, chunk_index, content, token_count, parent_id)
            VALUES (10, 1, 0, 'parent', 5, NULL)
            '''
        )
        await db.execute(
            '''
            INSERT INTO chunks (id, file_id, chunk_index, content, token_count, parent_id)
            VALUES (11, 1, 1, 'child-without-vector', 4, 10)
            '''
        )
        await db.execute(
            '''
            INSERT INTO chunks (id, file_id, chunk_index, content, token_count, parent_id)
            VALUES (12, 1, 2, 'child-missing-parent', 4, 999)
            '''
        )
        await db.execute(
            '''
            INSERT INTO chunks (id, file_id, chunk_index, content, token_count, parent_id)
            VALUES (13, 999, 0, 'orphan-chunk', 4, NULL)
            '''
        )
        await db.execute(
            '''
            INSERT INTO vec_chunks (chunk_id, file_id, file_path, chunk_text, vector, year, filename, extension, category)
            VALUES (10, 1, ?, ?, ?, NULL, ?, ?, ?)
            ''',
            ('/tmp/wrong-path.txt', 'wrong-parent-text', sqlite3.Binary(b'v1'), 'wrong-a.txt', '.md', 'data'),
        )
        await db.execute(
            '''
            INSERT INTO vec_chunks (chunk_id, file_id, file_path, chunk_text, vector, year, filename, extension, category)
            VALUES (999, 1, ?, ?, ?, NULL, ?, ?, ?)
            ''',
            ('/tmp/a.txt', 'orphan-vector', sqlite3.Binary(b'v2'), 'a.txt', '.txt', 'plaintext'),
        )
        await db.execute(
            '''
            INSERT INTO vec_chunks (chunk_id, file_id, file_path, chunk_text, vector, year, filename, extension, category)
            VALUES (998, 999, ?, ?, ?, NULL, ?, ?, ?)
            ''',
            ('/tmp/z.txt', 'orphan-vector-file', sqlite3.Binary(b'v3'), 'z.txt', '.txt', 'plaintext'),
        )
        await db.commit()
        await db.execute('PRAGMA foreign_keys=ON')

        issues_before = await get_index_integrity_issues(db)
        assert issues_before['orphan_chunks_missing_file'] == 1
        assert issues_before['orphan_vectors_missing_chunk'] >= 1
        assert issues_before['orphan_vectors_missing_file'] >= 1
        assert issues_before['child_chunks_missing_parent'] == 1
        assert issues_before['files_without_chunks'] == 1
        assert issues_before['child_chunks_without_vector'] >= 1
        assert issues_before['vec_file_path_mismatch'] >= 1
        assert issues_before['vec_filename_mismatch'] >= 1
        assert issues_before['vec_extension_mismatch'] >= 1
        assert issues_before['vec_category_mismatch'] >= 1
        assert issues_before['vec_chunk_text_mismatch'] >= 1

        repairs = await repair_index_integrity_issues(db)
        assert sum(repairs.values()) >= 1

        issues_after = await get_index_integrity_issues(db)
        assert sum(issues_after.values()) == 0
    finally:
        await db.close()
