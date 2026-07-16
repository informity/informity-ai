"""Test module for tests test chat messages schema migration."""

# pylint: disable=line-too-long

import sqlite3
from pathlib import Path

import pytest

from informity.config import settings
from informity.db.sqlite import SCHEMA_VERSION, get_chat, get_connection, init_db


def _create_legacy_chat_messages_table(
    conn: sqlite3.Connection,
    *,
    include_role_id: bool = False,
    include_specialization_id: bool = False,
    include_translation_columns: bool = False,
    include_scope_columns: bool = False,
) -> None:
    """Internal helper for create legacy chat messages table."""
    columns = [
        "id INTEGER PRIMARY KEY AUTOINCREMENT",
        "chat_id TEXT NOT NULL",
        "role TEXT NOT NULL",
        "content TEXT NOT NULL",
        "sources TEXT",
        "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
    ]
    if include_role_id:
        columns.append("role_id TEXT")
    if include_specialization_id:
        columns.append("specialization_id TEXT")
    if include_translation_columns:
        columns.extend(
            [
                "translated_from_message_id INTEGER",
                "translation_language TEXT",
                "translation_tone TEXT",
                "translation_source_hash TEXT NOT NULL DEFAULT ''",
                "translation_is_stale INTEGER DEFAULT 0",
            ]
        )
    if include_scope_columns:
        columns.extend(
            [
                "retrieval_scope_kind TEXT",
                "retrieval_scope_key TEXT",
                "model_filename TEXT",
                "is_internal INTEGER DEFAULT 0",
            ]
        )
    conn.execute(f"CREATE TABLE chat_messages ({', '.join(columns)})")
    conn.execute(
        "CREATE TABLE chats (chat_id TEXT PRIMARY KEY, title TEXT, updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
    )
    conn.execute(
        """
        INSERT INTO chats (chat_id, title)
        VALUES ('chat-legacy', 'Legacy chat')
        """
    )
    conn.execute(
        """
        INSERT INTO chat_messages (chat_id, role, content)
        VALUES ('chat-legacy', 'user', 'Hello from legacy schema')
        """
    )


def _chat_message_column_names(conn: sqlite3.Connection) -> set[str]:
    """Internal helper for chat message column names."""
    rows = conn.execute("PRAGMA table_info('chat_messages')").fetchall()
    return {str(row[1]) for row in rows}


@pytest.mark.asyncio
async def test_v7_database_missing_scope_columns_is_reconciled_on_startup(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Test v7 database missing scope columns is reconciled on startup."""
    db_path = tmp_path / "chat-migration-v7-missing-scope.db"
    monkeypatch.setattr(settings, "db_path", db_path)

    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE schema_version (version INTEGER PRIMARY KEY)")
    conn.execute("INSERT INTO schema_version (version) VALUES (7)")
    _create_legacy_chat_messages_table(
        conn,
        include_specialization_id=True,
        include_translation_columns=True,
        include_scope_columns=False,
    )
    conn.commit()
    conn.close()

    await init_db()

    verify_conn = sqlite3.connect(str(db_path))
    try:
        columns = _chat_message_column_names(verify_conn)
        assert "retrieval_scope_kind" in columns
        assert "retrieval_scope_key" in columns
        assert "model_filename" in columns
        assert "is_internal" in columns
        schema_row = verify_conn.execute("SELECT version FROM schema_version LIMIT 1").fetchone()
        assert schema_row is not None
        assert int(schema_row[0]) == SCHEMA_VERSION
    finally:
        verify_conn.close()

    db = await get_connection()
    try:
        messages = await get_chat(db, chat_id="chat-legacy")
        assert len(messages) == 1
        assert messages[0].content == "Hello from legacy schema"
        assert messages[0].retrieval_scope_kind is None
        assert messages[0].is_internal is False
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_v8_database_missing_scope_columns_is_reconciled_on_startup(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Test v8 database missing scope columns is reconciled on startup."""
    db_path = tmp_path / "chat-migration-v8-missing-scope.db"
    monkeypatch.setattr(settings, "db_path", db_path)

    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE schema_version (version INTEGER PRIMARY KEY)")
    conn.execute("INSERT INTO schema_version (version) VALUES (8)")
    _create_legacy_chat_messages_table(
        conn,
        include_specialization_id=True,
        include_translation_columns=True,
        include_scope_columns=False,
    )
    conn.commit()
    conn.close()

    await init_db()

    verify_conn = sqlite3.connect(str(db_path))
    try:
        columns = _chat_message_column_names(verify_conn)
        assert "retrieval_scope_kind" in columns
        assert "retrieval_scope_key" in columns
        assert "model_filename" in columns
        assert "is_internal" in columns
        schema_row = verify_conn.execute("SELECT version FROM schema_version LIMIT 1").fetchone()
        assert schema_row is not None
        assert int(schema_row[0]) == SCHEMA_VERSION
    finally:
        verify_conn.close()


@pytest.mark.asyncio
async def test_v9_database_missing_file_discovery_column_is_reconciled_on_startup(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Test v9 database missing file discovery column is reconciled on startup."""
    db_path = tmp_path / "chat-migration-v9-missing-file-discovery.db"
    monkeypatch.setattr(settings, "db_path", db_path)

    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE schema_version (version INTEGER PRIMARY KEY)")
    conn.execute("INSERT INTO schema_version (version) VALUES (9)")
    _create_legacy_chat_messages_table(
        conn,
        include_specialization_id=True,
        include_translation_columns=True,
        include_scope_columns=True,
    )
    conn.commit()
    conn.close()

    await init_db()

    verify_conn = sqlite3.connect(str(db_path))
    try:
        columns = _chat_message_column_names(verify_conn)
        assert "file_discovery" in columns
        schema_row = verify_conn.execute("SELECT version FROM schema_version LIMIT 1").fetchone()
        assert schema_row is not None
        assert int(schema_row[0]) == SCHEMA_VERSION
    finally:
        verify_conn.close()


@pytest.mark.asyncio
async def test_v3_database_migrates_chat_messages_through_full_chain(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Test v3 database migrates chat messages through full chain."""
    db_path = tmp_path / "chat-migration-v3.db"
    monkeypatch.setattr(settings, "db_path", db_path)

    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE schema_version (version INTEGER PRIMARY KEY)")
    conn.execute("INSERT INTO schema_version (version) VALUES (3)")
    _create_legacy_chat_messages_table(conn, include_role_id=True)
    conn.commit()
    conn.close()

    await init_db()

    verify_conn = sqlite3.connect(str(db_path))
    try:
        columns = _chat_message_column_names(verify_conn)
        assert "role_id" not in columns
        assert "specialization_id" in columns
        assert "retrieval_scope_kind" in columns
        assert "model_filename" in columns
        assert "is_internal" in columns
        assert "translated_from_message_id" in columns
        schema_row = verify_conn.execute("SELECT version FROM schema_version LIMIT 1").fetchone()
        assert schema_row is not None
        assert int(schema_row[0]) == SCHEMA_VERSION
    finally:
        verify_conn.close()


@pytest.mark.asyncio
async def test_minimal_legacy_chat_messages_table_is_reconciled_without_data_loss(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Test minimal legacy chat messages table is reconciled without data loss."""
    db_path = tmp_path / "chat-migration-minimal.db"
    monkeypatch.setattr(settings, "db_path", db_path)

    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE schema_version (version INTEGER PRIMARY KEY)")
    conn.execute("INSERT INTO schema_version (version) VALUES (1)")
    _create_legacy_chat_messages_table(conn)
    conn.commit()
    conn.close()

    await init_db()

    db = await get_connection()
    try:
        messages = await get_chat(db, chat_id="chat-legacy")
        assert len(messages) == 1
        assert messages[0].content == "Hello from legacy schema"
    finally:
        await db.close()
