"""Test module for tests test storage migrations."""

from __future__ import annotations

from pathlib import Path

import pytest

from informity.config import settings
from informity.db.sqlite import get_connection, init_db
from informity.storage_migrations import migrate_legacy_upload_storage_layout


@pytest.mark.asyncio
async def test_migrate_legacy_upload_storage_layout_moves_chat_and_translate_uploads(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Test migrate legacy upload storage layout moves chat and translate uploads."""
    app_data_dir = tmp_path / ".informity"
    db_path = app_data_dir / "db" / "informity.db"
    monkeypatch.setattr(settings, "app_data_dir", app_data_dir)
    monkeypatch.setattr(settings, "db_path", db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    await init_db()

    legacy_chat_file = (
        app_data_dir / "storage" / "upload" / "chat" / "chat-1" / "upload-1" / "chat.txt"
    )
    legacy_translate_file = (
        app_data_dir / "storage" / "upload" / "translate" / "tx-1" / "translate.txt"
    )
    legacy_chat_file.parent.mkdir(parents=True, exist_ok=True)
    legacy_translate_file.parent.mkdir(parents=True, exist_ok=True)
    legacy_chat_file.write_text("chat upload content", encoding="utf-8")
    legacy_translate_file.write_text("translate upload content", encoding="utf-8")

    db = await get_connection()
    try:
        await db.execute(
            """
            INSERT INTO files (
                source_provider, entity_type, source_item_id, path, filename, extension,
                size_bytes, content_hash, extracted_text_preview, category, modified_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "upload.local",
                "file",
                str(legacy_chat_file),
                str(legacy_chat_file),
                "chat.txt",
                ".txt",
                21,
                "hash-chat",
                "chat upload content",
                "plaintext",
                "2026-06-04T00:00:00+00:00",
            ),
        )
        await db.execute(
            """
            INSERT INTO files (
                source_provider, entity_type, source_item_id, path, filename, extension,
                size_bytes, content_hash, extracted_text_preview, category, modified_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "translate.local",
                "file",
                str(legacy_translate_file),
                str(legacy_translate_file),
                "translate.txt",
                ".txt",
                28,
                "hash-translate",
                "translate upload content",
                "plaintext",
                "2026-06-04T00:00:00+00:00",
            ),
        )
        await db.commit()

        result = await migrate_legacy_upload_storage_layout(db, app_data_dir)
        assert result == {"migrated_dirs": 2, "updated_file_rows": 2}

        updated_rows = await (
            await db.execute(
                "SELECT source_provider, source_item_id, path FROM files ORDER BY filename ASC"
            )
        ).fetchall()
        assert [str(row["path"]) for row in updated_rows] == [
            str(app_data_dir / "storage" / "uploads" / "chat-1" / "upload-1" / "chat.txt"),
            str(app_data_dir / "storage" / "translate" / "tx-1" / "translate.txt"),
        ]
        assert [str(row["source_item_id"]) for row in updated_rows] == [
            str(item["path"]) for item in updated_rows
        ]

        assert not (app_data_dir / "storage" / "upload" / "chat").exists()
        assert not (app_data_dir / "storage" / "upload" / "translate").exists()
        assert (app_data_dir / "storage" / "uploads" / "chat-1" / "upload-1" / "chat.txt").exists()
        assert (app_data_dir / "storage" / "translate" / "tx-1" / "translate.txt").exists()
    finally:
        await db.close()
