from __future__ import annotations

import shutil
from contextlib import suppress
from pathlib import Path

import aiosqlite
import structlog

log = structlog.get_logger(__name__)


async def migrate_legacy_upload_storage_layout(
    db: aiosqlite.Connection,
    app_data_dir: Path,
) -> dict[str, int]:
    """
    Migrate legacy `storage/upload/{chat,translate}` layouts to the current
    split roots used by chat uploads and translate uploads.

    This is a best-effort startup compatibility shim:
    - `storage/upload/chat/*`   -> `storage/uploads/*`
    - `storage/upload/translate/*` -> `storage/translate/*`

    It updates matching `files.path` and `files.source_item_id` rows so older
    uploads remain discoverable after the move.
    """
    storage_root = app_data_dir / 'storage'
    legacy_root = storage_root / 'upload'
    if not legacy_root.exists():
        return {'migrated_dirs': 0, 'updated_file_rows': 0}

    migrations = (
        (legacy_root / 'chat', storage_root / 'uploads'),
        (legacy_root / 'translate', storage_root / 'translate'),
    )

    migrated_dirs = 0
    updated_file_rows = 0

    for legacy_section_root, new_section_root in migrations:
        if not legacy_section_root.exists():
            continue
        new_section_root.mkdir(parents=True, exist_ok=True)

        for legacy_child in sorted(item for item in legacy_section_root.iterdir() if item.is_dir()):
            target_child = new_section_root / legacy_child.name
            if target_child.exists():
                log.info(
                    'legacy_upload_storage_target_exists',
                    legacy_path=str(legacy_child),
                    target_path=str(target_child),
                )
                continue

            shutil.move(str(legacy_child), str(target_child))
            migrated_dirs += 1

            old_prefix = str(legacy_child)
            new_prefix = str(target_child)
            cursor = await db.execute(
                'SELECT id, path FROM files WHERE path = ? OR path LIKE ?',
                (old_prefix, f'{old_prefix}/%'),
            )
            rows = await cursor.fetchall()
            for row in rows:
                old_path = str(row['path'] or '')
                if not old_path.startswith(old_prefix):
                    continue
                new_path = new_prefix + old_path[len(old_prefix):]
                if new_path == old_path:
                    continue
                await db.execute(
                    'UPDATE files SET path = ?, source_item_id = ? WHERE id = ?',
                    (new_path, new_path, int(row['id'])),
                )
                updated_file_rows += 1

        with suppress(OSError):
            if legacy_section_root.exists() and not any(legacy_section_root.iterdir()):
                legacy_section_root.rmdir()

    with suppress(OSError):
        if legacy_root.exists() and not any(legacy_root.iterdir()):
            legacy_root.rmdir()

    await db.commit()
    if migrated_dirs or updated_file_rows:
        log.info(
            'legacy_upload_storage_migrated',
            migrated_dirs=migrated_dirs,
            updated_file_rows=updated_file_rows,
        )
    return {
        'migrated_dirs': migrated_dirs,
        'updated_file_rows': updated_file_rows,
    }
