from __future__ import annotations

from pathlib import Path

import aiosqlite

from informity.db.models import IndexedFile
from informity.db.sqlite import get_file_by_path
from informity.indexer.pipeline import IndexResult, index_file
from informity.scanner.crawler import ScannedFile


async def index_uploaded_file(
    db: aiosqlite.Connection,
    scanned: ScannedFile,
    *,
    source_provider: str,
    entity_type: str,
) -> tuple[IndexResult, IndexedFile | None]:
    result = await index_file(
        db,
        scanned,
        source_provider=source_provider,
        entity_type=entity_type,
    )
    if not result.success:
        return result, None
    indexed = await get_file_by_path(db, str(Path(scanned.path)))
    return result, indexed
