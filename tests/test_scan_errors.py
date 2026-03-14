from datetime import UTC, datetime
from pathlib import Path

import pytest

from informity.api.routes_scan import get_scan_status
from informity.config import settings
from informity.db.models import ScanErrorRecord, ScanRecord, ScanStatus
from informity.db.sqlite import (
    get_connection,
    init_db,
    insert_scan_error_record,
    insert_scan_record,
)


@pytest.mark.asyncio
async def test_scan_status_includes_recent_errors_and_timeout_count(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    db_path = tmp_path / 'scan-errors-test.db'
    monkeypatch.setattr(settings, 'db_path', db_path)

    await init_db()
    db = await get_connection()
    try:
        scan = await insert_scan_record(
            db,
            ScanRecord(
                started_at=datetime.now(UTC),
                status=ScanStatus.RUNNING,
            ),
        )
        assert scan.id is not None

        await insert_scan_error_record(
            db,
            ScanErrorRecord(
                scan_id=scan.id,
                path='/tmp/a.pdf',
                filename='a.pdf',
                extension='.pdf',
                operation='indexing_file',
                error_code='scan_file_timeout',
                error_message='File processing exceeded timeout (90s)',
                is_timeout=True,
            ),
        )
        await insert_scan_error_record(
            db,
            ScanErrorRecord(
                scan_id=scan.id,
                path='/tmp/b.md',
                filename='b.md',
                extension='.md',
                operation='indexing_file',
                error_code='scan_processing_exception',
                error_message='mock extractor failure',
                is_timeout=False,
            ),
        )

        status = await get_scan_status(db)
        assert status.status == 'running'
        assert status.timeout_errors == 1
        assert len(status.recent_errors) == 2
        assert {item.filename for item in status.recent_errors} == {'a.pdf', 'b.md'}
        assert any(item.is_timeout for item in status.recent_errors)
    finally:
        await db.close()
