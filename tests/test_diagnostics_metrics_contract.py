from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from informity.config import settings
from informity.db.sqlite import (
    get_connection,
    get_diagnostics_metrics_since,
    init_db,
    insert_diagnostics_metrics,
)


@pytest.mark.asyncio
async def test_insert_diagnostics_metrics_normalizes_query_type_and_issues(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    db_path = tmp_path / 'diagnostics-contract.db'
    monkeypatch.setattr(settings, 'db_path', db_path)

    await init_db()
    db = await get_connection()
    try:
        metrics = SimpleNamespace(
            chat_id='chat-1',
            question='Question',
            model_filename='model.gguf',
            query_type='NON_CANONICAL_TYPE',
            raw_chunks_count=3,
            sources_count=2,
            generation_seconds=1.2,
            answer_length=120,
            timeout_occurred=False,
            has_empty_answer=False,
            has_refusal_pattern=False,
            unsupported_claim_count=0,
            evidence_coverage_rate=0.0,
            not_found_count=0,
        )
        await insert_diagnostics_metrics(
            db=db,
            metrics=metrics,
            detected_issues=['timeout', 'TIMEOUT', 'unknown_issue', ''],
            run_id=None,
        )

        rows = await get_diagnostics_metrics_since(db=db, days=30)
        assert len(rows) == 1
        assert rows[0]['type'] == 'user'
        assert rows[0]['query_type'] == 'unknown'
        assert rows[0]['detected_issues'] == ['timeout']
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_get_diagnostics_metrics_since_filters_non_canonical_issue_values(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    db_path = tmp_path / 'diagnostics-contract-read.db'
    monkeypatch.setattr(settings, 'db_path', db_path)

    await init_db()
    db = await get_connection()
    try:
        await db.execute(
            """
            INSERT INTO response_diagnostics_metrics (
              chat_id, question, type, model_filename, run_id, query_type,
              raw_chunks_count, sources_count, generation_seconds, answer_length,
              timeout_occurred, has_empty_answer, has_refusal_pattern,
              unsupported_claim_count, evidence_coverage_rate, not_found_count,
              detected_issues, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                'chat-2',
                'Q2',
                'evaluation',
                'model.gguf',
                'run-1',
                'focused',
                5,
                2,
                2.4,
                200,
                0,
                0,
                0,
                0,
                0.0,
                0,
                '["timeout", "unsupported_claims_detected", "bad_issue"]',
                datetime.now(UTC).isoformat(),
            ),
        )
        await db.commit()

        rows = await get_diagnostics_metrics_since(db=db, days=30, type_filter='evaluation', run_id_filter='run-1')
        assert len(rows) == 1
        assert rows[0]['detected_issues'] == ['timeout', 'unsupported_claims_detected']
    finally:
        await db.close()

