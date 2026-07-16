"""Test module for tests test routes translate."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from informity.api import routes_translate


class _DummyDB:
    pass


@pytest.mark.asyncio
async def test_cancel_translate_job_writes_cancel_token_and_cancelled_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test cancel translate job writes cancel token and cancelled event."""
    updates: list[dict[str, object]] = []
    emitted: list[dict[str, object]] = []

    async def _fake_get_translate_job(db, job_id):
        """Internal helper for fake get translate job."""
        _ = (db, job_id)
        return {
            "status": "running",
            "file_id": 17,
            "target_language": "Spanish",
            "completed_sections": 2,
            "section_count": 5,
        }

    async def _fake_update_translate_job(db, job_id, **kwargs):
        """Internal helper for fake update translate job."""
        _ = (db, job_id)
        updates.append(kwargs)

    async def _fake_get_file_by_id(db, file_id):
        """Internal helper for fake get file by id."""
        _ = (db, file_id)
        return SimpleNamespace(filename="example.pdf")

    async def _fake_emit_log_event(**kwargs):
        """Internal helper for fake emit log event."""
        emitted.append(kwargs)

    monkeypatch.setattr(routes_translate, "get_translate_job", _fake_get_translate_job)
    monkeypatch.setattr(routes_translate, "update_translate_job", _fake_update_translate_job)
    monkeypatch.setattr(routes_translate, "get_file_by_id", _fake_get_file_by_id)
    monkeypatch.setattr(routes_translate, "emit_log_event", _fake_emit_log_event)

    response = await routes_translate.cancel_translate_job("job-1", db=_DummyDB())

    assert response == {"cancelled": True}
    assert updates == [
        {"status": "stalled", "error": routes_translate.TRANSLATE_CANCEL_ERROR_TOKEN}
    ]
    assert emitted[0]["event_name"] == "translate_job_cancelled"
    assert (
        emitted[0]["details"]["cancel_error_token"] == routes_translate.TRANSLATE_CANCEL_ERROR_TOKEN
    )


def test_cancelled_translate_row_helper_detects_token() -> None:
    """Test cancelled translate row helper detects token."""
    assert (
        routes_translate._is_cancelled_translate_row(
            {"status": "stalled", "error": routes_translate.TRANSLATE_CANCEL_ERROR_TOKEN},
        )
        is True
    )
    assert (
        routes_translate._is_cancelled_translate_row(
            {"status": "stalled", "error": "No progress within stall window."},
        )
        is False
    )
    assert (
        routes_translate._is_cancelled_translate_row(
            {"status": "running", "error": routes_translate.TRANSLATE_CANCEL_ERROR_TOKEN},
        )
        is False
    )


def test_split_text_for_translation_breaks_flat_docs() -> None:
    """Test split text for translation breaks flat docs."""
    text = " ".join(
        f"Sentence {index} has enough words to count as a real sentence." for index in range(1, 26)
    )

    parts = routes_translate._split_text_for_translation(text, max_tokens=30)

    assert len(parts) > 1
    assert all(routes_translate._count_tokens(part) <= 30 for part in parts)
    assert "Sentence 1" in parts[0]
    assert "Sentence 25" in parts[-1]
