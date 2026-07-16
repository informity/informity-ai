"""Test module for tests test api security."""

import pytest
from fastapi import HTTPException

from informity.api.security import (
    INDEX_RESET_IN_PROGRESS_DETAIL,
    LLM_BUSY_DETAIL,
    TAURI_SESSION_HEADER,
    EndpointGuard,
    get_cors_allow_origins,
    get_tauri_session_token_from_env,
    is_loopback_host,
    is_tauri_desktop_mode,
    is_tauri_session_authorized,
    raise_if_index_reset_in_progress,
    raise_if_llm_busy,
)


@pytest.mark.asyncio
async def test_endpoint_guard_blocks_excess_in_flight() -> None:
    """Test endpoint guard blocks excess in flight."""
    guard = EndpointGuard(
        name="test",
        max_in_flight=1,
        max_requests_per_window=10,
        window_seconds=60,
    )

    first = guard.slot()
    await first.__aenter__()
    try:
        second = guard.slot()
        with pytest.raises(HTTPException) as exc_info:
            await second.__aenter__()
        assert exc_info.value.status_code == 429
    finally:
        await first.__aexit__(None, None, None)


@pytest.mark.asyncio
async def test_endpoint_guard_rate_limit() -> None:
    """Test endpoint guard rate limit."""
    guard = EndpointGuard(
        name="test-rate",
        max_in_flight=2,
        max_requests_per_window=1,
        window_seconds=60,
    )

    await guard.check_rate_limit()
    with pytest.raises(HTTPException) as exc_info:
        await guard.check_rate_limit()
    assert exc_info.value.status_code == 429


def test_get_tauri_session_token_from_env() -> None:
    """Test get tauri session token from env."""
    assert (
        get_tauri_session_token_from_env({"INFORMITY_TAURI_SESSION_TOKEN": "  token123  "})
        == "token123"
    )
    assert get_tauri_session_token_from_env({"INFORMITY_TAURI_SESSION_TOKEN": "   "}) is None
    assert get_tauri_session_token_from_env({}) is None


def test_is_tauri_desktop_mode() -> None:
    """Test is tauri desktop mode."""
    assert is_tauri_desktop_mode("token") is True
    assert is_tauri_desktop_mode(None) is False


def test_is_tauri_session_authorized() -> None:
    """Test is tauri session authorized."""
    assert is_tauri_session_authorized({}, None) is True
    assert is_tauri_session_authorized({TAURI_SESSION_HEADER: "abc"}, "abc") is True
    assert is_tauri_session_authorized({TAURI_SESSION_HEADER: "abc"}, "def") is False
    assert is_tauri_session_authorized({}, "abc") is False


def test_get_cors_allow_origins_desktop_mode() -> None:
    """Test get cors allow origins desktop mode."""
    origins = get_cors_allow_origins(8420, desktop_mode=True)
    assert origins == ["http://127.0.0.1:5173", "http://localhost:5173", "tauri://localhost"]


def test_get_cors_allow_origins_standard_mode() -> None:
    """Test get cors allow origins standard mode."""
    origins = get_cors_allow_origins(8420, desktop_mode=False)
    assert "http://localhost:8420" in origins
    assert "http://127.0.0.1:8420" in origins
    assert "http://localhost:3000" in origins
    assert "http://127.0.0.1:5173" in origins
    assert "http://localhost:5173" in origins
    assert "tauri://localhost" in origins


def test_is_loopback_host() -> None:
    """Test is loopback host."""
    assert is_loopback_host("127.0.0.1") is True
    assert is_loopback_host("localhost") is True
    assert is_loopback_host("::1") is True
    assert is_loopback_host("0.0.0.0") is False
    assert is_loopback_host("192.168.1.10") is False


def test_raise_if_llm_busy() -> None:
    """Test raise if llm busy."""
    raise_if_llm_busy(False)
    with pytest.raises(HTTPException) as exc_info:
        raise_if_llm_busy(True)
    assert exc_info.value.status_code == 409
    assert exc_info.value.detail == LLM_BUSY_DETAIL


def test_raise_if_index_reset_in_progress() -> None:
    """Test raise if index reset in progress."""
    raise_if_index_reset_in_progress(False)
    with pytest.raises(HTTPException) as exc_info:
        raise_if_index_reset_in_progress(True)
    assert exc_info.value.status_code == 409
    assert exc_info.value.detail == INDEX_RESET_IN_PROGRESS_DETAIL
