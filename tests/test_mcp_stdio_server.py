"""Test module for tests test mcp stdio server."""

from __future__ import annotations

import io
import json

import pytest

from informity.mcp import stdio_server


@pytest.mark.asyncio
async def test_initialize_with_zero_id_returns_response() -> None:
    """Test initialize with zero id returns response."""
    response = await stdio_server._handle_request({"id": 0, "method": "initialize"})
    assert response is not None
    assert response["id"] == 0
    assert response["result"]["serverInfo"]["name"] == stdio_server.SERVER_NAME


@pytest.mark.asyncio
async def test_notification_with_zero_id_still_responds_method_not_found() -> None:
    """Test notification with zero id still responds method not found."""
    response = await stdio_server._handle_request({"id": 0, "method": "notifications/unknown"})
    assert response is not None
    assert response["id"] == 0
    assert response["error"]["code"] == -32601


@pytest.mark.asyncio
async def test_tools_list_exposes_readonly_tools() -> None:
    """Test tools list exposes readonly tools."""
    response = await stdio_server._handle_request({"id": 1, "method": "tools/list"})
    assert response is not None
    tools = response["result"]["tools"]
    names = {tool["name"] for tool in tools}
    assert "informity_health" in names
    assert "informity_search_semantic" in names


def test_write_message_uses_newline_delimited_json_framing() -> None:
    """Test write message uses newline delimited json framing."""

    class _Buffer:
        def __init__(self) -> None:
            """Initialize the instance."""
            self.parts: list[bytes] = []

        def write(self, data: bytes) -> None:
            """Write."""
            self.parts.append(data)

        def flush(self) -> None:
            """Flush."""
            return None

    buffer = _Buffer()
    payload = {"jsonrpc": "2.0", "id": 7, "result": {"ok": True}}

    stdio_server._write_message(buffer, payload)

    raw = b"".join(buffer.parts)
    expected = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    assert raw == expected + b"\n"


def test_read_message_tolerates_non_json_noise_before_valid_json_line() -> None:
    """Test read message tolerates non json noise before valid json line."""
    stream = io.BytesIO(b'INFO noisy startup log\n{"jsonrpc":"2.0","id":1,"method":"ping"}\n')

    first = stdio_server._read_message(stream)
    second = stdio_server._read_message(stream)

    assert first is None
    assert second is not None
    assert second["method"] == "ping"
