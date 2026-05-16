from __future__ import annotations

import json

import pytest

from informity.mcp import stdio_server


@pytest.mark.asyncio
async def test_initialize_with_zero_id_returns_response() -> None:
    response = await stdio_server._handle_request({'id': 0, 'method': 'initialize'})
    assert response is not None
    assert response['id'] == 0
    assert response['result']['serverInfo']['name'] == stdio_server.SERVER_NAME


@pytest.mark.asyncio
async def test_notification_with_zero_id_still_responds_method_not_found() -> None:
    response = await stdio_server._handle_request({'id': 0, 'method': 'notifications/unknown'})
    assert response is not None
    assert response['id'] == 0
    assert response['error']['code'] == -32601


@pytest.mark.asyncio
async def test_tools_list_exposes_readonly_tools() -> None:
    response = await stdio_server._handle_request({'id': 1, 'method': 'tools/list'})
    assert response is not None
    tools = response['result']['tools']
    names = {tool['name'] for tool in tools}
    assert 'informity_health' in names
    assert 'informity_search_semantic' in names


def test_write_message_uses_newline_delimited_json_framing() -> None:
    class _Buffer:
        def __init__(self) -> None:
            self.parts: list[bytes] = []

        def write(self, data: bytes) -> None:
            self.parts.append(data)

        def flush(self) -> None:
            return None

    buffer = _Buffer()
    payload = {'jsonrpc': '2.0', 'id': 7, 'result': {'ok': True}}

    stdio_server._write_message(buffer, payload)

    raw = b''.join(buffer.parts)
    expected = json.dumps(payload, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
    assert raw == expected + b'\n'
