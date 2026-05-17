from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from informity.mcp import http_server


def test_mcp_http_rate_limit_returns_429(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('INFORMITY_MCP_TOKEN', 'test-token')
    monkeypatch.setattr(http_server, '_RATE_LIMIT_MAX_REQUESTS', 1)
    monkeypatch.setattr(http_server, '_RATE_LIMIT_WINDOW_SECONDS', 60.0)
    http_server._rate_limit_buckets.clear()

    app = http_server.create_http_app()
    client = TestClient(app)
    payload = {
        'jsonrpc': '2.0',
        'id': 1,
        'method': 'initialize',
        'params': {'protocolVersion': '2025-11-25', 'capabilities': {}, 'clientInfo': {'name': 'test', 'version': '1'}},
    }
    headers = {'Authorization': 'Bearer test-token'}

    first = client.post('/mcp', json=payload, headers=headers)
    second = client.post('/mcp', json=payload, headers=headers)

    assert first.status_code == 200
    assert second.status_code == 429
    assert second.json()['error']['message'] == 'Rate limit exceeded'

