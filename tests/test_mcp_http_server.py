from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from informity.mcp.http_server import create_http_app


@pytest.fixture
def http_client() -> TestClient:
    app = create_http_app()
    return TestClient(app)


def test_http_initialize_requires_bearer_token(http_client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('INFORMITY_MCP_TOKEN', 'expected-token')
    response = http_client.post(
        '/',
        json={
            'jsonrpc': '2.0',
            'id': 1,
            'method': 'initialize',
            'params': {'protocolVersion': '2025-11-25'},
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload['error']['code'] == -32001


def test_http_initialize_with_bearer_token_succeeds(http_client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('INFORMITY_MCP_TOKEN', 'expected-token')
    response = http_client.post(
        '/',
        headers={'Authorization': 'Bearer expected-token'},
        json={
            'jsonrpc': '2.0',
            'id': 1,
            'method': 'initialize',
            'params': {'protocolVersion': '2025-11-25'},
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload['id'] == 1
    assert payload['result']['serverInfo']['name'] == 'informity-mcp'


def test_http_tools_list_with_bearer_token_succeeds(http_client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('INFORMITY_MCP_TOKEN', 'expected-token')
    response = http_client.post(
        '/mcp',
        headers={'Authorization': 'Bearer expected-token'},
        json={
            'jsonrpc': '2.0',
            'id': 2,
            'method': 'tools/list',
            'params': {},
        },
    )
    assert response.status_code == 200
    payload = response.json()
    names = {tool['name'] for tool in payload['result']['tools']}
    assert 'informity_health' in names


def test_http_rejects_oversized_body(http_client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('INFORMITY_MCP_TOKEN', 'expected-token')
    large = 'x' * (600 * 1024)
    response = http_client.post(
        '/mcp',
        headers={'Authorization': 'Bearer expected-token'},
        content=large,
    )
    assert response.status_code == 413
    payload = response.json()
    assert payload['error']['code'] == -32600
    assert payload['error']['message'] == 'Request body too large'
