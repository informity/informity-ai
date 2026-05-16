from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from informity.mcp.authorization import McpAuthorizationError, authorize_mcp_request
from informity.mcp.server import mcp_readonly_server
from informity.mcp.tools_readonly import McpReadScope, tool_files_list, tool_search_semantic


def test_mcp_authorization_allows_stdio() -> None:
    authorize_mcp_request(transport='stdio', bearer_token=None)


def test_mcp_authorization_rejects_http_without_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv('INFORMITY_MCP_TOKEN', raising=False)
    with pytest.raises(McpAuthorizationError):
        authorize_mcp_request(transport='http', bearer_token='abc')


def test_mcp_authorization_rejects_invalid_http_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('INFORMITY_MCP_TOKEN', 'expected-token')
    with pytest.raises(McpAuthorizationError):
        authorize_mcp_request(transport='http', bearer_token='wrong-token')


@pytest.mark.asyncio
async def test_tool_files_list_metadata_only_hides_preview(monkeypatch: pytest.MonkeyPatch) -> None:
    from informity.mcp import tools_readonly as mod

    fake_file = SimpleNamespace(
        id=7,
        filename='notes.md',
        path='/docs/notes.md',
        category=SimpleNamespace(value='notes'),
        extension='.md',
        indexed_at=datetime.now(UTC),
        extracted_text_preview='Secret draft preview',
    )

    async def _fake_get_files(*_args, **_kwargs):
        return [fake_file], 1

    monkeypatch.setattr(mod, 'get_files', _fake_get_files)

    payload = await tool_files_list(
        db=SimpleNamespace(),
        scope=McpReadScope(mode='metadata_only'),
        limit=10,
    )
    assert payload['total'] == 1
    assert payload['results'][0]['preview'] is None


@pytest.mark.asyncio
async def test_tool_search_semantic_metadata_only_drops_preview(monkeypatch: pytest.MonkeyPatch) -> None:
    from informity.mcp import tools_readonly as mod

    fake_file = SimpleNamespace(
        id=3,
        filename='policy.pdf',
        path='/docs/policy.pdf',
        category=SimpleNamespace(value='compliance'),
        extension='.pdf',
    )

    monkeypatch.setattr(mod.embedder, 'embed_query', lambda _q: [0.1, 0.2, 0.3])
    monkeypatch.setattr(
        mod.vector_store,
        'search_similar',
        lambda *_args, **_kwargs: [{'file_id': 3, 'chunk_text': 'very secret content', 'score': 0.12}],
    )

    async def _fake_get_files_by_ids(*_args, **_kwargs):
        return {3: fake_file}

    monkeypatch.setattr(mod, 'get_files_by_ids', _fake_get_files_by_ids)

    payload = await tool_search_semantic(
        db=SimpleNamespace(),
        scope=McpReadScope(mode='metadata_only'),
        query='secret',
        limit=5,
    )
    assert payload['total'] == 1
    assert 'preview' not in payload['results'][0]


@pytest.mark.asyncio
async def test_mcp_server_health_tool_works() -> None:
    payload = await mcp_readonly_server.execute_tool(
        tool_name='informity_health',
        args={},
        transport='stdio',
    )
    assert payload['ok'] is True
    assert payload['component'] == 'informity.mcp.readonly'
