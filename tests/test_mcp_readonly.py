from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from informity.mcp.authorization import McpAuthorizationError, authorize_mcp_request
from informity.mcp.protocol import handle_jsonrpc_request
from informity.mcp.server import mcp_readonly_server
from informity.mcp.tools_readonly import (
    McpReadScope,
    tool_files_list,
    tool_index_status,
    tool_search_semantic,
)


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


@pytest.mark.asyncio
async def test_tool_search_semantic_excludes_upload_local_and_deduplicates_content_hash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from informity.mcp import tools_readonly as mod

    monkeypatch.setattr(mod.embedder, 'embed_query', lambda _q: [0.1, 0.2, 0.3])
    monkeypatch.setattr(
        mod.vector_store,
        'search_similar',
        lambda *_args, **_kwargs: [
            {'file_id': 10, 'chunk_text': 'termination clause A', 'score': 0.1},
            {'file_id': 11, 'chunk_text': 'termination clause A duplicate', 'score': 0.11},
            {'file_id': 12, 'chunk_text': 'upload copy should be hidden', 'score': 0.12},
        ],
    )

    file_a = SimpleNamespace(
        id=10,
        filename='Agreement.docx',
        path='/docs/Agreement.docx',
        category=SimpleNamespace(value='document'),
        extension='.docx',
        source_provider='filesystem',
        content_hash='hash-1',
    )
    file_a_dup = SimpleNamespace(
        id=11,
        filename='Agreement copy.docx',
        path='/docs/Agreement copy.docx',
        category=SimpleNamespace(value='document'),
        extension='.docx',
        source_provider='filesystem',
        content_hash='hash-1',
    )
    file_upload = SimpleNamespace(
        id=12,
        filename='Agreement upload.docx',
        path='/Users/me/.informity/storage/uploads/chat/upload.docx',
        category=SimpleNamespace(value='document'),
        extension='.docx',
        source_provider='upload.local',
        content_hash='hash-2',
    )

    async def _fake_get_files_by_ids(*_args, **_kwargs):
        return {10: file_a, 11: file_a_dup, 12: file_upload}

    monkeypatch.setattr(mod, 'get_files_by_ids', _fake_get_files_by_ids)

    payload = await tool_search_semantic(
        db=SimpleNamespace(),
        scope=McpReadScope(mode='search_snippets'),
        query='termination',
        limit=5,
    )
    assert payload['total'] == 1
    assert len(payload['results']) == 1
    assert payload['results'][0]['file_id'] == 10


@pytest.mark.asyncio
async def test_mcp_tools_call_times_out(monkeypatch: pytest.MonkeyPatch) -> None:
    from informity.mcp import protocol as mod

    class _SlowServer:
        async def execute_tool(self, **_kwargs):
            await asyncio.sleep(5.2)
            return {'ok': True}

    monkeypatch.setattr(mod, '_mcp_readonly_server', _SlowServer())
    monkeypatch.setattr(mod, '_mcp_tool_not_found_error', KeyError)
    monkeypatch.setattr(mod.settings, 'mcp_tool_call_timeout_seconds', 5.0)

    response = await handle_jsonrpc_request(
        {
            'jsonrpc': '2.0',
            'id': 9,
            'method': 'tools/call',
            'params': {'name': 'informity_health', 'arguments': {}},
        },
        transport='stdio',
        bearer_token=None,
    )
    assert response is not None
    assert response['error']['code'] == -32001
    assert response['error']['message'] == 'MCP tool call timed out'


@pytest.mark.asyncio
async def test_tool_files_list_default_limit_is_50_and_explicit_limit_allows_200(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from informity.mcp import tools_readonly as mod

    calls: list[int] = []

    async def _fake_get_files(*_args, **kwargs):
        calls.append(int(kwargs.get('limit', 0)))
        return [], 0

    monkeypatch.setattr(mod, 'get_files', _fake_get_files)

    await tool_files_list(
        db=SimpleNamespace(),
        scope=McpReadScope(mode='metadata_only'),
    )
    await tool_files_list(
        db=SimpleNamespace(),
        scope=McpReadScope(mode='metadata_only'),
        limit=200,
    )
    await tool_files_list(
        db=SimpleNamespace(),
        scope=McpReadScope(mode='metadata_only'),
        limit=500,
    )

    assert calls[0] == 50
    assert calls[1] == 200
    assert calls[2] == 200


@pytest.mark.asyncio
async def test_tool_index_status_excludes_upload_local_counts() -> None:
    class _FakeCursor:
        def __init__(self, count: int) -> None:
            self._count = count

        async def fetchone(self):
            return {'count': self._count}

    class _FakeDb:
        def __init__(self) -> None:
            self.queries: list[str] = []

        async def execute(self, sql: str, _params: tuple[str, ...]):
            self.queries.append(sql)
            if 'FROM files' in sql:
                return _FakeCursor(115)
            return _FakeCursor(37756)

    db = _FakeDb()
    payload = await tool_index_status(db)  # type: ignore[arg-type]

    assert payload['total_files'] == 115
    assert payload['total_chunks'] == 37756
    assert all("source_provider != ?" in query for query in db.queries)
