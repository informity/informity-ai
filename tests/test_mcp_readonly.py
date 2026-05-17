from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from informity.mcp.authorization import McpAuthorizationError, authorize_mcp_request
from informity.mcp.protocol import handle_jsonrpc_request
from informity.mcp.server import InformityMcpReadOnlyServer, mcp_readonly_server
from informity.mcp.tools_readonly import (
    McpReadScope,
    _coerce_response_size,
    tool_files_list,
    tool_filter_options,
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
async def test_mcp_tools_call_rejects_non_allowlisted_tool_before_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from informity.mcp import protocol as mod

    class _ExplodingServer:
        async def execute_tool(self, **_kwargs):
            raise AssertionError('dispatch should not be reached for non-allowlisted tools')

    monkeypatch.setattr(mod, '_mcp_readonly_server', _ExplodingServer())
    monkeypatch.setattr(mod, '_mcp_tool_not_found_error', KeyError)

    response = await handle_jsonrpc_request(
        {
            'jsonrpc': '2.0',
            'id': 21,
            'method': 'tools/call',
            'params': {'name': 'informity_delete_file', 'arguments': {'file_id': 1}},
        },
        transport='stdio',
        bearer_token=None,
    )
    assert response is not None
    assert response['error']['code'] == -32601
    assert response['error']['message'] == 'Unknown tool: informity_delete_file'


@pytest.mark.asyncio
async def test_mcp_tools_call_dispatches_legacy_alias_for_readonly_tool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from informity.mcp import protocol as mod

    captured_tool_names: list[str] = []

    class _CaptureServer:
        async def execute_tool(self, **kwargs):
            captured_tool_names.append(str(kwargs.get('tool_name', '')))
            return {'ok': True}

    monkeypatch.setattr(mod, '_mcp_readonly_server', _CaptureServer())
    monkeypatch.setattr(mod, '_mcp_tool_not_found_error', KeyError)

    response = await handle_jsonrpc_request(
        {
            'jsonrpc': '2.0',
            'id': 22,
            'method': 'tools/call',
            'params': {'name': 'informity.search.semantic', 'arguments': {'query': 'test'}},
        },
        transport='stdio',
        bearer_token=None,
    )
    assert response is not None
    assert response['result']['isError'] is False
    assert captured_tool_names == ['informity_search_semantic']


@pytest.mark.asyncio
async def test_mcp_server_uses_readonly_sqlite_connection(monkeypatch: pytest.MonkeyPatch) -> None:
    from informity.mcp import server as mod

    class _FakeConn:
        def __init__(self) -> None:
            self.row_factory = None
            self.pragmas: list[str] = []

        async def execute(self, sql: str):
            self.pragmas.append(sql)

    captured: dict[str, object] = {}

    async def _fake_connect(path: str, *, uri: bool):
        captured['path'] = path
        captured['uri'] = uri
        return _FakeConn()

    monkeypatch.setattr(mod.aiosqlite, 'connect', _fake_connect)

    server = InformityMcpReadOnlyServer()
    conn = await server._get_readonly_connection()
    assert captured['uri'] is True
    assert 'mode=ro' in str(captured['path'])
    assert conn.pragmas == [
        'PRAGMA query_only=ON',
        'PRAGMA foreign_keys=ON',
        'PRAGMA busy_timeout=5000',
    ]


@pytest.mark.asyncio
async def test_tool_search_semantic_normalizes_category_and_file_types(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from informity.mcp import tools_readonly as mod

    monkeypatch.setattr(mod.embedder, 'embed_query', lambda _q: [0.1, 0.2, 0.3])
    monkeypatch.setattr(
        mod.vector_store,
        'search_similar',
        lambda *_args, **_kwargs: [{'file_id': 8, 'chunk_text': 'contract term', 'score': 0.05}],
    )

    fake_file = SimpleNamespace(
        id=8,
        filename='Contract.PDF',
        path='/docs/Contract.PDF',
        category=SimpleNamespace(value='document'),
        extension='.pdf',
        source_provider='filesystem',
        content_hash='hash-8',
    )

    async def _fake_get_files_by_ids(*_args, **_kwargs):
        return {8: fake_file}

    monkeypatch.setattr(mod, 'get_files_by_ids', _fake_get_files_by_ids)

    payload = await tool_search_semantic(
        db=SimpleNamespace(),
        scope=McpReadScope(mode='search_snippets'),
        query='contract',
        limit=5,
        category='Document',
        file_types=['PDF'],
    )
    assert payload['total'] == 1
    assert payload['results'][0]['file_id'] == 8


@pytest.mark.asyncio
async def test_tool_search_semantic_no_results_returns_hints_for_filtered_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from informity.mcp import tools_readonly as mod

    class _FakeCursor:
        def __init__(self, rows):
            self._rows = rows

        async def fetchall(self):
            return self._rows

    class _FakeDb:
        async def execute(self, sql: str, _params):
            if 'category' in sql:
                return _FakeCursor([{'category': 'document'}])
            return _FakeCursor([{'extension': '.pdf'}, {'extension': '.docx'}])

    monkeypatch.setattr(mod.embedder, 'embed_query', lambda _q: [0.1, 0.2, 0.3])
    monkeypatch.setattr(mod.vector_store, 'search_similar', lambda *_args, **_kwargs: [])
    async def _fake_get_files_by_ids(*_args, **_kwargs):
        return {}

    monkeypatch.setattr(mod, 'get_files_by_ids', _fake_get_files_by_ids)

    payload = await tool_search_semantic(
        db=_FakeDb(),
        scope=McpReadScope(mode='search_snippets'),
        query='contract',
        limit=5,
        category='Document',
        file_types=['pdf'],
    )
    assert payload['total'] == 0
    assert 'hints' in payload
    assert payload['hints']['applied_filters']['category'] == 'document'
    assert payload['hints']['applied_filters']['file_types'] == ['.pdf']
    assert payload['hints']['valid_categories'] == ['document']
    assert payload['hints']['valid_file_types'] == ['.pdf', '.docx']
    assert payload['hints']['unknown_filters']['unknown_category'] is False
    assert payload['hints']['unknown_filters']['unknown_file_types'] == []


@pytest.mark.asyncio
async def test_tool_search_semantic_hints_include_unknown_filter_details(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from informity.mcp import tools_readonly as mod

    class _FakeCursor:
        def __init__(self, rows):
            self._rows = rows

        async def fetchall(self):
            return self._rows

    class _FakeDb:
        async def execute(self, sql: str, _params):
            if 'category' in sql:
                return _FakeCursor([{'category': 'document'}])
            return _FakeCursor([{'extension': '.pdf'}])

    monkeypatch.setattr(mod.embedder, 'embed_query', lambda _q: [0.1, 0.2, 0.3])
    monkeypatch.setattr(mod.vector_store, 'search_similar', lambda *_args, **_kwargs: [])

    async def _fake_get_files_by_ids(*_args, **_kwargs):
        return {}

    monkeypatch.setattr(mod, 'get_files_by_ids', _fake_get_files_by_ids)

    payload = await tool_search_semantic(
        db=_FakeDb(),
        scope=McpReadScope(mode='search_snippets'),
        query='contract',
        limit=5,
        category='other',
        file_types=['zzz'],
    )
    assert payload['total'] == 0
    assert payload['hints']['unknown_filters']['unknown_category'] is True
    assert payload['hints']['unknown_filters']['unknown_file_types'] == ['.zzz']


def test_coerce_response_size_trims_results_list() -> None:
    payload = {
        'results': [
            {'file_id': 1, 'preview': 'a' * 120},
            {'file_id': 2, 'preview': 'b' * 120},
            {'file_id': 3, 'preview': 'c' * 120},
        ],
        'total': 3,
    }
    coerced = _coerce_response_size(payload, max_bytes=220)
    assert coerced['truncated'] is True
    assert coerced['returned'] < 3
    assert coerced['total_before_truncation'] == 3
    assert len(str(coerced).encode('utf-8', errors='ignore')) <= 220


@pytest.mark.asyncio
async def test_tool_filter_options_returns_distinct_categories_and_file_types() -> None:
    class _FakeCursor:
        def __init__(self, rows):
            self._rows = rows

        async def fetchall(self):
            return self._rows

    class _FakeDb:
        async def execute(self, sql: str, _params):
            if 'category' in sql:
                return _FakeCursor(
                    [
                        {'category': 'document'},
                        {'category': 'web'},
                    ]
                )
            return _FakeCursor(
                [
                    {'extension': '.docx'},
                    {'extension': '.pdf'},
                ]
            )

    payload = await tool_filter_options(_FakeDb())
    assert payload['categories'] == ['document', 'web']
    assert payload['file_types'] == ['.docx', '.pdf']


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
