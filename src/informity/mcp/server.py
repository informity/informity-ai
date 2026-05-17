from __future__ import annotations

from typing import Any

from informity.config import settings
from informity.db.sqlite import get_connection
from informity.mcp.authorization import authorize_mcp_request
from informity.mcp.tool_registry import (
    LEGACY_TOOL_ALIASES,
    TOOL_FILES_LIST,
    TOOL_HEALTH,
    TOOL_INDEX_STATUS,
    TOOL_SCAN_STATUS,
    TOOL_SEARCH_SEMANTIC,
)
from informity.mcp.tools_readonly import (
    McpReadScope,
    tool_files_list,
    tool_health,
    tool_index_status,
    tool_scan_status,
    tool_search_semantic,
)


class McpToolNotFoundError(KeyError):
    """Raised when an unknown MCP tool name is requested."""


class InformityMcpReadOnlyServer:
    """
    Read-only MCP tool dispatcher.

    Phase-2 implementation intentionally exposes only read-only tools.
    """

    async def execute_tool(
        self,
        *,
        tool_name: str,
        args: dict[str, Any] | None = None,
        transport: str = 'stdio',
        bearer_token: str | None = None,
        skip_authorization: bool = False,
    ) -> dict[str, Any]:
        if not skip_authorization:
            authorize_mcp_request(transport=transport, bearer_token=bearer_token)
        payload = args or {}
        normalized_tool_name = LEGACY_TOOL_ALIASES.get(tool_name, tool_name)
        scope = McpReadScope(mode=settings.mcp_scope_mode).normalize()

        if normalized_tool_name == TOOL_HEALTH:
            return await tool_health()

        db = await get_connection()
        try:
            if normalized_tool_name == TOOL_FILES_LIST:
                return await tool_files_list(
                    db,
                    scope=scope,
                    limit=int(payload.get('limit', 50)),
                    offset=int(payload.get('offset', 0)),
                    search=str(payload.get('search')) if payload.get('search') is not None else None,
                )
            if normalized_tool_name == TOOL_SEARCH_SEMANTIC:
                file_types = payload.get('file_types')
                return await tool_search_semantic(
                    db,
                    scope=scope,
                    query=str(payload.get('query') or ''),
                    limit=int(payload.get('limit', 50)),
                    category=str(payload.get('category')) if payload.get('category') is not None else None,
                    file_types=list(file_types) if isinstance(file_types, list) else None,
                )
            if normalized_tool_name == TOOL_INDEX_STATUS:
                return await tool_index_status(db)
            if normalized_tool_name == TOOL_SCAN_STATUS:
                return await tool_scan_status(db)
        finally:
            await db.close()

        raise McpToolNotFoundError(f'Unknown MCP tool: {tool_name}')


mcp_readonly_server = InformityMcpReadOnlyServer()
