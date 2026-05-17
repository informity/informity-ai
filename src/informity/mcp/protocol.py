from __future__ import annotations

import asyncio
import json
from typing import Any

import structlog

from informity.config import settings
from informity.mcp.authorization import McpAuthorizationError, authorize_mcp_request
from informity.mcp.tool_registry import LEGACY_TOOL_ALIASES, READONLY_TOOL_NAMES, TOOLS

log = structlog.get_logger(__name__)

JSON = dict[str, Any]

SERVER_NAME = 'informity-mcp'
SERVER_VERSION = '0.1.0'
DEFAULT_PROTOCOL_VERSION = '2025-11-25'

_mcp_readonly_server: Any | None = None
_mcp_tool_not_found_error: type[Exception] | None = None


def _get_tool_dispatch() -> tuple[Any, type[Exception]]:
    global _mcp_readonly_server, _mcp_tool_not_found_error
    if _mcp_readonly_server is None or _mcp_tool_not_found_error is None:
        from informity.mcp.server import McpToolNotFoundError, mcp_readonly_server

        _mcp_readonly_server = mcp_readonly_server
        _mcp_tool_not_found_error = McpToolNotFoundError
    return _mcp_readonly_server, _mcp_tool_not_found_error


def error_response(request_id: Any, code: int, message: str) -> JSON:
    return {
        'jsonrpc': '2.0',
        'id': request_id,
        'error': {
            'code': code,
            'message': message,
        },
    }


async def handle_jsonrpc_request(
    payload: JSON,
    *,
    transport: str,
    bearer_token: str | None,
) -> JSON | None:
    request_id = payload.get('id')
    method = str(payload.get('method') or '').strip()
    params = payload.get('params')
    if params is None:
        params_obj: JSON = {}
    elif isinstance(params, dict):
        params_obj = params
    else:
        params_obj = {}

    if request_id is None and method.startswith('notifications/'):
        return None
    if method == 'notifications/initialized':
        return None
    if transport == 'http':
        try:
            authorize_mcp_request(transport='http', bearer_token=bearer_token)
        except McpAuthorizationError as exc:
            return error_response(request_id, -32001, str(exc))

    if method == 'initialize':
        requested_protocol = str(params_obj.get('protocolVersion') or '').strip()
        negotiated_protocol = requested_protocol or DEFAULT_PROTOCOL_VERSION
        return {
            'jsonrpc': '2.0',
            'id': request_id,
            'result': {
                'protocolVersion': negotiated_protocol,
                'serverInfo': {
                    'name': SERVER_NAME,
                    'version': SERVER_VERSION,
                },
                'capabilities': {
                    'tools': {'listChanged': False},
                },
            },
        }

    if method == 'ping':
        return {
            'jsonrpc': '2.0',
            'id': request_id,
            'result': {},
        }

    if method == 'tools/list':
        return {
            'jsonrpc': '2.0',
            'id': request_id,
            'result': {
                'tools': TOOLS,
            },
        }

    if method == 'tools/call':
        tool_server, tool_not_found_error = _get_tool_dispatch()
        tool_name = str(params_obj.get('name') or '').strip()
        normalized_tool_name = LEGACY_TOOL_ALIASES.get(tool_name, tool_name)
        arguments = params_obj.get('arguments')
        args_obj = arguments if isinstance(arguments, dict) else {}
        if not tool_name:
            return error_response(request_id, -32602, 'Missing tool name')
        if normalized_tool_name not in READONLY_TOOL_NAMES:
            return error_response(request_id, -32601, f'Unknown tool: {tool_name}')
        try:
            timeout_seconds = max(5.0, float(getattr(settings, 'mcp_tool_call_timeout_seconds', 30.0) or 30.0))
            result = await asyncio.wait_for(
                tool_server.execute_tool(
                    tool_name=normalized_tool_name,
                    args=args_obj,
                    transport=transport,
                    bearer_token=bearer_token,
                    skip_authorization=(transport == 'http'),
                ),
                timeout=timeout_seconds,
            )
            return {
                'jsonrpc': '2.0',
                'id': request_id,
                'result': {
                    'content': [
                        {
                            'type': 'text',
                            'text': json.dumps(result, ensure_ascii=False),
                        },
                    ],
                    'isError': False,
                },
            }
        except tool_not_found_error:
            return error_response(request_id, -32601, f'Unknown tool: {tool_name}')
        except McpAuthorizationError as exc:
            return error_response(request_id, -32001, str(exc))
        except TimeoutError:
            return error_response(request_id, -32001, 'MCP tool call timed out')
        except ValueError as exc:
            return error_response(request_id, -32602, str(exc))
        except Exception as exc:  # pragma: no cover - defensive server boundary
            log.exception('mcp_tool_call_failed', tool_name=tool_name, transport=transport, error=str(exc))
            return error_response(request_id, -32603, 'Internal MCP tool error')

    return error_response(request_id, -32601, f'Method not found: {method}')
