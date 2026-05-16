from __future__ import annotations

import asyncio
import json
import os
import sys
from collections.abc import Mapping
from typing import Any

import structlog

from informity.mcp.tool_registry import TOOLS

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


def _read_message(stdin: Any) -> JSON | None:
    line = stdin.readline()
    if not line:
        return None

    # Primary MCP stdio mode: newline-delimited JSON-RPC.
    try:
        decoded = line.decode('utf-8').strip()
    except UnicodeDecodeError:
        return None
    if decoded.startswith('{'):
        try:
            payload = json.loads(decoded)
        except json.JSONDecodeError:
            return None
        return payload if isinstance(payload, dict) else None

    # Compatibility fallback: Content-Length framed payload.
    headers: dict[str, str] = {}
    if ':' in decoded:
        name, value = decoded.split(':', 1)
        headers[name.strip().lower()] = value.strip()
    while True:
        line = stdin.readline()
        if not line:
            return None
        if line in (b'\r\n', b'\n'):
            break
        try:
            text = line.decode('utf-8').strip()
        except UnicodeDecodeError:
            continue
        if ':' not in text:
            continue
        name, value = text.split(':', 1)
        headers[name.strip().lower()] = value.strip()

    length_raw = headers.get('content-length')
    if not length_raw:
        return None
    try:
        length = int(length_raw)
    except ValueError:
        return None
    if length <= 0:
        return None

    body = stdin.read(length)
    if not body:
        return None
    try:
        payload = json.loads(body.decode('utf-8'))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _write_message(stdout: Any, payload: Mapping[str, Any]) -> None:
    body = json.dumps(payload, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
    stdout.write(body + b'\n')
    stdout.flush()


def _error_response(request_id: Any, code: int, message: str) -> JSON:
    return {
        'jsonrpc': '2.0',
        'id': request_id,
        'error': {
            'code': code,
            'message': message,
        },
    }


async def _handle_request(payload: JSON) -> JSON | None:
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
        arguments = params_obj.get('arguments')
        args_obj = arguments if isinstance(arguments, dict) else {}
        if not tool_name:
            return _error_response(request_id, -32602, 'Missing tool name')
        try:
            result = await tool_server.execute_tool(
                tool_name=tool_name,
                args=args_obj,
                transport='stdio',
                bearer_token=None,
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
            return _error_response(request_id, -32601, f'Unknown tool: {tool_name}')
        except ValueError as exc:
            return _error_response(request_id, -32602, str(exc))
        except Exception as exc:  # pragma: no cover - defensive server boundary
            log.exception('mcp_stdio_tool_call_failed', tool_name=tool_name, error=str(exc))
            return _error_response(request_id, -32603, 'Internal MCP tool error')

    return _error_response(request_id, -32601, f'Method not found: {method}')


def main() -> None:
    # Keep a dedicated binary handle to original stdout for MCP frames only.
    protocol_stdout = os.fdopen(os.dup(1), 'wb', closefd=True)
    # Redirect process stdout to stderr so incidental logs/prints cannot corrupt MCP framing.
    os.dup2(2, 1)

    stdin = sys.stdin.buffer
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        while True:
            message = _read_message(stdin)
            if message is None:
                break
            response = loop.run_until_complete(_handle_request(message))
            if response is not None:
                _write_message(protocol_stdout, response)
    finally:
        protocol_stdout.close()
        loop.close()


if __name__ == '__main__':
    main()
