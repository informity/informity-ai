from __future__ import annotations

import asyncio
import json
import os
import sys
from collections.abc import Mapping
from typing import Any

JSON = dict[str, Any]
SERVER_NAME = 'informity-mcp'


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
    if ':' not in decoded:
        return None
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


async def _handle_request(payload: JSON) -> JSON | None:
    from informity.mcp.protocol import error_response, handle_jsonrpc_request

    try:
        return await handle_jsonrpc_request(payload, transport='stdio', bearer_token=None)
    except Exception:
        request_id = payload.get('id')
        return error_response(request_id, -32603, 'Internal MCP protocol error')


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
