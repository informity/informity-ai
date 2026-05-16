from __future__ import annotations

from collections.abc import Mapping

import structlog
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from informity.mcp.protocol import error_response, handle_jsonrpc_request

log = structlog.get_logger(__name__)


def _extract_bearer_token(request: Request) -> str | None:
    auth_header = str(request.headers.get('authorization') or '').strip()
    if not auth_header:
        return None
    if not auth_header.lower().startswith('bearer '):
        return None
    token = auth_header[7:].strip()
    return token or None


def create_http_app() -> FastAPI:
    app = FastAPI(
        title='Informity MCP (HTTP)',
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )

    @app.post('/')
    @app.post('/mcp')
    async def mcp_rpc(request: Request) -> JSONResponse:
        try:
            payload = await request.json()
        except Exception:
            return JSONResponse(error_response(None, -32700, 'Parse error'))
        if not isinstance(payload, Mapping):
            return JSONResponse(error_response(None, -32600, 'Invalid request'))

        bearer_token = _extract_bearer_token(request)
        response_payload = await handle_jsonrpc_request(
            dict(payload),
            transport='http',
            bearer_token=bearer_token,
        )
        if response_payload is None:
            return JSONResponse(status_code=204, content=None)
        return JSONResponse(response_payload)

    return app
