from __future__ import annotations

import json
import time
from collections import deque
from collections.abc import Mapping

import structlog
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from informity.config import settings
from informity.mcp.protocol import error_response, handle_jsonrpc_request

log = structlog.get_logger(__name__)
_RATE_LIMIT_WINDOW_SECONDS = 60.0
_RATE_LIMIT_MAX_REQUESTS = 120
_rate_limit_buckets: dict[str, deque[float]] = {}


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
        client_host = str((request.client.host if request.client else '') or 'unknown')
        now = time.monotonic()
        bucket = _rate_limit_buckets.get(client_host)
        if bucket is None:
            bucket = deque()
            _rate_limit_buckets[client_host] = bucket
        cutoff = now - _RATE_LIMIT_WINDOW_SECONDS
        while bucket and bucket[0] < cutoff:
            bucket.popleft()
        if len(bucket) >= _RATE_LIMIT_MAX_REQUESTS:
            log.warning('mcp_http_rate_limited', client_host=client_host, window_seconds=_RATE_LIMIT_WINDOW_SECONDS)
            return JSONResponse(
                status_code=429,
                content=error_response(None, -32001, 'Rate limit exceeded'),
            )
        bucket.append(now)

        max_body_bytes = max(16_384, int(getattr(settings, 'mcp_http_max_body_bytes', 512 * 1024) or (512 * 1024)))
        content_length = str(request.headers.get('content-length') or '').strip()
        if content_length:
            try:
                if int(content_length) > max_body_bytes:
                    return JSONResponse(
                        status_code=413,
                        content=error_response(None, -32600, 'Request body too large'),
                    )
            except ValueError:
                pass

        try:
            raw_body = await request.body()
        except Exception:
            return JSONResponse(error_response(None, -32700, 'Parse error'))
        if len(raw_body) > max_body_bytes:
            return JSONResponse(
                status_code=413,
                content=error_response(None, -32600, 'Request body too large'),
            )
        try:
            payload = json.loads(raw_body)
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
