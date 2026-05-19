from __future__ import annotations

import os
import secrets

from informity.config import settings
from informity.log_events import emit_log_event


class McpAuthorizationError(PermissionError):
    """Raised when an MCP request fails authorization."""


def authorize_mcp_request(*, transport: str, bearer_token: str | None) -> None:
    """
    Authorize an MCP request by transport and current auth mode.

    Phase-2 policy:
    - stdio transport is trusted as local process mediation.
    - http transport requires a bearer token that matches INFORMITY_MCP_TOKEN.
    """
    normalized_transport = str(transport or '').strip().lower()
    if normalized_transport == 'stdio':
        return

    if normalized_transport != 'http':
        try:
            import asyncio
            loop = asyncio.get_running_loop()
            loop.create_task(
                emit_log_event(
                    event_name='mcp_policy_violation',
                    source='MCP Auth',
                    message='MCP request used unsupported transport.',
                    details={'transport': normalized_transport},
                    dedupe_bucket_seconds=120,
                )
            )
        except RuntimeError:
            pass
        raise McpAuthorizationError(f'Unsupported MCP transport: {transport}')

    if settings.mcp_auth_mode != 'token_required':
        try:
            import asyncio
            loop = asyncio.get_running_loop()
            loop.create_task(
                emit_log_event(
                    event_name='mcp_policy_violation',
                    source='MCP Auth',
                    message='MCP auth mode is unsupported.',
                    details={'auth_mode': getattr(settings, 'mcp_auth_mode', None)},
                    dedupe_bucket_seconds=120,
                )
            )
        except RuntimeError:
            pass
        raise McpAuthorizationError('Unsupported MCP auth mode configuration')

    expected = str(os.environ.get('INFORMITY_MCP_TOKEN') or '').strip()
    if not expected:
        expected = str(getattr(settings, 'mcp_access_token', '') or '').strip()
    provided = str(bearer_token or '').strip()
    if not expected:
        try:
            import asyncio
            loop = asyncio.get_running_loop()
            loop.create_task(
                emit_log_event(
                    event_name='mcp_auth_failed',
                    source='MCP Auth',
                    message='MCP token is required but not configured.',
                    dedupe_bucket_seconds=120,
                )
            )
        except RuntimeError:
            pass
        raise McpAuthorizationError(
            'MCP HTTP authorization is enabled but INFORMITY_MCP_TOKEN is not configured'
        )
    if not provided or not secrets.compare_digest(provided, expected):
        try:
            import asyncio
            loop = asyncio.get_running_loop()
            loop.create_task(
                emit_log_event(
                    event_name='mcp_auth_failed',
                    source='MCP Auth',
                    message='MCP bearer token is missing or invalid.',
                    dedupe_bucket_seconds=60,
                )
            )
        except RuntimeError:
            pass
        raise McpAuthorizationError('Missing or invalid MCP bearer token')
