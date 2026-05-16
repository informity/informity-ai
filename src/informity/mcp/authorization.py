from __future__ import annotations

import os
import secrets

from informity.config import settings


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
        raise McpAuthorizationError(f'Unsupported MCP transport: {transport}')

    if settings.mcp_auth_mode != 'token_required':
        raise McpAuthorizationError('Unsupported MCP auth mode configuration')

    expected = str(os.environ.get('INFORMITY_MCP_TOKEN') or '').strip()
    if not expected:
        expected = str(getattr(settings, 'mcp_access_token', '') or '').strip()
    provided = str(bearer_token or '').strip()
    if not expected:
        raise McpAuthorizationError(
            'MCP HTTP authorization is enabled but INFORMITY_MCP_TOKEN is not configured'
        )
    if not provided or not secrets.compare_digest(provided, expected):
        raise McpAuthorizationError('Missing or invalid MCP bearer token')
