from __future__ import annotations

import secrets

MCP_TOKEN_PREFIX = 'imcp_'
MCP_TOKEN_RANDOM_LENGTH = 32
MCP_TOKEN_ALPHABET = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_'


def generate_mcp_access_token() -> str:
    return MCP_TOKEN_PREFIX + ''.join(secrets.choice(MCP_TOKEN_ALPHABET) for _ in range(MCP_TOKEN_RANDOM_LENGTH))
