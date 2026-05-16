from __future__ import annotations

import socket

import pytest

from informity import config
from informity.mcp.lifecycle import McpLifecycleManager


def _get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(('127.0.0.1', 0))
        return int(sock.getsockname()[1])


def _can_connect(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        return sock.connect_ex((host, port)) == 0


@pytest.mark.asyncio
async def test_http_lifecycle_binds_and_unbinds_port() -> None:
    manager = McpLifecycleManager()
    host = '127.0.0.1'
    port = _get_free_port()

    original = (
        config.settings.mcp_enabled,
        config.settings.mcp_transport,
        config.settings.mcp_http_host,
        config.settings.mcp_http_port,
    )
    try:
        config.settings.mcp_enabled = True
        config.settings.mcp_transport = 'http'
        config.settings.mcp_http_host = host
        config.settings.mcp_http_port = port

        await manager.start_from_settings()
        assert manager.running is True
        assert manager.last_error is None
        assert _can_connect(host, port) is True

        await manager.stop()
        assert manager.running is False
        assert _can_connect(host, port) is False
    finally:
        config.settings.mcp_enabled, config.settings.mcp_transport, config.settings.mcp_http_host, config.settings.mcp_http_port = original
        await manager.stop()


@pytest.mark.asyncio
async def test_http_lifecycle_reports_error_on_port_contention() -> None:
    manager = McpLifecycleManager()
    host = '127.0.0.1'
    port = _get_free_port()

    blocker = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    blocker.bind((host, port))
    blocker.listen(1)

    original = (
        config.settings.mcp_enabled,
        config.settings.mcp_transport,
        config.settings.mcp_http_host,
        config.settings.mcp_http_port,
    )
    try:
        config.settings.mcp_enabled = True
        config.settings.mcp_transport = 'http'
        config.settings.mcp_http_host = host
        config.settings.mcp_http_port = port

        await manager.start_from_settings()
        assert manager.running is False
        assert manager.last_error is not None
        assert 'Failed to start MCP HTTP server' in manager.last_error
    finally:
        blocker.close()
        config.settings.mcp_enabled, config.settings.mcp_transport, config.settings.mcp_http_host, config.settings.mcp_http_port = original
        await manager.stop()

