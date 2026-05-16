from __future__ import annotations

import asyncio

import structlog

from informity.api.security import is_loopback_host
from informity.config import settings

log = structlog.get_logger(__name__)


class McpLifecycleManager:
    """
    Phase-1 MCP lifecycle scaffold.

    This intentionally does not expose tool handlers yet; it provides deterministic
    start/stop/restart semantics so settings wiring and app lifecycle integration
    can be safely implemented before transport/tool expansion.
    """

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._running = False
        self._last_error: str | None = None

    @property
    def running(self) -> bool:
        return self._running

    @property
    def last_error(self) -> str | None:
        return self._last_error

    async def start_from_settings(self) -> None:
        async with self._lock:
            if self._running:
                return
            if not settings.mcp_enabled:
                self._last_error = None
                return
            if settings.mcp_transport == 'http' and not is_loopback_host(settings.mcp_http_host):
                self._last_error = (
                    'MCP HTTP host must be loopback (127.0.0.1/localhost/::1). '
                    f'Configured: {settings.mcp_http_host}'
                )
                log.warning('mcp_start_denied_non_loopback_host', host=settings.mcp_http_host)
                return

            # Phase 1 behavior: lifecycle state only (no transport runtime yet).
            self._running = True
            self._last_error = None
            log.info(
                'mcp_lifecycle_started',
                transport=settings.mcp_transport,
                host=settings.mcp_http_host,
                port=settings.mcp_http_port,
                scope_mode=settings.mcp_scope_mode,
            )

    async def stop(self) -> None:
        async with self._lock:
            if not self._running:
                return
            self._running = False
            log.info('mcp_lifecycle_stopped')

    async def restart_from_settings(self) -> None:
        async with self._lock:
            was_running = self._running
        if was_running:
            await self.stop()
        await self.start_from_settings()

    def snapshot(self) -> dict[str, str | None]:
        state = 'running' if self._running else 'disabled'
        if self._last_error:
            state = 'error'
        return {
            'state': state,
            'error': self._last_error,
        }


mcp_lifecycle = McpLifecycleManager()
