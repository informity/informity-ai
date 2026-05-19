from __future__ import annotations

import asyncio
import socket
from contextlib import suppress

import structlog
import uvicorn

from informity.api.security import is_loopback_host
from informity.config import settings
from informity.log_events import emit_log_event
from informity.mcp.http_server import create_http_app

log = structlog.get_logger(__name__)


class McpLifecycleManager:
    """
    MCP lifecycle manager for transport runtime state.

    Owns deterministic start/stop/restart semantics for configured MCP transport.
    """

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._running = False
        self._last_error: str | None = None
        self._http_server: uvicorn.Server | None = None
        self._http_task: asyncio.Task[None] | None = None
        self._http_host: str | None = None
        self._http_port: int | None = None

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
                await emit_log_event(
                    event_name='mcp_scope_denied',
                    source='MCP Server',
                    message='MCP server start denied because host is not loopback.',
                    details={'host': settings.mcp_http_host},
                    dedupe_bucket_seconds=300,
                )
                return

            if settings.mcp_transport == 'http':
                try:
                    await self._start_http_server_locked()
                except OSError as exc:
                    self._running = False
                    self._last_error = f'Failed to start MCP HTTP server: {exc}'
                    log.error(
                        'mcp_http_start_failed',
                        host=settings.mcp_http_host,
                        port=settings.mcp_http_port,
                        error=str(exc),
                    )
                    await emit_log_event(
                        event_name='mcp_server_failed',
                        source='MCP Server',
                        message='MCP server failed to start.',
                        details={'error': str(exc), 'host': settings.mcp_http_host, 'port': settings.mcp_http_port},
                        dedupe_bucket_seconds=120,
                    )
                    return

            self._running = True
            self._last_error = None
            log.info(
                'mcp_lifecycle_started',
                transport=settings.mcp_transport,
                host=settings.mcp_http_host,
                port=settings.mcp_http_port,
                scope_mode=settings.mcp_scope_mode,
            )
            await emit_log_event(
                event_name='mcp_server_started',
                source='MCP Server',
                message='MCP server started successfully.',
                details={
                    'transport': settings.mcp_transport,
                    'host': settings.mcp_http_host,
                    'port': settings.mcp_http_port,
                    'scope_mode': settings.mcp_scope_mode,
                },
            )

    async def stop(self) -> None:
        async with self._lock:
            if not self._running:
                return
            await self._stop_http_server_locked()
            self._running = False
            log.info('mcp_lifecycle_stopped')
            await emit_log_event(
                event_name='mcp_server_stopped',
                source='MCP Server',
                message='MCP server stopped.',
            )

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

    async def _start_http_server_locked(self) -> None:
        self._ensure_http_bindable(settings.mcp_http_host, int(settings.mcp_http_port))
        app = create_http_app()
        config = uvicorn.Config(
            app=app,
            host=settings.mcp_http_host,
            port=int(settings.mcp_http_port),
            log_level='warning',
            loop='asyncio',
            lifespan='off',
            access_log=False,
        )
        server = uvicorn.Server(config)
        task = asyncio.create_task(server.serve(), name='informity-mcp-http-server')
        await self._wait_for_http_startup(server, task)
        self._http_server = server
        self._http_task = task
        self._http_host = settings.mcp_http_host
        self._http_port = int(settings.mcp_http_port)

    def _ensure_http_bindable(self, host: str, port: int) -> None:
        infos = socket.getaddrinfo(host, port, socket.AF_UNSPEC, socket.SOCK_STREAM)
        last_error: OSError | None = None
        for family, socktype, proto, _canon, sockaddr in infos:
            sock = socket.socket(family, socktype, proto)
            try:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind(sockaddr)
                return
            except OSError as exc:
                last_error = exc
            finally:
                sock.close()
        if last_error is not None:
            raise OSError(f'Port bind check failed for {host}:{port}: {last_error}') from last_error

    async def _wait_for_http_startup(self, server: uvicorn.Server, task: asyncio.Task[None]) -> None:
        async def _poll() -> None:
            while not server.started:
                if task.done():
                    break
                await asyncio.sleep(0.02)

        await asyncio.wait_for(_poll(), timeout=5.0)
        if task.done() and not server.started:
            try:
                task.result()
            except BaseException as exc:  # noqa: BLE001
                raise OSError(f'MCP HTTP server startup failed: {exc}') from exc
        if not server.started:
            raise OSError('MCP HTTP server did not reach started state')

    async def _stop_http_server_locked(self) -> None:
        server = self._http_server
        task = self._http_task
        self._http_server = None
        self._http_task = None
        self._http_host = None
        self._http_port = None
        if server is not None:
            server.should_exit = True
        if task is not None:
            with suppress(asyncio.TimeoutError):
                await asyncio.wait_for(task, timeout=5.0)


mcp_lifecycle = McpLifecycleManager()
