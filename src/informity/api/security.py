# ==============================================================================
# Informity AI — API Security Guards
# In-process rate limiting and concurrency caps for expensive endpoints.
# ==============================================================================

import asyncio
import os
import secrets
import time
from collections import deque
from collections.abc import AsyncGenerator, Mapping
from contextlib import asynccontextmanager

from fastapi import HTTPException

TAURI_SESSION_HEADER = 'X-Informity-Session'
_TAURI_DEV_ORIGINS: tuple[str, ...] = (
    'http://127.0.0.1:5173',
    'http://localhost:5173',
)
_TAURI_ORIGIN = 'tauri://localhost'


def get_tauri_session_token_from_env(env: Mapping[str, str] | None = None) -> str | None:
    # Read optional per-launch desktop session token injected by the Tauri host process.
    values = env if env is not None else os.environ
    token = (values.get('INFORMITY_TAURI_SESSION_TOKEN') or '').strip()
    return token or None


def is_tauri_desktop_mode(session_token: str | None) -> bool:
    return bool(session_token)


def is_tauri_session_authorized(
    headers: Mapping[str, str],
    expected_token: str | None,
) -> bool:
    if not expected_token:
        return True
    request_token = (headers.get(TAURI_SESSION_HEADER) or '').strip()
    if not request_token:
        return False
    return secrets.compare_digest(request_token, expected_token)


def get_cors_allow_origins(port: int, *, desktop_mode: bool) -> list[str]:
    # In desktop session mode we only allow Tauri and local Vite dev origins.
    if desktop_mode:
        return [*_TAURI_DEV_ORIGINS, _TAURI_ORIGIN]

    # Default web/dev server origins.
    return [
        f'http://localhost:{port}',
        f'http://127.0.0.1:{port}',
        'http://localhost:3000',
        *_TAURI_DEV_ORIGINS,
        _TAURI_ORIGIN,
    ]


class EndpointGuard:
    # Simple in-memory guard for rate and in-flight request limits.
    def __init__(
        self,
        *,
        name: str,
        max_in_flight: int,
        max_requests_per_window: int,
        window_seconds: int,
    ) -> None:
        self.name = name
        self.max_in_flight = max_in_flight
        self.max_requests_per_window = max_requests_per_window
        self.window_seconds = window_seconds
        self._lock = asyncio.Lock()
        self._in_flight = 0
        self._request_timestamps: deque[float] = deque()

    def _prune_window(self, now: float) -> None:
        cutoff = now - float(self.window_seconds)
        while self._request_timestamps and self._request_timestamps[0] < cutoff:
            self._request_timestamps.popleft()

    async def check_rate_limit(self) -> None:
        now = time.monotonic()
        async with self._lock:
            self._prune_window(now)
            if len(self._request_timestamps) >= self.max_requests_per_window:
                raise HTTPException(
                    status_code=429,
                    detail=f'Rate limit exceeded for {self.name}. Please retry shortly.',
                )
            self._request_timestamps.append(now)

    async def _acquire_slot(self) -> None:
        async with self._lock:
            if self._in_flight >= self.max_in_flight:
                raise HTTPException(
                    status_code=429,
                    detail=f'Too many concurrent {self.name} requests. Please retry shortly.',
                )
            self._in_flight += 1

    async def _release_slot(self) -> None:
        async with self._lock:
            if self._in_flight > 0:
                self._in_flight -= 1

    @asynccontextmanager
    async def slot(self, *, check_rate: bool = True) -> AsyncGenerator[None]:
        if check_rate:
            await self.check_rate_limit()
        await self._acquire_slot()
        try:
            yield
        finally:
            await self._release_slot()
