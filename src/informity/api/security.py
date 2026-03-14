# ==============================================================================
# Informity AI — API Security Guards
# In-process rate limiting and concurrency caps for expensive endpoints.
# ==============================================================================

import asyncio
import time
from collections import deque
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import HTTPException


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
