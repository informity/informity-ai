import pytest
from fastapi import HTTPException

from informity.api.security import EndpointGuard


@pytest.mark.asyncio
async def test_endpoint_guard_blocks_excess_in_flight() -> None:
    guard = EndpointGuard(
        name='test',
        max_in_flight=1,
        max_requests_per_window=10,
        window_seconds=60,
    )

    first = guard.slot()
    await first.__aenter__()
    try:
        second = guard.slot()
        with pytest.raises(HTTPException) as exc_info:
            await second.__aenter__()
        assert exc_info.value.status_code == 429
    finally:
        await first.__aexit__(None, None, None)


@pytest.mark.asyncio
async def test_endpoint_guard_rate_limit() -> None:
    guard = EndpointGuard(
        name='test-rate',
        max_in_flight=2,
        max_requests_per_window=1,
        window_seconds=60,
    )

    await guard.check_rate_limit()
    with pytest.raises(HTTPException) as exc_info:
        await guard.check_rate_limit()
    assert exc_info.value.status_code == 429
