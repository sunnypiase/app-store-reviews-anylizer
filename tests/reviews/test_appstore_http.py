import httpx
import pytest

from app.reviews.appstore.errors import AppStoreUnavailableError
from app.reviews.appstore.http import get_with_retry


async def test_retries_on_429_then_succeeds(monkeypatch):
    sleep_calls: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleep_calls.append(seconds)

    monkeypatch.setattr("app.reviews.appstore.http.asyncio.sleep", fake_sleep)

    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            return httpx.Response(429)
        return httpx.Response(200, json={"ok": True})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        response = await get_with_retry(client, "https://example.com", max_attempts=5, base_delay=0.1)

    assert response.status_code == 200
    assert call_count == 3
    assert len(sleep_calls) == 2


async def test_raises_after_exhausting_retries_on_transport_error(monkeypatch):
    async def fake_sleep(seconds: float) -> None:
        return None

    monkeypatch.setattr("app.reviews.appstore.http.asyncio.sleep", fake_sleep)

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(AppStoreUnavailableError):
            await get_with_retry(client, "https://example.com", max_attempts=2, base_delay=0.01)


async def test_non_retryable_status_returned_immediately(monkeypatch):
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(404)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        response = await get_with_retry(client, "https://example.com", max_attempts=5, base_delay=0.01)

    assert response.status_code == 404
    assert calls == 1


async def test_raises_after_exhausting_retries_on_retryable_status(monkeypatch):
    """Every attempt gets a retryable status (not a transport error) — this
    must still raise AppStoreUnavailableError once retries are exhausted,
    not silently hand the caller back a bad 429/5xx response."""

    async def fake_sleep(seconds: float) -> None:
        return None

    monkeypatch.setattr("app.reviews.appstore.http.asyncio.sleep", fake_sleep)

    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(503)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(AppStoreUnavailableError):
            await get_with_retry(client, "https://example.com", max_attempts=3, base_delay=0.01)

    assert calls == 3
