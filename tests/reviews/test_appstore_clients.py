import asyncio
from urllib.parse import urlparse

import httpx
import pytest

from app.reviews.appstore.collector import AppStoreCollector
from app.reviews.appstore.errors import AppNotFoundError, InvalidCountryCodeError
from app.reviews.appstore.lookup_client import AppStoreLookupClient
from app.reviews.appstore.reviews_client import AppStoreReviewsClient


def _entry(entry_id: str, rating: str = "5") -> dict:
    return {
        "id": {"label": entry_id},
        "author": {"name": {"label": "jappleseed"}},
        "updated": {"label": "2026-01-01T00:00:00-07:00"},
        "title": {"label": "Great app"},
        "content": {"label": "Works well."},
        "im:rating": {"label": rating},
        "im:version": {"label": "2.3.1"},
    }


@pytest.fixture(autouse=True)
def _no_real_sleep(monkeypatch):
    async def fake_sleep(seconds: float) -> None:
        return None

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)


async def test_lookup_raises_app_not_found_on_zero_results():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"resultCount": 0, "results": []})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = AppStoreLookupClient(http, max_retries=2, retry_base_delay=0.01)
        with pytest.raises(AppNotFoundError):
            await client.verify_app_exists(999999999999, "us")


async def test_lookup_raises_invalid_country_on_structured_400():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400, json={"errorMessage": "Invalid value(s) for key(s): [country]", "queryParameters": {}}
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = AppStoreLookupClient(http, max_retries=2, retry_base_delay=0.01)
        with pytest.raises(InvalidCountryCodeError):
            await client.verify_app_exists(123, "zz")


async def test_lookup_passes_when_app_exists():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"resultCount": 1, "results": [{}]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = AppStoreLookupClient(http, max_retries=2, retry_base_delay=0.01)
        await client.verify_app_exists(310633997, "us")  # no exception


async def test_reviews_client_stops_at_page_10():
    requested_pages: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        page = int(urlparse(str(request.url)).path.split("page=")[1].split("/")[0])
        requested_pages.append(page)
        entries = [_entry(f"{page}-{i}") for i in range(50)]  # always full pages
        return httpx.Response(200, json={"feed": {"entry": entries}})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = AppStoreReviewsClient(http, max_retries=2, retry_base_delay=0.01, throttle_delay=0.01)
        results = await client.fetch_reviews(123, "us", limit=1000)

    assert requested_pages == list(range(1, 11))
    assert len(results) == 500  # 10 pages * 50, the documented hard cap


async def test_reviews_client_empty_feed_is_not_an_error():
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json={"feed": {}})  # no "entry" key at all

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = AppStoreReviewsClient(http, max_retries=2, retry_base_delay=0.01, throttle_delay=0.01)
        results = await client.fetch_reviews(123, "us", limit=100)

    assert results == []
    assert calls == 1


async def test_reviews_client_stops_on_partial_page():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"feed": {"entry": [_entry("only-one")]}})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = AppStoreReviewsClient(http, max_retries=2, retry_base_delay=0.01, throttle_delay=0.01)
        results = await client.fetch_reviews(123, "us", limit=100)

    assert len(results) == 1


async def test_reviews_client_stops_on_400_page_depth_exceeded():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, text="CustomerReviews RSS page depth is limited to 10")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = AppStoreReviewsClient(http, max_retries=2, retry_base_delay=0.01, throttle_delay=0.01)
        results = await client.fetch_reviews(123, "us", limit=100)

    assert results == []


async def test_collector_skips_malformed_entry_but_keeps_the_rest():
    def handler(request: httpx.Request) -> httpx.Response:
        path = str(request.url)
        if "lookup" in path:
            return httpx.Response(200, json={"resultCount": 1, "results": [{}]})
        return httpx.Response(
            200,
            json={
                "feed": {
                    "entry": [
                        _entry("good-1", rating="4"),
                        _entry("bad-rating", rating="not-a-number"),
                    ]
                }
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        collector = AppStoreCollector(
            lookup=AppStoreLookupClient(http, max_retries=2, retry_base_delay=0.01),
            reviews=AppStoreReviewsClient(http, max_retries=2, retry_base_delay=0.01, throttle_delay=0.01),
        )
        results = await collector.collect(310633997, "us", limit=10)

    assert len(results) == 1
    assert results[0].store_review_id == "good-1"
