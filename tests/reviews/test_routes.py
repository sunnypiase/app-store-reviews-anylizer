import uuid
from datetime import datetime, timezone

from app.main import app as fastapi_app
from app.reviews.appstore.errors import (
    AppNotFoundError,
    AppStoreUnavailableError,
    InvalidCountryCodeError,
)
from app.reviews.collector import CollectedReview
from app.reviews.dependencies import get_review_collector


class _FakeCollector:
    def __init__(self, reviews=None, error=None):
        self._reviews = reviews or []
        self._error = error

    async def collect(self, app_id: int, country_code: str, limit: int) -> list[CollectedReview]:
        if self._error is not None:
            raise self._error
        return self._reviews


def _collected_review(rating: int = 5) -> CollectedReview:
    return CollectedReview(
        store_review_id=str(uuid.uuid4()),
        date=datetime.now(timezone.utc),
        user_name="jappleseed",
        title="Great",
        content="Loved it",
        rating=rating,
        app_version="1.0",
    )


async def test_collect_reviews_returns_created_sample(client):
    fastapi_app.dependency_overrides[get_review_collector] = lambda: _FakeCollector(
        reviews=[_collected_review(5), _collected_review(1)]
    )

    response = await client.post(
        "/api/v1/reviews", json={"app_id": 123, "country_code": "us", "sample_size": 100}
    )

    assert response.status_code == 201
    body = response.json()
    assert body["app_id"] == 123
    assert body["country_code"] == "us"
    assert len(body["reviews"]) == 2


async def test_collect_reviews_app_not_found_returns_404(client):
    fastapi_app.dependency_overrides[get_review_collector] = lambda: _FakeCollector(
        error=AppNotFoundError(123, "us")
    )

    response = await client.post(
        "/api/v1/reviews", json={"app_id": 123, "country_code": "us", "sample_size": 100}
    )

    assert response.status_code == 404


async def test_collect_reviews_invalid_country_returns_400(client):
    fastapi_app.dependency_overrides[get_review_collector] = lambda: _FakeCollector(
        error=InvalidCountryCodeError("zz", "Invalid value(s) for key(s): [country]")
    )

    response = await client.post(
        "/api/v1/reviews", json={"app_id": 123, "country_code": "zz", "sample_size": 100}
    )

    assert response.status_code == 400


async def test_collect_reviews_unavailable_returns_503(client):
    fastapi_app.dependency_overrides[get_review_collector] = lambda: _FakeCollector(
        error=AppStoreUnavailableError("App Store request failed after 4 attempts")
    )

    response = await client.post(
        "/api/v1/reviews", json={"app_id": 123, "country_code": "us", "sample_size": 100}
    )

    assert response.status_code == 503


async def test_collect_reviews_rejects_invalid_country_code_shape(client):
    response = await client.post(
        "/api/v1/reviews", json={"app_id": 123, "country_code": "USA", "sample_size": 100}
    )

    assert response.status_code == 422


async def test_get_reviews_sample_not_found_returns_404(client):
    response = await client.get(f"/api/v1/reviews/{uuid.uuid4()}")

    assert response.status_code == 404
