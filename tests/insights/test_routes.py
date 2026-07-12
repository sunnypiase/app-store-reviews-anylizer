import uuid

from app.insights.dependencies import get_insight_generator, get_sentiment_classifier
from app.insights.gemini_classifier import GeminiClassificationError
from app.insights.gemini_insight_generator import (
    GeminiInsightDraft,
    GeminiInsightGenerationError,
    GeminiThemeDraft,
)
from app.main import app as fastapi_app


class _FakeGeminiClassifier:
    def __init__(self, sentiment_by_id=None, error: Exception | None = None):
        self._sentiment_by_id = sentiment_by_id or {}
        self._error = error

    async def classify(self, reviews):
        if self._error is not None:
            raise self._error
        return {review.id: self._sentiment_by_id[review.id] for review in reviews}


class _FakeInsightGenerator:
    def __init__(self, draft: GeminiInsightDraft | None = None, error: Exception | None = None):
        self._draft = draft
        self._error = error

    async def generate(self, reviews):
        if self._error is not None:
            raise self._error
        return self._draft


async def test_get_insight_for_existing_sample(client, make_sample):
    sample_id = await make_sample(
        [
            {
                "rating": 1,
                "title": "Billing issue",
                "content": "Customer service never responds, charged twice",
            },
            {
                "rating": 5,
                "title": "Great",
                "content": "This app is fantastic, I use it every day",
            },
        ]
    )

    response = await client.get(f"/api/v1/insights/{sample_id}")

    assert response.status_code == 200
    body = response.json()
    assert body["review_count"] == 2
    distribution = body["sentiment_distribution"]
    assert distribution["positive"] + distribution["neutral"] + distribution["negative"] == 2
    # Expanded contract: summary + provenance are always present.
    assert body["actionable_insights_source"] == "rule_based_fallback"
    assert body["executive_summary"]


async def test_get_insight_not_found_returns_404(client):
    response = await client.get(f"/api/v1/insights/{uuid.uuid4()}")

    assert response.status_code == 404


async def test_get_insight_uses_gemini_classifier_when_configured(client, make_sample):
    sample_id = await make_sample([{"rating": 5, "title": "Great", "content": "Great app, love it"}])
    reviews_response = await client.get(f"/api/v1/reviews/{sample_id}")
    review_id = uuid.UUID(reviews_response.json()["reviews"][0]["id"])

    fastapi_app.dependency_overrides[get_sentiment_classifier] = lambda: _FakeGeminiClassifier(
        sentiment_by_id={review_id: "negative"}
    )

    response = await client.get(f"/api/v1/insights/{sample_id}")

    assert response.status_code == 200
    assert response.json()["sentiment_distribution"] == {"positive": 0, "neutral": 0, "negative": 1}


async def test_get_insight_uses_insight_generator_when_configured(client, make_sample):
    sample_id = await make_sample(
        [
            {"rating": 1, "title": "Charged after cancel", "content": "Canceled but still charged"},
            {"rating": 1, "title": "Cannot cancel", "content": "No way to cancel the subscription"},
        ]
    )
    reviews_response = await client.get(f"/api/v1/reviews/{sample_id}")
    review_ids = [uuid.UUID(review["id"]) for review in reviews_response.json()["reviews"]]
    draft = GeminiInsightDraft(
        executive_summary="Cancellation complaints dominate the negative feedback.",
        themes=[
            GeminiThemeDraft(
                theme="Subscription cancellation",
                problem_summary="Users report charges after attempting to cancel.",
                suggestion="Audit the cancellation flow.",
                evidence_review_ids=review_ids,
            )
        ],
    )
    fastapi_app.dependency_overrides[get_insight_generator] = lambda: _FakeInsightGenerator(
        draft=draft
    )

    response = await client.get(f"/api/v1/insights/{sample_id}")

    assert response.status_code == 200
    body = response.json()
    assert body["actionable_insights_source"] == "gemini"
    assert body["executive_summary"] == "Cancellation complaints dominate the negative feedback."
    [insight] = body["actionable_insights"]
    assert insight["theme"] == "Subscription cancellation"
    assert insight["evidence_count"] == 2
    assert insight["evidence_review_ids"] == [str(review_id) for review_id in review_ids]


async def test_get_insight_returns_200_fallback_when_insight_generator_fails(client, make_sample):
    sample_id = await make_sample(
        [
            {"rating": 1, "title": "Billing issue", "content": "Charged twice, customer service silent"},
            {"rating": 2, "title": "Billing again", "content": "Charged twice again, billing is broken"},
        ]
    )
    fastapi_app.dependency_overrides[get_insight_generator] = lambda: _FakeInsightGenerator(
        error=GeminiInsightGenerationError("boom")
    )

    response = await client.get(f"/api/v1/insights/{sample_id}")

    assert response.status_code == 200
    body = response.json()
    assert body["actionable_insights_source"] == "rule_based_fallback"
    assert "local fallback" in body["executive_summary"]


async def test_get_insight_all_positive_sample_returns_none_source(client, make_sample):
    sample_id = await make_sample(
        [
            {"rating": 5, "title": "Great", "content": "I love this app, it's fantastic"},
            {"rating": 5, "title": "Excellent", "content": "Works great, wonderful design"},
        ]
    )

    response = await client.get(f"/api/v1/insights/{sample_id}")

    assert response.status_code == 200
    body = response.json()
    assert body["actionable_insights_source"] == "none"
    assert body["actionable_insights"] == []


async def test_get_insight_falls_back_to_vader_when_gemini_fails(client, make_sample):
    sample_id = await make_sample(
        [
            {
                "rating": 5,
                "title": "Great",
                "content": "I love this app, it's fantastic and works great",
            }
        ]
    )
    fastapi_app.dependency_overrides[get_sentiment_classifier] = lambda: _FakeGeminiClassifier(
        error=GeminiClassificationError("boom")
    )

    response = await client.get(f"/api/v1/insights/{sample_id}")

    assert response.status_code == 200
    assert response.json()["sentiment_distribution"]["positive"] == 1
