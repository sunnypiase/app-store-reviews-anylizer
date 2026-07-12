import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock

from google.genai import errors as genai_errors

from app.insights import gemini_classifier as gemini_classifier_module
from app.insights.gemini_classifier import GeminiClassificationError, GeminiSentimentClassifier
from app.reviews.schemas import Review


def _review(*, rating: int, title: str, content: str) -> Review:
    return Review(
        id=uuid.uuid4(),
        date=datetime(2026, 1, 1, tzinfo=timezone.utc),
        user_name="u",
        title=title,
        content=content,
        rating=rating,
        app_version="1.0",
    )


class _FakeResponse:
    def __init__(self, parsed):
        self.parsed = parsed
        self.text = "[]"


def _sentiment_response(reviews, sentiment: str) -> _FakeResponse:
    return _FakeResponse(
        [
            gemini_classifier_module._ReviewSentiment(review_id=str(review.id), sentiment=sentiment)
            for review in reviews
        ]
    )


async def test_classify_maps_response_back_to_review_ids():
    reviews = [
        _review(rating=1, title="bad", content="bad"),
        _review(rating=5, title="good", content="good"),
    ]
    client = AsyncMock()
    client.aio.models.generate_content = AsyncMock(
        side_effect=lambda **_: _sentiment_response(reviews, "positive")
    )
    classifier = GeminiSentimentClassifier(client, "gemini-flash-lite-latest")

    result = await classifier.classify(reviews)

    assert result == {review.id: "positive" for review in reviews}


async def test_classify_does_not_send_star_rating_to_gemini():
    review = _review(rating=1, title="loved it", content="best app ever")
    captured_prompts = []

    async def _capture(**kwargs):
        captured_prompts.append(kwargs["contents"])
        return _sentiment_response([review], "positive")

    client = AsyncMock()
    client.aio.models.generate_content = AsyncMock(side_effect=_capture)
    classifier = GeminiSentimentClassifier(client, "gemini-flash-lite-latest")

    await classifier.classify([review])

    prompt = captured_prompts[0]
    assert str(review.id) in prompt
    assert review.title in prompt
    assert review.content in prompt
    assert "rating" not in prompt.lower()
    assert "star" not in prompt.lower()


async def test_classify_raises_when_response_ids_dont_match_request():
    reviews = [_review(rating=1, title="bad", content="bad")]
    client = AsyncMock()
    client.aio.models.generate_content = AsyncMock(
        return_value=_FakeResponse(
            [gemini_classifier_module._ReviewSentiment(review_id=str(uuid.uuid4()), sentiment="negative")]
        )
    )
    classifier = GeminiSentimentClassifier(client, "gemini-flash-lite-latest")

    try:
        await classifier.classify(reviews)
        raise AssertionError("expected GeminiClassificationError")
    except GeminiClassificationError:
        pass


async def test_classify_retries_retryable_errors_then_succeeds(monkeypatch):
    monkeypatch.setattr(gemini_classifier_module, "_BASE_DELAY_SECONDS", 0.0)
    review = _review(rating=1, title="bad", content="bad")
    client = AsyncMock()
    client.aio.models.generate_content = AsyncMock(
        side_effect=[
            genai_errors.APIError(429, {"error": {"message": "rate limited"}}, None),
            _sentiment_response([review], "negative"),
        ]
    )
    classifier = GeminiSentimentClassifier(client, "gemini-flash-lite-latest")

    result = await classifier.classify([review])

    assert result == {review.id: "negative"}
    assert client.aio.models.generate_content.await_count == 2


async def test_classify_raises_on_non_retryable_error():
    review = _review(rating=1, title="bad", content="bad")
    client = AsyncMock()
    client.aio.models.generate_content = AsyncMock(
        side_effect=genai_errors.APIError(400, {"error": {"message": "bad request"}}, None)
    )
    classifier = GeminiSentimentClassifier(client, "gemini-flash-lite-latest")

    try:
        await classifier.classify([review])
        raise AssertionError("expected GeminiClassificationError")
    except GeminiClassificationError:
        pass

    assert client.aio.models.generate_content.await_count == 1
