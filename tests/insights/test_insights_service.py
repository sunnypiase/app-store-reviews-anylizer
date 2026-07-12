import uuid
from datetime import datetime, timezone

from app.insights.service import classify_sentiment, compute_insights, find_disagreements
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


def test_classify_sentiment_positive_and_negative():
    positive = _review(rating=5, title="Amazing", content="I love this app, it's fantastic and works great")
    negative = _review(rating=1, title="Terrible", content="Worst app ever, awful experience, I hate it")

    assert classify_sentiment(positive) == "positive"
    assert classify_sentiment(negative) == "negative"


def test_find_disagreements_flags_high_rating_negative_text():
    review = _review(
        rating=5, title="ok", content="Great app but they keep charging me even after I canceled, awful"
    )

    disagreements = find_disagreements([review], ["negative"])

    assert len(disagreements) == 1
    assert disagreements[0].review_id == review.id
    assert disagreements[0].rating == 5
    assert disagreements[0].sentiment == "negative"


def test_find_disagreements_ignores_matching_rating_and_sentiment():
    review = _review(rating=5, title="ok", content="ok")

    assert find_disagreements([review], ["positive"]) == []


def test_compute_insights_extracts_keywords_from_negative_reviews_only():
    reviews = [
        _review(
            rating=1, title="Billing issue", content="Customer service never responds, charged twice"
        ),
        _review(
            rating=2,
            title="Cancel subscription",
            content="Cancel subscription is impossible, customer service ignored me",
        ),
        _review(rating=5, title="Great", content="This app is fantastic, I use it every day"),
    ]

    insight = compute_insights(uuid.uuid4(), reviews)

    assert insight.review_count == 3
    assert len(insight.negative_keywords) > 0
    phrases = {kw.phrase for kw in insight.negative_keywords}
    assert "customer service" in phrases
    # Actionable insight themes must not overlap (no unigram alongside its own bigram).
    themes = [insight_.theme for insight_ in insight.actionable_insights]
    assert len(themes) == len(set(themes))


def test_compute_insights_empty_sample():
    insight = compute_insights(uuid.uuid4(), [])

    assert insight.review_count == 0
    assert insight.negative_keywords == []
    assert insight.actionable_insights == []
    assert insight.sentiment_distribution.positive == 0
    assert insight.sentiment_distribution.neutral == 0
    assert insight.sentiment_distribution.negative == 0
