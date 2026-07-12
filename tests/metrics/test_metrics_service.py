import uuid
from datetime import datetime, timezone

from app.metrics.service import compute_metrics
from app.reviews.schemas import Review


def _review(rating: int) -> Review:
    return Review(
        id=uuid.uuid4(),
        date=datetime(2026, 1, 1, tzinfo=timezone.utc),
        user_name="u",
        title="t",
        content="c",
        rating=rating,
        app_version="1.0",
    )


def test_computes_average_and_distribution():
    sample_id = uuid.uuid4()
    reviews = [_review(5), _review(5), _review(1), _review(3)]

    metrics = compute_metrics(sample_id, reviews)

    assert metrics.sample_id == sample_id
    assert metrics.review_count == 4
    assert metrics.average_rating == 3.5
    assert metrics.rating_distribution[5].count == 2
    assert metrics.rating_distribution[5].percent == 50.0
    assert metrics.rating_distribution[1].count == 1
    assert metrics.rating_distribution[2].count == 0
    assert set(metrics.rating_distribution.keys()) == {1, 2, 3, 4, 5}


def test_empty_sample_has_zero_average_and_all_zero_buckets():
    sample_id = uuid.uuid4()

    metrics = compute_metrics(sample_id, [])

    assert metrics.review_count == 0
    assert metrics.average_rating == 0.0
    assert all(bucket.count == 0 for bucket in metrics.rating_distribution.values())
