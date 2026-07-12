import uuid

from app.metrics.schemas import Metrics, RatingBucket
from app.reviews import schemas as review_schemas


def compute_metrics(sample_id: uuid.UUID, reviews: list[review_schemas.Review]) -> Metrics:
    review_count = len(reviews)
    counts = dict.fromkeys(range(1, 6), 0)
    for review in reviews:
        counts[review.rating] += 1

    average_rating = sum(review.rating for review in reviews) / review_count if review_count else 0.0
    rating_distribution = {
        star: RatingBucket(
            count=counts[star],
            percent=round(counts[star] / review_count * 100, 2) if review_count else 0.0,
        )
        for star in range(1, 6)
    }
    return Metrics(
        sample_id=sample_id,
        review_count=review_count,
        average_rating=round(average_rating, 2),
        rating_distribution=rating_distribution,
    )
