from fastapi import APIRouter

from app.metrics.schemas import Metrics, RatingBucket

metric_router = APIRouter()


@metric_router.get("/{sample_id}", response_model=Metrics)
async def get_metrics(sample_id: int) -> Metrics:
    return Metrics(
        sample_id=sample_id,
        review_count=100,
        average_rating=3.42,
        rating_distribution={
            1: RatingBucket(count=20, percent=20.0),
            2: RatingBucket(count=10, percent=10.0),
            3: RatingBucket(count=15, percent=15.0),
            4: RatingBucket(count=18, percent=18.0),
            5: RatingBucket(count=37, percent=37.0),
        },
    )
