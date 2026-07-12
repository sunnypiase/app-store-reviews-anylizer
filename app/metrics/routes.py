from uuid import UUID

from fastapi import APIRouter, HTTPException

from app.dependencies import DbSession
from app.metrics import service
from app.metrics.schemas import Metrics
from app.reviews import schemas as review_schemas
from app.reviews import service as review_service

metric_router = APIRouter()


@metric_router.get("/{sample_id}", response_model=Metrics)
async def get_metrics(sample_id: UUID, session: DbSession) -> Metrics:
    sample = await review_service.get_sample_with_reviews(session, sample_id)
    if sample is None:
        raise HTTPException(status_code=404, detail="Sample not found")
    reviews = [review_schemas.Review.model_validate(review) for review in sample.reviews]
    return service.compute_metrics(sample_id, reviews)
