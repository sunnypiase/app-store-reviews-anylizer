from uuid import UUID

from fastapi import APIRouter, HTTPException

from app.dependencies import DbSession
from app.insights import service
from app.insights.dependencies import InsightGeneratorDep, SentimentClassifierDep
from app.insights.schemas import Insight
from app.reviews import schemas as review_schemas
from app.reviews import service as review_service

insight_router = APIRouter()


@insight_router.get("/{sample_id}", response_model=Insight)
async def get_insight(
    sample_id: UUID,
    session: DbSession,
    gemini_classifier: SentimentClassifierDep,
    insight_generator: InsightGeneratorDep,
) -> Insight:
    sample = await review_service.get_sample_with_reviews(session, sample_id)
    if sample is None:
        raise HTTPException(status_code=404, detail="Sample not found")
    reviews = [review_schemas.Review.model_validate(review) for review in sample.reviews]
    return await service.compute_insights(sample_id, reviews, gemini_classifier, insight_generator)
