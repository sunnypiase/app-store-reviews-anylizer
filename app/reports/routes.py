from uuid import UUID

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse

from app.dependencies import DbSession
from app.insights import service as insight_service
from app.insights.dependencies import InsightGeneratorDep, SentimentClassifierDep
from app.metrics import service as metrics_service
from app.reports import service
from app.reviews import schemas as review_schemas
from app.reviews import service as review_service

report_router = APIRouter()


@report_router.get("/{sample_id}", response_class=HTMLResponse)
async def get_report(
    sample_id: UUID,
    session: DbSession,
    gemini_classifier: SentimentClassifierDep,
    insight_generator: InsightGeneratorDep,
) -> HTMLResponse:
    sample = await review_service.get_sample_with_reviews(session, sample_id)
    if sample is None:
        raise HTTPException(status_code=404, detail="Sample not found")
    reviews = [review_schemas.Review.model_validate(review) for review in sample.reviews]
    metrics = metrics_service.compute_metrics(sample_id, reviews)
    insight = await insight_service.compute_insights(
        sample_id, reviews, gemini_classifier, insight_generator
    )
    sample_schema = review_schemas.ReviewsSample.model_validate(sample)
    html = service.render_report(sample_schema, metrics, insight)
    return HTMLResponse(html)
