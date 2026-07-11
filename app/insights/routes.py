from datetime import datetime, timezone

from fastapi import APIRouter

from app.insights.schemas import (
    ActionableInsight,
    Disagreement,
    Insight,
    InsightsJob,
    InsightsJobCreate,
    InsightsJobId,
    NegativeKeyword,
    SentimentDistribution,
)

insight_router = APIRouter()


@insight_router.post("/jobs", response_model=InsightsJobId, status_code=202)
async def create_insights_job(job_create: InsightsJobCreate) -> InsightsJobId:
    return InsightsJobId(job_id=1)


@insight_router.get("/jobs/{job_id}", response_model=InsightsJob)
async def get_insights_job(job_id: int) -> InsightsJob:
    return InsightsJob(status="done", insight_id=1)


@insight_router.get("/{insight_id}", response_model=Insight)
async def get_insight(insight_id: int) -> Insight:
    return Insight(
        id=insight_id,
        sample_id=1,
        created_at=datetime.now(timezone.utc),
        sentiment_distribution=SentimentDistribution(
            positive=41, neutral=22, negative=37
        ),
        sentiment_rating_disagreement=[
            Disagreement(
                review_id=14284892659,
                rating=1,
                sentiment="positive",
                title="Great app but they keep charging me",
            ),
        ],
        negative_keywords=[
            NegativeKeyword(phrase="customer service", count=14),
            NegativeKeyword(phrase="charged", count=11),
            NegativeKeyword(phrase="cancel subscription", count=9),
        ],
        actionable_insights=[
            ActionableInsight(
                theme="Billing continues after cancellation",
                evidence_count=12,
                suggestion=(
                    "Audit the cancellation flow; send a confirmation email "
                    "with the effective end date."
                ),
                example_review_ids=[14284892659, 14283110021],
            ),
        ],
    )
