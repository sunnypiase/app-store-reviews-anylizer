from datetime import datetime, timezone
from typing import Literal

from fastapi import APIRouter, Response

from app.reviews.schemas import (
    ReviewsJob,
    ReviewsJobCreate,
    ReviewsJobId,
    ReviewsSample,
)

review_router = APIRouter()


@review_router.post("/jobs", response_model=ReviewsJobId, status_code=202)
async def create_reviews_job(job_create: ReviewsJobCreate) -> ReviewsJobId:
    return ReviewsJobId(job_id=1)


@review_router.get("/jobs/{job_id}", response_model=ReviewsJob)
async def get_reviews_job(job_id: int) -> ReviewsJob:
    return ReviewsJob(status="pending")


@review_router.get("/samples/{sample_id}", response_model=ReviewsSample)
async def get_reviews_sample(sample_id: int) -> ReviewsSample:
    return ReviewsSample(
        id=1,
        app_id=1,
        country_code="us",
        created_at=datetime.now(timezone.utc),
        reviews=[],
    )


@review_router.get("/samples/{sample_id}/download")
async def download_reviews_sample(
    sample_id: int, format: Literal["json", "csv"] = "json"
) -> Response:
    if format == "csv":
        content = "id,date,user_name,title,content,rating,app_version\n"
        media_type = "text/csv"
    else:
        content = '{"id": 1, "app_id": 1, "country_code": "us", "reviews": []}'
        media_type = "application/json"
    return Response(
        content=content,
        media_type=media_type,
        headers={
            "Content-Disposition": f"attachment; filename=reviews_sample_{sample_id}.{format}"
        },
    )
