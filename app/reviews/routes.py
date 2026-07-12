from typing import Literal
from uuid import UUID

from fastapi import APIRouter, HTTPException, Response

from app.dependencies import DbSession
from app.reviews import service
from app.reviews.dependencies import CollectorDep
from app.reviews.schemas import ReviewsCollectRequest, ReviewsSample

review_router = APIRouter()


@review_router.post("", response_model=ReviewsSample, status_code=201)
async def collect_reviews(
    collect_request: ReviewsCollectRequest, session: DbSession, collector: CollectorDep
) -> ReviewsSample:
    collected = await collector.collect(
        collect_request.app_id, collect_request.country_code, collect_request.sample_size
    )
    sample = await service.create_sample_with_reviews(
        session,
        app_id=collect_request.app_id,
        country_code=collect_request.country_code,
        reviews=collected,
    )
    await session.commit()
    return ReviewsSample.model_validate(sample)


@review_router.get("/{sample_id}", response_model=ReviewsSample)
async def get_reviews_sample(sample_id: UUID, session: DbSession) -> ReviewsSample:
    sample = await service.get_sample_with_reviews(session, sample_id)
    if sample is None:
        raise HTTPException(status_code=404, detail="Sample not found")
    return ReviewsSample.model_validate(sample)


@review_router.get("/{sample_id}/download")
async def download_reviews_sample(
    sample_id: UUID, session: DbSession, format: Literal["json", "csv"] = "json"
) -> Response:
    sample = await service.get_sample_with_reviews(session, sample_id)
    if sample is None:
        raise HTTPException(status_code=404, detail="Sample not found")
    if format == "csv":
        content = service.render_csv(sample)
        media_type = "text/csv"
    else:
        content = ReviewsSample.model_validate(sample).model_dump_json()
        media_type = "application/json"
    return Response(
        content=content,
        media_type=media_type,
        headers={
            "Content-Disposition": f"attachment; filename=reviews_sample_{sample_id}.{format}"
        },
    )
