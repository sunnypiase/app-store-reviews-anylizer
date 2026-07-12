import csv
import io
import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.reviews.collector import CollectedReview
from app.reviews.models import Review, ReviewSample


async def create_sample_with_reviews(
    session: AsyncSession, *, app_id: int, country_code: str, reviews: list[CollectedReview]
) -> ReviewSample:
    sample = ReviewSample(
        app_id=app_id,
        country_code=country_code,
        reviews=[
            Review(
                store_review_id=review.store_review_id,
                date=review.date,
                user_name=review.user_name,
                title=review.title,
                content=review.content,
                rating=review.rating,
                app_version=review.app_version,
            )
            for review in reviews
        ],
    )
    session.add(sample)
    return sample


async def get_sample_with_reviews(
    session: AsyncSession, sample_id: uuid.UUID
) -> ReviewSample | None:
    result = await session.execute(
        select(ReviewSample)
        .where(ReviewSample.id == sample_id)
        .options(selectinload(ReviewSample.reviews))
    )
    return result.scalar_one_or_none()


def render_csv(sample: ReviewSample) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["id", "date", "user_name", "title", "content", "rating", "app_version"])
    for review in sample.reviews:
        writer.writerow(
            [
                review.id,
                review.date.isoformat(),
                review.user_name,
                review.title,
                review.content,
                review.rating,
                review.app_version,
            ]
        )
    return buffer.getvalue()
