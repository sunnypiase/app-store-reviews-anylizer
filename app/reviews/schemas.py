from datetime import datetime
from typing import Literal
from pydantic import BaseModel, Field


# TODO: DB layer (app/reviews/models.py) now uses a UUID PK with a separate
# store_review_id; these API schemas still use a plain int id. Reconcile once
# the service layer wires real persistence into routes.py.
class Review(BaseModel):
    id: int
    date: datetime
    user_name: str
    title: str
    content: str
    rating: int = Field(ge=1, le=5)
    app_version: str


class ReviewsSample(BaseModel):
    id: int
    app_id: int
    country_code: str = Field(
        pattern=r"^[a-z]{2}$", description="ISO 3166-1 alpha-2, e.g. 'us'"
    )
    created_at: datetime
    reviews: list[Review]


class ReviewsJobCreate(BaseModel):
    app_id: int
    country_code: str = Field(
        pattern=r"^[a-z]{2}$", description="ISO 3166-1 alpha-2, e.g. 'us'"
    )
    sample_size: int = Field(default=100, ge=1, le=500)


class ReviewsJobId(BaseModel):
    job_id: int


class ReviewsJob(BaseModel):
    status: Literal["pending", "running", "done", "failed"]
    sample_id: int | None = None
    error: str | None = None
