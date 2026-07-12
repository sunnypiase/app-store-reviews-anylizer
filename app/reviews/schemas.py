import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class Review(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    date: datetime
    user_name: str
    title: str
    content: str
    rating: int = Field(ge=1, le=5)
    app_version: str


class ReviewsSample(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    app_id: int  # Apple's numeric app id — distinct concept from our own PKs
    country_code: str = Field(
        pattern=r"^[a-z]{2}$", description="ISO 3166-1 alpha-2, e.g. 'us'"
    )
    created_at: datetime
    reviews: list[Review]


class ReviewsCollectRequest(BaseModel):
    app_id: int = Field(gt=0)
    country_code: str = Field(
        pattern=r"^[a-z]{2}$", description="ISO 3166-1 alpha-2, e.g. 'us'"
    )
    sample_size: int = Field(default=100, ge=1, le=500)
