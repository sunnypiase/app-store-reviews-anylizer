import uuid

from pydantic import BaseModel, Field


class RatingBucket(BaseModel):
    count: int = Field(ge=0)
    percent: float = Field(ge=0, le=100)


class Metrics(BaseModel):
    sample_id: uuid.UUID
    review_count: int = Field(ge=0)
    average_rating: float = Field(ge=0, le=5, description="0 when the sample has no reviews")
    rating_distribution: dict[int, RatingBucket] = Field(
        description="Keyed by star rating 1-5; all five keys always present"
    )
