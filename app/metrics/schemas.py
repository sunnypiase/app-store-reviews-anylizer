from pydantic import BaseModel, Field


class RatingBucket(BaseModel):
    count: int = Field(ge=0)
    percent: float = Field(ge=0, le=100)


class Metrics(BaseModel):
    sample_id: int
    review_count: int = Field(ge=0)
    average_rating: float = Field(ge=1, le=5)
    rating_distribution: dict[int, RatingBucket] = Field(
        description="Keyed by star rating 1-5; all five keys always present"
    )
