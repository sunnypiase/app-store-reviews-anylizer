from datetime import datetime
from typing import Protocol

from pydantic import BaseModel, Field


class CollectedReview(BaseModel):
    """A fetched review, not yet persisted (id is assigned by the database)."""

    store_review_id: str
    date: datetime
    user_name: str
    title: str
    content: str
    rating: int = Field(ge=1, le=5)
    app_version: str


class ReviewCollector(Protocol):
    """Source-agnostic collection seam; AppStoreCollector is the only implementation."""

    async def collect(
        self, app_id: int, country_code: str, limit: int
    ) -> list[CollectedReview]: ...
