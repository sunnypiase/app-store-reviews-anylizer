from datetime import datetime
from typing import Protocol

from pydantic import BaseModel, Field


class CollectedReview(BaseModel):
    """A review fetched from some external source, not yet persisted (no id
    — that's assigned by the database on insert)."""

    store_review_id: str
    date: datetime
    user_name: str
    title: str
    content: str
    rating: int = Field(ge=1, le=5)
    app_version: str


class ReviewCollector(Protocol):
    """The seam between "fetched from somewhere" and "persisted here" — not
    Apple-specific. app.reviews.appstore.collector.AppStoreCollector is the
    only implementation today.
    """

    async def collect(
        self, app_id: int, country_code: str, limit: int
    ) -> list[CollectedReview]: ...
