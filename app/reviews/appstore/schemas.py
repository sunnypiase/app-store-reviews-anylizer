"""Pydantic models for Apple's own wire format (iTunes Lookup + RSS reviews
feed) — see docs/APPSTORE_RSS_RESEARCH.md for the researched shape and
gotchas. Kept separate from app.reviews.schemas, which is this app's own
public API contract, not Apple's.
"""

from pydantic import BaseModel, ConfigDict, Field


class AppleLabel(BaseModel):
    label: str


class AppleAuthor(BaseModel):
    name: AppleLabel


class AppleFeedEntry(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    entry_id: AppleLabel = Field(alias="id")
    author: AppleAuthor
    updated: AppleLabel
    title: AppleLabel
    content: AppleLabel
    rating: AppleLabel = Field(alias="im:rating")
    version: AppleLabel = Field(alias="im:version")


class AppleReviewsFeed(BaseModel):
    # Apple omits the "entry" key entirely on an empty/exhausted feed — a
    # default of [] means that parses as "no reviews", not a validation error.
    entry: list[AppleFeedEntry] = Field(default_factory=list)


class AppleReviewsResponse(BaseModel):
    feed: AppleReviewsFeed


class AppleLookupResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    result_count: int = Field(alias="resultCount")


class AppleLookupError(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    error_message: str = Field(alias="errorMessage")
