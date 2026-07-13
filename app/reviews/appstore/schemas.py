"""Apple's wire format (iTunes Lookup + RSS feed), separate from this app's
own API contract in app.reviews.schemas."""

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
    # Apple omits "entry" on an exhausted feed; default [] parses that as "no reviews".
    entry: list[AppleFeedEntry] = Field(default_factory=list)


class AppleReviewsResponse(BaseModel):
    feed: AppleReviewsFeed


class AppleLookupResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    result_count: int = Field(alias="resultCount")


class AppleLookupError(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    error_message: str = Field(alias="errorMessage")
