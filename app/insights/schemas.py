from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

Sentiment = Literal["positive", "neutral", "negative"]


class SentimentDistribution(BaseModel):
    positive: int = Field(ge=0)
    neutral: int = Field(ge=0)
    negative: int = Field(ge=0)


class Disagreement(BaseModel):
    """A review whose text sentiment contradicts its star rating."""

    review_id: int
    rating: int = Field(ge=1, le=5)
    sentiment: Sentiment
    title: str


class NegativeKeyword(BaseModel):
    phrase: str
    count: int = Field(ge=1)


class ActionableInsight(BaseModel):
    theme: str
    evidence_count: int = Field(ge=1)
    suggestion: str
    example_review_ids: list[int]


class Insight(BaseModel):
    id: int
    sample_id: int
    created_at: datetime
    sentiment_distribution: SentimentDistribution
    sentiment_rating_disagreement: list[Disagreement]
    negative_keywords: list[NegativeKeyword]
    actionable_insights: list[ActionableInsight]


class InsightsJobCreate(BaseModel):
    sample_id: int


class InsightsJobId(BaseModel):
    job_id: int


class InsightsJob(BaseModel):
    status: Literal["pending", "running", "done", "failed"]
    insight_id: int | None = None
    error: str | None = None
