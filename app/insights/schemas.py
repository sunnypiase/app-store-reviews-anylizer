import uuid
from typing import Literal

from pydantic import BaseModel, Field

Sentiment = Literal["positive", "neutral", "negative"]


class SentimentDistribution(BaseModel):
    positive: int = Field(ge=0)
    neutral: int = Field(ge=0)
    negative: int = Field(ge=0)


class Disagreement(BaseModel):
    """A review whose text sentiment contradicts its star rating."""

    review_id: uuid.UUID
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
    example_review_ids: list[uuid.UUID]


class Insight(BaseModel):
    sample_id: uuid.UUID
    review_count: int = Field(ge=0)
    sentiment_distribution: SentimentDistribution
    sentiment_rating_disagreement: list[Disagreement]
    negative_keywords: list[NegativeKeyword]
    actionable_insights: list[ActionableInsight]
