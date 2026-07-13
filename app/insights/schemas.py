import uuid
from typing import Literal

from pydantic import BaseModel, Field

Sentiment = Literal["positive", "neutral", "negative"]
ActionableInsightsSource = Literal["gemini", "rule_based_fallback", "none"]

# Hard caps on model-influenced text; the service truncates before building the response.
THEME_MAX_LENGTH = 120
SUGGESTION_MAX_LENGTH = 1000
EXECUTIVE_SUMMARY_MAX_LENGTH = 600


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
    theme: str = Field(max_length=THEME_MAX_LENGTH)
    evidence_count: int = Field(ge=1)
    suggestion: str = Field(max_length=SUGGESTION_MAX_LENGTH)
    evidence_review_ids: list[uuid.UUID] = Field(
        description="Every review supporting this theme, not a truncated sample"
    )


class Insight(BaseModel):
    sample_id: uuid.UUID
    review_count: int = Field(ge=0)
    sentiment_distribution: SentimentDistribution
    sentiment_rating_disagreement: list[Disagreement]
    negative_keywords: list[NegativeKeyword]
    actionable_insights: list[ActionableInsight]
    executive_summary: str = Field(max_length=EXECUTIVE_SUMMARY_MAX_LENGTH)
    actionable_insights_source: ActionableInsightsSource
