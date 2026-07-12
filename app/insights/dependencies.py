from typing import Annotated

from fastapi import Depends
from google import genai

from app.config import GeminiConfig, gemini_config
from app.insights.gemini_classifier import GeminiSentimentClassifier
from app.insights.gemini_insight_generator import GeminiInsightGenerator


def _get_gemini_config() -> GeminiConfig:
    return gemini_config


def get_sentiment_classifier(
    config: Annotated[GeminiConfig, Depends(_get_gemini_config)],
) -> GeminiSentimentClassifier | None:
    if not config.api_key:
        return None
    return GeminiSentimentClassifier(genai.Client(api_key=config.api_key), config.model_name)


def get_insight_generator(
    config: Annotated[GeminiConfig, Depends(_get_gemini_config)],
) -> GeminiInsightGenerator | None:
    if not config.api_key:
        return None
    return GeminiInsightGenerator(
        genai.Client(api_key=config.api_key),
        config.insights_model_name,
        config.insights_request_timeout_seconds,
    )


SentimentClassifierDep = Annotated[
    GeminiSentimentClassifier | None, Depends(get_sentiment_classifier)
]
InsightGeneratorDep = Annotated[GeminiInsightGenerator | None, Depends(get_insight_generator)]
