from typing import Annotated

from fastapi import Depends, Request
from google import genai

from app.config import GeminiConfig, gemini_config
from app.insights.gemini_classifier import GeminiSentimentClassifier
from app.insights.gemini_insight_generator import GeminiInsightGenerator


def _get_gemini_config() -> GeminiConfig:
    return gemini_config


def get_gemini_client(request: Request) -> genai.Client | None:
    return request.app.state.gemini_client


GeminiClientDep = Annotated[genai.Client | None, Depends(get_gemini_client)]


def get_sentiment_classifier(
    client: GeminiClientDep,
    config: Annotated[GeminiConfig, Depends(_get_gemini_config)],
) -> GeminiSentimentClassifier | None:
    if client is None:
        return None
    return GeminiSentimentClassifier(client, config.model_name)


def get_insight_generator(
    client: GeminiClientDep,
    config: Annotated[GeminiConfig, Depends(_get_gemini_config)],
) -> GeminiInsightGenerator | None:
    if client is None:
        return None
    return GeminiInsightGenerator(
        client,
        config.insights_model_name,
        config.insights_request_timeout_seconds,
    )


SentimentClassifierDep = Annotated[
    GeminiSentimentClassifier | None, Depends(get_sentiment_classifier)
]
InsightGeneratorDep = Annotated[GeminiInsightGenerator | None, Depends(get_insight_generator)]
