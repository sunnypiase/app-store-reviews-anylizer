from typing import Annotated

from fastapi import Depends
from google import genai

from app.config import GeminiConfig, gemini_config
from app.insights.gemini_classifier import GeminiSentimentClassifier


def _get_gemini_config() -> GeminiConfig:
    return gemini_config


def get_sentiment_classifier(
    config: Annotated[GeminiConfig, Depends(_get_gemini_config)],
) -> GeminiSentimentClassifier | None:
    if not config.api_key:
        return None
    return GeminiSentimentClassifier(genai.Client(api_key=config.api_key), config.model_name)


SentimentClassifierDep = Annotated[
    GeminiSentimentClassifier | None, Depends(get_sentiment_classifier)
]
