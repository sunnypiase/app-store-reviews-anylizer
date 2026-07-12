"""Gemini-backed sentiment classification for the live insights endpoint.

Batching/retry approach ported from the eval tooling
(scripts/sentiment_eval/gemini_sentiment.py), adapted to the async
`google-genai` client so the insights pipeline stays non-blocking.
"""

import asyncio
import json
import logging
import random
import uuid
from typing import Literal

from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from pydantic import BaseModel

from app.insights.schemas import Sentiment
from app.reviews import schemas as review_schemas

logger = logging.getLogger(__name__)

_BATCH_SIZE = 150
_MAX_ATTEMPTS = 4
_BASE_DELAY_SECONDS = 10.0
_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}

# Only title/content are ever sent to Gemini -- never the star rating -- so
# the classifier can't take a shortcut that a real deployment (rating
# unknown at review time) wouldn't have.
_PROMPT_TEMPLATE = """You are classifying App Store review sentiment.
For each review below, decide whether the sentiment expressed in the title \
and content is "positive", "negative", or "neutral" (mixed or no clear \
opinion counts as neutral). Judge the text itself alone.

Return one entry per review, in the same order, each with the review's \
review_id and your sentiment label.

Reviews:
{reviews_block}
"""


class _ReviewSentiment(BaseModel):
    review_id: str
    sentiment: Literal["positive", "negative", "neutral"]


class GeminiClassificationError(Exception):
    """Gemini failed to classify a batch after retries, or returned a
    response that doesn't cover every review that was sent."""


def _render_batch(reviews: list[review_schemas.Review]) -> str:
    lines = []
    for review in reviews:
        lines.append(f"- id: {review.id}\n  title: {review.title}\n  content: {review.content}")
    return "\n".join(lines)


class GeminiSentimentClassifier:
    def __init__(self, client: genai.Client, model_name: str) -> None:
        self._client = client
        self._model_name = model_name

    async def classify(self, reviews: list[review_schemas.Review]) -> dict[uuid.UUID, Sentiment]:
        if not reviews:
            return {}
        batches = [
            reviews[start : start + _BATCH_SIZE] for start in range(0, len(reviews), _BATCH_SIZE)
        ]
        scored_batches = await asyncio.gather(*(self._score_batch(batch) for batch in batches))

        sentiments: dict[uuid.UUID, Sentiment] = {}
        for batch, scored in zip(batches, scored_batches, strict=True):
            expected_ids = {str(review.id) for review in batch}
            got_ids = {item.review_id for item in scored}
            if got_ids != expected_ids:
                raise GeminiClassificationError(
                    f"Gemini response ids {got_ids} don't match requested ids {expected_ids}"
                )
            for item in scored:
                sentiments[uuid.UUID(item.review_id)] = item.sentiment
        return sentiments

    async def _score_batch(
        self, reviews: list[review_schemas.Review]
    ) -> list[_ReviewSentiment]:
        prompt = _PROMPT_TEMPLATE.format(reviews_block=_render_batch(reviews))

        last_error: Exception | None = None
        for attempt in range(1, _MAX_ATTEMPTS + 1):
            try:
                response = await self._client.aio.models.generate_content(
                    model=self._model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=list[_ReviewSentiment],
                    ),
                )
                parsed = response.parsed
                if parsed is None:
                    parsed = [_ReviewSentiment(**item) for item in json.loads(response.text)]
                return parsed
            except genai_errors.APIError as exc:
                last_error = exc
                status = getattr(exc, "code", None)
                if status not in _RETRYABLE_STATUS_CODES or attempt == _MAX_ATTEMPTS:
                    raise GeminiClassificationError(f"Gemini batch scoring failed: {exc}") from exc
                delay = _BASE_DELAY_SECONDS * (2 ** (attempt - 1)) + random.uniform(0, 1)
                logger.warning(
                    "Gemini batch scoring failed (%s), retrying in %.1fs (attempt %d/%d)",
                    exc,
                    delay,
                    attempt,
                    _MAX_ATTEMPTS,
                )
                await asyncio.sleep(delay)
        raise GeminiClassificationError(
            f"Gemini batch scoring failed after {_MAX_ATTEMPTS} attempts"
        ) from last_error
