"""Score the golden dataset with the Gemini API (free tier).

Setup:
1. Get a free API key at https://aistudio.google.com/apikey
2. Put it in .env at the repo root: GEMINI_API_KEY=...
   (optionally also GEMINI_MODEL_NAME, default: gemini-flash-lite-latest)

Reviews are sent in large batches (150/request) with a structured-JSON
response schema, so 1500 reviews costs 10 requests -- all fired concurrently
via a thread pool, since 10 concurrent requests is comfortably under the
free tier's 15-requests/minute cap for flash-lite models.

Usage:
    uv run python -m scripts.sentiment_eval.gemini_sentiment
"""

import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Literal

from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from pydantic import BaseModel

from scripts.sentiment_eval.config import gemini_config

DATA_DIR = Path(__file__).parent / "data"
GOLDEN_PATH = DATA_DIR / "golden_dataset.json"
OUTPUT_PATH = DATA_DIR / "predictions_gemini.json"

BATCH_SIZE = 150
MAX_ATTEMPTS = 4
BASE_DELAY_SECONDS = 10.0
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}

PROMPT_TEMPLATE = """You are classifying App Store review sentiment.
For each review below, decide whether the sentiment expressed in the title \
and content is "positive", "negative", or "neutral" (mixed or no clear \
opinion counts as neutral). Judge the text itself, not the star rating.

Return one entry per review, in the same order, each with the review's \
store_review_id and your sentiment label.

Reviews:
{reviews_block}
"""


class ReviewSentiment(BaseModel):
    store_review_id: str
    sentiment: Literal["positive", "negative", "neutral"]


def render_batch(reviews: list[dict]) -> str:
    lines = []
    for review in reviews:
        lines.append(
            f"- id: {review['store_review_id']}\n"
            f"  title: {review['title']}\n"
            f"  content: {review['content']}"
        )
    return "\n".join(lines)


def score_batch(client: genai.Client, reviews: list[dict]) -> list[ReviewSentiment]:
    prompt = PROMPT_TEMPLATE.format(reviews_block=render_batch(reviews))

    last_error: Exception | None = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = client.models.generate_content(
                model=gemini_config.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=list[ReviewSentiment],
                ),
            )
            parsed = response.parsed
            if parsed is None:
                parsed = [ReviewSentiment(**item) for item in json.loads(response.text)]
            return parsed
        except genai_errors.APIError as exc:
            last_error = exc
            status = getattr(exc, "code", None)
            if status not in RETRYABLE_STATUS_CODES or attempt == MAX_ATTEMPTS:
                raise
            delay = BASE_DELAY_SECONDS * (2 ** (attempt - 1)) + random.uniform(0, 1)
            print(f"  batch failed ({exc}), retrying in {delay:.1f}s (attempt {attempt}/{MAX_ATTEMPTS})")
            time.sleep(delay)
    raise RuntimeError(f"batch scoring failed after {MAX_ATTEMPTS} attempts") from last_error


def main() -> None:
    if not gemini_config.api_key:
        print(
            "error: GEMINI_API_KEY is not set.\n"
            "Get a free key at https://aistudio.google.com/apikey and add it to .env:\n"
            "  GEMINI_API_KEY=your-key-here",
            file=sys.stderr,
        )
        sys.exit(1)

    reviews = json.loads(GOLDEN_PATH.read_text())
    client = genai.Client(api_key=gemini_config.api_key)

    predictions: list[dict] = []
    if OUTPUT_PATH.exists():
        existing = json.loads(OUTPUT_PATH.read_text())
        golden_ids = {r["store_review_id"] for r in reviews}
        predictions = [p for p in existing if p["store_review_id"] in golden_ids]
        if predictions:
            print(f"resuming: {len(predictions)} predictions already on disk for this golden set")

    already_scored = {p["store_review_id"] for p in predictions}
    remaining = [r for r in reviews if r["store_review_id"] not in already_scored]

    chunks = [remaining[start : start + BATCH_SIZE] for start in range(0, len(remaining), BATCH_SIZE)]
    print(f"scoring {len(remaining)} remaining reviews in {len(chunks)} concurrent request(s)...")

    failures: list[Exception] = []
    with ThreadPoolExecutor(max_workers=len(chunks) or 1) as pool:
        futures = {pool.submit(score_batch, client, chunk): chunk for chunk in chunks}
        for future in as_completed(futures):
            chunk = futures[future]
            try:
                results = future.result()
            except Exception as exc:
                print(f"error: a batch of {len(chunk)} reviews failed permanently ({exc})", file=sys.stderr)
                failures.append(exc)
                continue
            predictions.extend(
                {"store_review_id": r.store_review_id, "predicted_sentiment": r.sentiment} for r in results
            )
            OUTPUT_PATH.write_text(json.dumps(predictions, indent=2))
            print(f"  batch done ({len(results)} reviews) -- {len(predictions)}/{len(reviews)} total so far")

    if failures:
        print(f"warning: {len(failures)} batch(es) failed -- rerun this script to retry them", file=sys.stderr)

    predicted_ids = {p["store_review_id"] for p in predictions}
    expected_ids = {r["store_review_id"] for r in reviews}
    if predicted_ids != expected_ids:
        missing = expected_ids - predicted_ids
        extra = predicted_ids - expected_ids
        print(f"warning: id mismatch -- missing={missing} extra={extra}", file=sys.stderr)

    print(f"wrote {len(predictions)} Gemini predictions to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
