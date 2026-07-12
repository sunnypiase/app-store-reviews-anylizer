"""Collect reviews for several apps in one run and merge them into a single
tagged JSON file, reusing the same collection path as collect_reviews.py
(app.reviews.appstore.collector.AppStoreCollector) -- just looped per app and
with app_id/app_name stamped onto each review so the merged golden dataset
can still be sliced back out per app.

Usage:
    uv run python -m scripts.sentiment_eval.collect_multi
"""

import asyncio
import json
from pathlib import Path

import httpx

from app.config import appstore_client_config
from app.reviews.appstore.collector import AppStoreCollector
from app.reviews.appstore.lookup_client import AppStoreLookupClient
from app.reviews.appstore.reviews_client import AppStoreReviewsClient

DEFAULT_OUTPUT = Path(__file__).parent / "data" / "raw_reviews.json"
COUNTRY = "us"
LIMIT_PER_APP = 500

APPS = [
    (1459969523, "Nebula: Spiritual Guidance"),
    (938003185, "Robinhood: Trade Anything"),
    (570060128, "Duolingo: Language Lessons"),
]


async def collect_one(app_id: int, country_code: str, limit: int) -> list[dict]:
    async with httpx.AsyncClient(
        timeout=appstore_client_config.request_timeout_seconds
    ) as http:
        collector = AppStoreCollector(
            lookup=AppStoreLookupClient(
                http,
                max_retries=appstore_client_config.max_retries,
                retry_base_delay=appstore_client_config.retry_base_delay_seconds,
            ),
            reviews=AppStoreReviewsClient(
                http,
                max_retries=appstore_client_config.max_retries,
                retry_base_delay=appstore_client_config.retry_base_delay_seconds,
                throttle_delay=appstore_client_config.throttle_delay_seconds,
            ),
        )
        reviews = await collector.collect(app_id, country_code, limit)
        return [r.model_dump(mode="json") for r in reviews]


async def collect_all() -> list[dict]:
    merged: list[dict] = []
    for app_id, app_name in APPS:
        print(f"collecting up to {LIMIT_PER_APP} reviews for {app_name} ({app_id})...")
        reviews = await collect_one(app_id, COUNTRY, LIMIT_PER_APP)
        for r in reviews:
            r["app_id"] = app_id
            r["app_name"] = app_name
        print(f"  got {len(reviews)} reviews")
        merged.extend(reviews)
    return merged


def main() -> None:
    reviews = asyncio.run(collect_all())
    DEFAULT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_OUTPUT.write_text(json.dumps(reviews, indent=2))
    print(f"wrote {len(reviews)} reviews total to {DEFAULT_OUTPUT}")


if __name__ == "__main__":
    main()
