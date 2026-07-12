"""Collect ~N App Store reviews for one app, straight to JSON.

Reuses app.reviews.appstore.collector.AppStoreCollector directly -- no DB, no
job queue, no FastAPI involved. Run as:

    uv run python -m scripts.sentiment_eval.collect_reviews \
        --app-id 1459969523 --country us --limit 100
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

import httpx

from app.config import appstore_client_config
from app.reviews.appstore.collector import AppStoreCollector
from app.reviews.appstore.errors import (
    AppNotFoundError,
    AppStoreUnavailableError,
    InvalidCountryCodeError,
)
from app.reviews.appstore.lookup_client import AppStoreLookupClient
from app.reviews.appstore.reviews_client import AppStoreReviewsClient

DEFAULT_OUTPUT = Path(__file__).parent / "data" / "raw_reviews.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--app-id", type=int, default=1459969523)
    parser.add_argument("--country", default="us")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


async def collect(app_id: int, country_code: str, limit: int) -> list[dict]:
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


def main() -> None:
    args = parse_args()
    try:
        reviews = asyncio.run(collect(args.app_id, args.country, args.limit))
    except AppNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
    except InvalidCountryCodeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
    except AppStoreUnavailableError as exc:
        print(f"error: {exc} -- App Store API unreachable after retries", file=sys.stderr)
        sys.exit(1)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(reviews, indent=2))
    print(f"wrote {len(reviews)} reviews to {args.output}")


if __name__ == "__main__":
    main()
