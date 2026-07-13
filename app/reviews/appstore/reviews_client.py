import asyncio
import logging
import random

import httpx

from app.reviews.appstore.http import get_with_retry
from app.reviews.appstore.schemas import AppleFeedEntry, AppleReviewsResponse

logger = logging.getLogger(__name__)

_REVIEWS_PAGE_SIZE = 50
_MAX_PAGE = 10


class AppStoreReviewsClient:
    """Paginates the RSS feed, stopping at `limit`, page 10 (Apple's hard cap),
    or a partial/empty page."""

    def __init__(
        self,
        http: httpx.AsyncClient,
        *,
        max_retries: int,
        retry_base_delay: float,
        throttle_delay: float,
    ) -> None:
        self._http = http
        self._max_retries = max_retries
        self._retry_base_delay = retry_base_delay
        self._throttle_delay = throttle_delay

    async def fetch_reviews(
        self, app_id: int, country_code: str, limit: int
    ) -> list[AppleFeedEntry]:
        logger.info(
            "Fetching up to %d reviews for app %s in %s App Store", limit, app_id, country_code
        )
        collected: list[AppleFeedEntry] = []
        for page in range(1, _MAX_PAGE + 1):
            if len(collected) >= limit:
                break
            url = (
                f"https://itunes.apple.com/{country_code}/rss/customerreviews/"
                f"page={page}/id={app_id}/sortby=mostrecent/json"
            )
            response = await get_with_retry(
                self._http, url, max_attempts=self._max_retries, base_delay=self._retry_base_delay
            )
            if response.status_code == 400:
                # Page-depth exceeded — expected end of pagination, not an error.
                logger.info(
                    "App %s: page-depth exceeded at page %d, stopping pagination", app_id, page
                )
                break
            response.raise_for_status()
            feed = AppleReviewsResponse.model_validate(response.json()).feed
            if not feed.entry:
                # No entry key at all, or an exhausted feed — valid, empty result.
                logger.info("App %s: page %d returned no reviews, stopping pagination", app_id, page)
                break
            collected.extend(feed.entry)
            logger.info(
                "App %s: fetched page %d (%d reviews, %d total so far)",
                app_id,
                page,
                len(feed.entry),
                len(collected),
            )
            if len(feed.entry) < _REVIEWS_PAGE_SIZE:
                break  # partial page — last page reached
            await asyncio.sleep(self._throttle_delay + random.uniform(0, self._throttle_delay))
        logger.info("App %s: collected %d reviews total", app_id, min(len(collected), limit))
        return collected[:limit]
