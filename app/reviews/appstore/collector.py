import logging
from datetime import datetime

from pydantic import ValidationError

from app.reviews.appstore.lookup_client import AppStoreLookupClient
from app.reviews.appstore.reviews_client import AppStoreReviewsClient
from app.reviews.appstore.schemas import AppleFeedEntry
from app.reviews.collector import CollectedReview

logger = logging.getLogger(__name__)


class AppStoreCollector:
    """Validate via Lookup, paginate the RSS feed, map to CollectedReview;
    a malformed entry is skipped and logged, not fatal."""

    def __init__(self, lookup: AppStoreLookupClient, reviews: AppStoreReviewsClient) -> None:
        self._lookup = lookup
        self._reviews = reviews

    async def collect(
        self, app_id: int, country_code: str, limit: int
    ) -> list[CollectedReview]:
        await self._lookup.verify_app_exists(app_id, country_code)
        entries = await self._reviews.fetch_reviews(app_id, country_code, limit)

        results: list[CollectedReview] = []
        for entry in entries:
            try:
                results.append(_to_collected_review(entry))
            except (ValueError, ValidationError) as exc:
                logger.warning(
                    "Skipping malformed review entry %s: %s", entry.entry_id.label, exc
                )
        return results


def _to_collected_review(entry: AppleFeedEntry) -> CollectedReview:
    return CollectedReview(
        store_review_id=entry.entry_id.label,
        date=datetime.fromisoformat(entry.updated.label),
        user_name=entry.author.name.label,
        title=entry.title.label,
        content=entry.content.label,
        rating=int(entry.rating.label),
        app_version=entry.version.label,
    )
