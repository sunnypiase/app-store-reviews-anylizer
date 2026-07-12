from typing import Annotated

from fastapi import Depends

from app.config import AppStoreClientConfig, appstore_client_config
from app.dependencies import HttpClientDep
from app.reviews.appstore.collector import AppStoreCollector
from app.reviews.appstore.lookup_client import AppStoreLookupClient
from app.reviews.appstore.reviews_client import AppStoreReviewsClient
from app.reviews.collector import ReviewCollector


def _get_appstore_client_config() -> AppStoreClientConfig:
    return appstore_client_config


def get_review_collector(
    http: HttpClientDep,
    config: Annotated[AppStoreClientConfig, Depends(_get_appstore_client_config)],
) -> ReviewCollector:
    return AppStoreCollector(
        lookup=AppStoreLookupClient(
            http, max_retries=config.max_retries, retry_base_delay=config.retry_base_delay_seconds
        ),
        reviews=AppStoreReviewsClient(
            http,
            max_retries=config.max_retries,
            retry_base_delay=config.retry_base_delay_seconds,
            throttle_delay=config.throttle_delay_seconds,
        ),
    )


CollectorDep = Annotated[ReviewCollector, Depends(get_review_collector)]
