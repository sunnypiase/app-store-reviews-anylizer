import logging

import httpx

from app.reviews.appstore.errors import AppNotFoundError, InvalidCountryCodeError
from app.reviews.appstore.http import get_with_retry
from app.reviews.appstore.schemas import AppleLookupError, AppleLookupResponse

logger = logging.getLogger(__name__)

_LOOKUP_URL = "https://itunes.apple.com/lookup"


class AppStoreLookupClient:
    """Validates (app_id, country_code) via the iTunes Lookup API — the RSS feed
    itself returns 200 for nonexistent apps, so it can't validate."""

    def __init__(self, http: httpx.AsyncClient, *, max_retries: int, retry_base_delay: float) -> None:
        self._http = http
        self._max_retries = max_retries
        self._retry_base_delay = retry_base_delay

    async def verify_app_exists(self, app_id: int, country_code: str) -> None:
        logger.info("Verifying app %s exists in %s App Store", app_id, country_code)
        response = await get_with_retry(
            self._http,
            _LOOKUP_URL,
            params={"id": app_id, "country": country_code},
            max_attempts=self._max_retries,
            base_delay=self._retry_base_delay,
        )
        if response.status_code == 400:
            error = AppleLookupError.model_validate(response.json())
            logger.warning(
                "Invalid country code '%s' for app %s: %s", country_code, app_id, error.error_message
            )
            raise InvalidCountryCodeError(country_code, error.error_message)
        response.raise_for_status()
        parsed = AppleLookupResponse.model_validate(response.json())
        if parsed.result_count == 0:
            logger.warning("App %s not found in %s App Store", app_id, country_code)
            raise AppNotFoundError(app_id, country_code)
