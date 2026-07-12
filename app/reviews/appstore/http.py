import asyncio
import logging
import random

import httpx

from app.reviews.appstore.errors import AppStoreUnavailableError

logger = logging.getLogger(__name__)

# Apple's rate limiting is undocumented and IP-based, with no rate-limit
# headers to detect it proactively (docs/APPSTORE_RSS_RESEARCH.md) — treat
# 403/429 the same as a transient 5xx and retry with backoff+jitter.
RETRYABLE_STATUSES = {403, 429, 500, 502, 503, 504}


async def get_with_retry(
    client: httpx.AsyncClient,
    url: str,
    *,
    params: dict | None = None,
    max_attempts: int,
    base_delay: float,
) -> httpx.Response:
    response: httpx.Response | None = None
    last_exc: httpx.TransportError | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            response = await client.get(url, params=params)
        except httpx.TransportError as exc:
            last_exc = exc
            reason = f"transport error: {exc}"
        else:
            if response.status_code not in RETRYABLE_STATUSES:
                return response
            reason = f"HTTP {response.status_code}"
        if attempt < max_attempts:
            delay = base_delay * (2 ** (attempt - 1)) + random.uniform(0, base_delay)
            logger.warning(
                "Retrying App Store request to %s (attempt %d/%d) after %s; sleeping %.2fs",
                url,
                attempt,
                max_attempts,
                reason,
                delay,
            )
            await asyncio.sleep(delay)
    logger.error(
        "App Store request to %s failed after %d attempts (last: %s)",
        url,
        max_attempts,
        reason,
    )
    raise AppStoreUnavailableError(
        f"App Store request to {url} failed after {max_attempts} attempts: "
        f"{last_exc if last_exc is not None else reason}"
    )
