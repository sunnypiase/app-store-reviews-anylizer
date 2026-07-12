class AppStoreCollectionError(Exception):
    """Base for App Store collection failures that should be reported to the
    caller as a specific HTTP error, rather than a raw 500."""


class AppNotFoundError(AppStoreCollectionError):
    def __init__(self, app_id: int, country_code: str) -> None:
        super().__init__(f"App {app_id} was not found in the {country_code} App Store")


class InvalidCountryCodeError(AppStoreCollectionError):
    def __init__(self, country_code: str, apple_message: str) -> None:
        super().__init__(f"Invalid country code '{country_code}': {apple_message}")


class AppStoreUnavailableError(AppStoreCollectionError):
    """The RSS/Lookup API kept returning 403/429/5xx (or was unreachable)
    after all retries — transient; the caller may want to retry later."""
