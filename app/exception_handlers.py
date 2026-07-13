import logging

from fastapi import FastAPI, Request
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.reviews.appstore.errors import (
    AppNotFoundError,
    AppStoreCollectionError,
    AppStoreUnavailableError,
    InvalidCountryCodeError,
)

logger = logging.getLogger(__name__)


async def log_validation_errors(request: Request, exc: RequestValidationError):
    logger.info(
        "Invalid request %s %s: %s", request.method, request.url.path, exc.errors()
    )
    return await request_validation_exception_handler(request, exc)


async def handle_app_not_found(request: Request, exc: AppNotFoundError) -> JSONResponse:
    logger.warning("App not found for %s %s: %s", request.method, request.url.path, exc)
    return JSONResponse(status_code=404, content={"detail": str(exc)})


async def handle_invalid_country_code(
    request: Request, exc: InvalidCountryCodeError
) -> JSONResponse:
    logger.warning("Invalid country code for %s %s: %s", request.method, request.url.path, exc)
    return JSONResponse(status_code=400, content={"detail": str(exc)})


async def handle_app_store_unavailable(
    request: Request, exc: AppStoreUnavailableError
) -> JSONResponse:
    logger.error("App Store unavailable for %s %s: %s", request.method, request.url.path, exc)
    return JSONResponse(status_code=503, content={"detail": str(exc)})


async def handle_app_store_collection_error(
    request: Request, exc: AppStoreCollectionError
) -> JSONResponse:
    logger.error(
        "Unhandled App Store collection error for %s %s: %s",
        request.method,
        request.url.path,
        exc,
    )
    return JSONResponse(status_code=502, content={"detail": str(exc)})


def register_exception_handlers(app: FastAPI) -> None:
    app.add_exception_handler(RequestValidationError, log_validation_errors)
    app.add_exception_handler(AppNotFoundError, handle_app_not_found)
    app.add_exception_handler(InvalidCountryCodeError, handle_invalid_country_code)
    app.add_exception_handler(AppStoreUnavailableError, handle_app_store_unavailable)
    app.add_exception_handler(AppStoreCollectionError, handle_app_store_collection_error)
