import logging
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.config import appstore_client_config
from app.database import engine, test_db_connection
from app.logging_config import setup_logging
from app.insights.routes import insight_router
from app.metrics.routes import metric_router
from app.reports.routes import report_router
from app.reviews.appstore.errors import (
    AppNotFoundError,
    AppStoreCollectionError,
    AppStoreUnavailableError,
    InvalidCountryCodeError,
)
from app.reviews.routes import review_router

setup_logging()
logger = logging.getLogger(__name__)

API_V1 = "/api/v1"


@asynccontextmanager
async def lifespan(app: FastAPI):
    await test_db_connection()
    async with httpx.AsyncClient(
        timeout=appstore_client_config.request_timeout_seconds
    ) as http_client:
        app.state.http_client = http_client
        yield
    await engine.dispose()


app = FastAPI(lifespan=lifespan, title="App Store Review Analyzer", version="1.0.0")


@app.exception_handler(RequestValidationError)
async def log_validation_errors(request: Request, exc: RequestValidationError):
    logger.info(
        "Invalid request %s %s: %s", request.method, request.url.path, exc.errors()
    )
    return await request_validation_exception_handler(request, exc)


@app.exception_handler(AppNotFoundError)
async def handle_app_not_found(request: Request, exc: AppNotFoundError) -> JSONResponse:
    logger.warning("App not found for %s %s: %s", request.method, request.url.path, exc)
    return JSONResponse(status_code=404, content={"detail": str(exc)})


@app.exception_handler(InvalidCountryCodeError)
async def handle_invalid_country_code(
    request: Request, exc: InvalidCountryCodeError
) -> JSONResponse:
    logger.warning("Invalid country code for %s %s: %s", request.method, request.url.path, exc)
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(AppStoreUnavailableError)
async def handle_app_store_unavailable(
    request: Request, exc: AppStoreUnavailableError
) -> JSONResponse:
    logger.error("App Store unavailable for %s %s: %s", request.method, request.url.path, exc)
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.exception_handler(AppStoreCollectionError)
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


app.include_router(review_router, prefix=f"{API_V1}/reviews")
app.include_router(metric_router, prefix=f"{API_V1}/metrics")
app.include_router(insight_router, prefix=f"{API_V1}/insights")
app.include_router(report_router, prefix=f"{API_V1}/reports")
