from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI, Request
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError

from app.database import engine, test_db_connection
from app.logging_config import setup_logging
from app.insights.routes import insight_router
from app.metrics.routes import metric_router
from app.reports.routes import report_router
from app.reviews.routes import review_router

setup_logging()
logger = logging.getLogger(__name__)

API_V1 = "/api/v1"


@asynccontextmanager
async def lifespan(app: FastAPI):
    await test_db_connection()
    yield
    await engine.dispose()


app = FastAPI(lifespan=lifespan, title="App Store Review Analyzer", version="1.0.0")


@app.exception_handler(RequestValidationError)
async def log_validation_errors(request: Request, exc: RequestValidationError):
    logger.info(
        "Invalid request %s %s: %s", request.method, request.url.path, exc.errors()
    )
    return await request_validation_exception_handler(request, exc)


app.include_router(review_router, prefix=f"{API_V1}/reviews")
app.include_router(metric_router, prefix=f"{API_V1}/metrics")
app.include_router(insight_router, prefix=f"{API_V1}/insights")
app.include_router(report_router, prefix=f"{API_V1}/reports")
