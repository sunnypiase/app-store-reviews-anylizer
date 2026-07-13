import logging
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI
from google import genai

from app.config import appstore_client_config, gemini_config
from app.database import engine, test_db_connection
from app.exception_handlers import register_exception_handlers
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
    app.state.gemini_client = (
        genai.Client(api_key=gemini_config.api_key) if gemini_config.api_key else None
    )
    async with httpx.AsyncClient(
        timeout=appstore_client_config.request_timeout_seconds
    ) as http_client:
        app.state.http_client = http_client
        yield
    if app.state.gemini_client is not None:
        await app.state.gemini_client.aio.aclose()
    await engine.dispose()


app = FastAPI(lifespan=lifespan, title="App Store Review Analyzer", version="1.0.0")

register_exception_handlers(app)

app.include_router(review_router, prefix=f"{API_V1}/reviews")
app.include_router(metric_router, prefix=f"{API_V1}/metrics")
app.include_router(insight_router, prefix=f"{API_V1}/insights")
app.include_router(report_router, prefix=f"{API_V1}/reports")
