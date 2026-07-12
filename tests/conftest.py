import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from datetime import datetime, timezone

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.database import Base, get_db_session
from app.dependencies import get_http_client
from app.insights.dependencies import get_sentiment_classifier
from app.main import app as fastapi_app
from app.reviews import models as reviews_models  # noqa: F401 - registers tables on Base.metadata
from app.reviews.models import Review, ReviewSample

# A dedicated Postgres database, separate from the dev "reviews" database
# (see docs/ALEMBIC_GUIDE.md) — created once via
# `docker compose exec db psql -U postgres -c "CREATE DATABASE reviews_test;"`.
TEST_DATABASE_URL = "postgresql+asyncpg://postgres:postgres@localhost:5432/reviews_test"

test_engine = create_async_engine(TEST_DATABASE_URL)
db_session_factory = async_sessionmaker(test_engine, expire_on_commit=False)


@pytest.fixture(scope="session", autouse=True)
async def _schema():
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    yield
    await test_engine.dispose()


@pytest.fixture(autouse=True)
async def _clean_tables():
    yield
    async with test_engine.begin() as conn:
        await conn.execute(
            text("TRUNCATE reviews, review_samples RESTART IDENTITY CASCADE")
        )


@pytest.fixture
async def db_session() -> AsyncIterator[AsyncSession]:
    """A session bound to a single connection wrapped in an outer transaction
    that's rolled back at teardown. session.commit() inside code under test
    only commits a SAVEPOINT (join_transaction_mode="create_savepoint"), so
    the outer transaction survives multiple commits — fast, isolated tests
    with no data actually persisted (per docs/SQLALCHEMY_DEEP_DIVE.md §7).
    """
    async with test_engine.connect() as conn:
        async with conn.begin() as outer_transaction:
            session = AsyncSession(
                bind=conn, join_transaction_mode="create_savepoint", expire_on_commit=False
            )
            yield session
            await session.close()
            await outer_transaction.rollback()


@pytest.fixture
async def client(db_session: AsyncSession) -> AsyncIterator[AsyncClient]:
    """An HTTP client for the FastAPI app with its DB session swapped for
    `db_session` (rolled back per test). Deliberately skips ASGI lifespan
    (no real Postgres ping, no httpx.AsyncClient on app.state) — route tests
    that hit the App Store client must override `get_review_collector`
    themselves with a fake. `get_http_client` is stubbed out here too: it's
    a sub-dependency of `get_review_collector`, and FastAPI resolves a
    dependant's sub-dependencies from its original signature even when the
    dependant itself is overridden, so leaving it unset would still try
    (and fail) to read the lifespan-only `app.state.http_client`.

    `get_sentiment_classifier` defaults to `None` (VADER fallback) here too:
    `.env` has a real `GEMINI_API_KEY` for the eval scripts, so leaving this
    unstubbed would make every insights/reports test that doesn't care about
    Gemini fire a real network call. Tests exercising the Gemini path
    override it themselves with a fake.
    """
    async def _override_get_db_session() -> AsyncIterator[AsyncSession]:
        yield db_session

    fastapi_app.dependency_overrides[get_db_session] = _override_get_db_session
    fastapi_app.dependency_overrides[get_http_client] = lambda: None
    fastapi_app.dependency_overrides[get_sentiment_classifier] = lambda: None
    async with AsyncClient(transport=ASGITransport(app=fastapi_app), base_url="http://test") as ac:
        yield ac
    fastapi_app.dependency_overrides.clear()


@pytest.fixture
def make_sample(
    db_session: AsyncSession,
) -> Callable[[list[dict]], Awaitable[uuid.UUID]]:
    """Persists a ReviewSample with the given reviews (list of dicts with at
    least `rating`, `title`, `content`) into `db_session` and returns its id.
    """

    async def _make(reviews: list[dict]) -> uuid.UUID:
        sample = ReviewSample(app_id=123, country_code="us")
        sample.reviews = [
            Review(
                store_review_id=str(index),
                date=review.get("date", datetime.now(timezone.utc)),
                user_name=review.get("user_name", "u"),
                title=review["title"],
                content=review["content"],
                rating=review["rating"],
                app_version=review.get("app_version", "1.0"),
            )
            for index, review in enumerate(reviews)
        ]
        db_session.add(sample)
        await db_session.flush()
        return sample.id

    return _make
