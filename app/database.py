from collections.abc import AsyncIterator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import text

from app.config import db_config


class Base(DeclarativeBase):
    pass


engine = create_async_engine(
    str(db_config.database_url),
    pool_size=10,
    max_overflow=10,
    pool_pre_ping=True,
    pool_recycle=1800,
)

async_session_factory = async_sessionmaker(engine, expire_on_commit=False)


async def get_db_session() -> AsyncIterator[AsyncSession]:
    async with async_session_factory() as session:
        yield session

async def test_db_connection() -> None:
    async with engine.connect() as conn:
        await conn.execute(text("SELECT 1"))