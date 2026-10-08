"""Shared test fixtures.

Database tests run against the REAL PostgreSQL (docker compose up -d db, plus
`alembic upgrade head`). SQLite cannot stand in: it has no pgvector and no
tsvector, which are exactly the parts worth testing.

Each test gets its own session and everything it writes is rolled back
afterwards, so tests never leave rows behind and never touch ingested data.
"""

from collections.abc import AsyncIterator

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from app.core.config import settings


@pytest.fixture
async def session() -> AsyncIterator[AsyncSession]:
    # NullPool + a fresh engine per test: pytest-asyncio gives each test its own
    # event loop, and pooled connections must not leak from one loop to another.
    engine = create_async_engine(settings.database_url, poolclass=NullPool)
    async with async_sessionmaker(engine, expire_on_commit=False)() as db_session:
        yield db_session
        await db_session.rollback()  # undo everything the test wrote
    await engine.dispose()
