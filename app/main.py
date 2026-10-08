"""FastAPI application entrypoint.

Run locally:   uv run uvicorn app.main:app --reload --port 8080
Then open:     http://localhost:8080/docs
"""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.models.db import engine
from app.routers import ask, health

# Print log records as bare messages. The /ask log line is already JSON, which
# Cloud Logging turns into a structured, searchable entry.
logging.basicConfig(level=logging.INFO, format="%(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Code before `yield` runs at startup, code after it runs at shutdown."""
    yield
    # Cloud Run sends SIGTERM before stopping an instance. Closing the pool here
    # releases this instance's database connections immediately, instead of
    # leaving them open on the Cloud SQL side until they time out.
    await engine.dispose()


app = FastAPI(
    title="NewsRAG",
    description="Multi-hop question answering over news articles, with citations.",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(health.router)
app.include_router(ask.router)
