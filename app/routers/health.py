"""Health endpoints.

Two endpoints on purpose, because they answer different questions:

    GET /health   "Is the process alive?"      never touches the database
    GET /ready    "Can it serve real traffic?" checks the database

Why split them: if the database goes down, restarting the API container does not
fix anything. A liveness check that queried the database would make the platform
restart healthy containers in a loop. Readiness is where dependency checks belong.
"""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.db import get_session

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])

# FastAPI's recommended way to declare a dependency: the type says what you get,
# Depends(...) says how FastAPI obtains it.
SessionDep = Annotated[AsyncSession, Depends(get_session)]


@router.get("/health")
async def health() -> dict[str, str]:
    """Liveness: the process is running and can answer HTTP."""
    return {"status": "ok"}


@router.get("/ready")
async def ready(session: SessionDep) -> dict[str, str]:
    """Readiness: the database is reachable. Returns 503 if it is not."""
    try:
        await session.execute(text("SELECT 1"))
    except (SQLAlchemyError, OSError) as exc:
        # OSError covers "connection refused" before SQLAlchemy wraps anything.
        logger.warning("readiness check failed: %s", exc)
        raise HTTPException(status_code=503, detail="database unavailable") from exc
    return {"status": "ready"}
