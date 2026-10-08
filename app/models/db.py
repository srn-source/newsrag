"""Database tables and the async session factory.

Two tables:

    documents   one row per news article
    chunks      one row per ~1000-character slice of an article, plus its embedding

A chunk stores the text *and* its vector side by side, so a single SQL query can
rank by vector similarity and still return the text the LLM needs for citations.
"""

from collections.abc import AsyncGenerator
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import Computed, DateTime, ForeignKey, Index, Text, func
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from app.core.config import settings

# Length of one embedding vector. This MUST match the embedding model's output:
#   text-multilingual-embedding-002 -> 768
# Changing the model means changing this number and writing a new migration.
EMBEDDING_DIM = 768


class Base(DeclarativeBase):
    """Parent of every table. Alembic reads `Base.metadata` to generate migrations."""


class Document(Base):
    """One news article from the MultiHop-RAG corpus."""

    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(primary_key=True)

    # Stored EXACTLY as the dataset provides it. Evaluation matches the dataset's
    # gold evidence titles against this column by exact string comparison, so do
    # not strip, lowercase, or truncate it.
    title: Mapped[str] = mapped_column(Text)

    source: Mapped[str] = mapped_column(Text)  # news outlet, e.g. "Mashable"
    category: Mapped[str] = mapped_column(Text)  # e.g. "entertainment"
    author: Mapped[str | None] = mapped_column(Text, default=None)

    # Unique so ingestion can be re-run safely: delete the row with this url,
    # then insert the fresh version.
    url: Mapped[str] = mapped_column(Text, unique=True)

    # The dataset gives a timezone-aware ISO string ("2023-11-27T08:45:59+00:00"),
    # so the column must be timezone-aware too.
    published_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    # Deleting an article also deletes all of its chunks.
    chunks: Mapped[list["Chunk"]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )


class Chunk(Base):
    """A slice of one article, embedded so it can be found by vector search."""

    __tablename__ = "chunks"

    id: Mapped[int] = mapped_column(primary_key=True)

    document_id: Mapped[int] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    chunk_index: Mapped[int]  # position inside the article, starting at 0

    # The header ("Source: ... | Date: ... | Title: ...") followed by the text slice.
    # This one string is what we embed, what the LLM reads, and what we cite — keeping
    # it as a single column means those three can never disagree.
    content: Mapped[str] = mapped_column(Text)

    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIM))
    embedding_model: Mapped[str] = mapped_column(Text)  # which model produced the vector

    # The attribute is `meta` because `metadata` is a reserved name on SQLAlchemy
    # declarative classes. The database column is still called "metadata".
    meta: Mapped[dict] = mapped_column("metadata", JSONB, default=dict)

    # A GENERATED column: PostgreSQL fills it from `content` automatically, so we
    # never write to it and it can never drift out of sync. The 'english' config
    # gives us stemming ("running" -> "run") and stopword removal for free.
    search_vector = mapped_column(
        TSVECTOR, Computed("to_tsvector('english', content)", persisted=True)
    )

    document: Mapped["Document"] = relationship(back_populates="chunks")

    __table_args__ = (
        # GIN index is what makes keyword (full-text) search fast.
        Index("ix_chunks_search_vector", "search_vector", postgresql_using="gin"),
    )


# --- Engine and sessions ----------------------------------------------------

# One engine per process. It owns the connection pool, so it is created once at
# import time and shared by every request.
#
# Pool sizes are set explicitly, never left at SQLAlchemy's defaults (5 + 10
# overflow = up to 15 connections per instance). Cloud Run runs several instances,
# and every one of them gets its own pool — see db_pool_size in app/core/config.py.
engine = create_async_engine(
    settings.database_url,
    pool_size=settings.db_pool_size,
    max_overflow=settings.db_max_overflow,
    pool_timeout=settings.db_pool_timeout,
    pool_pre_ping=True,  # test a pooled connection before using it; Cloud SQL drops idle ones
)

# expire_on_commit=False lets us keep reading an object's attributes after commit
# without SQLAlchemy firing another SELECT.
SessionLocal = async_sessionmaker(engine, expire_on_commit=False)

# How to use sessions without starving the pool:
#
#   A session holds a pooled connection from its first query until it is closed.
#   A /ask request spends ~50 ms in the database and 2-5 s waiting for the LLM, so
#   a session that stays open for the whole request holds a connection ~99% idle.
#
#   Rule: open a session, run the queries, close it — BEFORE calling the LLM.
#   The retriever does exactly that with `async with SessionLocal() as session:`.


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency for short, database-only endpoints such as /ready.

    Do not use it for endpoints that call the LLM: the session (and its
    connection) would stay open until the response is sent.
    """
    async with SessionLocal() as session:
        yield session
