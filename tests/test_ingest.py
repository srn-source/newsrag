"""Integration tests for ingestion against the real database (rolled back after each test).

Requires: docker compose up -d db   and   uv run alembic upgrade head
"""

from datetime import UTC, datetime

from sqlalchemy import func, select

from app.models.db import Chunk, Document
from app.services.embeddings import FakeEmbedding
from app.services.repository import DocumentRepository
from scripts.ingest import Article, ingest_article

# A url that cannot collide with a real dataset article.
URL = "https://example.invalid/ingest-test"


def make_article(body: str, title: str = "Test  title with  double  spaces ") -> Article:
    return Article(
        title=title,
        source="TestSource",
        url=URL,
        category="tech",
        author=None,
        published_at=datetime(2023, 11, 27, 8, 45, 59, tzinfo=UTC),
        body=body,
    )


LONG_BODY = " ".join(f"Sentence {i} of the article." for i in range(200))


async def count_chunks(session) -> int:
    stmt = select(func.count()).select_from(Chunk).join(Document).where(Document.url == URL)
    return await session.scalar(stmt)


async def test_ingest_stores_document_and_chunks(session) -> None:
    chunk_count = await ingest_article(session, FakeEmbedding(), make_article(LONG_BODY))

    assert chunk_count > 1
    assert await count_chunks(session) == chunk_count

    chunk = await session.scalar(
        select(Chunk).join(Document).where(Document.url == URL, Chunk.chunk_index == 0)
    )
    assert chunk.content.startswith("Source: TestSource | Date: 2023-11-27 | Title: ")
    assert chunk.embedding_model == "fake"


async def test_ingest_is_idempotent(session) -> None:
    first = await ingest_article(session, FakeEmbedding(), make_article(LONG_BODY))
    second = await ingest_article(session, FakeEmbedding(), make_article(LONG_BODY))

    assert first == second
    documents = await session.scalar(
        select(func.count()).select_from(Document).where(Document.url == URL)
    )
    assert documents == 1  # replaced, not duplicated
    assert await count_chunks(session) == first


async def test_title_is_stored_verbatim(session) -> None:
    # Evaluation joins on exact titles, so not even whitespace may change.
    title = "  Odd   spacing, kept as-is "
    await ingest_article(session, FakeEmbedding(), make_article(LONG_BODY, title=title))

    stored = await session.scalar(select(Document.title).where(Document.url == URL))
    assert stored == title


async def test_empty_body_stores_nothing(session) -> None:
    assert await ingest_article(session, FakeEmbedding(), make_article("")) == 0
    assert await count_chunks(session) == 0


async def test_exists_checks_the_embedding_model(session) -> None:
    await ingest_article(session, FakeEmbedding(), make_article(LONG_BODY))
    repo = DocumentRepository(session)

    assert await repo.exists(URL, "fake") is True
    # Same article embedded by another model counts as NOT done -> will be re-embedded.
    assert await repo.exists(URL, "text-multilingual-embedding-002") is False
