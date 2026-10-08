"""Load the MultiHop-RAG news corpus into PostgreSQL.

For every article:  chunk -> embed -> store (document row + chunk rows with vectors)

Usage (database running, migration applied):
    uv run python -m scripts.ingest --limit 5     # quick check with 5 articles
    uv run python -m scripts.ingest               # all 609 articles
    uv run python -m scripts.ingest --force       # re-embed articles already stored

The embedding provider comes from .env (EMBEDDING_PROVIDER=fake|vertex).
Run with `fake` first: it is instant and free, and proves the pipeline works.

Safety properties:
  - Idempotent : re-running replaces an article (delete by url, then insert) and
                 never duplicates it.
  - Resumable  : articles already stored with the same embedding model are
                 skipped, so a re-run after a failure only does the missing work.
  - Isolated   : one transaction per article. Article 300 failing does not undo
                 the 299 before it, and does not stop the rest.
  - Embed first: vectors are computed BEFORE the old rows are deleted, so a
                 failed embedding call leaves the previous version intact.

Dataset: MultiHop-RAG (Tang & Yang, 2024, arXiv:2401.15391), ODC-BY license.
"""

import argparse
import asyncio
import sys
import time
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models.db import EMBEDDING_DIM, Chunk, Document, SessionLocal, engine
from app.services.chunker import chunk_article
from app.services.embeddings import EmbeddingProvider, create_embedder
from app.services.repository import DocumentRepository

DATASET = "yixuantt/MultiHopRAG"

# Texts sent to the embedding API per request. One article (~15 chunks) usually
# fits in a single request; this stays well under Vertex AI's per-request limits.
EMBED_BATCH_SIZE = 32


@dataclass
class Article:
    title: str
    source: str
    url: str
    category: str
    author: str | None
    published_at: datetime
    body: str


def load_articles() -> list[Article]:
    """Download the `corpus` subset from Hugging Face (cached after the first run)."""
    # Imported here so this module can be imported (e.g. by tests) without the
    # `ingest` dependency group, which only the ingestion image installs.
    from datasets import load_dataset

    rows = load_dataset(DATASET, "corpus", split="train")

    articles: dict[str, Article] = {}  # keyed by url -> duplicates collapse to one
    for row in rows:
        articles[row["url"]] = Article(
            # title is kept EXACTLY as given: evaluation matches it against the
            # dataset's gold evidence titles by exact string comparison.
            title=row["title"],
            source=row["source"],
            url=row["url"],
            category=row["category"],
            author=row["author"] or None,
            # e.g. "2023-11-27T08:45:59+00:00" -> timezone-aware datetime
            published_at=datetime.fromisoformat(row["published_at"]),
            body=row["body"],
        )
    return list(articles.values())


async def embed_in_batches(embedder: EmbeddingProvider, texts: list[str]) -> list[list[float]]:
    vectors: list[list[float]] = []
    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        vectors += await embedder.embed(texts[i : i + EMBED_BATCH_SIZE])
    return vectors


async def ingest_article(
    session: AsyncSession, embedder: EmbeddingProvider, article: Article
) -> int:
    """Store one article (replacing any previous version). Returns the chunk count.

    Does NOT commit — the caller owns the transaction. That keeps this function
    testable: a test can call it and then roll everything back.
    """
    contents = chunk_article(article.source, article.published_at, article.title, article.body)
    if not contents:
        return 0  # empty body: nothing to store

    # 1. Embed first. If this fails, nothing has been deleted yet.
    vectors = await embed_in_batches(embedder, contents)
    if any(len(vector) != EMBEDDING_DIM for vector in vectors):
        # Fail loudly and early instead of an obscure pgvector error at insert time.
        raise ValueError(
            f"embedding model returned {len(vectors[0])} dimensions, "
            f"but the database column is vector({EMBEDDING_DIM})"
        )

    # 2. Replace the article: delete the old version, insert the new one.
    repo = DocumentRepository(session)
    await repo.delete_by_url(article.url)

    document = Document(
        title=article.title,
        source=article.source,
        url=article.url,
        category=article.category,
        author=article.author,
        published_at=article.published_at,
    )
    document.chunks = [
        Chunk(
            chunk_index=index,
            content=content,
            embedding=vector,
            embedding_model=embedder.model_name,
        )
        for index, (content, vector) in enumerate(zip(contents, vectors, strict=True))
    ]
    await repo.add(document)
    return len(contents)


async def main(limit: int | None, force: bool) -> int:
    embedder = create_embedder(settings)
    print(f"Embedding provider: {settings.embedding_provider} (model: {embedder.model_name})")

    articles = load_articles()
    if limit:
        articles = articles[:limit]
    print(f"Articles to process: {len(articles)}")

    stored = skipped = empty = total_chunks = 0
    failed: list[str] = []
    started = time.perf_counter()

    for number, article in enumerate(articles, start=1):
        # A fresh session per article: its own transaction, and the database
        # connection goes back to the pool as soon as the article is done.
        async with SessionLocal() as session:
            try:
                repo = DocumentRepository(session)
                if not force and await repo.exists(article.url, embedder.model_name):
                    skipped += 1
                else:
                    chunk_count = await ingest_article(session, embedder, article)
                    if chunk_count == 0:
                        empty += 1
                    else:
                        await session.commit()
                        stored += 1
                        total_chunks += chunk_count
            except Exception as exc:  # one bad article must not stop the other 600
                await session.rollback()
                failed.append(article.url)
                print(f"  FAILED {article.url}: {exc}")

        if number % 25 == 0 or number == len(articles):
            elapsed = time.perf_counter() - started
            print(f"  {number}/{len(articles)} processed ({elapsed:.0f}s)")

    async with SessionLocal() as session:
        documents_in_db, chunks_in_db = await DocumentRepository(session).counts()
    await engine.dispose()

    print()
    print(f"Stored : {stored} articles, {total_chunks} chunks")
    print(f"Skipped: {skipped} (already stored with this model; use --force to redo)")
    print(f"Empty  : {empty} (no body text)")
    print(f"Failed : {len(failed)}")
    print(f"Database now holds {documents_in_db} documents and {chunks_in_db} chunks")
    print(f"Took {time.perf_counter() - started:.0f}s")

    return 1 if failed else 0  # non-zero exit code -> a Cloud Run Job shows as failed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest the MultiHop-RAG news corpus.")
    parser.add_argument("--limit", type=int, help="only process the first N articles")
    parser.add_argument("--force", action="store_true", help="re-embed already stored articles")
    args = parser.parse_args()
    sys.exit(asyncio.run(main(args.limit, args.force)))
