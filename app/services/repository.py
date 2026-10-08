"""Repository pattern: every SQL statement about documents and chunks lives here.

Rule of this project: if you find `select(`, `delete(` or raw SQL outside this
file, it is in the wrong place. Callers ask the repository for what they need
("delete this article", "count rows") and never build queries themselves.

Benefits:
  - One place to read, test and optimise the SQL.
  - Callers (ingestion script, retriever, API) stay free of database details,
    so the storage could change without rewriting them.

The repository never commits. The caller decides when a unit of work is complete
(e.g. ingestion commits once per article), so several repository calls can be
grouped into one transaction.

Write side (ingestion): exists, delete_by_url, add, counts
Read side (retrieval):  vector_search, keyword_search
"""

from typing import Any

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.db import Chunk, Document
from app.models.schemas import RetrievedChunk

# The columns a search returns. Deliberately NOT the whole Chunk row: that would also
# load each 768-number embedding, which the caller never needs.
_HIT_COLUMNS = (
    Chunk.id,
    Chunk.content,
    Document.title,
    Document.source,
    Document.url,
    Document.published_at,
)


def _to_retrieved(row: Any, score: float) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=row.id,
        title=row.title,
        source=row.source,
        url=row.url,
        published_at=row.published_at.date().isoformat(),
        content=row.content,
        score=float(score),
    )


class DocumentRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def exists(self, url: str, embedding_model: str) -> bool:
        """True if this article is already stored with vectors from this model.

        Lets ingestion resume after a failure without paying to re-embed articles
        that are already done.
        """
        stmt = (
            select(Chunk.id)
            .join(Document)
            .where(Document.url == url, Chunk.embedding_model == embedding_model)
            .limit(1)
        )
        return await self.session.scalar(stmt) is not None

    async def delete_by_url(self, url: str) -> None:
        """Delete an article. Its chunks go too, via ON DELETE CASCADE in the schema."""
        await self.session.execute(delete(Document).where(Document.url == url))

    async def add(self, document: Document) -> None:
        """Insert an article together with the chunks attached to `document.chunks`."""
        self.session.add(document)
        # flush = send the INSERTs now (still inside the transaction, not committed),
        # so a database error surfaces here, next to the code that caused it.
        await self.session.flush()

    async def counts(self) -> tuple[int, int]:
        """(number of documents, number of chunks) — printed after ingestion."""
        documents = await self.session.scalar(select(func.count()).select_from(Document))
        chunks = await self.session.scalar(select(func.count()).select_from(Chunk))
        return documents or 0, chunks or 0

    # --- read side ------------------------------------------------------------

    async def vector_search(self, embedding: list[float], top_k: int) -> list[RetrievedChunk]:
        """The `top_k` chunks whose embeddings point in the most similar direction.

        Cosine distance: 0 = same direction (same meaning), 2 = opposite.
        We return score = 1 - distance, so a higher score means more similar.
        No index is needed at this size: PostgreSQL compares against every chunk
        (~9,000), which is fast and always finds the true nearest neighbours.
        """
        distance = Chunk.embedding.cosine_distance(embedding).label("distance")
        stmt = (
            select(*_HIT_COLUMNS, distance)
            .select_from(Chunk)
            .join(Document)
            .order_by(distance)
            .limit(top_k)
        )
        rows = await self.session.execute(stmt)
        return [_to_retrieved(row, 1.0 - row.distance) for row in rows]

    async def keyword_search(self, terms: list[str], top_k: int) -> list[RetrievedChunk]:
        """The `top_k` chunks that contain the most of these words (full-text search).

        Terms are joined with OR: a chunk matching only some of the words still counts.
        (AND would demand every word of a long question, and almost nothing matches.)
        `websearch_to_tsquery` never raises on odd input, and applies the same
        'english' stemming as the search_vector column ("running" -> "run").
        """
        if not terms:
            return []  # e.g. a Thai question with no English words in it

        query = func.websearch_to_tsquery("english", " or ".join(terms))
        rank = func.ts_rank(Chunk.search_vector, query).label("rank")
        stmt = (
            select(*_HIT_COLUMNS, rank)
            .select_from(Chunk)
            .join(Document)
            .where(Chunk.search_vector.op("@@")(query))  # @@ = "matches", uses the GIN index
            .order_by(rank.desc())
            .limit(top_k)
        )
        rows = await self.session.execute(stmt)
        return [_to_retrieved(row, row.rank) for row in rows]
