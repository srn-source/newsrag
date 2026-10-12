"""Hybrid retrieval: vector search + keyword search, merged with Reciprocal Rank Fusion.

Why two searches:
    vector search   finds the same MEANING in different words
                    ("automaker acquired" for "car company bought")
    keyword search  finds EXACT names the embedding may blur
                    ("TechCrunch", "Bankman-Fried")
News questions need both: they mix a described event with named outlets and people.

Why RRF to merge them: the two scores are on unrelated scales (cosine similarity vs
ts_rank), so they cannot be added. RRF ignores the scores and uses only each
chunk's RANK in each list. No weights to tune.
"""

import re
from collections import Counter
from dataclasses import replace

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.models.schemas import RetrievedChunk
from app.services.embeddings import EmbeddingProvider
from app.services.repository import DocumentRepository

# The standard RRF constant from the original paper (Cormack et al., 2009). It damps
# the difference between rank 1 and rank 2, so one list cannot dominate the other.
RRF_K = 60

# English words and numbers only. A Thai question still yields the English names
# people type inside it ("TechCrunch รายงานว่า..." -> ["techcrunch"]).
_WORD = re.compile(r"[A-Za-z0-9]+")


def reciprocal_rank_fusion(
    result_lists: list[list[RetrievedChunk]],
    top_k: int,
    k: int = RRF_K,
    max_per_article: int = 0,
) -> list[RetrievedChunk]:
    """Merge ranked lists: each appearance at rank r adds 1 / (k + r) to a chunk's score.

    A chunk ranked well in SEVERAL lists beats one ranked first in only one list:
        rank 3 in list A + rank 1 in list B = 1/63 + 1/61 = 0.0323
        rank 1 in list A only               = 1/61        = 0.0164

    max_per_article > 0 keeps at most that many chunks from any one article, so a
    single long article cannot fill every slot (seen in evaluation: 7 of 8 excerpts
    from one article on a question that needed three). 0 = no limit.

    Pure function (lists in, list out) -> easy to unit test.
    """
    scores: dict[int, float] = {}
    chunks: dict[int, RetrievedChunk] = {}
    for results in result_lists:
        for rank, chunk in enumerate(results, start=1):
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + 1.0 / (k + rank)
            chunks[chunk.chunk_id] = chunk

    best_first = sorted(scores, key=lambda chunk_id: scores[chunk_id], reverse=True)

    selected: list[RetrievedChunk] = []
    taken_per_article: Counter[str] = Counter()
    for chunk_id in best_first:
        url = chunks[chunk_id].url  # one URL = one article
        if max_per_article and taken_per_article[url] >= max_per_article:
            continue  # this article already has its share; give the slot to another
        taken_per_article[url] += 1
        selected.append(replace(chunks[chunk_id], score=scores[chunk_id]))
        if len(selected) == top_k:
            break
    return selected


def query_terms(text: str) -> list[str]:
    """Lower-cased English words longer than 2 letters, without duplicates, in order."""
    words = [word.lower() for word in _WORD.findall(text) if len(word) > 2]
    return list(dict.fromkeys(words))  # dict keeps first-seen order -> ordered de-duplication


class HybridRetriever:
    """Vector + keyword search for one query, fused with RRF."""

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        embedder: EmbeddingProvider,
        candidate_multiplier: int = 4,
        max_per_article: int = 0,
    ) -> None:
        # A session FACTORY, not a session: retrieve() opens its own short session and
        # closes it before returning, so no database connection is held while the
        # graph waits for the LLM.
        self.session_factory = session_factory
        self.embedder = embedder
        # Each search returns top_k x 4 candidates; fusion then keeps the best top_k.
        # A chunk ranked 12th by vector and 2nd by keyword must be visible to RRF.
        # The surplus also leaves room to skip chunks once an article hits its limit.
        self.candidate_multiplier = candidate_multiplier
        self.max_per_article = max_per_article

    async def retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]:
        candidates = top_k * self.candidate_multiplier

        # Embed BEFORE opening a session: it is a network call to Vertex AI and does
        # not need the database.
        [query_vector] = await self.embedder.embed([query], is_query=True)

        async with self.session_factory() as session:  # borrow a pooled connection
            repo = DocumentRepository(session)
            # One after the other, not asyncio.gather: a single AsyncSession must not
            # run two queries at the same time.
            vector_hits = await repo.vector_search(query_vector, candidates)
            keyword_hits = await repo.keyword_search(query_terms(query), candidates)
        # Leaving `async with` returned the connection to the pool.

        return reciprocal_rank_fusion(
            [vector_hits, keyword_hits], top_k, max_per_article=self.max_per_article
        )
