"""Tests for retrieval: RRF and query terms (pure), and the two searches (real database).

Database tests insert their own rows inside a transaction that is rolled back,
so they work whether or not the corpus has been ingested.
"""

import math
from datetime import UTC, datetime

from app.models.db import EMBEDDING_DIM, Chunk, Document
from app.models.schemas import RetrievedChunk
from app.services.repository import DocumentRepository
from app.services.retriever import RRF_K, query_terms, reciprocal_rank_fusion

# --- RRF (pure) --------------------------------------------------------------------


def hit(chunk_id: int) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        title=f"title {chunk_id}",
        source="Test",
        url=f"https://example.invalid/{chunk_id}",
        published_at="2023-11-27",
        content=f"content {chunk_id}",
        score=0.0,
    )


def test_rrf_prefers_a_chunk_found_by_both_searches() -> None:
    vector = [hit(1), hit(2), hit(3)]
    keyword = [hit(3), hit(4)]
    fused = reciprocal_rank_fusion([vector, keyword], top_k=4)
    # Chunk 3 is only 3rd in one list but appears in both -> it wins.
    assert fused[0].chunk_id == 3
    assert {chunk.chunk_id for chunk in fused} == {1, 2, 3, 4}


def test_rrf_score_is_the_sum_of_reciprocal_ranks() -> None:
    fused = reciprocal_rank_fusion([[hit(1), hit(2)], [hit(2)]], top_k=2)
    by_id = {chunk.chunk_id: chunk.score for chunk in fused}
    assert math.isclose(by_id[2], 1 / (RRF_K + 2) + 1 / (RRF_K + 1))
    assert math.isclose(by_id[1], 1 / (RRF_K + 1))


def test_rrf_keeps_only_top_k() -> None:
    fused = reciprocal_rank_fusion([[hit(i) for i in range(10)]], top_k=3)
    assert [chunk.chunk_id for chunk in fused] == [0, 1, 2]


def test_rrf_of_nothing_is_empty() -> None:
    assert reciprocal_rank_fusion([[], []], top_k=5) == []


def from_article(chunk_id: int, article: str) -> RetrievedChunk:
    """A chunk belonging to `article` (chunks of one article share its URL)."""
    return RetrievedChunk(
        chunk_id=chunk_id,
        title=article,
        source="Test",
        url=f"https://example.invalid/{article}",
        published_at="2023-11-27",
        content=f"content {chunk_id}",
        score=0.0,
    )


# Five chunks of article A rank above everything else, like the Google lawsuit
# article that took 7 of 8 slots in evaluation.
CROWDED = [
    from_article(1, "A"),
    from_article(2, "A"),
    from_article(3, "A"),
    from_article(4, "A"),
    from_article(5, "A"),
    from_article(6, "B"),
    from_article(7, "C"),
]


def test_without_a_limit_one_article_can_take_every_slot() -> None:
    fused = reciprocal_rank_fusion([CROWDED], top_k=4)
    assert {chunk.title for chunk in fused} == {"A"}


def test_limit_per_article_makes_room_for_other_articles() -> None:
    fused = reciprocal_rank_fusion([CROWDED], top_k=4, max_per_article=2)
    assert [chunk.chunk_id for chunk in fused] == [1, 2, 6, 7]  # best 2 of A, then B, C


def test_limit_returns_fewer_than_top_k_when_candidates_run_out() -> None:
    fused = reciprocal_rank_fusion([CROWDED], top_k=10, max_per_article=1)
    assert [chunk.title for chunk in fused] == ["A", "B", "C"]


# --- query terms (pure) --------------------------------------------------------------


def test_query_terms_lowercases_drops_short_words_and_duplicates() -> None:
    terms = query_terms("Did TechCrunch report on Sam Bankman-Fried? TechCrunch did.")
    assert terms == ["did", "techcrunch", "report", "sam", "bankman", "fried"]


def test_query_terms_keeps_english_names_inside_a_thai_question() -> None:
    assert query_terms("TechCrunch รายงานว่าใครถูกฟ้องคดีฉ้อโกง") == ["techcrunch"]


def test_query_terms_of_pure_thai_is_empty() -> None:
    assert query_terms("ใครถูกฟ้องคดีฉ้อโกงคริปโต") == []


# --- repository searches (real database) -------------------------------------------


async def add_document(session, url: str, contents_and_vectors: list[tuple[str, list[float]]]):
    document = Document(
        title=f"Test {url}",
        source="TestSource",
        url=url,
        category="tech",
        published_at=datetime(2023, 11, 27, tzinfo=UTC),
    )
    document.chunks = [
        Chunk(chunk_index=i, content=content, embedding=vector, embedding_model="test")
        for i, (content, vector) in enumerate(contents_and_vectors)
    ]
    await DocumentRepository(session).add(document)


def one_hot(position: int) -> list[float]:
    """A vector pointing along one axis: unlike every real embedding."""
    vector = [0.0] * EMBEDDING_DIM
    vector[position] = 1.0
    return vector


async def test_keyword_search_matches_any_term(session) -> None:
    # Made-up words, so they cannot collide with the real corpus.
    await add_document(
        session,
        "https://example.invalid/kw",
        [("zzqbank zzqlicense", one_hot(0)), ("zzqinsure only", one_hot(1))],
    )
    hits = await DocumentRepository(session).keyword_search(["zzqbank", "zzqnothing"], top_k=5)
    # Found although "zzqnothing" appears nowhere -> terms are OR-ed, not AND-ed.
    assert [h.content for h in hits] == ["zzqbank zzqlicense"]
    assert hits[0].source == "TestSource"
    assert hits[0].published_at == "2023-11-27"


async def test_keyword_search_stems_words(session) -> None:
    await add_document(session, "https://example.invalid/stem", [("zzqwidgets", one_hot(0))])
    # The stored word is plural, the search word singular. They only match because
    # the 'english' configuration reduces both to the same stem.
    hits = await DocumentRepository(session).keyword_search(["zzqwidget"], top_k=5)
    assert [h.content for h in hits] == ["zzqwidgets"]


async def test_keyword_search_with_no_terms_returns_nothing(session) -> None:
    assert await DocumentRepository(session).keyword_search([], top_k=5) == []


async def test_vector_search_ranks_the_closest_vector_first(session) -> None:
    target = one_hot(5)
    await add_document(session, "https://example.invalid/vec", [("the target chunk", target)])

    hits = await DocumentRepository(session).vector_search(target, top_k=3)

    assert hits[0].content == "the target chunk"
    assert math.isclose(hits[0].score, 1.0, abs_tol=1e-6)  # identical vector -> similarity 1
    assert len(hits) <= 3
