"""Every path through the RAG graph, with stubs instead of the database and LLM.

No database, no network: each test builds the graph with a stub retriever, a
FakeListChatModel that returns a scripted answer, and a stub planner.
"""

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from app.graph.nodes import NOT_FOUND
from app.graph.workflow import build_graph
from app.models.schemas import RetrievedChunk, SubQueries

QUESTION = "Who is facing a fraud trial according to TechCrunch and The Verge?"

CHUNK = RetrievedChunk(
    chunk_id=1,
    title="FTX founder on trial",
    source="TechCrunch",
    url="https://example.invalid/ftx",
    published_at="2023-10-01",
    content="Source: TechCrunch | Date: 2023-10-01 | Title: FTX founder on trial\nThe trial...",
    score=1.0,
)


class StubRetriever:
    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self.chunks = chunks
        self.queries: list[str] = []  # records what the graph searched for

    async def retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]:
        self.queries.append(query)
        return self.chunks[:top_k]


class StubPlanner:
    def __init__(self, queries: list[str]) -> None:
        self.queries = queries

    async def ainvoke(self, prompt: str, /) -> SubQueries:
        return SubQueries(queries=self.queries)


class FailingPlanner:
    async def ainvoke(self, prompt: str, /) -> SubQueries:
        raise RuntimeError("LLM unavailable")


async def run(
    answer: str, mode: str = "baseline", chunks=(CHUNK,), planner=None, question=QUESTION
):
    retriever = StubRetriever(list(chunks))
    graph = build_graph(
        retriever=retriever,
        llm=FakeListChatModel(responses=[answer]),
        planner=planner or StubPlanner([]),
    )
    state = await graph.ainvoke({"question": question, "mode": mode, "top_k": 8})
    return state, retriever


async def test_baseline_answers_with_one_search() -> None:
    state, retriever = await run("Sam Bankman-Fried\nHe is on trial [1].")
    assert state["steps"] == ["plan:skip", "retrieve:1", "generate", "validate:ok"]
    assert state["status"] == "answered"
    assert retriever.queries == [QUESTION]


async def test_multihop_searches_the_question_plus_sub_queries() -> None:
    planner = StubPlanner(["FTX trial TechCrunch", "FTX trial The Verge"])
    state, retriever = await run("Sam Bankman-Fried\n[1]", mode="multihop", planner=planner)
    assert state["steps"] == ["plan:3", "retrieve:3", "generate", "validate:ok"]
    # The original question is always searched first.
    assert retriever.queries == [QUESTION, "FTX trial TechCrunch", "FTX trial The Verge"]


async def test_multihop_drops_blank_and_duplicate_sub_queries() -> None:
    planner = StubPlanner([QUESTION, "   ", "FTX trial"])
    state, retriever = await run("Sam Bankman-Fried\n[1]", mode="multihop", planner=planner)
    assert state["steps"][0] == "plan:2"
    assert retriever.queries == [QUESTION, "FTX trial"]


async def test_multihop_falls_back_to_the_question_when_the_planner_fails() -> None:
    state, retriever = await run(
        "Sam Bankman-Fried\n[1]", mode="multihop", planner=FailingPlanner()
    )
    assert state["steps"][0] == "plan:failed"
    assert retriever.queries == [QUESTION]
    assert state["status"] == "answered"


async def test_no_chunks_refuses_without_calling_the_llm() -> None:
    state, _ = await run("this answer must never be used", chunks=())
    assert state["steps"] == ["plan:skip", "retrieve:1", "refuse"]
    assert state["status"] == "refused"
    assert state["answer"] == NOT_FOUND


async def test_llm_refusal_is_reported_as_refused() -> None:
    state, _ = await run(f"{NOT_FOUND}\nThe excerpts do not say.")
    assert state["status"] == "refused"
    assert state["steps"][-1] == "validate:skip"


async def test_answer_without_citation_is_rejected() -> None:
    state, _ = await run("Sam Bankman-Fried\nEveryone knows this.")
    assert state["steps"][-1] == "validate:fail"
    assert state["status"] == "refused"
    assert state["answer"] == NOT_FOUND
    assert state["draft"].startswith("Sam Bankman-Fried")  # kept for evaluation


async def test_citation_of_a_missing_excerpt_is_rejected() -> None:
    state, _ = await run("Sam Bankman-Fried\nSee [9].")  # only 1 excerpt exists
    assert state["steps"][-1] == "validate:fail"


async def test_a_year_in_brackets_is_not_a_citation() -> None:
    state, _ = await run("Yes\nIn [2023] the trial began [1].")
    assert state["steps"][-1] == "validate:ok"


@pytest.mark.parametrize("mode", ["baseline", "multihop"])
async def test_thai_question_runs_through_the_same_graph(mode: str) -> None:
    state, _ = await run(
        "Sam Bankman-Fried\nเขาถูกฟ้องคดีฉ้อโกง [1]",
        mode=mode,
        question="TechCrunch รายงานว่าใครถูกฟ้องคดีฉ้อโกง",
    )
    assert state["status"] == "answered"
