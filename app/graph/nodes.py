"""The graph's nodes, and the prompts they use.

    plan      LLM     split the question into search queries (multihop mode only)
    retrieve  -       search every query, merge the results with RRF
    grade     -       choose the next node: generate, or refuse when nothing was found
    generate  LLM     answer from the excerpts, citing them as [1], [2], ...
    validate  -       check the citations point at real excerpts; reject the answer if not
    refuse    -       answer "Insufficient information."

Each node is a plain async function: state in, the keys it changes out.
The nodes never touch the database — they receive a retriever, an LLM and a
planner from workflow.py, which is what lets tests run them with stubs.
"""

import logging
import re
from typing import Any, Protocol

from langchain_core.language_models import BaseChatModel

from app.graph.state import RAGState
from app.models.schemas import RetrievedChunk
from app.services.retriever import reciprocal_rank_fusion

logger = logging.getLogger(__name__)

# Exactly the answer the dataset uses for unanswerable (null_query) questions,
# so evaluation can compare strings directly.
NOT_FOUND = "Insufficient information."

MAX_SUB_QUERIES = 3

# [1] .. [99]. Two digits at most, so a year in brackets like "[2023]" is not a citation.
CITATION = re.compile(r"\[(\d{1,2})\]")


class Retriever(Protocol):
    """Anything with this method: HybridRetriever in production, a stub in tests."""

    async def retrieve(self, query: str, top_k: int) -> list[RetrievedChunk]: ...


class Planner(Protocol):
    """Anything with this method: the structured-output LLM, or PassThroughPlanner."""

    async def ainvoke(self, prompt: str, /) -> Any: ...


# --- prompts -------------------------------------------------------------------

PLAN_PROMPT = """Break the question below into at most {max_queries} short English search \
queries for a news search engine.
- Write one query per news outlet, person, company or event the question mentions.
- Keep outlet names and dates in the queries.
- If the question is not in English, write the queries in English anyway.
- If the question is simple, return a single query.

Question: {question}"""

ANSWER_PROMPT = """You are a news question-answering assistant. Answer ONLY from the \
numbered news excerpts below. Each excerpt starts with its source, date and title.

Answer format:
- Line 1: the short answer only: a name, "Yes", "No", or exactly "{not_found}" if the \
excerpts do not contain the answer. If the question offers options (e.g. "consistent or \
inconsistent", "before or after"), answer with the option word itself, not "Yes" or "No".
- Line 2 onward: a brief explanation, written in the same language as the question.
- Cite the excerpts that support each claim, like [1] or [2][3]. Every answer must cite \
at least one excerpt, except "{not_found}", which cites nothing.

Excerpts:
{excerpts}

Question: {question}
Answer:"""


def format_excerpts(chunks: list[RetrievedChunk]) -> str:
    # Each chunk's content already starts with "Source: ... | Date: ... | Title: ...".
    return "\n\n".join(f"[{number}] {chunk.content}" for number, chunk in enumerate(chunks, 1))


def first_line(text: str) -> str:
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    return lines[0] if lines else ""


def is_refusal(answer: str) -> bool:
    return NOT_FOUND.lower() in first_line(answer).lower()


def cited_numbers(answer: str) -> set[int]:
    """The excerpt numbers an answer cites: "Yes [1][3]" -> {1, 3}."""
    return {int(number) for number in CITATION.findall(answer)}


# --- nodes -----------------------------------------------------------------------


async def plan(state: RAGState, planner: Planner) -> RAGState:
    question = state["question"]

    if state["mode"] == "baseline":  # the A/B switch: baseline skips planning entirely
        return {"queries": [question], "steps": ["plan:skip"]}

    try:
        result = await planner.ainvoke(
            PLAN_PROMPT.format(max_queries=MAX_SUB_QUERIES, question=question)
        )
        proposed = [query.strip() for query in result.queries]
    except Exception:  # a planning failure should degrade to baseline, not fail the request
        logger.warning("planner failed; searching with the original question only", exc_info=True)
        return {"queries": [question], "steps": ["plan:failed"]}

    extra = [query for query in proposed if query and query != question][:MAX_SUB_QUERIES]
    # The original question is ALWAYS searched too: if the planner splits badly,
    # the results of the plain question are still in the mix.
    queries = [question, *extra]
    return {"queries": queries, "steps": [f"plan:{len(queries)}"]}


async def retrieve(state: RAGState, retriever: Retriever, max_per_article: int = 0) -> RAGState:
    top_k = state["top_k"]
    results = [await retriever.retrieve(query, top_k) for query in state["queries"]]
    # Several queries -> several ranked lists -> merge with the same RRF used inside
    # the hybrid retriever. Excerpts found by several sub-queries rise to the top.
    # The per-article limit applies here too, or one article could again take
    # every slot after merging.
    chunks = (
        results[0]
        if len(results) == 1
        else reciprocal_rank_fusion(results, top_k, max_per_article=max_per_article)
    )
    return {"chunks": chunks, "steps": [f"retrieve:{len(results)}"]}


def grade(state: RAGState) -> str:
    """Conditional edge: which node runs after `retrieve`.

    Only a safety guard: vector search always returns rows when the table has data,
    so `chunks` is empty only if nothing was ingested. Deciding whether the excerpts
    actually answer the question is left to the LLM (it can answer NOT_FOUND) and
    to `validate`. A score threshold here would be meaningless: RRF scores of the
    top hit are always >= 1/61.
    """
    return "generate" if state["chunks"] else "refuse"


async def generate(state: RAGState, llm: BaseChatModel) -> RAGState:
    prompt = ANSWER_PROMPT.format(
        not_found=NOT_FOUND,
        excerpts=format_excerpts(state["chunks"]),
        question=state["question"],
    )
    message = await llm.ainvoke(prompt)
    # .text always gives a plain string, even when a model returns content blocks.
    answer = str(message.text).strip()
    status = "refused" if is_refusal(answer) else "answered"
    return {"answer": answer, "status": status, "steps": ["generate"]}


def validate(state: RAGState) -> RAGState:
    """Deterministic citation check — no LLM call.

    Catches an answer that cites nothing (the model answered from its own memory)
    or cites an excerpt that does not exist ("[9]" with 8 excerpts).
    It checks citation INTEGRITY, not faithfulness: it cannot tell whether excerpt
    [1] really supports the claim. That is measured offline, in evaluation.
    """
    if state["status"] != "answered":
        return {"steps": ["validate:skip"]}  # the LLM already refused

    cited = cited_numbers(state["answer"])
    if cited and all(1 <= number <= len(state["chunks"]) for number in cited):
        return {"steps": ["validate:ok"]}

    return {
        "draft": state["answer"],  # kept so evaluation can see what was rejected
        "answer": NOT_FOUND,
        "status": "refused",
        "steps": ["validate:fail"],
    }


def refuse(state: RAGState) -> RAGState:
    return {"answer": NOT_FOUND, "status": "refused", "steps": ["refuse"]}
