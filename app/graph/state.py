"""The state that flows through the graph.

LangGraph passes one dict from node to node. Each node returns ONLY the keys it
changes; LangGraph merges them into the state. Think of it as a shared form that
every node fills in a part of.
"""

import operator
from typing import Annotated, TypedDict

from app.models.schemas import RetrievedChunk


class RAGState(TypedDict, total=False):  # total=False: keys appear as nodes fill them
    # --- input (set by the /ask endpoint) ---
    question: str
    mode: str  # "baseline" | "multihop"
    top_k: int

    # --- filled in by the nodes ---
    queries: list[str]  # plan: the searches to run (always includes the question itself)
    chunks: list[RetrievedChunk]  # retrieve: the excerpts the LLM will read
    answer: str  # generate / refuse / validate
    status: str  # "answered" | "refused"
    draft: str  # validate: the LLM's original answer, if validate rejected it

    # A reducer: instead of replacing the list, LangGraph APPENDS what each node
    # returns. So every node returns just its own step, e.g. ["generate"], and the
    # final state holds the full trace: ["plan:3", "retrieve:3", "generate", ...].
    steps: Annotated[list[str], operator.add]
