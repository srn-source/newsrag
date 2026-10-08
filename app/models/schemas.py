"""Data shapes passed between layers and returned by the API.

RetrievedChunk   retrieval -> graph        plain data, no database objects
SubQueries       planner LLM -> graph      structured output the LLM must follow
AskRequest       client -> POST /ask       validated by FastAPI (bad input -> 422)
AskResponse      POST /ask -> client
"""

from dataclasses import dataclass
from typing import Annotated, Literal

from pydantic import BaseModel, Field, StringConstraints

EXAMPLE_QUESTION = (
    "Who is the individual associated with the cryptocurrency industry facing a criminal "
    "trial on fraud and conspiracy charges, as reported by both The Verge and TechCrunch, "
    "and is accused by prosecutors of committing fraud for personal gain?"
)


@dataclass(frozen=True)
class RetrievedChunk:
    """One search hit. Frozen: retrieval results are facts, nobody should edit them."""

    chunk_id: int
    title: str
    source: str
    url: str
    published_at: str  # "YYYY-MM-DD"
    content: str  # header + text: exactly what the LLM reads
    score: float  # higher = more relevant (meaning depends on the search that produced it)


class SubQueries(BaseModel):
    """What the planner LLM must return. LangChain turns this class into a JSON schema."""

    queries: list[str] = Field(
        default_factory=list,
        description="Up to 3 short English search queries, one per outlet, person or event.",
    )


class AskRequest(BaseModel):
    question: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=1000)
    ]
    mode: Literal["baseline", "multihop"] | None = Field(
        default=None,
        description="baseline = search once; multihop = split the question first. "
        "Empty = the server default.",
    )
    top_k: int = Field(
        default=8,
        ge=1,
        le=20,
        description="How many excerpts to give the LLM. Multi-hop questions need several.",
    )

    model_config = {"json_schema_extra": {"examples": [{"question": EXAMPLE_QUESTION}]}}


class Citation(BaseModel):
    index: int = Field(description="The [n] number used in the answer text.")
    title: str
    source: str
    url: str
    published_at: str
    excerpt: str
    score: float


class AskResponse(BaseModel):
    answer: str
    # Clients branch on this field instead of parsing the answer text.
    status: Literal["answered", "refused"]
    mode: str
    citations: list[Citation]
    steps: list[str] = Field(description="Which graph nodes ran, in order (execution trace).")
