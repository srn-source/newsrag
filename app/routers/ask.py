"""POST /ask — answer a question about the news, with citations.

This file only deals with HTTP: check the API key, run the graph, turn the
final graph state into the response, log one line. All the RAG logic lives in
app/graph and app/services.
"""

import json
import logging
import secrets
import time
import uuid
from functools import lru_cache
from typing import Annotated

from fastapi import APIRouter, Depends, Header, HTTPException

from app.core.config import settings
from app.graph.nodes import cited_numbers
from app.graph.state import RAGState
from app.models.schemas import AskRequest, AskResponse, Citation
from app.services.rag import build_rag_graph

logger = logging.getLogger(__name__)

router = APIRouter(tags=["rag"])

EXCERPT_CHARS = 300  # how much of each cited chunk to show in the response


def verify_api_key(x_api_key: Annotated[str | None, Header()] = None) -> None:
    """When API_KEY is set (on Cloud Run), require the header  X-API-Key: <value>.

    Stops a public URL from spending LLM quota. Locally API_KEY is empty -> no check.
    """
    if not settings.api_key:
        return
    # compare_digest takes the same time whether the first or the last character
    # differs, so the key cannot be guessed one character at a time from timings.
    if x_api_key is None or not secrets.compare_digest(
        x_api_key.encode("utf-8"), settings.api_key.encode("utf-8")
    ):
        raise HTTPException(status_code=401, detail="invalid or missing X-API-Key header")


@lru_cache
def get_graph():
    """Build the graph once per process, on the first request, then reuse it.

    The graph holds no per-request data: the retriever opens its own database
    session for each search, and the Vertex AI clients are safe to share.
    """
    return build_rag_graph(settings)


def to_response(state: RAGState, mode: str) -> AskResponse:
    """Final graph state -> API response. Only excerpts the answer actually cites
    are returned, numbered as in the answer text."""
    citations: list[Citation] = []
    if state["status"] == "answered":
        chunks = state["chunks"]
        for number in sorted(cited_numbers(state["answer"])):
            chunk = chunks[number - 1]  # validate already checked the number exists
            body = chunk.content.split("\n", 1)[-1]  # drop the "Source | Date | Title" line
            citations.append(
                Citation(
                    index=number,
                    title=chunk.title,
                    source=chunk.source,
                    url=chunk.url,
                    published_at=chunk.published_at,
                    excerpt=body[:EXCERPT_CHARS],
                    score=chunk.score,
                )
            )
    return AskResponse(
        answer=state["answer"],
        status=state["status"],
        mode=mode,
        citations=citations,
        steps=state["steps"],
    )


@router.post("/ask", response_model=AskResponse, dependencies=[Depends(verify_api_key)])
async def ask(request: AskRequest) -> AskResponse:
    mode = request.mode or settings.default_mode
    started = time.perf_counter()

    try:
        state = await get_graph().ainvoke(
            {"question": request.question, "mode": mode, "top_k": request.top_k}
        )
    except Exception as exc:
        # Vertex AI or the database failed. Log the details for us; give the client a
        # clear "try again" instead of a raw 500 with a stack trace.
        logger.exception("ask failed")
        raise HTTPException(
            status_code=503, detail="The question could not be answered right now. Try again."
        ) from exc

    response = to_response(state, mode)

    # One JSON line per request. On Cloud Run, stdout JSON becomes a structured log
    # entry, filterable by any field. The question text itself is NOT logged: user
    # input may contain personal data; its length is enough for debugging.
    logger.info(
        json.dumps(
            {
                "event": "ask",
                "request_id": str(uuid.uuid4()),
                "mode": mode,
                "status": response.status,
                "steps": response.steps,
                "question_chars": len(request.question),
                "chunks_retrieved": len(state.get("chunks", [])),
                "citations": len(response.citations),
                "llm_model": settings.llm_model,
                "latency_ms": round((time.perf_counter() - started) * 1000),
            }
        )
    )
    return response
