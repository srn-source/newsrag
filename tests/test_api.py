"""Tests for the /ask HTTP layer: input validation, API key, response building.

None of these reach the graph, so they need no database and no LLM:
FastAPI rejects bad input (422) and a bad key (401) before the endpoint runs.
"""

import pytest
from fastapi.testclient import TestClient

from app.core.config import settings
from app.graph.nodes import NOT_FOUND
from app.main import app
from app.models.schemas import RetrievedChunk
from app.routers.ask import to_response

client = TestClient(app)


@pytest.fixture(autouse=True)
def no_api_key(monkeypatch) -> None:
    """Run every test with the API key check off, whatever .env says.
    The API key test below switches it on for itself."""
    monkeypatch.setattr(settings, "api_key", None)


def test_empty_question_is_rejected() -> None:
    assert client.post("/ask", json={"question": ""}).status_code == 422


def test_whitespace_only_question_is_rejected() -> None:
    assert client.post("/ask", json={"question": "    "}).status_code == 422


def test_top_k_out_of_range_is_rejected() -> None:
    assert client.post("/ask", json={"question": "hi", "top_k": 0}).status_code == 422
    assert client.post("/ask", json={"question": "hi", "top_k": 21}).status_code == 422


def test_unknown_mode_is_rejected() -> None:
    assert client.post("/ask", json={"question": "hi", "mode": "magic"}).status_code == 422


def test_api_key_is_required_when_configured(monkeypatch) -> None:
    monkeypatch.setattr(settings, "api_key", "secret-test-key")

    missing = client.post("/ask", json={"question": "hi"})
    wrong = client.post("/ask", json={"question": "hi"}, headers={"X-API-Key": "nope"})

    assert missing.status_code == 401
    assert wrong.status_code == 401


def chunk(chunk_id: int) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        title=f"Title {chunk_id}",
        source="TechCrunch",
        url=f"https://example.invalid/{chunk_id}",
        published_at="2023-10-01",
        content=f"Source: TechCrunch | Date: 2023-10-01 | Title: Title {chunk_id}\nBody {chunk_id}",
        score=0.5,
    )


def test_response_lists_only_the_cited_excerpts() -> None:
    state = {
        "answer": "Yes\nBoth outlets reported it [3][1].",
        "status": "answered",
        "chunks": [chunk(10), chunk(20), chunk(30)],
        "steps": ["plan:skip", "retrieve:1", "generate", "validate:ok"],
    }
    response = to_response(state, "baseline")

    assert [c.index for c in response.citations] == [1, 3]
    assert response.citations[1].title == "Title 30"
    assert response.citations[0].excerpt == "Body 10"  # header line removed


def test_refused_response_has_no_citations() -> None:
    state = {
        "answer": NOT_FOUND,
        "status": "refused",
        "chunks": [chunk(1)],
        "steps": ["plan:skip", "retrieve:1", "generate", "validate:fail"],
    }
    assert to_response(state, "baseline").citations == []
