"""Check that the Vertex AI models named in .env work, before ingesting or asking.

    uv run python -m scripts.check_vertex

Makes three tiny calls (costs a fraction of a cent):
    1. embed one sentence     -> model reachable, returns EMBEDDING_DIM numbers
    2. ask Gemini to say OK   -> chat model reachable
    3. plan one question      -> structured output works (used by multihop mode)

Run it after `gcloud auth application-default login` and setting GCP_PROJECT.
It tests the models in .env even if the providers there are still `fake`.
"""

import asyncio
import sys

from app.core.config import settings
from app.graph.nodes import MAX_SUB_QUERIES, PLAN_PROMPT
from app.models.db import EMBEDDING_DIM
from app.models.schemas import EXAMPLE_QUESTION
from app.services.embeddings import create_embedder
from app.services.llm import create_chat_model, create_planner

HINTS = """
Common causes:
  - Not logged in          -> gcloud auth application-default login
  - API not enabled        -> gcloud services enable aiplatform.googleapis.com
  - Model not found (404)  -> the model ID is retired or not offered in GCP_LOCATION;
                              try GCP_LOCATION=us-central1, or a newer model ID from
                              the Vertex AI Model Garden, in .env
"""


async def main() -> int:
    if not settings.gcp_project:
        print("GCP_PROJECT is empty in .env — set it to your project ID first.")
        return 1

    # Force the real providers for this check, whatever .env says.
    vertex = settings.model_copy(update={"embedding_provider": "vertex", "llm_provider": "vertex"})
    print(f"Project: {vertex.gcp_project}   Location: {vertex.gcp_location}\n")
    ok = True

    print(f"1. Embedding model: {vertex.embedding_model}")
    try:
        [vector] = await create_embedder(vertex).embed(["Vertex AI connectivity check"])
        if len(vector) == EMBEDDING_DIM:
            print(f"   OK — {len(vector)} dimensions")
        else:
            ok = False
            print(f"   WRONG SIZE — got {len(vector)}, database expects {EMBEDDING_DIM}")
    except Exception as exc:
        ok = False
        print(f"   FAILED — {exc}")

    print(f"2. Chat model: {vertex.llm_model}")
    chat_model = None
    try:
        chat_model = create_chat_model(vertex)
        reply = await chat_model.ainvoke("Reply with exactly one word: OK")
        print(f"   OK — replied: {str(reply.text).strip()!r}")
    except Exception as exc:
        ok = False
        print(f"   FAILED — {exc}")

    print("3. Structured output (planner)")
    if chat_model is None:
        ok = False
        print("   SKIPPED — the chat model could not be created")
    else:
        try:
            planner = create_planner(vertex, chat_model)
            result = await planner.ainvoke(
                PLAN_PROMPT.format(max_queries=MAX_SUB_QUERIES, question=EXAMPLE_QUESTION)
            )
            print(f"   OK — sub-queries: {result.queries}")
        except Exception as exc:
            ok = False
            print(f"   FAILED — {exc}")

    print("\nAll checks passed." if ok else HINTS)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
