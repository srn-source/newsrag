"""LLM providers — Factory pattern, like embeddings.py.

Two objects are built here:

    chat model   writes the final answer               (graph node `generate`)
    planner      splits a question into sub-queries    (graph node `plan`)

Both come in a fake and a real (Gemini on Vertex AI) version, chosen by
LLM_PROVIDER in .env. Tests and CI use the fakes: no network, no GCP account, no cost.
"""

from langchain_core.language_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.runnables import Runnable

from app.core.config import Settings
from app.models.schemas import SubQueries

# What the fake chat model always answers. It cites [1] so it passes the `validate`
# node, which lets the whole graph run end to end in tests without a real LLM.
FAKE_ANSWER = "Fake answer\nThis placeholder comes from the fake LLM, not from a real model [1]."


class PassThroughPlanner:
    """Planner for LLM_PROVIDER=fake: proposes no sub-queries.

    The fake chat model cannot produce structured output, so multihop mode falls
    back to searching with the original question only.
    """

    async def ainvoke(self, prompt: str, /) -> SubQueries:
        return SubQueries(queries=[])


def create_chat_model(settings: Settings) -> BaseChatModel:
    """Factory: the only place a chat model is created."""
    if settings.llm_provider == "fake":
        return FakeListChatModel(responses=[FAKE_ANSWER])

    if settings.llm_provider == "vertex":
        if not settings.gcp_project:
            raise ValueError("GCP_PROJECT must be set when LLM_PROVIDER=vertex")

        # Imported here so tests and CI never load the Google SDK.
        # (ChatVertexAI from langchain-google-vertexai is deprecated; this is its
        # replacement. vertexai=True + project = Vertex AI with Application Default
        # Credentials: no API key in code, the service account on Cloud Run.)
        from langchain_google_genai import ChatGoogleGenerativeAI

        options = {}
        if settings.llm_thinking_level:  # None -> let the model use its default
            options["thinking_level"] = settings.llm_thinking_level

        return ChatGoogleGenerativeAI(
            model=settings.llm_model,
            vertexai=True,
            project=settings.gcp_project,
            location=settings.gcp_location,
            temperature=0,  # same question -> (nearly) same answer: needed for evaluation
            timeout=settings.llm_timeout_seconds,  # library default: wait forever
            max_retries=settings.llm_max_retries,  # library default: 6 retries
            **options,
        )

    raise ValueError(f"unknown llm provider: {settings.llm_provider}")


def create_planner(settings: Settings, chat_model: BaseChatModel) -> Runnable | PassThroughPlanner:
    """Factory for the planner used by the `plan` node."""
    if settings.llm_provider == "fake":
        return PassThroughPlanner()
    # with_structured_output makes the model return a SubQueries object instead of
    # free text, so the graph never has to parse a list out of a sentence.
    return chat_model.with_structured_output(SubQueries)
