"""Embedding providers — Factory pattern.

An embedding turns text into a list of numbers (a vector) so that texts with
similar meaning end up close together. We store one vector per chunk and, at
question time, look for the chunks whose vectors are closest to the question's.

Two providers, one interface:

    FakeEmbedding    deterministic, offline, free     -> tests and CI
    VertexEmbedding  real semantic vectors (Google)   -> ingestion, demo, evaluation

The rest of the app never creates a provider directly; it calls `create_embedder`.
Switching providers is therefore one line in `.env`: EMBEDDING_PROVIDER=fake|vertex
"""

import hashlib
import random
from typing import Protocol

from app.core.config import Settings
from app.models.db import EMBEDDING_DIM


class EmbeddingProvider(Protocol):
    """What every provider must offer. Any class with these members fits."""

    model_name: str  # stored next to each vector, so we always know what produced it

    async def embed(self, texts: list[str], is_query: bool = False) -> list[list[float]]:
        """Return one vector per input text, in the same order."""
        ...


class FakeEmbedding:
    """Pseudo-random vectors: the same text always gets the same vector.

    The vectors carry NO meaning — two similar sentences get unrelated vectors.
    Use it to test the plumbing (chunking, storage, SQL) without a network call
    or a GCP account. Never use it to measure retrieval quality.
    """

    model_name = "fake"

    def __init__(self, dim: int = EMBEDDING_DIM) -> None:
        self.dim = dim

    async def embed(self, texts: list[str], is_query: bool = False) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def _vector(self, text: str) -> list[float]:
        # Seed a random generator with a hash of the text -> deterministic output.
        seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16)
        rng = random.Random(seed)
        return [rng.uniform(-1.0, 1.0) for _ in range(self.dim)]


class VertexEmbedding:
    """Google Vertex AI embeddings, e.g. text-multilingual-embedding-002.

    "Multilingual" matters here: a Thai question and an English article about the
    same thing get similar vectors, so Thai questions can find English news.

    Authentication uses Application Default Credentials — no API key anywhere:
      - locally:   `gcloud auth application-default login`
      - Cloud Run: the service account attached to the service
    """

    def __init__(self, model: str, project: str, location: str, dim: int = EMBEDDING_DIM) -> None:
        # Imported here, not at the top of the file, so tests and CI that only
        # use FakeEmbedding never load the Google SDK.
        from google import genai

        self.model_name = model
        self.dim = dim
        self._client = genai.Client(vertexai=True, project=project, location=location)

    async def embed(self, texts: list[str], is_query: bool = False) -> list[list[float]]:
        from google.genai import types

        config = types.EmbedContentConfig(
            # Questions and passages are embedded slightly differently ("asymmetric"
            # retrieval): a short question should land near the passage that
            # ANSWERS it, not near passages that merely look like questions.
            task_type="RETRIEVAL_QUERY" if is_query else "RETRIEVAL_DOCUMENT",
            output_dimensionality=self.dim,
            # Our chunks are ~300 tokens, far below the model's input limit; this
            # only guards against a pathological input failing the whole batch.
            auto_truncate=True,
        )
        response = await self._client.aio.models.embed_content(
            model=self.model_name, contents=texts, config=config
        )
        vectors = [embedding.values for embedding in response.embeddings or []]
        if len(vectors) != len(texts):
            raise RuntimeError(f"asked for {len(texts)} embeddings, got {len(vectors)}")
        return vectors


def create_embedder(settings: Settings) -> EmbeddingProvider:
    """Factory: build the provider named in settings. The only place providers are created."""
    if settings.embedding_provider == "fake":
        return FakeEmbedding()

    if settings.embedding_provider == "vertex":
        if not settings.gcp_project:
            raise ValueError("GCP_PROJECT must be set when EMBEDDING_PROVIDER=vertex")
        return VertexEmbedding(
            model=settings.embedding_model,
            project=settings.gcp_project,
            location=settings.gcp_location,
        )

    raise ValueError(f"unknown embedding provider: {settings.embedding_provider}")
