"""Unit tests for the embedding factory and the fake provider — no network, no database."""

import pytest

from app.core.config import Settings
from app.models.db import EMBEDDING_DIM
from app.services.embeddings import FakeEmbedding, create_embedder


async def test_fake_vectors_have_the_database_dimension() -> None:
    [vector] = await FakeEmbedding().embed(["hello"])
    assert len(vector) == EMBEDDING_DIM


async def test_fake_is_deterministic() -> None:
    embedder = FakeEmbedding()
    first = await embedder.embed(["same text"])
    second = await embedder.embed(["same text"])
    assert first == second


async def test_fake_gives_different_texts_different_vectors() -> None:
    a, b = await FakeEmbedding().embed(["text one", "text two"])
    assert a != b


async def test_one_vector_per_input_in_order() -> None:
    texts = ["a", "b", "c"]
    vectors = await FakeEmbedding().embed(texts)
    assert len(vectors) == 3
    assert vectors[1] == (await FakeEmbedding().embed(["b"]))[0]


def test_factory_returns_fake() -> None:
    settings = Settings(database_url="postgresql+asyncpg://x", embedding_provider="fake")
    assert isinstance(create_embedder(settings), FakeEmbedding)


def test_factory_vertex_requires_gcp_project() -> None:
    settings = Settings(
        database_url="postgresql+asyncpg://x", embedding_provider="vertex", gcp_project=""
    )
    with pytest.raises(ValueError, match="GCP_PROJECT"):
        create_embedder(settings)
