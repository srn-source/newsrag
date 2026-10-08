"""Assemble the production RAG graph from settings.

Used by BOTH the /ask endpoint and the evaluation script, so the numbers in the
evaluation describe exactly the pipeline the API serves — not a look-alike copy.
"""

from app.core.config import Settings
from app.graph.workflow import build_graph
from app.models.db import SessionLocal
from app.services.embeddings import create_embedder
from app.services.llm import create_chat_model, create_planner
from app.services.retriever import HybridRetriever


def build_rag_graph(settings: Settings):
    embedder = create_embedder(settings)
    chat_model = create_chat_model(settings)
    limit = settings.max_chunks_per_article
    return build_graph(
        retriever=HybridRetriever(SessionLocal, embedder, max_per_article=limit),
        llm=chat_model,
        planner=create_planner(settings, chat_model),
        max_per_article=limit,
    )
