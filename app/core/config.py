"""Application settings.

Every configurable value in the app is defined here and read from the
environment. Nothing is hardcoded anywhere else.

Where the values come from:
  - local dev : the `.env` file (git-ignored, never committed)
  - Cloud Run : environment variables, with DATABASE_URL injected from
                Secret Manager

`.env.example` is the committed template that documents every variable.
"""

from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

THINKING_LEVELS = {"minimal", "low", "medium", "high"}


class Settings(BaseSettings):
    """Validated settings. Reading an unknown value fails at startup, not mid-request."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # ignore env vars we don't declare, instead of crashing
    )

    # --- Database -----------------------------------------------------------
    # Must use the async driver, e.g.
    #   postgresql+asyncpg://rag:rag@localhost:5432/rag
    # No default on purpose: the app should refuse to start without a database.
    database_url: str

    # Connection pool, PER Cloud Run instance. The database sees the total:
    #   max connections used = Cloud Run max-instances x (pool_size + max_overflow)
    # Cloud SQL's smallest tier allows only 25 connections, so with the defaults
    # below and `--max-instances 3` we use at most 15, leaving room for the
    # ingest job, Alembic, and an admin psql session.
    db_pool_size: int = 5
    db_max_overflow: int = 0
    # Seconds a request waits for a free connection before failing. A short
    # timeout turns pool exhaustion into a fast, visible error instead of a hang.
    db_pool_timeout: int = 10

    # --- Embeddings ---------------------------------------------------------
    # "fake" produces meaningless vectors, but lets tests and CI run with no
    # GCP credentials. Never use "fake" when measuring evaluation numbers.
    embedding_provider: Literal["fake", "vertex"] = "fake"
    embedding_model: str = "text-multilingual-embedding-002"

    # --- LLM ----------------------------------------------------------------
    llm_provider: Literal["fake", "vertex"] = "fake"
    llm_model: str = "gemini-3.8-flash"
    # How long Gemini 3+ "thinks" before answering: minimal | low | medium | high.
    # Thinking tokens are slow AND billed as output. Reading 8 excerpts and giving a
    # short answer needs little of it: with the model default, one answer took ~40 s.
    # Leave empty to use the model's default (required for Gemini 2.5 models, which
    # do not accept this setting).
    llm_thinking_level: str | None = "low"
    # Bound every Gemini call. The library defaults are NO timeout and 6 retries
    # with back-off, so one slow or rate-limited call could hang a request for
    # minutes. With 40 s and no retry, multihop (2 calls) finishes or fails in
    # ~85 s, inside Cloud Run's 120 s request timeout; the client may retry.
    llm_timeout_seconds: float = Field(default=40, gt=0)
    # 1 = a single attempt. (In the Google SDK, 0 would mean "its default of 5".)
    llm_max_retries: int = Field(default=1, ge=1)

    # --- RAG behaviour ------------------------------------------------------
    # "baseline" = retrieve once.
    # "multihop" = split the question into sub-queries, then retrieve each.
    default_mode: Literal["baseline", "multihop"] = "multihop"

    # At most this many chunks from one article among the excerpts the LLM reads.
    # 0 = no limit. A multi-hop question needs evidence from 2-4 articles; without a
    # limit, one long article can fill most of the 8 slots.
    # 2 was measured on two independent 75-question samples: AllHit@10 +0.19 and
    # +0.15, 25 questions improved, none got worse (eval/results/ret-*, s7-*).
    max_chunks_per_article: int = Field(default=2, ge=0)

    # --- Auth ---------------------------------------------------------------
    # When set, /ask requires the header:  X-API-Key: <value>
    # Left empty locally; set on Cloud Run so a public URL can't burn LLM quota.
    api_key: str | None = None

    # --- GCP ----------------------------------------------------------------
    # Only needed when a provider above is "vertex".
    gcp_project: str | None = None
    # "global" because the newest Gemini models are served from the global endpoint
    # first (gemini-3.8-flash returned 404 in asia-southeast1). The embedding model
    # works there too. Trade-off: a regional endpoint keeps data in one region.
    gcp_location: str = "global"

    @field_validator("llm_thinking_level")
    @classmethod
    def _check_thinking_level(cls, value: str | None) -> str | None:
        if not value:
            return None  # empty in .env -> the model's default
        if value not in THINKING_LEVELS:
            raise ValueError(f"LLM_THINKING_LEVEL must be one of {sorted(THINKING_LEVELS)}")
        return value


# Import this object everywhere; it is built once when the module is first imported.
settings = Settings()
