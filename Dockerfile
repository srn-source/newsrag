# syntax=docker/dockerfile:1
#
# One Dockerfile, two images (build targets):
#
#   docker build --target api    -t newsrag-api .     # FastAPI service  -> Cloud Run service
#   docker build --target ingest -t newsrag-ingest .  # migrate + ingest -> Cloud Run Job
#
# Plain `docker build .` builds the LAST stage, which is `api`.
# Both images share the same code; only the installed dependency groups differ,
# so the API image does not carry `datasets` / `pandas`, which only ingestion uses.

# --- base: Python + uv, shared by both targets -----------------------------
FROM python:3.12-slim AS base

# Pin uv to the same version used locally, so `uv.lock` resolves identically.
COPY --from=ghcr.io/astral-sh/uv:0.12.23 /uv /uvx /bin/

ENV PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PATH="/app/.venv/bin:$PATH"
# PYTHONUNBUFFERED      : print/log lines reach Cloud Logging immediately
# UV_COMPILE_BYTECODE   : precompile .pyc at build time -> faster cold starts
# UV_PYTHON_DOWNLOADS   : use this image's Python, never download another
# PATH                  : run `uvicorn` / `alembic` from the venv without `uv run`

WORKDIR /app

# Unprivileged user: if the app is ever compromised, it is not root in the container.
RUN useradd --create-home --uid 1000 app


# --- ingest: Cloud Run Job (run migrations, then load the dataset) ----------
# Needs scripts/ingest.py, which arrives on Day 2. Building the `api` target does
# not touch this stage, so its absence does not break the API build.
FROM base AS ingest

# Dependencies first, code second: editing code then reuses the cached dependency
# layer instead of reinstalling ~180 packages on every build.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-default-groups --group ingest

COPY app/ app/
COPY migrations/ migrations/
COPY alembic.ini ./
COPY scripts/ scripts/

USER app
CMD ["sh", "-c", "alembic upgrade head && python -m scripts.ingest"]


# --- api: Cloud Run service --------------------------------------------------
FROM base AS api

COPY pyproject.toml uv.lock ./
# --frozen            : install exactly what uv.lock says; fail if it is out of date
# --no-default-groups : skip dev/ingest/frontend -> main runtime dependencies only
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-default-groups

COPY app/ app/

USER app

# Cloud Run tells the container which port to listen on via $PORT.
ENV PORT=8080
EXPOSE 8080

# Shell form so ${PORT} is expanded; `exec` replaces the shell with uvicorn, so
# uvicorn is PID 1 and receives Cloud Run's SIGTERM -> graceful shutdown, which
# runs the lifespan hook in app/main.py that closes the database pool.
CMD exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT}
