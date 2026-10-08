# syntax=docker/dockerfile:1
#
# Chainlit UI image. Build from the REPOSITORY ROOT (it needs pyproject.toml + uv.lock):
#   docker build -f frontend/Dockerfile -t newsrag-ui .
#
# Installs ONLY the `frontend` dependency group (chainlit + httpx): no SQLAlchemy,
# LangGraph or Google SDK, because the UI just calls the API over HTTP.

FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.12.23 /uv /uvx /bin/

ENV PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app
RUN useradd --create-home --uid 1000 app

COPY pyproject.toml uv.lock ./
# --only-group: the frontend group and nothing else (not even the API's main deps).
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --only-group frontend

COPY frontend/ frontend/
# Chainlit writes its config (.chainlit/) and uploads (.files/) next to app.py.
RUN chown -R app:app /app/frontend

USER app
WORKDIR /app/frontend

ENV PORT=8000
EXPOSE 8000

# --headless: do not try to open a browser inside the container.
CMD exec chainlit run app.py --headless --host 0.0.0.0 --port ${PORT}
