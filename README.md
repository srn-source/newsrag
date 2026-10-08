# NewsRAG — multi-hop question answering over the news

Ask a question that needs evidence from **several news articles** and get a short
answer with **citations to the exact excerpts** it is based on. If the articles do
not contain the answer, the system says so instead of guessing.

Built with FastAPI, PostgreSQL + pgvector, LangGraph and Gemini on Vertex AI, and
deployed on Google Cloud Run with Cloud SQL.

| | |
|---|---|
| **Chat UI (live)** | `<UI_URL>` — login details are sent separately |
| **API docs (live)** | `<API_URL>/docs` — `POST /ask` needs the `X-API-Key` header, sent separately |
| **Demo video** | `<VIDEO_URL>` |
| **Architecture** | [ARCHITECTURE.md](ARCHITECTURE.md) |
| **Evaluation** | [EVALUATION.md](EVALUATION.md) |

```text
Q: Who is the individual associated with the cryptocurrency industry facing a criminal
   trial on fraud and conspiracy charges, as reported by both The Verge and TechCrunch?

A: Sam Bankman-Fried
   Both The Verge and TechCrunch report that Sam Bankman-Fried, the founder of FTX,
   is facing a criminal trial on charges including wire fraud and conspiracy [1][4] ...

   [1] The Verge   · 2023-09-28 · The FTX trial is bigger than Sam Bankman-Fried
   [4] TechCrunch  · 2023-10-01 · SBF's trial starts soon, but how did he — and FTX — get here?
```

## Highlights

- **Hybrid retrieval.** pgvector similarity search and PostgreSQL full-text search,
  merged with Reciprocal Rank Fusion. News questions mix described events with named
  outlets and people: embeddings catch the first, keywords the second.
- **A per-article cap found through error analysis.** One long article was filling
  7 of 8 excerpt slots on a question that needed three. Capping each article at 2
  chunks raised "all evidence found" from 41% to 60%. The change was re-checked on a
  held-out sample: 25 questions improved and none got worse.
- **Multi-hop query planning with LangGraph.** In `multihop` mode, Gemini splits the
  question into one search per outlet or event (`plan → retrieve → generate →
  validate`). Every step is recorded in the response.
- **Grounded answers.** Answers must cite excerpts. A deterministic `validate` step
  rejects answers with missing or invented citations. On unanswerable questions the
  system refused 25 of 25 times; the LLM alone invented an answer 7 times.
- **Measured, not assumed.** Retrieval and answer quality were evaluated on 100
  stratified questions with 95% confidence intervals and paired comparisons. The
  results are reported as measured, including the negative ones.
- **Production-minded deployment.** Stateless Cloud Run services, Cloud SQL, Secret
  Manager, least-privilege service accounts, a connection pool sized against Cloud
  SQL's connection limit, and bounded LLM timeouts.

## Results at a glance

Full report with methodology and limitations: [EVALUATION.md](EVALUATION.md).

| | LLM alone | RAG baseline | RAG multihop |
|---|---|---|---|
| Accuracy, all 100 questions | 0.55 [0.45, 0.64] | 0.55 [0.45, 0.64] | **0.63** [0.53, 0.72] |
| Comparison questions | 0.24 | 0.40 | **0.68** |
| Unanswerable questions correctly refused | 0.72 | **1.00** | **1.00** |
| Correct when it answers | 0.76 | 0.75 | **0.84** |

| Retrieval (75 answerable questions) | vector | keyword | **hybrid + cap** |
|---|---|---|---|
| All evidence articles found in top 10 | 0.31 | 0.40 | **0.60** |

**Honest reading:** RAG ties the LLM alone on overall accuracy. The LLM already knows
these famous 2023 stories and scores 0.96 on "which company..." inference questions
from memory. RAG wins clearly where the answer depends on what *specific articles* said
(comparison, temporal, unanswerable). The main weakness is over-refusal when only part
of the evidence is retrieved. [EVALUATION.md](EVALUATION.md) explains both and lists the
next improvements.

## How it works

```mermaid
flowchart LR
    U["User"] --> UI["Chainlit UI<br/>(Cloud Run)"]
    UI -->|"POST /ask"| API["FastAPI + LangGraph<br/>(Cloud Run)"]
    API -->|"hybrid search"| DB[("Cloud SQL<br/>PostgreSQL + pgvector")]
    API -->|"embed · plan · answer"| VX["Vertex AI<br/>Gemini + embeddings"]
    JOB["Cloud Run Job<br/>migrate + ingest"] --> DB
```

1. **Ingestion** (once, offline): 609 articles are split into ~1,000-character chunks.
   Each chunk is prefixed with `Source | Date | Title`, embedded, and stored with a
   generated full-text index.
2. **`POST /ask`** (online): `plan` (multihop only) → `retrieve` (hybrid + RRF, ≤2
   chunks per article) → `generate` (Gemini answers only from 8 numbered excerpts) →
   `validate` (citations must point at real excerpts) → answer + cited excerpts.

Design, GCP deployment, scaling and security are covered in
[ARCHITECTURE.md](ARCHITECTURE.md).

## Run it locally

**Prerequisites:** [uv](https://docs.astral.sh/uv/) and Docker Desktop. A Google Cloud
project is optional: without one, `fake` providers stand in for Vertex AI. The whole
pipeline runs end to end, but the answers are placeholders.

```bash
git clone <GITHUB_REPO_URL> && cd <repo>
cp .env.example .env                  # fake providers: no GCP account needed
uv sync                               # Python 3.12 + all dependencies from uv.lock

docker compose up -d db               # PostgreSQL 16 + pgvector
uv run alembic upgrade head           # create the extension, tables and indexes
uv run python -m scripts.ingest       # download the corpus, chunk, embed, store

uv run uvicorn app.main:app --port 8080
# → http://localhost:8080/docs
```

Chat UI, in a second terminal:

```bash
cd frontend
uv run chainlit run app.py --port 8000
# → http://localhost:8000
```

Tests (need the database from `docker compose up -d db`):

```bash
uv run pytest
uv run ruff check .
```

### With real models (Vertex AI)

```bash
gcloud auth application-default login       # credentials for local Python; no API keys in code
```

In `.env`, set `EMBEDDING_PROVIDER=vertex`, `LLM_PROVIDER=vertex` and
`GCP_PROJECT=<your-project>`. Then:

```bash
uv run python -m scripts.check_vertex       # 3 tiny calls: embedding, chat, structured output
uv run python -m scripts.ingest             # re-embeds every article with the real model
```

### Everything in containers

```bash
docker compose up --build                   # db + api (fake providers from .env)

# with Vertex AI: mount your local credentials read-only (see docker-compose.vertex.yml)
export GCLOUD_ADC_FILE=~/.config/gcloud/application_default_credentials.json
docker compose -f docker-compose.yml -f docker-compose.vertex.yml up --build -d api
docker compose -f docker-compose.yml -f docker-compose.vertex.yml --profile jobs run --rm ingest
```

## API

`POST /ask`

```json
{ "question": "Did TechCrunch and The Verge both report on the FTX trial?",
  "mode": "multihop",
  "top_k": 8 }
```

| field | |
|---|---|
| `question` | 1–1,000 characters. English or Thai: the search is multilingual. |
| `mode` | `multihop` (split the question first; default) or `baseline` (one search) |
| `top_k` | excerpts given to the LLM, 1–20 (default 8) |

Response:

| field | |
|---|---|
| `answer` | Line 1 is the short answer; then an explanation with `[n]` citations |
| `status` | `answered` or `refused`, so clients never parse the text |
| `citations` | Only the excerpts the answer cites. `index` matches `[n]`; also outlet, date, title, URL, excerpt |
| `steps` | Execution trace, e.g. `["plan:3", "retrieve:3", "generate", "validate:ok"]` |

Other endpoints:

| endpoint | |
|---|---|
| `GET /health` | Liveness check. Never touches the database. |
| `GET /ready` | Readiness check. Returns 503 if the database is unreachable. |

```bash
curl -X POST "$API_URL/ask" -H "X-API-Key: $API_KEY" -H "Content-Type: application/json" \
     -d '{"question": "Who is facing a fraud trial according to TechCrunch and The Verge?"}'
```

## Configuration

All settings come from environment variables (`.env` locally, Cloud Run variables and
Secret Manager in production). Nothing is hard-coded. See [.env.example](.env.example).

| variable | default | |
|---|---|---|
| `DATABASE_URL` | – | `postgresql+asyncpg://…`. On Cloud Run it comes from Secret Manager. |
| `EMBEDDING_PROVIDER` / `LLM_PROVIDER` | `fake` | `fake` for tests and CI, `vertex` for real models |
| `EMBEDDING_MODEL` | `text-multilingual-embedding-002` | 768 dimensions, matching the database column |
| `LLM_MODEL` | `gemini-3.8-flash` | |
| `LLM_THINKING_LEVEL` | `low` | `minimal` / `low` / `medium` / `high`. Higher is slower and costs more. |
| `LLM_TIMEOUT_SECONDS` / `LLM_MAX_RETRIES` | `40` / `1` | Bound each Gemini call: fail fast instead of hanging |
| `DEFAULT_MODE` | `multihop` | used when a request omits `mode` |
| `MAX_CHUNKS_PER_ARTICLE` | `2` | `0` = no limit |
| `DB_POOL_SIZE` / `DB_MAX_OVERFLOW` | `5` / `0` | per instance; sized against Cloud SQL's 25 connections |
| `API_KEY` | empty | when set, `/ask` requires `X-API-Key` |
| `GCP_PROJECT` / `GCP_LOCATION` | – / `global` | Vertex AI project and endpoint |

## Project structure

```text
app/
  core/config.py          settings from the environment (validated at startup)
  models/db.py            SQLAlchemy tables, engine and session
  models/schemas.py       API request/response and internal data shapes
  services/repository.py  Repository pattern: every SQL query lives here
  services/retriever.py   hybrid search, RRF, per-article cap
  services/embeddings.py  Factory: fake | Vertex AI embeddings
  services/llm.py         Factory: fake | Gemini chat model and planner
  services/chunker.py     chunking with source/date/title headers
  services/rag.py         assembles the graph (shared by the API and evaluation)
  graph/                  LangGraph state, nodes, prompts and wiring
  routers/                /health, /ready, /ask
scripts/
  ingest.py               idempotent, resumable ingestion (also the Cloud Run Job)
  check_vertex.py         verify the Vertex AI models in .env
  inspect_question.py     show sub-queries, excerpts and answer for one question
eval/                     evaluation sets, metrics, runner, saved results
frontend/                 Chainlit UI (its own image and Cloud Run service)
migrations/               Alembic (first migration hand-written: CREATE EXTENSION vector)
tests/                    unit tests, plus integration tests against real PostgreSQL
```

## Evaluation

```bash
uv run python -m eval.build_eval_set                 # 100 stratified questions, seed 42
uv run python -m eval.run_eval --limit 4             # smoke test, about 1 minute
uv run python -m eval.run_eval --tag final           # full run, about 1 hour, real models
uv run python -m eval.run_eval --report-only --tag final   # rebuild the tables, no API calls
```

See [EVALUATION.md](EVALUATION.md) for the method, results, error analysis and limitations.

## Limitations

- **The corpus is 609 articles from late 2023**, so recent events are out of scope by design.
- **The LLM already knows much of this news**, which inflates the LLM-alone baseline (see EVALUATION.md).
- **The system leans towards refusing** when only part of the evidence is retrieved.
- **Latency is 14–20 s per question (p50).** Reasoning models are slow, even at a low thinking level.
- **The demo deployment is a single-zone Cloud SQL instance with no backups.**
  ARCHITECTURE.md lists what production would change.

## Dataset and credits

[MultiHop-RAG](https://huggingface.co/datasets/yixuantt/MultiHopRAG): Yixuan Tang and
Yi Yang, *MultiHop-RAG: Benchmarking Retrieval-Augmented Generation for Multi-Hop
Queries*, 2024 ([arXiv:2401.15391](https://arxiv.org/abs/2401.15391)). Licensed under
ODC-BY.
