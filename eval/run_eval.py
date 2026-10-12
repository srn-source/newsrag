"""Evaluate retrieval and answers on eval/queries.jsonl.

    uv run python -m eval.run_eval --limit 4          # smoke test: 1 question per type (~1 min)
    uv run python -m eval.run_eval                    # all 100 questions (~20-30 min)
    uv run python -m eval.run_eval --only retrieval   # retrieval table only: no Gemini calls
    uv run python -m eval.run_eval --report-only      # rebuild the tables from saved results: free
    uv run python -m eval.run_eval --tag v1           # save under eval/results/v1/ (default: latest)

Every score is reported with a 95% confidence interval, and modes / strategies are
compared with paired-bootstrap intervals of their difference (see eval/metrics.py).

Needs EMBEDDING_PROVIDER=vertex and LLM_PROVIDER=vertex: fake vectors and fake
answers give meaningless numbers. (--allow-fake runs anyway, to test the script.)

Two experiments:

  Retrieval  vector vs keyword vs hybrid: do the top 10 chunks contain the gold
             evidence articles? No LLM involved.
  Answers    llm_only vs baseline vs multihop: is line 1 of the answer correct?
             llm_only asks Gemini WITHOUT retrieval. If it scores as well as RAG,
             the model already knew these (famous, 2023) stories and RAG adds little.

Writes eval/results/<tag>/:
  retrieval.jsonl, generation.jsonl   one line per question x strategy/mode,
                                      to inspect individual failures
  summary.md                          the tables, with N, seed, models and thinking level
"""

import argparse
import asyncio
import json
import statistics
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from app.core.config import settings
from app.graph.nodes import NOT_FOUND, is_refusal
from app.models.db import SessionLocal, engine
from app.services.embeddings import create_embedder
from app.services.llm import create_chat_model
from app.services.rag import build_rag_graph
from app.services.repository import DocumentRepository
from app.services.retriever import HybridRetriever, query_terms
from eval.metrics import (
    BOOTSTRAP_ITERATIONS,
    bootstrap_mean_ci,
    evidence_recall,
    is_correct,
    paired_bootstrap_diff,
    retrieval_scores,
    token_f1,
    unique_urls,
    wilson_interval,
)

HERE = Path(__file__).parent
QUERIES = HERE / "queries.jsonl"
RESULTS = HERE / "results"

TYPES = ["inference_query", "comparison_query", "temporal_query", "null_query"]
SHORT = {
    "inference_query": "inference",
    "comparison_query": "comparison",
    "temporal_query": "temporal",
    "null_query": "null",
}
MODES = ["llm_only", "baseline", "multihop"]
STRATEGIES = ["vector", "keyword", "hybrid"]
RETRIEVAL_K = 10  # chunks compared against the gold evidence articles
TOP_K = 8  # same as the API default: excerpts the LLM reads
RETRIES = 3

LLM_ONLY_PROMPT = """Answer the question from your own knowledge. Do not invent facts.

Answer format:
- Line 1: the short answer only: a name, "Yes", "No", or exactly "{not_found}" if you \
do not know. If the question offers options (e.g. "consistent or inconsistent", "before \
or after"), answer with the option word itself, not "Yes" or "No".
- Line 2 onward: a brief explanation.

Question: {question}
Answer:"""


# --- helpers ---------------------------------------------------------------------


async def with_retries(func, *args, **kwargs):
    """Call `await func(...)`, retrying transient failures (e.g. 429 rate limits)."""
    for attempt in range(1, RETRIES + 1):
        try:
            return await func(*args, **kwargs)
        except Exception as exc:
            if attempt == RETRIES:
                raise
            wait = 2**attempt  # 2 s, then 4 s
            print(f"    retry {attempt}/{RETRIES - 1} in {wait}s — {type(exc).__name__}: {exc}"[:200])
            await asyncio.sleep(wait)


def load_questions(limit: int | None) -> list[dict]:
    if not QUERIES.exists():
        sys.exit(f"{QUERIES} not found — run:  uv run python -m eval.build_eval_set")
    with QUERIES.open(encoding="utf-8") as file:
        questions = [json.loads(line) for line in file if line.strip()]
    return questions[:limit] if limit else questions


def progress(number: int, total: int, label: str) -> None:
    if number % 10 == 0 or number == total:
        print(f"  {label}: {number}/{total}")


def mean(values) -> float | None:
    present = [float(value) for value in values if value is not None]
    return statistics.fmean(present) if present else None


def fmt(value: float | None, digits: int = 2) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def with_interval(value: float | None, interval: tuple[float, float] | None, digits: int = 2) -> str:
    """0.8, (0.71, 0.87) -> '0.80 [0.71, 0.87]'"""
    if value is None:
        return "-"
    if interval is None:
        return fmt(value, digits)
    low, high = interval
    return f"{value:.{digits}f} [{low:.{digits}f}, {high:.{digits}f}]"


def proportion_cell(flags: list[bool]) -> str:
    """Accuracy-style cell (share of True) with its Wilson 95% interval."""
    if not flags:
        return "-"
    return with_interval(sum(flags) / len(flags), wilson_interval(sum(flags), len(flags)))


def mean_cell(values: list[float], digits: int = 2) -> str:
    """Average of any scores (recall, MRR) with a bootstrap 95% interval."""
    return with_interval(mean(values), bootstrap_mean_ci(values), digits)


def difference_table(
    records: list[dict],
    group_field: str,
    comparisons: list[tuple[str, str]],
    metrics: list[tuple[str, str]],
) -> str:
    """Paired comparison: is A really better than B on the SAME questions?

    For each (metric, A vs B): the mean difference A - B over the questions both
    were run on, and its paired-bootstrap 95% interval. An interval that contains 0
    means this sample cannot tell the two apart.
    """
    rows = []
    for label, field in metrics:
        for a, b in comparisons:
            scores_a = {r["id"]: float(r[field]) for r in records if r[group_field] == a}
            scores_b = {r["id"]: float(r[field]) for r in records if r[group_field] == b}
            shared = sorted(scores_a.keys() & scores_b.keys())
            if not shared:
                continue
            difference, low, high = paired_bootstrap_diff(
                [scores_a[i] for i in shared], [scores_b[i] for i in shared]
            )
            if low > 0:
                verdict = f"{a} better"
            elif high < 0:
                verdict = f"{b} better"
            else:
                verdict = "no clear difference"
            rows.append(
                [label, f"{a} - {b}", str(len(shared)), f"{difference:+.2f}",
                 f"[{low:+.2f}, {high:+.2f}]", verdict]
            )
    return markdown_table(
        ["metric", "comparison", "n", "difference", "95% CI", "verdict"], rows
    )


# --- experiment 1: retrieval -------------------------------------------------------


async def evaluate_retrieval(questions: list[dict], embedder) -> list[dict]:
    """Score each search strategy on the questions that have gold evidence.

    vector and keyword are called directly; hybrid goes through HybridRetriever,
    i.e. the exact code the API uses (which fetches 4x candidates before fusing).
    """
    hybrid = HybridRetriever(
        SessionLocal, embedder, max_per_article=settings.max_chunks_per_article
    )
    answerable = [question for question in questions if question["evidence"]]
    records: list[dict] = []

    for number, question in enumerate(answerable, start=1):
        gold = {item["url"] for item in question["evidence"]}
        text = question["question"]

        [vector] = await with_retries(embedder.embed, [text], is_query=True)
        async with SessionLocal() as session:
            repo = DocumentRepository(session)
            hits = {
                "vector": await repo.vector_search(vector, RETRIEVAL_K),
                "keyword": await repo.keyword_search(query_terms(text), RETRIEVAL_K),
            }
        hits["hybrid"] = await with_retries(hybrid.retrieve, text, RETRIEVAL_K)

        for strategy in STRATEGIES:
            urls = unique_urls(hits[strategy])
            recall, all_hit, reciprocal_rank = retrieval_scores(urls, gold)
            records.append(
                {
                    "id": question["id"],
                    "type": question["type"],
                    "strategy": strategy,
                    "recall": recall,
                    "all_hit": all_hit,
                    "reciprocal_rank": reciprocal_rank,
                    "gold_articles": len(gold),
                    "retrieved_articles": len(urls),
                }
            )
        progress(number, len(answerable), "retrieval")
    return records


def retrieval_table(records: list[dict]) -> str:
    rows = []
    for strategy in STRATEGIES:
        group = [record for record in records if record["strategy"] == strategy]
        if group:
            rows.append(
                [
                    strategy,
                    str(len(group)),
                    mean_cell([record["recall"] for record in group]),
                    proportion_cell([record["all_hit"] for record in group]),
                    mean_cell([record["reciprocal_rank"] for record in group], digits=3),
                    fmt(mean(record["retrieved_articles"] for record in group), 1),
                ]
            )
    return markdown_table(
        [
            "strategy",
            "n",
            f"Recall@{RETRIEVAL_K}",
            f"AllHit@{RETRIEVAL_K}",
            "MRR",
            f"articles@{RETRIEVAL_K}",  # distinct articles among the chunks: diversity
        ],
        rows,
    )


def retrieval_differences(records: list[dict]) -> str:
    return difference_table(
        records,
        group_field="strategy",
        comparisons=[("hybrid", "vector"), ("hybrid", "keyword")],
        metrics=[(f"AllHit@{RETRIEVAL_K}", "all_hit"), (f"Recall@{RETRIEVAL_K}", "recall")],
    )


# --- experiment 2: answers ---------------------------------------------------------


async def answer_once(mode: str, question: str, graph, chat_model) -> dict:
    started = time.perf_counter()
    if mode == "llm_only":
        message = await chat_model.ainvoke(
            LLM_ONLY_PROMPT.format(not_found=NOT_FOUND, question=question)
        )
        state = {"answer": str(message.text).strip(), "chunks": [], "steps": ["generate"]}
    else:
        state = dict(await graph.ainvoke({"question": question, "mode": mode, "top_k": TOP_K}))
    state["latency_ms"] = round((time.perf_counter() - started) * 1000)
    return state


def count_llm_calls(steps: list[str]) -> int:
    """Gemini calls made: every `generate`, plus `plan` unless it was skipped."""
    return sum(
        1 for step in steps if step == "generate" or (step.startswith("plan:") and step != "plan:skip")
    )


async def evaluate_generation(questions: list[dict], graph, chat_model, modes, out_file) -> list[dict]:
    records: list[dict] = []
    for number, question in enumerate(questions, start=1):
        gold_urls = {item["url"] for item in question["evidence"]}

        # All modes for one question, then the next question: a slow minute or a
        # rate-limit burst then hits every mode equally instead of one mode's block.
        for mode in modes:
            try:
                state = await with_retries(
                    answer_once, mode, question["question"], graph, chat_model
                )
                error = None
            except Exception as exc:  # recorded and counted, never silently dropped
                state = {"answer": "", "chunks": [], "steps": [], "latency_ms": None}
                error = f"{type(exc).__name__}: {exc}"[:300]

            draft = state.get("draft")
            record = {
                "id": question["id"],
                "type": question["type"],
                "mode": mode,
                "question": question["question"],
                "gold": question["answer"],
                "answer": state["answer"],
                "correct": error is None
                and is_correct(state["answer"], question["answer"], question["type"]),
                "token_f1": 0.0
                if error
                else token_f1(state["answer"], question["answer"], question["type"]),
                # validate replaced this answer with a refusal: was the original right?
                "draft": draft,
                "draft_correct": draft is not None
                and is_correct(draft, question["answer"], question["type"]),
                "evidence_recall": None
                if mode == "llm_only"
                else evidence_recall(state["chunks"], gold_urls),
                "latency_ms": state["latency_ms"],
                "llm_calls": count_llm_calls(state["steps"]),
                "steps": state["steps"],
                "error": error,
                # What the LLM was shown, for error analysis: evidence_recall is per
                # ARTICLE, so it cannot tell whether the chunk holding the answer was
                # among the excerpts. These fields can.
                "queries": state.get("queries", []),
                "excerpts": [
                    {
                        "chunk_id": chunk.chunk_id,
                        "url": chunk.url,
                        "source": chunk.source,
                        "is_evidence": chunk.url in gold_urls,
                    }
                    for chunk in state["chunks"]
                ],
            }
            records.append(record)
            out_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_file.flush()  # a crash mid-run keeps everything done so far
        progress(number, len(questions), "answers")
    return records


def generation_table(records: list[dict], modes: list[str]) -> str:
    first_mode = [record for record in records if record["mode"] == modes[0]]
    counts = {qtype: sum(record["type"] == qtype for record in first_mode) for qtype in TYPES}
    headers = [
        "mode",
        *[f"{SHORT[qtype]} (n={counts[qtype]})" for qtype in TYPES if counts[qtype]],
        "overall EM",
        "overall token F1",
        "evidence recall",
        "latency p50 (ms)",
        "LLM calls",
        "validate rejects (draft correct)",
        "errors",
    ]
    rows = []
    for mode in modes:
        group = [record for record in records if record["mode"] == mode]
        latencies = [record["latency_ms"] for record in group if record["latency_ms"] is not None]
        rejected = [record for record in group if "validate:fail" in record["steps"]]
        rows.append(
            [
                mode,
                *[
                    proportion_cell([r["correct"] for r in group if r["type"] == qtype])
                    for qtype in TYPES
                    if counts[qtype]
                ],
                proportion_cell([record["correct"] for record in group]),
                mean_cell([record["token_f1"] for record in group]),
                mean_cell(
                    [r["evidence_recall"] for r in group if r["evidence_recall"] is not None]
                ),
                str(round(statistics.median(latencies))) if latencies else "-",
                fmt(mean(record["llm_calls"] for record in group), 1),
                f"{len(rejected)} ({sum(r['draft_correct'] for r in rejected)})",
                str(sum(record["error"] is not None for record in group)),
            ]
        )
    return markdown_table(headers, rows)


def generation_differences(records: list[dict], modes: list[str]) -> str:
    wanted = [("multihop", "baseline"), ("baseline", "llm_only"), ("multihop", "llm_only")]
    return difference_table(
        records,
        group_field="mode",
        comparisons=[(a, b) for a, b in wanted if a in modes and b in modes],
        metrics=[("accuracy (EM)", "correct"), ("token F1", "token_f1")],
    )


def answering_table(records: list[dict], modes: list[str]) -> str:
    """Accuracy split into how often each mode answers and how often it is right when it does.

    A system can raise accuracy by answering more (risking made-up answers) or by
    answering better. These columns separate the two:
      coverage    answerable questions it answered instead of refusing
      precision   answers given (to any question) that were correct
      null refused unanswerable questions it correctly refused
    """
    rows = []
    for mode in modes:
        group = [record for record in records if record["mode"] == mode]
        answerable = [r for r in group if r["type"] != "null_query"]
        null = [r for r in group if r["type"] == "null_query"]
        answered = [r for r in group if r["answer"] and not is_refusal(r["answer"])]
        rows.append(
            [
                mode,
                f"{len(answered)} of {len(group)}",
                proportion_cell([bool(r["answer"]) and not is_refusal(r["answer"]) for r in answerable]),
                proportion_cell([r["correct"] for r in answered]),
                proportion_cell([is_refusal(r["answer"]) for r in null]),
            ]
        )
    return markdown_table(
        ["mode", "answered", "coverage (answerable)", "precision (when answering)", "null refused"],
        rows,
    )


def _wrong_cell(records: list[dict]) -> str:
    """'5 (3 refused)' = 5 wrong answers, 3 of them refusals."""
    return f"{len(records)} ({sum(is_refusal(r['answer']) for r in records)} refused)"


def error_analysis_table(records: list[dict], modes: list[str]) -> str:
    """Where do the wrong answers come from: retrieval or generation?

    Wrong answers to answerable questions, split by how much of the gold evidence
    the LLM was shown, each "N (R refused)":
      none found   -> retrieval failed; the LLM never had a chance
      some found   -> retrieval partly failed (common for multi-hop)
      all found    -> retrieval worked; the LLM refused or answered wrongly
    Plus null questions the system answered instead of refusing (made-up answers).
    """
    rows = []
    for mode in modes:
        group = [record for record in records if record["mode"] == mode]
        answerable = [record for record in group if record["type"] != "null_query"]
        wrong = [record for record in answerable if not record["correct"]]

        if mode == "llm_only":  # no retrieval -> no evidence to split by
            buckets = ["-", "-", "-"]
        else:
            scored = [r for r in wrong if r["evidence_recall"] is not None]
            buckets = [
                _wrong_cell([r for r in scored if r["evidence_recall"] == 0]),
                _wrong_cell([r for r in scored if 0 < r["evidence_recall"] < 1]),
                _wrong_cell([r for r in scored if r["evidence_recall"] == 1]),
            ]
        made_up = sum(1 for r in group if r["type"] == "null_query" and not r["correct"])
        null_total = sum(1 for r in group if r["type"] == "null_query")
        rows.append(
            [mode, f"{len(wrong)} of {len(answerable)}", *buckets, f"{made_up} of {null_total}"]
        )
    return markdown_table(
        [
            "mode",
            "wrong (answerable)",
            "evidence: none found",
            "evidence: some found",
            "evidence: all found",
            "null questions answered (made up)",
        ],
        rows,
    )


HOW_TO_READ = f"""## How to read these numbers

- `0.80 [0.71, 0.87]` = the measured score and its 95% confidence interval: with a
  different sample of questions, the score would most likely land in that range.
  Proportions (AllHit, EM, precision, coverage, null refused) use the Wilson
  interval; means (Recall, MRR, token F1, evidence recall) use a bootstrap
  ({BOOTSTRAP_ITERATIONS:,} resamples). Latency is a cost, reported without one.
- Answers: EM (exact match) = line 1 matches the gold answer, 1 or 0. Token F1 =
  word overlap with the gold answer (SQuAD-style), giving partial credit, e.g.
  "Everton" vs "Everton Football Club" = 0.50. Yes/No and null questions score
  the same under both.
- Each per-type column shows its own n. With ~25 questions per type the intervals
  are wide: read them as trends. The overall column is the headline number.
- Differences are paired (same questions for both sides). If the interval contains
  0, this sample cannot show that one is better than the other.
- Error analysis: wrong answers with "evidence: none/some found" point at
  retrieval; with "all found" they point at the prompt or the model.
"""


# --- main ----------------------------------------------------------------------------


def write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def report_sections(retrieval: list[dict], generation: list[dict]) -> list[tuple[str, str]]:
    sections: list[tuple[str, str]] = []
    if retrieval:
        sections.append(("Retrieval (answerable questions only)", retrieval_table(retrieval)))
        sections.append(("Retrieval: paired differences", retrieval_differences(retrieval)))
    if generation:
        # Results saved before token F1 existed: score them now (no API calls needed).
        for record in generation:
            if "token_f1" not in record:
                record["token_f1"] = (
                    0.0
                    if record["error"]
                    else token_f1(record["answer"], record["gold"], record["type"])
                )
        modes = [mode for mode in MODES if any(r["mode"] == mode for r in generation)]
        sections.append(("Answers (accuracy of line 1)", generation_table(generation, modes)))
        if len(modes) > 1:
            sections.append(("Answers: paired differences", generation_differences(generation, modes)))
        sections.append(("Answers: answering vs refusing", answering_table(generation, modes)))
        sections.append(("Answers: error analysis", error_analysis_table(generation, modes)))
    return sections


def write_summary(
    sections: list[tuple[str, str]], question_count: int, out_dir: Path, note: str = ""
) -> None:
    seed = next((q.get("seed") for q in load_questions(1)), None)
    header = [
        f"# Evaluation results ({out_dir.name})",
        "",
        f"- Date: {datetime.now(UTC):%Y-%m-%d %H:%M} UTC{note}",
        f"- Questions: {question_count} from `{QUERIES.name}` (stratified sample, seed {seed})",
        f"- Embedding: `{settings.embedding_model}` · LLM: `{settings.llm_model}`"
        f" (thinking: `{settings.llm_thinking_level or 'model default'}`)"
        f" · location: `{settings.gcp_location}`",
        f"- Excerpts given to the LLM: {TOP_K} · retrieval scored at {RETRIEVAL_K} chunks"
        f" · max chunks per article: {settings.max_chunks_per_article or 'no limit'}",
    ]
    body = [f"\n## {title}\n\n{table}" for title, table in sections]
    summary = "\n".join([*header, *body, "", HOW_TO_READ])
    (out_dir / "summary.md").write_text(summary, encoding="utf-8")
    print("\n" + summary)
    print(f"Per-question details: {out_dir}")


def report_only(out_dir: Path) -> int:
    """Rebuild summary.md from the saved .jsonl results. No Vertex AI calls, no cost.

    Note: the header shows the models in the CURRENT .env, which may differ from
    the ones the saved results were produced with.
    """
    retrieval = read_jsonl(out_dir / "retrieval.jsonl")
    generation = read_jsonl(out_dir / "generation.jsonl")
    if not retrieval and not generation:
        print(f"No saved results in {out_dir} — run the evaluation first.")
        return 1
    question_count = len({record["id"] for record in [*retrieval, *generation]})
    sections = report_sections(retrieval, generation)
    write_summary(sections, question_count, out_dir, " (rebuilt from saved results)")
    return 0


async def main(
    limit: int | None, only: str | None, modes: list[str], allow_fake: bool, out_dir: Path
) -> int:
    providers = (settings.embedding_provider, settings.llm_provider)
    if "fake" in providers and not allow_fake:
        print("EMBEDDING_PROVIDER and LLM_PROVIDER must both be `vertex` for real numbers.")
        print("(Use --allow-fake only to test that the script runs.)")
        return 1

    questions = load_questions(limit)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Questions: {len(questions)}   providers: {providers}   output: {out_dir}\n")
    retrieval: list[dict] = []
    generation: list[dict] = []

    if only in (None, "retrieval"):
        print("Retrieval: vector vs keyword vs hybrid")
        retrieval = await evaluate_retrieval(questions, create_embedder(settings))
        write_jsonl(out_dir / "retrieval.jsonl", retrieval)

    if only in (None, "generation"):
        print(f"Answers: {', '.join(modes)}")
        graph = build_rag_graph(settings)
        chat_model = create_chat_model(settings)
        with (out_dir / "generation.jsonl").open("w", encoding="utf-8") as out_file:
            generation = await evaluate_generation(questions, graph, chat_model, modes, out_file)

    await engine.dispose()
    write_summary(report_sections(retrieval, generation), len(questions), out_dir)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate retrieval and answers.")
    parser.add_argument("--limit", type=int, help="only the first N questions (4 = one per type)")
    parser.add_argument("--only", choices=["retrieval", "generation"], help="run one experiment")
    parser.add_argument("--modes", nargs="+", choices=MODES, default=MODES)
    parser.add_argument("--allow-fake", action="store_true", help="run with fake providers")
    parser.add_argument(
        "--report-only", action="store_true", help="rebuild summary.md from saved results (free)"
    )
    parser.add_argument(
        "--tag",
        default="latest",
        help="results folder name under eval/results/, e.g. v1 — keeps runs apart for comparison",
    )
    parser.add_argument("--queries", type=Path, help="question file (default: eval/queries.jsonl)")
    args = parser.parse_args()
    if args.queries:
        QUERIES = args.queries  # module level: every function reads this name at call time
    out_dir = RESULTS / args.tag
    if args.report_only:
        sys.exit(report_only(out_dir))
    sys.exit(asyncio.run(main(args.limit, args.only, args.modes, args.allow_fake, out_dir)))
