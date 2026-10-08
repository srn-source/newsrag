"""Look inside the pipeline for one question: what was searched, what was found, what was answered.

    uv run python -m scripts.inspect_question --id 566                 # a question from eval/queries.jsonl
    uv run python -m scripts.inspect_question --id 566 --modes multihop
    uv run python -m scripts.inspect_question "Who is facing a fraud trial?"

For each mode prints:
  - the search queries (multihop: the original question + the planner's sub-queries)
  - the excerpts given to the LLM, marked EVIDENCE when they come from a gold article
  - the answer and the steps

Costs one or two Gemini calls per mode. Use it to explain a wrong answer in the
evaluation: wrong excerpts -> retrieval problem; right excerpts -> prompt / model.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

from app.core.config import settings
from app.models.db import engine
from app.services.rag import build_rag_graph

QUERIES = Path(__file__).parent.parent / "eval" / "queries.jsonl"
PREVIEW_CHARS = 220


def find_question(question_id: int) -> dict:
    with QUERIES.open(encoding="utf-8") as file:
        for line in file:
            record = json.loads(line)
            if record["id"] == question_id:
                return record
    sys.exit(f"Question id {question_id} not found in {QUERIES}")


async def main(question: str, gold: str | None, evidence: list[dict], modes: list[str]) -> None:
    gold_urls = {item["url"] for item in evidence}
    print(f"QUESTION: {question}")
    if gold is not None:
        print(f"GOLD ANSWER: {gold}")
    for item in evidence:
        print(f"  evidence: {item['source']} | {item['title']}")

    graph = build_rag_graph(settings)
    for mode in modes:
        state = await graph.ainvoke({"question": question, "mode": mode, "top_k": 8})
        print(f"\n{'=' * 30} {mode.upper()} {'=' * 30}")
        print(f"steps: {state['steps']}")
        print("search queries:")
        for query in state.get("queries", []):
            print(f"  - {query}")

        found = {chunk.url for chunk in state["chunks"]} & gold_urls
        if gold_urls:
            print(f"evidence articles found: {len(found)} of {len(gold_urls)}")
        print("excerpts given to the LLM:")
        for number, chunk in enumerate(state["chunks"], start=1):
            marker = "EVIDENCE" if chunk.url in gold_urls else "        "
            body = chunk.content.split("\n", 1)[-1][:PREVIEW_CHARS]
            print(f"  [{number}] {marker} {chunk.source} | {chunk.published_at} | {chunk.title}")
            print(f"       chunk {chunk.chunk_id}, score {chunk.score:.4f}: {body}...")

        print(f"answer ({state['status']}):")
        for line in state["answer"].splitlines():
            print(f"  {line}")

    await engine.dispose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect the RAG pipeline for one question.")
    parser.add_argument("question", nargs="?", help="free-text question (or use --id)")
    parser.add_argument("--id", type=int, help="question id from eval/queries.jsonl")
    parser.add_argument(
        "--modes", nargs="+", choices=["baseline", "multihop"], default=["baseline", "multihop"]
    )
    args = parser.parse_args()

    if args.id is not None:
        record = find_question(args.id)
        question, gold, evidence = record["question"], record["answer"], record["evidence"]
    elif args.question:
        question, gold, evidence = args.question, None, []
    else:
        parser.error("give a question or --id")

    asyncio.run(main(question, gold, evidence, args.modes))
