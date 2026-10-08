"""Build eval/queries.jsonl: a fixed, stratified sample of MultiHop-RAG questions.

    uv run python -m eval.build_eval_set                # 25 per type = 100 questions

Stratified: the same number of questions from each of the 4 question types, so
the overall score is not dominated by whichever type is most common.
Fixed seed: anyone who re-runs this gets exactly the same 100 questions.

The output is committed to git, so the evaluation is reproducible without
re-downloading or re-sampling.
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

DATASET = "yixuantt/MultiHopRAG"
OUTPUT = Path(__file__).parent / "queries.jsonl"
TYPES = ["inference_query", "comparison_query", "temporal_query", "null_query"]
DEFAULT_SEED = 42
DEFAULT_PER_TYPE = 25


def main(per_type: int, seed: int, output: Path) -> None:
    from datasets import load_dataset  # ingest dependency group

    questions = load_dataset(DATASET, "MultiHopRAG", split="train")
    corpus_urls = {row["url"] for row in load_dataset(DATASET, "corpus", split="train")}

    by_type: dict[str, list[tuple[int, dict]]] = defaultdict(list)
    for index, row in enumerate(questions):
        by_type[row["question_type"]].append((index, row))

    unknown = set(by_type) - set(TYPES)
    if unknown:
        print(f"Note: ignoring unexpected question types: {sorted(unknown)}")

    rng = random.Random(seed)
    samples = {qtype: rng.sample(by_type[qtype], min(per_type, len(by_type[qtype]))) for qtype in TYPES}

    # Interleave the types (inference, comparison, temporal, null, inference, ...) so
    # that the first N lines cover every type: `run_eval --limit 4` tests all four.
    ordered = [samples[qtype][i] for i in range(per_type) for qtype in TYPES if i < len(samples[qtype])]

    evidence_total = evidence_missing = 0
    with output.open("w", encoding="utf-8") as file:
        for index, row in ordered:
            # One article can appear several times in evidence_list (different facts);
            # retrieval is scored per article, so keep each URL once.
            evidence = {
                item["url"]: {"title": item["title"], "source": item["source"], "url": item["url"]}
                for item in row["evidence_list"]
            }
            evidence_total += len(evidence)
            evidence_missing += sum(url not in corpus_urls for url in evidence)
            record = {
                "id": index,  # row number in the dataset, to trace a question back
                "type": row["question_type"],
                "question": row["query"],
                "answer": row["answer"],
                "evidence": list(evidence.values()),
                "seed": seed,
            }
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Wrote {len(ordered)} questions to {output}")
    for qtype in TYPES:
        print(f"  {qtype:<18} {len(samples[qtype])}  (of {len(by_type[qtype])} in the dataset)")
    # Sanity check of the join between questions and corpus: retrieval scoring
    # matches evidence URLs against the URLs of ingested articles.
    print(f"Evidence articles: {evidence_total}, missing from the corpus: {evidence_missing}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample the evaluation questions.")
    parser.add_argument("--per-type", type=int, default=DEFAULT_PER_TYPE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    # A different file for a held-out set, e.g. eval/queries_seed7.jsonl, so the
    # main eval/queries.jsonl is never overwritten.
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    main(args.per_type, args.seed, args.output)
