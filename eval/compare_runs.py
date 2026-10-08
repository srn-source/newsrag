"""Compare two evaluation runs on the same questions: did a change really help?

    uv run python -m eval.compare_runs v1 v2

Pairs each question with itself across the two runs and reports `after - before`
with a paired-bootstrap 95% interval. If the interval contains 0, this sample
cannot show that the change helped (or hurt).

Free: reads the saved results in eval/results/<tag>/, no Vertex AI calls.
Writes eval/results/compare_<before>_vs_<after>.md
"""

import argparse
import sys
from pathlib import Path

from eval.run_eval import MODES, RESULTS, RETRIEVAL_K, difference_table, read_jsonl


def labelled(records: list[dict], run: str, **match) -> list[dict]:
    """Records matching e.g. strategy="hybrid", tagged with the run they came from."""
    return [
        {**record, "run": run}
        for record in records
        if all(record.get(field) == value for field, value in match.items())
    ]


def run_settings(run_dir: Path) -> list[str]:
    """The '- ...' header lines of a run's summary.md: models, thinking, limits."""
    summary = run_dir / "summary.md"
    if not summary.exists():
        return []
    return [line for line in summary.read_text(encoding="utf-8").splitlines() if line.startswith("- ")]


def main(before: str, after: str) -> int:
    before_dir, after_dir = RESULTS / before, RESULTS / after
    comparison = [(after, before)]
    sections: list[str] = [f"# {after} vs {before}", ""]

    for tag, run_dir in [(before, before_dir), (after, after_dir)]:
        sections.append(f"**{tag}**")
        sections += run_settings(run_dir) or [f"- (no summary.md in {run_dir})"]
        sections.append("")

    compared = False

    # --- retrieval: the hybrid strategy is what the API uses -------------------
    old = read_jsonl(before_dir / "retrieval.jsonl")
    new = read_jsonl(after_dir / "retrieval.jsonl")
    if old and new:
        compared = True
        records = labelled(old, before, strategy="hybrid") + labelled(new, after, strategy="hybrid")
        sections += [
            "## Retrieval (hybrid)",
            "",
            difference_table(
                records,
                group_field="run",
                comparisons=comparison,
                metrics=[
                    (f"AllHit@{RETRIEVAL_K}", "all_hit"),
                    (f"Recall@{RETRIEVAL_K}", "recall"),
                    ("MRR", "reciprocal_rank"),
                    (f"articles@{RETRIEVAL_K}", "retrieved_articles"),
                ],
            ),
            "",
        ]

    # --- answers: each mode separately -----------------------------------------
    old = read_jsonl(before_dir / "generation.jsonl")
    new = read_jsonl(after_dir / "generation.jsonl")
    if old and new:
        compared = True
        for mode in MODES:
            records = labelled(old, before, mode=mode) + labelled(new, after, mode=mode)
            if not records:
                continue
            # evidence_recall is empty for llm_only and for null questions: compare
            # only the records that have it.
            with_evidence = [r for r in records if r["evidence_recall"] is not None]
            metrics_tables = [
                difference_table(records, "run", comparison, [("accuracy", "correct")])
            ]
            if with_evidence:
                metrics_tables.append(
                    difference_table(
                        with_evidence, "run", comparison, [("evidence recall", "evidence_recall")]
                    )
                )
            sections += [f"## Answers: {mode}", "", *metrics_tables, ""]

    if not compared:
        print(f"Nothing to compare: need results in both {before_dir} and {after_dir}.")
        return 1

    report = "\n".join(sections)
    out_path = RESULTS / f"compare_{before}_vs_{after}.md"
    out_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"Saved: {out_path}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare two evaluation runs (paired).")
    parser.add_argument("before", help="tag of the earlier run, e.g. v1")
    parser.add_argument("after", help="tag of the later run, e.g. v2")
    args = parser.parse_args()
    sys.exit(main(args.before, args.after))
