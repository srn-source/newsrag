# Evaluation

How well does NewsRAG find the evidence, and how well does it answer? Every number
is measured on 100 questions, reported with a 95% confidence interval, and
reproducible from `eval/results/final/`.

## 1. Setup

| | |
|---|---|
| Dataset | [MultiHop-RAG](https://huggingface.co/datasets/yixuantt/MultiHopRAG) (Tang & Yang, 2024): 609 news articles, questions needing 2–4 articles |
| Sample | 100 questions, 25 per type: inference, comparison, temporal, null (unanswerable). Seed 42. A held-out sample (seed 7) confirms the retrieval change |
| Models | Gemini `gemini-3.8-flash` (thinking `low`, temperature 0) · embeddings `text-multilingual-embedding-002` (768-d) |
| Systems compared | `llm_only`: Gemini with no retrieval · `baseline`: one hybrid search · `multihop` (deployed): Gemini splits the question into ≤3 sub-queries |

**Scoring is deterministic: no LLM judge.**
- **Retrieval** is scored against the dataset's gold evidence articles.
- **Answers** are scored on line 1 of the response, which the prompt forces to be the short answer:
  - **Exact match (EM):** Yes/No must match, names must be contained, null questions must be refused.
  - **Token F1 (SQuAD-style):** partial credit, e.g. "Everton" for "Everton Football Club".
  - BLEU and ROUGE are not used, because they degenerate on 1–3-word answers.
- **Intervals:** Wilson for proportions, bootstrap (10,000 resamples) for means, and a *paired* bootstrap for differences between systems.

## 2. Retrieval results

75 answerable questions, top 10 chunks.

| strategy | Recall@10 | AllHit@10 | MRR |
|---|---|---|---|
| vector | 0.60 [0.53, 0.68] | 0.31 [0.21, 0.42] | 0.758 [0.672, 0.838] |
| keyword | 0.71 [0.64, 0.77] | 0.40 [0.30, 0.51] | 0.833 [0.763, 0.896] |
| **hybrid + cap (deployed)** | **0.79** [0.72, 0.85] | **0.60** [0.49, 0.70] | **0.853** [0.784, 0.917] |

- **Recall@10:** the share of gold articles found.
- **AllHit@10:** the share of questions where *every* gold article was found.
- **MRR:** 1 / rank of the first gold article.

**Per-article cap (at most 2 chunks per article).** Error analysis found one long
article filling 7 of 8 excerpt slots on a question that needed three articles.

| sample | AllHit@10 without → with cap | questions improved / worsened |
|---|---|---|
| seed 42 (main) | 0.41 → **0.60** | **14 / 0** |
| seed 7 (held-out) | 0.35 → **0.49** | **11 / 0** |

**What it means**
- **Hybrid is clearly the best strategy.** On AllHit@10 it beats vector by +0.29 [+0.20, +0.40] and keyword by +0.20 [+0.11, +0.31]. Vector and keyword find different evidence, and fusing them recovers both.
- **The cap matters as much as hybrid itself.** Without it, hybrid was no better than keyword (+0.01). The gain repeated on held-out questions and no question got worse, so it is not overfitted to the test set.
- **Finding the first article is easy; finding all of them is hard.** MRR is high for every strategy (0.76–0.85), but 40% of questions still miss at least one article. That is the main bottleneck for multi-hop answers.

## 3. Generation results

100 questions. Per-type columns are exact match.

| system | inference | comparison | temporal | null | **EM** | **token F1** | latency p50 |
|---|---|---|---|---|---|---|---|
| llm_only | **0.96** | 0.24 | 0.28 | 0.72 | 0.55 [0.45, 0.64] | 0.55 [0.45, 0.65] | 6.5 s |
| baseline | 0.36 | 0.40 | 0.44 | **1.00** | 0.55 [0.45, 0.64] | 0.56 [0.46, 0.65] | 13.8 s |
| **multihop (deployed)** | 0.36 | **0.68** | **0.48** | **1.00** | **0.63** [0.53, 0.72] | **0.63** [0.53, 0.72] | 19.5 s |

| system | precision: correct when it answers | coverage: answerable questions answered | unanswerable questions refused |
|---|---|---|---|
| llm_only | 0.76 [0.62, 0.85] | 0.56 [0.45, 0.67] | 0.72: **7 made-up answers** |
| baseline | 0.75 [0.60, 0.86] | 0.53 [0.42, 0.64] | **1.00**: 0 made up |
| **multihop** | **0.84** [0.71, 0.92] | **0.60** [0.49, 0.70] | **1.00**: 0 made up |

Paired differences on EM: multihop − baseline **+0.08** [0.00, +0.16]; baseline − llm_only
+0.00 [−0.12, +0.13]. Token F1 gives the same picture to within 0.01.

**What it means**
- **Multihop is the best system, but the margin over baseline is not yet conclusive.**
  The interval just touches 0 at n = 100. The gain comes from comparison questions
  (0.68 vs 0.40), where one search per outlet retrieves both sides. It costs a second
  LLM call and about 6 s more latency.
- **RAG makes the system trustworthy.** It never invented an answer to an unanswerable
  question (vs 7 of 25 without retrieval), and when multihop answers, 84% of answers are correct.
  It wins clearly on comparison and temporal questions, which ask what *specific
  articles* said.
- **The overall tie with the LLM alone hides two effects:**
  - **Inference questions (0.96 vs 0.36) measure memory, not retrieval.** They describe famous entities (Google, Sam Bankman-Fried) that Gemini knows from its training data. On private or recent documents, the LLM-alone score would collapse.
  - **RAG over-refuses.** It declines 40–47% of answerable questions. Most errors are refusals made when only *some* of the evidence was retrieved. That ties back to AllHit@10 = 0.60 above.

## 4. Reproduce

```bash
uv run python -m eval.build_eval_set                       # eval/queries.jsonl (seed 42)
uv run python -m eval.run_eval --tag final                 # ~1 h: retrieval + generation
uv run python -m eval.run_eval --report-only --tag final   # rebuild tables, no API calls

# per-article cap experiment (retrieval only, no LLM calls)
uv run python -m eval.build_eval_set --seed 7 --output eval/queries_seed7.jsonl
MAX_CHUNKS_PER_ARTICLE=0 uv run python -m eval.run_eval --only retrieval --tag ret-nocap
MAX_CHUNKS_PER_ARTICLE=2 uv run python -m eval.run_eval --only retrieval --tag ret-cap2
uv run python -m eval.compare_runs ret-nocap ret-cap2

# inspect one question: sub-queries, excerpts, answer
uv run python -m scripts.inspect_question --id 566
```

Per-question results, including the excerpts behind each answer, are in
`eval/results/<tag>/retrieval.jsonl` and `generation.jsonl`.
