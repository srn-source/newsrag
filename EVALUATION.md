# Evaluation

How well does NewsRAG retrieve evidence and answer multi-hop news questions, and
does retrieval actually help compared with asking the LLM directly?

All numbers below are measured, reported with 95% confidence intervals, and
reproducible from the files in `eval/results/` (commands at the end).

## TL;DR

- **Retrieval: hybrid search with a per-article cap is clearly the best strategy.**
  It finds every evidence article for 60% of questions (AllHit@10 = 0.60), versus
  40% for keyword and 31% for vector search. The cap was found through error analysis
  and confirmed on an independent second sample.
- **Answers: overall accuracy is 0.55 (LLM alone), 0.55 (baseline RAG) and 0.63
  (multi-hop RAG).** At n = 100, none of these differences is statistically
  conclusive. Multi-hop vs baseline (+0.08, CI [0.00, +0.16]) is suggestive.
- **The overall tie hides opposite effects.** RAG is far better on comparison,
  temporal and unanswerable questions. It is far worse on inference questions, where
  the LLM already knows the famous answer from training data (0.96 without retrieval
  vs 0.36 with it).
- **RAG never invented an answer to an unanswerable question** (0 of 25, vs 7 of 25
  for the LLM alone). When it does answer, it is right 75–84% of the time.
- **The main weakness is over-refusal.** RAG declines 30–35 of the 75 answerable
  questions, mostly when retrieval found only part of the evidence. This is the first
  thing to improve (see [Next steps](#7-next-steps)).

## 1. Setup

| | |
|---|---|
| Dataset | [MultiHop-RAG](https://huggingface.co/datasets/yixuantt/MultiHopRAG) (Tang & Yang, 2024): 609 news articles, questions needing 2–4 articles |
| Sample | 100 questions, stratified: 25 each of inference, comparison, temporal, null (unanswerable). Seed 42, `eval/queries.jsonl` |
| Held-out sample | 100 more questions, seed 7, `eval/queries_seed7.jsonl` (used to confirm the retrieval change) |
| Embedding | Vertex AI `text-multilingual-embedding-002`, 768 dimensions |
| LLM | `gemini-3.8-flash`, thinking level `low`, temperature 0 |
| Retrieval | hybrid = pgvector cosine + PostgreSQL full-text, fused with RRF; at most 2 chunks per article; 8 excerpts to the LLM |
| Modes | `llm_only`: Gemini without retrieval · `baseline`: one search · `multihop`: LLM splits the question into ≤3 sub-queries, results fused |

**Scoring is deterministic.** No LLM judge is used.
- **Retrieval** is scored on *articles* (chunks of the same article count once) against the dataset's gold evidence URLs.
- **Answers:** the prompt forces line 1 to be the short answer only. Line 1 is matched against the gold answer:
  - Yes/No answers: the first word must match.
  - Names: whole-word containment.
  - Null questions: correct only if the system refused.

**Confidence intervals:**
- Proportions use the Wilson interval.
- Means use a bootstrap with 10,000 resamples.
- Comparisons use a *paired* bootstrap on the same questions. An interval containing 0 means "no reliable difference at this sample size".

## 2. Retrieval

75 answerable questions. Metrics are computed on the top 10 chunks.

| strategy | Recall@10 | AllHit@10 | MRR | articles@10 |
|---|---|---|---|---|
| vector | 0.60 [0.53, 0.68] | 0.31 [0.21, 0.42] | 0.758 [0.672, 0.838] | 3.6 |
| keyword | 0.71 [0.64, 0.77] | 0.40 [0.30, 0.51] | 0.833 [0.763, 0.896] | 5.0 |
| **hybrid (used)** | **0.79 [0.72, 0.85]** | **0.60 [0.49, 0.70]** | **0.853 [0.784, 0.917]** | 6.7 |

- **Recall@10:** share of gold articles found.
- **AllHit@10:** share of questions where *every* gold article was found. This is the key metric for multi-hop: one missing article can make the question unanswerable.
- **MRR:** 1 / rank of the first gold article.
- **articles@10:** distinct articles among the 10 chunks.

Paired differences: hybrid beats vector by +0.29 AllHit [+0.20, +0.40] and keyword by +0.20 [+0.11, +0.31].

### The per-article cap: found by error analysis, confirmed on held-out data

Inspecting a wrong answer showed one long article filling 7 of the 8 excerpt slots
for a question that needed three articles. Every chunk of that article matched both
searches, so fusion ranked them all highly. Capping each article at 2 chunks:

| sample | AllHit@10 without cap → with cap | Recall@10 | improved / worsened questions | MRR |
|---|---|---|---|---|
| seed 42 (main) | 0.41 → **0.60** | 0.69 → 0.79 | **14 / 0** | 0.853 → 0.853 |
| seed 7 (held-out) | 0.35 → **0.49** | 0.67 → 0.78 | **11 / 0** | 0.810 → 0.810 |

- **No question got worse** in either sample.
- The improvement repeats on questions the change was never tuned on: the held-out sample shares only 4 questions with the main one, and none of those 4 changed.
- MRR is identical by construction. The cap only removes *repeat* chunks of an article already ranked, so it never reorders where each article first appears.
- The cap also explains an earlier puzzle. Without it, hybrid was **no better than keyword alone** (+0.01). With it, hybrid leads by +0.20.

## 3. Answers

100 questions, accuracy of line 1.

| mode | inference | comparison | temporal | null | **overall** | evidence recall | latency p50 | LLM calls |
|---|---|---|---|---|---|---|---|---|
| llm_only | **0.96** [0.80, 0.99] | 0.24 [0.11, 0.43] | 0.28 [0.14, 0.48] | 0.72 [0.52, 0.86] | 0.55 [0.45, 0.64] | – | 6.5 s | 1 |
| baseline | 0.36 [0.20, 0.55] | 0.40 [0.23, 0.59] | 0.44 [0.27, 0.63] | **1.00** [0.87, 1.00] | 0.55 [0.45, 0.64] | 0.73 | 13.8 s | 1 |
| multihop | 0.36 [0.20, 0.55] | **0.68** [0.48, 0.83] | **0.48** [0.30, 0.67] | **1.00** [0.87, 1.00] | **0.63** [0.53, 0.72] | 0.79 | 19.5 s | 2 |

Paired differences in overall accuracy:

| comparison | difference | 95% CI | verdict |
|---|---|---|---|
| multihop − baseline | +0.08 | [0.00, +0.16] | suggestive, not conclusive |
| baseline − llm_only | +0.00 | [−0.12, +0.13] | no overall difference |
| multihop − llm_only | +0.08 | [−0.05, +0.20] | no overall difference |

### Answering vs refusing

The system may refuse ("Insufficient information."). Splitting accuracy into *how
often it answers* and *how often it is right when it does* shows two different
profiles:

| mode | answered (of 100) | correct when answered | answerable questions refused (of 75) | unanswerable questions answered (of 25) |
|---|---|---|---|---|
| llm_only | 49 | 37 / 49 = **0.76** | 33 | **7** (made up) |
| baseline | 40 | 30 / 40 = **0.75** | 35 | **0** |
| multihop | 45 | 38 / 45 = **0.84** | 30 | **0** |

### What the per-type results mean

- **Null (unanswerable): RAG 1.00 vs LLM alone 0.72.** Grounding works. Without
  retrieval, Gemini answered 7 of 25 questions whose answer is not in the corpus. One
  example: asked for the letter that "the CEO of Tesla" starts with, it answered "E"
  from general knowledge. With retrieval, it never did.
- **Comparison and temporal: RAG wins clearly**, and multihop wins most (comparison
  0.68 vs 0.40 baseline vs 0.24 LLM alone). These questions ask about what
  *specific articles* said on specific dates. The LLM cannot know that, and splitting
  the question into one search per outlet or date retrieves both sides.
- **Inference: the LLM alone scores 0.96, RAG only 0.36.** This is the largest effect
  in the evaluation, and it is **not** a retrieval win for the LLM:
  - Inference questions describe a famous entity through several clues ("which company... known for ranking local search results... sued by news publishers?" → Google).
  - Gemini recognises the entity from its training data. The news is from 2023, before its training cutoff.
  - RAG instead requires the *excerpts* to support the answer. When some clues' articles are missing, it refuses. That is exactly what the error analysis shows.

  So the dataset's inference questions mostly measure model knowledge, not
  retrieval. On private or post-cutoff data, which is the reason to use RAG at all,
  the LLM-alone score would collapse.

### Error analysis

Wrong answers to the 75 answerable questions, split by how much gold evidence the
LLM was shown. Each cell is wrong answers (of which refusals).

| mode | wrong | evidence: none found | some found | all found |
|---|---|---|---|---|
| baseline | 45 | 3 (3) | 27 (21) | 15 (11) |
| multihop | 37 | 1 (1) | 26 (22) | 10 (7) |

- **Most errors are refusals** (35 of 45 for baseline, 30 of 37 for multihop), not wrong answers.
- **Most errors happen with partial evidence** ("some found", 26–27). Retrieving every article of a 3–4-article question is still the bottleneck: AllHit is 0.60.
- **7–11 refusals happened with *all* evidence present.** The prompt makes the model too cautious when it must combine facts across excerpts.

### Judge audit

The wrong *non-refusal* answers were read by hand:

- **3 false negatives from string matching:** gold "Everton Football Club", answer
  "Everton" / "Everton F.C." — correct but scored wrong, in all three modes.
- **2 debatable:** gold "no", answer "Different amounts" (the question asked whether
  two outlets reported the same amount). The meaning is right, but the format is not.
- The other wrong answers are genuine: a reversed Yes/No, or the wrong outlet when
  one article was missing.

So the string judge **under-counts by about 1–2 points per mode**. That does not change
any conclusion above.

## 4. Positive results

1. Hybrid retrieval with the per-article cap is clearly the best strategy, and the
   cap's effect replicated on held-out data with zero regressions.
2. Retrieval eliminates made-up answers to unanswerable questions (7 → 0 of 25).
3. When RAG answers, it is usually right: 84% for multihop.
4. Query decomposition helps where it was designed to: comparison questions across
   outlets (+0.28 over baseline on that type).

## 5. Negative results, stated plainly

1. **RAG does not beat the LLM alone on overall accuracy here** (0.55 / 0.63 vs 0.55).
   The reason is that the model already knows these 2023 headlines, which mostly
   affects inference questions. This would not hold for private or recent documents.
2. **The system over-refuses:** 40–47% of answerable questions get "Insufficient
   information." It is tuned towards never inventing an answer, which costs coverage.
3. **Multihop's gain over baseline is not conclusive at n = 100**, and it costs a
   second LLM call and about 40% more latency (19.5 s vs 13.8 s p50).
4. **Latency is high for an interactive API** (p50 14–20 s) even with thinking set to `low`.
   With the model's default thinking it was about 40 s.

## 6. Limitations of this evaluation

- **Sample size.** 100 questions give ±8–10-point intervals overall and ±15–20 per
  type. A reliable test of a 5-point difference needs about 1,000 questions. The
  dataset has 2,556, so this is a time-and-cost choice, not a data limit.
- **Knowledge contamination.** The LLM-alone baseline is inflated by memorised 2023
  news. It measures "does RAG beat a model that already read the news", not "does RAG
  help on unseen documents".
- **String-match judge.** It misses aliases ("Everton" vs "Everton Football Club"),
  so it slightly under-counts.
- **Single run.** A thinking model is not fully deterministic even at temperature 0.
  During development, question 566 was answered correctly from 8 excerpts and refused
  from the *same* 8 excerpts in a different order. Several answers also changed
  between two identical smoke runs. Run-to-run variance was not measured.
- **The retrieval change was designed from errors in the main sample.** The held-out
  replication mitigates this. Prompt changes were deliberately *not* tuned on the
  test set.

## 7. Next steps

System, in order of expected impact:

1. **Calibrate refusals.** Allow an answer when the excerpts identify it even if not
   every clue is confirmed. Accept it only if null-question accuracy stays near 1.00:
   that is the guard against inventing answers. The target is the 7–11 refusals with
   all evidence present, and part of the 21–22 with partial evidence.
2. **Metadata filtering.** Extract outlet names and dates from the question
   ("Sporting News on 2023-10-02") and add a filtered search
   (`WHERE source = … AND published_at = …`). Temporal questions identify articles
   by date, which neither embeddings nor keyword search match well.
3. **More excerpts in multihop mode** (e.g. 12 instead of 8). More sub-queries need
   more slots to keep evidence from each.
4. **Lower latency.** Try thinking level `minimal`, and measure the accuracy cost.

Evaluation:

1. **Scale to 300–500 questions,** running several requests in parallel within the
   Vertex AI quota, to make the multihop-vs-baseline comparison conclusive.
2. **Report coverage and precision automatically** (the "answering vs refusing"
   table above is computed by hand).
3. **Add an alias-aware judge,** and check it against a hand-labelled subset of about
   50 answers.
4. **Run each configuration 3 times** to measure run-to-run variance.
5. **Control contamination:** evaluate on articles published after the model's
   training cutoff, or on synthetic documents, where the LLM-alone score shows what
   retrieval truly adds.

## 8. Reproduce

```bash
uv run python -m eval.build_eval_set                       # eval/queries.jsonl (seed 42)
uv run python -m eval.run_eval --tag final                 # ~1 h, all experiments
uv run python -m eval.run_eval --report-only --tag final   # rebuild tables, no API calls

# per-article cap experiment, main and held-out samples (retrieval only, no LLM)
uv run python -m eval.build_eval_set --seed 7 --output eval/queries_seed7.jsonl
MAX_CHUNKS_PER_ARTICLE=0 uv run python -m eval.run_eval --only retrieval --tag ret-nocap
MAX_CHUNKS_PER_ARTICLE=2 uv run python -m eval.run_eval --only retrieval --tag ret-cap2
uv run python -m eval.compare_runs ret-nocap ret-cap2

# look inside one question: sub-queries, excerpts, answer
uv run python -m scripts.inspect_question --id 566
```

Per-question results, including the excerpts each answer was based on, are in
`eval/results/<tag>/generation.jsonl` and `retrieval.jsonl`.
