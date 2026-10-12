# Evaluation results (final)

- Date: 2026-10-09 06:09 UTC (rebuilt from saved results)
- Questions: 100 from `queries.jsonl` (stratified sample, seed 42)
- Embedding: `text-multilingual-embedding-002` · LLM: `gemini-3.8-flash` (thinking: `low`) · location: `global`
- Excerpts given to the LLM: 8 · retrieval scored at 10 chunks · max chunks per article: 2

## Retrieval (answerable questions only)

| strategy | n | Recall@10 | AllHit@10 | MRR | articles@10 |
|---|---|---|---|---|---|
| vector | 75 | 0.60 [0.53, 0.68] | 0.31 [0.21, 0.42] | 0.758 [0.672, 0.838] | 3.6 |
| keyword | 75 | 0.71 [0.64, 0.77] | 0.40 [0.30, 0.51] | 0.833 [0.763, 0.896] | 5.0 |
| hybrid | 75 | 0.79 [0.72, 0.85] | 0.60 [0.49, 0.70] | 0.853 [0.784, 0.917] | 6.7 |

## Retrieval: paired differences

| metric | comparison | n | difference | 95% CI | verdict |
|---|---|---|---|---|---|
| AllHit@10 | hybrid - vector | 75 | +0.29 | [+0.20, +0.40] | hybrid better |
| AllHit@10 | hybrid - keyword | 75 | +0.20 | [+0.11, +0.31] | hybrid better |
| Recall@10 | hybrid - vector | 75 | +0.18 | [+0.12, +0.24] | hybrid better |
| Recall@10 | hybrid - keyword | 75 | +0.08 | [+0.01, +0.14] | hybrid better |

## Answers (accuracy of line 1)

| mode | inference (n=25) | comparison (n=25) | temporal (n=25) | null (n=25) | overall EM | overall token F1 | evidence recall | latency p50 (ms) | LLM calls | validate rejects (draft correct) | errors |
|---|---|---|---|---|---|---|---|---|---|---|---|
| llm_only | 0.96 [0.80, 0.99] | 0.24 [0.11, 0.43] | 0.28 [0.14, 0.48] | 0.72 [0.52, 0.86] | 0.55 [0.45, 0.64] | 0.55 [0.45, 0.65] | - | 6462 | 1.0 | 0 (0) | 0 |
| baseline | 0.36 [0.20, 0.55] | 0.40 [0.23, 0.59] | 0.44 [0.27, 0.63] | 1.00 [0.87, 1.00] | 0.55 [0.45, 0.64] | 0.56 [0.46, 0.65] | 0.73 [0.66, 0.80] | 13846 | 1.0 | 0 (0) | 0 |
| multihop | 0.36 [0.20, 0.55] | 0.68 [0.48, 0.83] | 0.48 [0.30, 0.67] | 1.00 [0.87, 1.00] | 0.63 [0.53, 0.72] | 0.63 [0.53, 0.72] | 0.79 [0.73, 0.85] | 19466 | 2.0 | 0 (0) | 0 |

## Answers: paired differences

| metric | comparison | n | difference | 95% CI | verdict |
|---|---|---|---|---|---|
| accuracy (EM) | multihop - baseline | 100 | +0.08 | [+0.00, +0.16] | no clear difference |
| accuracy (EM) | baseline - llm_only | 100 | +0.00 | [-0.12, +0.13] | no clear difference |
| accuracy (EM) | multihop - llm_only | 100 | +0.08 | [-0.05, +0.20] | no clear difference |
| token F1 | multihop - baseline | 100 | +0.07 | [-0.01, +0.15] | no clear difference |
| token F1 | baseline - llm_only | 100 | +0.00 | [-0.12, +0.13] | no clear difference |
| token F1 | multihop - llm_only | 100 | +0.07 | [-0.06, +0.20] | no clear difference |

## Answers: answering vs refusing

| mode | answered | coverage (answerable) | precision (when answering) | null refused |
|---|---|---|---|---|
| llm_only | 49 of 100 | 0.56 [0.45, 0.67] | 0.76 [0.62, 0.85] | 0.72 [0.52, 0.86] |
| baseline | 40 of 100 | 0.53 [0.42, 0.64] | 0.75 [0.60, 0.86] | 1.00 [0.87, 1.00] |
| multihop | 45 of 100 | 0.60 [0.49, 0.70] | 0.84 [0.71, 0.92] | 1.00 [0.87, 1.00] |

## Answers: error analysis

| mode | wrong (answerable) | evidence: none found | evidence: some found | evidence: all found | null questions answered (made up) |
|---|---|---|---|---|---|
| llm_only | 38 of 75 | - | - | - | 7 of 25 |
| baseline | 45 of 75 | 3 (3 refused) | 27 (21 refused) | 15 (11 refused) | 0 of 25 |
| multihop | 37 of 75 | 1 (1 refused) | 26 (22 refused) | 10 (7 refused) | 0 of 25 |

## How to read these numbers

- `0.80 [0.71, 0.87]` = the measured score and its 95% confidence interval: with a
  different sample of questions, the score would most likely land in that range.
  Proportions (AllHit, EM, precision, coverage, null refused) use the Wilson
  interval; means (Recall, MRR, token F1, evidence recall) use a bootstrap
  (10,000 resamples). Latency is a cost, reported without one.
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
