# Evaluation results (s7-cap2)

- Date: 2026-10-08 14:20 UTC
- Questions: 100 from `queries_seed7.jsonl` (stratified sample, seed 7)
- Embedding: `text-multilingual-embedding-002` · LLM: `gemini-3.8-flash` (thinking: `low`) · location: `global`
- Excerpts given to the LLM: 8 · retrieval scored at 10 chunks · max chunks per article: 2

## Retrieval (answerable questions only)

| strategy | n | Recall@10 | AllHit@10 | MRR | articles@10 |
|---|---|---|---|---|---|
| vector | 75 | 0.55 [0.48, 0.62] | 0.24 [0.16, 0.35] | 0.728 [0.638, 0.814] | 3.6 |
| keyword | 75 | 0.68 [0.62, 0.74] | 0.33 [0.24, 0.45] | 0.844 [0.774, 0.906] | 4.5 |
| hybrid | 75 | 0.78 [0.72, 0.84] | 0.49 [0.38, 0.60] | 0.810 [0.737, 0.877] | 6.6 |

## Retrieval: paired differences

| metric | comparison | n | difference | 95% CI | verdict |
|---|---|---|---|---|---|
| AllHit@10 | hybrid - vector | 75 | +0.25 | [+0.16, +0.36] | hybrid better |
| AllHit@10 | hybrid - keyword | 75 | +0.16 | [+0.05, +0.28] | hybrid better |
| Recall@10 | hybrid - vector | 75 | +0.23 | [+0.17, +0.29] | hybrid better |
| Recall@10 | hybrid - keyword | 75 | +0.11 | [+0.04, +0.17] | hybrid better |

## How to read these numbers

- `0.80 [0.71, 0.87]` = the measured score and its 95% confidence interval: with a
  different sample of questions, the score would most likely land in that range.
  Accuracy and AllHit use the Wilson interval; Recall and MRR use a bootstrap
  (10,000 resamples).
- Each per-type column shows its own n. With ~25 questions per type the intervals
  are wide: read them as trends. The overall column is the headline number.
- Differences are paired (same questions for both sides). If the interval contains
  0, this sample cannot show that one is better than the other.
- Error analysis: wrong answers with "evidence: none/some found" point at
  retrieval; with "all found" they point at the prompt or the model.
