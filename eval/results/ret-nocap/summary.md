# Evaluation results (ret-nocap)

- Date: 2026-10-08 14:04 UTC
- Questions: 100 from `eval/queries.jsonl` (stratified sample, seed 42)
- Embedding: `text-multilingual-embedding-002` · LLM: `gemini-3.8-flash` (thinking: `low`) · location: `global`
- Excerpts given to the LLM: 8 · retrieval scored at 10 chunks · max chunks per article: no limit

## Retrieval (answerable questions only)

| strategy | n | Recall@10 | AllHit@10 | MRR | articles@10 |
|---|---|---|---|---|---|
| vector | 75 | 0.60 [0.53, 0.68] | 0.31 [0.21, 0.42] | 0.758 [0.672, 0.838] | 3.6 |
| keyword | 75 | 0.71 [0.64, 0.77] | 0.40 [0.30, 0.51] | 0.833 [0.763, 0.896] | 5.0 |
| hybrid | 75 | 0.69 [0.62, 0.75] | 0.41 [0.31, 0.53] | 0.853 [0.784, 0.917] | 4.2 |

## Retrieval: paired differences

| metric | comparison | n | difference | 95% CI | verdict |
|---|---|---|---|---|---|
| AllHit@10 | hybrid - vector | 75 | +0.11 | [+0.03, +0.19] | hybrid better |
| AllHit@10 | hybrid - keyword | 75 | +0.01 | [-0.08, +0.11] | no clear difference |
| Recall@10 | hybrid - vector | 75 | +0.08 | [+0.03, +0.13] | hybrid better |
| Recall@10 | hybrid - keyword | 75 | -0.02 | [-0.08, +0.03] | no clear difference |

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
