"""Scoring functions for evaluation. Pure functions, covered by tests/test_eval_metrics.py.

Answer correctness is decided by string comparison, not by another LLM:
the dataset's gold answers are short ("Sam Bankman-Fried", "Yes", "No"), and the
answer prompt forces line 1 to be the short answer only. Deterministic, free, and
the same verdict every run.
"""

import math
import random
import re
from collections import Counter
from collections.abc import Sequence

from app.graph.nodes import first_line, is_refusal
from app.models.schemas import RetrievedChunk

_NON_ALNUM = re.compile(r"[^a-z0-9]+")

Z_95 = 1.959964  # standard normal quantile for a two-sided 95% interval
BOOTSTRAP_ITERATIONS = 10_000
BOOTSTRAP_SEED = 0  # fixed: re-running the report gives identical intervals


def normalize(text: str) -> str:
    """'Sam Bankman-Fried!' -> 'sam bankman fried' (lower case, punctuation -> spaces)."""
    return _NON_ALNUM.sub(" ", text.lower()).strip()


def is_correct(answer: str, gold: str, question_type: str) -> bool:
    """Does the answer's first line match the gold answer?

    null_query  -> correct only if the system refused (the corpus has no answer).
    Yes / No    -> the first WORD must be the gold word ("Yes, both..." counts).
    names etc.  -> the gold answer must appear as whole words in line 1
                   ("Meta" does not match "Metaverse").
    """
    refused = is_refusal(answer)
    if question_type == "null_query":
        return refused
    if refused:
        return False

    predicted = normalize(first_line(answer))
    expected = normalize(gold)
    if not expected:
        return False
    if expected in {"yes", "no"}:
        return predicted.split()[:1] == [expected]
    return f" {expected} " in f" {predicted} "


_ARTICLES = {"a", "an", "the"}


def token_f1(answer: str, gold: str, question_type: str) -> float:
    """SQuAD-style token F1 between line 1 of the answer and the gold answer (0-1).

    Gives partial credit where exact match gives none:
        gold "Everton Football Club", answer "Everton"  -> EM 0, F1 0.50
        gold "Sam Bankman-Fried",     answer "Bankman-Fried" -> EM 0, F1 0.80
    precision = shared words / words in the answer
    recall    = shared words / words in the gold answer
    F1        = 2 * precision * recall / (precision + recall)

    Yes/No and null questions have no partial answers, so they score exactly like
    is_correct (1 or 0). Normalisation follows SQuAD: lower case, no punctuation,
    no articles.
    """
    refused = is_refusal(answer)
    if question_type == "null_query":
        return 1.0 if refused else 0.0
    if refused:
        return 0.0

    expected = [word for word in normalize(gold).split() if word not in _ARTICLES]
    if expected in (["yes"], ["no"]):
        return 1.0 if is_correct(answer, gold, question_type) else 0.0

    predicted = [word for word in normalize(first_line(answer)).split() if word not in _ARTICLES]
    if not predicted or not expected:
        return 0.0
    shared = sum((Counter(predicted) & Counter(expected)).values())
    if shared == 0:
        return 0.0
    precision = shared / len(predicted)
    recall = shared / len(expected)
    return 2 * precision * recall / (precision + recall)


def unique_urls(chunks: list[RetrievedChunk]) -> list[str]:
    """Article URLs in rank order, each once. Several chunks of one article = one hit."""
    return list(dict.fromkeys(chunk.url for chunk in chunks))


def retrieval_scores(retrieved: list[str], gold: set[str]) -> tuple[float, bool, float]:
    """(recall, all_hit, reciprocal_rank), all measured on ARTICLES, not chunks.

    recall           share of the gold articles that were retrieved
    all_hit          True only if EVERY gold article was retrieved — the strict
                     multi-hop metric: one missing article can make the answer wrong
    reciprocal_rank  1 / position of the first gold article (0 if none found)
    """
    found = gold & set(retrieved)
    recall = len(found) / len(gold)
    all_hit = found == gold
    reciprocal_rank = next(
        (1.0 / position for position, url in enumerate(retrieved, start=1) if url in gold), 0.0
    )
    return recall, all_hit, reciprocal_rank


def evidence_recall(chunks: list[RetrievedChunk], gold: set[str]) -> float | None:
    """Share of gold articles among the excerpts the LLM actually read (None if no gold)."""
    if not gold:
        return None
    return len(gold & set(unique_urls(chunks))) / len(gold)


# --- confidence intervals ----------------------------------------------------------
#
# With 25-100 questions, a score of 0.80 could easily have been 0.72 or 0.86 on a
# different sample. The intervals say how much the numbers can be trusted.


def wilson_interval(successes: int, n: int, z: float = Z_95) -> tuple[float, float] | None:
    """95% interval for a proportion (e.g. accuracy = 20 correct out of 25).

    Wilson rather than the textbook "p ± 1.96·sqrt(p(1-p)/n)": that formula breaks at
    small n and at 0% / 100% (it gives an interval of width 0 for 10/10, or limits
    outside 0-1). Wilson stays inside 0-1 and is reliable down to small samples.
    """
    if n == 0:
        return None
    p = successes / n
    denominator = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denominator
    margin = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def _percentile(sorted_values: list[float], q: float) -> float:
    index = round(q * (len(sorted_values) - 1))
    return sorted_values[index]


def bootstrap_mean_ci(
    values: Sequence[float],
    iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float] | None:
    """95% interval for a mean of any values (e.g. recall 0.5, 1.0, 0.5 ...).

    Bootstrap: resample the questions WITH replacement many times, take the mean of
    each resample, and read off the middle 95% of those means. No formula or
    distribution assumption needed, so it also works for recall and MRR.
    """
    if not values:
        return None
    numbers = [float(value) for value in values]  # True/False -> 1.0/0.0
    rng = random.Random(seed)
    n = len(numbers)
    means = sorted(sum(rng.choices(numbers, k=n)) / n for _ in range(iterations))
    return _percentile(means, 0.025), _percentile(means, 0.975)


def paired_bootstrap_diff(
    a: Sequence[float],
    b: Sequence[float],
    iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float, float]:
    """(difference, low, high): 95% interval for mean(a) - mean(b), on the SAME questions.

    a[i] and b[i] must be the scores of two systems on question i. Each resample
    draws whole questions, keeping the pairs together, so a question that is hard
    for both systems does not inflate the noise. That makes this far more sensitive
    than checking whether two separate intervals overlap.

    If the interval contains 0, this sample cannot tell the two systems apart.
    """
    if len(a) != len(b):
        raise ValueError("paired scores must have the same length")
    if not a:
        raise ValueError("no paired scores")
    n = len(a)
    differences = [x - y for x, y in zip(a, b, strict=True)]
    rng = random.Random(seed)
    resampled = sorted(sum(rng.choices(differences, k=n)) / n for _ in range(iterations))
    return (
        sum(differences) / n,
        _percentile(resampled, 0.025),
        _percentile(resampled, 0.975),
    )
