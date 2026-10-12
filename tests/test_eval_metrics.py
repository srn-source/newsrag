"""Unit tests for the evaluation scoring functions. A bug here would silently
change every number in the evaluation, so the rules are pinned down explicitly."""

import pytest

from app.graph.nodes import NOT_FOUND
from app.models.schemas import RetrievedChunk
from eval.metrics import (
    bootstrap_mean_ci,
    evidence_recall,
    is_correct,
    normalize,
    paired_bootstrap_diff,
    retrieval_scores,
    token_f1,
    unique_urls,
    wilson_interval,
)


def test_normalize_lowercases_and_turns_punctuation_into_spaces() -> None:
    assert normalize("  Sam Bankman-Fried! ") == "sam bankman fried"


@pytest.mark.parametrize(
    ("answer", "gold", "expected"),
    [
        ("Sam Bankman-Fried\nHe is on trial [1].", "Sam Bankman-Fried", True),
        ("sam bankman fried", "Sam Bankman-Fried", True),  # case and punctuation ignored
        ("Caroline Ellison\n[1]", "Sam Bankman-Fried", False),
        ("Metaverse\n[1]", "Meta", False),  # whole words only
        ("The answer is Meta\n[1]", "Meta", True),
    ],
)
def test_name_answers(answer: str, gold: str, expected: bool) -> None:
    assert is_correct(answer, gold, "inference_query") is expected


@pytest.mark.parametrize(
    ("answer", "gold", "expected"),
    [
        ("Yes\nBoth reported it [1].", "Yes", True),
        ("Yes, both outlets agree [1]", "Yes", True),  # only the first word counts
        ("No\n[1]", "Yes", False),
        ("Nothing suggests that [1]", "No", False),  # "Nothing" is not "No"
    ],
)
def test_yes_no_answers(answer: str, gold: str, expected: bool) -> None:
    assert is_correct(answer, gold, "comparison_query") is expected


def test_a_refusal_is_wrong_for_an_answerable_question() -> None:
    assert is_correct(NOT_FOUND, "Sam Bankman-Fried", "inference_query") is False


def test_null_questions_are_correct_only_when_refused() -> None:
    assert is_correct(f"{NOT_FOUND}\nNothing in the excerpts.", NOT_FOUND, "null_query") is True
    assert is_correct("Sam Bankman-Fried\n[1]", NOT_FOUND, "null_query") is False


@pytest.mark.parametrize(
    ("answer", "gold", "expected"),
    [
        ("Google\n[1]", "Google", 1.0),
        ("Everton\n[1]", "Everton Football Club", 0.5),  # P 1/1, R 1/3 -> 0.50
        ("Bankman-Fried\n[1]", "Sam Bankman-Fried", 0.8),  # P 2/2, R 2/3 -> 0.80
        ("The Verge\n[1]", "Verge", 1.0),  # articles are ignored, as in SQuAD
        ("Caroline Ellison\n[1]", "Sam Bankman-Fried", 0.0),
    ],
)
def test_token_f1_gives_partial_credit_on_names(answer: str, gold: str, expected: float) -> None:
    assert token_f1(answer, gold, "inference_query") == pytest.approx(expected)


def test_token_f1_is_all_or_nothing_for_yes_no() -> None:
    assert token_f1("Yes, both outlets agree [1]", "Yes", "comparison_query") == 1.0
    assert token_f1("No\n[1]", "Yes", "comparison_query") == 0.0


def test_token_f1_for_refusals() -> None:
    assert token_f1(NOT_FOUND, NOT_FOUND, "null_query") == 1.0
    assert token_f1("Google\n[1]", NOT_FOUND, "null_query") == 0.0
    assert token_f1(NOT_FOUND, "Google", "inference_query") == 0.0


def chunk(url: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=hash(url), title="t", source="s", url=url, published_at="2023-01-01",
        content="c", score=0.0,
    )


def test_unique_urls_keeps_rank_order_and_drops_repeats() -> None:
    assert unique_urls([chunk("a"), chunk("b"), chunk("a"), chunk("c")]) == ["a", "b", "c"]


def test_retrieval_scores() -> None:
    recall, all_hit, reciprocal_rank = retrieval_scores(["x", "b", "y"], {"b", "d"})
    assert recall == 0.5  # found b, missed d
    assert all_hit is False
    assert reciprocal_rank == 0.5  # first gold article at position 2


def test_retrieval_scores_all_found_at_the_top() -> None:
    assert retrieval_scores(["a", "b"], {"a", "b"}) == (1.0, True, 1.0)


def test_retrieval_scores_nothing_found() -> None:
    assert retrieval_scores(["x"], {"a"}) == (0.0, False, 0.0)


def test_evidence_recall() -> None:
    assert evidence_recall([chunk("a"), chunk("x")], {"a", "b"}) == 0.5
    assert evidence_recall([chunk("a")], set()) is None  # null questions have no gold


# --- confidence intervals ----------------------------------------------------------


@pytest.mark.parametrize(
    ("successes", "n", "low", "high"),
    [
        (20, 25, 0.609, 0.911),  # 80% of 25: wide
        (80, 100, 0.711, 0.866),  # 80% of 100: narrower
        (4, 5, 0.376, 0.964),  # 80% of 5: nearly useless
    ],
)
def test_wilson_interval_matches_reference_values(successes, n, low, high) -> None:
    interval = wilson_interval(successes, n)
    assert interval == pytest.approx((low, high), abs=0.001)


def test_wilson_interval_stays_inside_0_and_1_at_the_extremes() -> None:
    low, high = wilson_interval(10, 10)
    assert high == pytest.approx(1.0)
    assert 0.70 < low < 0.75  # 10/10 is still not proof of 100%
    low, high = wilson_interval(0, 10)
    assert low == pytest.approx(0.0)
    assert 0.25 < high < 0.30


def test_wilson_interval_of_nothing_is_none() -> None:
    assert wilson_interval(0, 0) is None


def test_bootstrap_of_identical_values_has_no_width() -> None:
    assert bootstrap_mean_ci([0.5] * 20) == (0.5, 0.5)


def test_bootstrap_is_reproducible_and_contains_the_mean() -> None:
    values = [1.0, 0.0, 0.5, 1.0, 1.0, 0.0, 0.5, 1.0]
    first = bootstrap_mean_ci(values)
    assert first == bootstrap_mean_ci(values)  # fixed seed -> same interval every run
    low, high = first
    assert low <= sum(values) / len(values) <= high


def test_paired_difference_of_identical_systems_is_zero() -> None:
    scores = [1.0, 0.0, 1.0, 1.0, 0.0]
    assert paired_bootstrap_diff(scores, scores) == (0.0, 0.0, 0.0)


def test_paired_difference_when_one_system_always_wins() -> None:
    assert paired_bootstrap_diff([1.0] * 10, [0.0] * 10) == (1.0, 1.0, 1.0)


def test_paired_difference_of_a_small_gain_crosses_zero() -> None:
    # A wins on 2 questions, B on 1, the other 22 are ties: +0.04 but not convincing.
    a = [1.0, 1.0, 0.0] + [1.0] * 22
    b = [0.0, 0.0, 1.0] + [1.0] * 22
    difference, low, high = paired_bootstrap_diff(a, b)
    assert difference == pytest.approx(0.04)
    assert low < 0 < high


def test_paired_difference_needs_equal_lengths() -> None:
    with pytest.raises(ValueError):
        paired_bootstrap_diff([1.0, 0.0], [1.0])
