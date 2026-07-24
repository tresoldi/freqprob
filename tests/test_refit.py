"""Re-fitting an estimator must behave identically to a fresh fit on the same data."""

# mypy: disable-error-code="arg-type,no-untyped-def"

import pytest

import freqprob

# data B differs from data A: some shared keys, some A-only, some B-only
DATA_A = {"a": 5, "b": 3, "c": 2}
DATA_B = {"b": 4, "c": 1, "d": 6, "e": 2}
PROBE = ["a", "b", "c", "d", "e", "unseen"]

# Chosen so a stale (c, w1) key from A (~0.333) differs from B's unseen mass
# (~0.5), making the stale-key bug observable rather than coincidentally masked.
BIGRAMS_A = {("c", "w1"): 1, ("c", "w2"): 1, ("c", "w3"): 1}
BIGRAMS_B = {("x", "b"): 5, ("y", "b"): 2, ("z", "d"): 1}
BIGRAM_PROBE = [("c", "w1"), ("x", "b"), ("z", "d"), ("q", "q")]


def _refit(factory, data_a, data_b):
    """Return a scorer built on A then re-fit on B, forcing a real recompute.

    Caches are cleared before the re-fit so the second fit(B) is a cache miss
    that actually re-runs _compute_probabilities on top of A's state (this is
    what surfaces stale-key bugs; a cache hit would mask them).
    """
    scorer = factory(data_a)
    freqprob.clear_all_caches()
    scorer.fit(data_b)
    return scorer


@pytest.mark.parametrize(
    ("factory", "data_a", "data_b", "probe"),
    [
        (lambda d: freqprob.MLE(d, logprob=False), DATA_A, DATA_B, PROBE),
        (lambda d: freqprob.Uniform(d, logprob=False), DATA_A, DATA_B, PROBE),
        (lambda d: freqprob.Laplace(d, bins=100, logprob=False), DATA_A, DATA_B, PROBE),
        (lambda d: freqprob.Lidstone(d, gamma=0.5, bins=100, logprob=False), DATA_A, DATA_B, PROBE),
        (lambda d: freqprob.ELE(d, bins=100, logprob=False), DATA_A, DATA_B, PROBE),
        (lambda d: freqprob.Bayesian(d, alpha=1.0, logprob=False), DATA_A, DATA_B, PROBE),
        (lambda d: freqprob.CertaintyDegree(d, logprob=False), DATA_A, DATA_B, PROBE),
        (lambda d: freqprob.WittenBell(d, logprob=False), DATA_A, DATA_B, PROBE),
        (lambda d: freqprob.SimpleGoodTuring(d, logprob=False), DATA_A, DATA_B, PROBE),
        (
            lambda d: freqprob.KneserNey(d, discount=0.5, logprob=False),
            BIGRAMS_A,
            BIGRAMS_B,
            BIGRAM_PROBE,
        ),
    ],
)
def test_refit_matches_fresh_fit(factory, data_a, data_b, probe) -> None:
    """Fitting on A then B must equal a fresh estimator fitted only on B."""
    freqprob.clear_all_caches()
    fresh = factory(data_b)
    refit = _refit(factory, data_a, data_b)
    for element in probe:
        assert refit(element) == pytest.approx(fresh(element)), f"mismatch at {element!r}"


def test_refit_logprob_does_not_raise() -> None:
    """Re-fitting in log-space must not raise a math domain error (the reported bug)."""
    scorer = freqprob.MLE({"a": 1})  # logprob=True by default
    scorer.fit({"a": 2, "b": 1})  # must not raise
    assert scorer.score("a") == scorer("a")
