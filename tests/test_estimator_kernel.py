"""What every estimator gets from the base rather than writing itself.

``ScoringMethod`` used to be a protocol: its one hook told a subclass to
populate two private attributes, so all eighteen estimators re-derived the same
mechanics -- the log/plain fork, the ``log(0)`` floor, the total count -- and
each wrote its formula twice, once per output space.

The hook now returns the estimates instead. ``_estimate`` works in plain
probability space and ``fit`` installs the result, so the duality lives in
:meth:`~freqprob.base.ScoringMethod._score_of` alone. These tests hold that
line: an estimator's log-scores must be exactly the logs of its plain scores,
for every estimator, because both come from one computation.
"""

import math

import pytest

from freqprob import (
    ELE,
    MLE,
    AbsoluteDiscounting,
    Bayesian,
    CertaintyDegree,
    KatzBackoff,
    KneserNey,
    Laplace,
    Lidstone,
    ModifiedKneserNey,
    PitmanYor,
    Random,
    SimpleGoodTuring,
    StupidBackoff,
    Uniform,
    WittenBell,
    clear_all_caches,
)
from freqprob.base import MIN_PROBABILITY, ScoringMethod

UNIGRAMS = {"a": 8, "b": 5, "c": 3, "d": 2, "e": 1, "f": 1}
BIGRAMS = {
    ("the", "cat"): 5,
    ("the", "dog"): 3,
    ("a", "cat"): 2,
    ("a", "dog"): 1,
    ("big", "cat"): 1,
    ("small", "dog"): 1,
}

# Every estimator, with the arguments it needs and a distribution it accepts.
ESTIMATORS = [
    ("Uniform", lambda logprob: Uniform(UNIGRAMS, logprob=logprob)),
    ("Random", lambda logprob: Random(UNIGRAMS, seed=42, logprob=logprob)),
    ("MLE", lambda logprob: MLE(UNIGRAMS, logprob=logprob)),
    ("Lidstone", lambda logprob: Lidstone(UNIGRAMS, gamma=0.3, bins=50, logprob=logprob)),
    ("Laplace", lambda logprob: Laplace(UNIGRAMS, bins=50, logprob=logprob)),
    ("ELE", lambda logprob: ELE(UNIGRAMS, bins=50, logprob=logprob)),
    ("Bayesian", lambda logprob: Bayesian(UNIGRAMS, alpha=0.5, logprob=logprob)),
    ("WittenBell", lambda logprob: WittenBell(UNIGRAMS, bins=50, logprob=logprob)),
    ("CertaintyDegree", lambda logprob: CertaintyDegree(UNIGRAMS, bins=50, logprob=logprob)),
    ("SimpleGoodTuring", lambda logprob: SimpleGoodTuring(UNIGRAMS, bins=50, logprob=logprob)),
    ("KneserNey", lambda logprob: KneserNey(BIGRAMS, discount=0.75, logprob=logprob)),
    ("ModifiedKneserNey", lambda logprob: ModifiedKneserNey(BIGRAMS, logprob=logprob)),
    ("AbsoluteDiscounting", lambda logprob: AbsoluteDiscounting(BIGRAMS, logprob=logprob)),
    ("PitmanYor", lambda logprob: PitmanYor(BIGRAMS, strength=1.0, logprob=logprob)),
    ("KatzBackoff", lambda logprob: KatzBackoff(BIGRAMS, logprob=logprob)),
    ("StupidBackoff", lambda logprob: StupidBackoff(BIGRAMS, logprob=logprob)),
]

IDS = [name for name, _ in ESTIMATORS]
FACTORIES = [factory for _, factory in ESTIMATORS]


@pytest.fixture(autouse=True)
def _clear_caches() -> None:
    """Keep memoized estimates from leaking between parametrized cases."""
    clear_all_caches()


@pytest.mark.parametrize("factory", FACTORIES, ids=IDS)
class TestOneComputationTwoOutputSpaces:
    """A log-scorer and a plain scorer must agree exactly."""

    def test_log_scores_are_logs_of_plain_scores(self, factory: object) -> None:
        """The fork is gone, so the two output spaces cannot drift apart."""
        plain = factory(False)  # type: ignore[operator]
        logged = factory(True)  # type: ignore[operator]

        probed = [*plain.support(), "definitely-unseen", ("no", "such", "gram")]
        for element in probed:
            expected = plain(element)
            if expected <= 0.0:
                expected = MIN_PROBABILITY
            assert logged(element) == pytest.approx(math.log(expected))

    def test_plain_scores_are_probabilities(self, factory: object) -> None:
        """Nothing in the plain output space escapes ``[0, 1]``."""
        plain = factory(False)  # type: ignore[operator]

        for element in plain.support():
            score = plain(element)
            assert 0.0 <= score <= 1.0
            assert not math.isnan(score)

    def test_log_scores_are_finite(self, factory: object) -> None:
        """The floor keeps ``log`` in its domain without producing infinities."""
        logged = factory(True)  # type: ignore[operator]

        for element in [*logged.support(), "definitely-unseen"]:
            score = logged(element)
            assert math.isfinite(score)
            assert score <= 0.0

    def test_refit_equals_a_fresh_fit(self, factory: object) -> None:
        """``fit`` resets every piece of state an estimator carries."""
        refitted = factory(False)  # type: ignore[operator]
        fresh = factory(False)  # type: ignore[operator]

        # Fit to something else entirely, then back to the original.
        refitted.fit({"zzz": 4, "yyy": 1})
        refitted.fit(BIGRAMS if isinstance(next(iter(fresh.support())), tuple) else UNIGRAMS)

        assert sorted(map(str, refitted.support())) == sorted(map(str, fresh.support()))
        for element in fresh.support():
            assert refitted(element) == pytest.approx(fresh(element))


class TestEstimateHookContract:
    """``_estimate`` returns plain probabilities; the base does the rest."""

    @pytest.mark.parametrize("factory", FACTORIES, ids=IDS)
    def test_estimate_returns_plain_probabilities(self, factory: object) -> None:
        """Whatever the output space, the hook reports probabilities."""
        logged = factory(True)  # type: ignore[operator]

        table, reserved = logged._estimate(
            BIGRAMS if isinstance(next(iter(logged.support())), tuple) else UNIGRAMS
        )

        assert 0.0 <= reserved <= 1.0
        for prob in table.values():
            assert 0.0 <= prob <= 1.0

    def test_estimate_does_not_install_anything(self) -> None:
        """The hook is a computation, not a mutation: ``fit`` installs."""
        scorer = MLE(UNIGRAMS, logprob=False)
        before = dict(scorer._prob)

        scorer._estimate({"totally": 1, "different": 2})

        assert dict(scorer._prob) == before

    def test_score_of_floors_only_zero(self) -> None:
        """A small-but-genuine probability survives; only zero is substituted."""
        logged = MLE(UNIGRAMS, logprob=True)

        assert logged._score_of(2.5e-11) == pytest.approx(math.log(2.5e-11))
        assert logged._score_of(0.0) == pytest.approx(math.log(MIN_PROBABILITY))

    def test_score_of_is_identity_for_plain_scorers(self) -> None:
        """A plain scorer can still report an exact zero."""
        plain = MLE(UNIGRAMS, logprob=False)

        assert plain._score_of(0.0) == 0.0
        assert plain._score_of(0.25) == 0.25

    def test_subclasses_implement_only_estimate(self) -> None:
        """``_estimate`` is the single abstract hook on the base."""
        abstract = ScoringMethod.__abstractmethods__
        assert abstract == frozenset({"_estimate"})


class TestSharedCacheAcrossOutputSpaces:
    """Because ``_estimate`` is logprob-independent, one entry serves both."""

    def test_a_plain_scorer_reuses_a_log_scorers_estimates(self) -> None:
        """The cache key no longer splits on ``logprob``."""
        clear_all_caches()

        SimpleGoodTuring(UNIGRAMS, bins=50, logprob=True)
        after_first = _sgt_cache_size()

        SimpleGoodTuring(UNIGRAMS, bins=50, logprob=False)
        after_second = _sgt_cache_size()

        assert after_first == 1
        assert after_second == after_first

    def test_cached_estimates_are_not_shared_between_instances(self) -> None:
        """A cache hit hands back a copy, so one scorer cannot corrupt another."""
        clear_all_caches()

        first = SimpleGoodTuring(UNIGRAMS, bins=50, logprob=False)
        second = SimpleGoodTuring(UNIGRAMS, bins=50, logprob=False)

        assert first._prob is not second._prob
        assert first._prob == second._prob

    def test_total_unseen_mass_survives_a_cache_hit(self) -> None:
        """Simple Good-Turing's extra output is cached alongside the estimates."""
        clear_all_caches()

        first = SimpleGoodTuring(UNIGRAMS, bins=50, logprob=False)
        second = SimpleGoodTuring(UNIGRAMS, bins=50, logprob=False)

        assert second.total_unseen_mass == pytest.approx(first.total_unseen_mass)
        assert second.total_unseen_mass > 0.0


def _sgt_cache_size() -> int:
    """Return the number of entries in the Simple Good-Turing cache."""
    from freqprob import get_cache_stats

    return int(get_cache_stats()["sgt_cache_size"])
