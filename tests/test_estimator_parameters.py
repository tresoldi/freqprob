"""Method-specific parameters: where they live, and that they are validated.

Each method used to carry a ``*Config`` dataclass of its own -- eight of them,
adding one or two fields each and no behaviour. None overrode ``__post_init__``,
so the shared config validated only the fields no subclass cared about, and
every method-specific range check sat in the estimator's ``__init__`` instead.
Reading such a field needed a ``# type: ignore`` at fifteen call sites, because
``self.config`` is typed as the base class.

A method's own parameters are now attributes on the estimator. These tests hold
that: the parameter is reachable and typed, it is validated where it arrives,
and the memoization key still depends on it.
"""

import pytest

from freqprob import (
    MLE,
    AbsoluteDiscounting,
    Bayesian,
    CertaintyDegree,
    Interpolated,
    KatzBackoff,
    KneserNey,
    Laplace,
    Lidstone,
    PitmanYor,
    Random,
    SimpleGoodTuring,
    StupidBackoff,
    Uniform,
    clear_all_caches,
    get_cache_stats,
)
from freqprob.base import ScoringMethodConfig

UNIGRAMS = {"a": 8, "b": 5, "c": 3, "d": 2, "e": 1, "f": 1}
BIGRAMS = {
    ("the", "cat"): 5,
    ("the", "dog"): 3,
    ("a", "cat"): 2,
    ("a", "dog"): 1,
    ("big", "cat"): 1,
    ("small", "dog"): 1,
}
TRIGRAMS = {("the", "big", "cat"): 3, ("a", "big", "dog"): 2}
LOW_ORDER = {("big", "cat"): 5, ("big", "dog"): 3}


class TestParametersLiveOnTheEstimator:
    """A method's own parameter is an attribute of that method."""

    @pytest.mark.parametrize(
        ("scorer", "attribute", "expected"),
        [
            (Random(UNIGRAMS, seed=42, logprob=False), "seed", 42),
            (Bayesian(UNIGRAMS, alpha=0.25, logprob=False), "alpha", 0.25),
            (Lidstone(UNIGRAMS, gamma=1.5, logprob=False), "gamma", 1.5),
            (KneserNey(BIGRAMS, discount=0.6, logprob=False), "discount", 0.6),
            (AbsoluteDiscounting(BIGRAMS, discount=0.6, logprob=False), "discount", 0.6),
            (PitmanYor(BIGRAMS, discount=0.4, strength=2.0, logprob=False), "strength", 2.0),
            (KatzBackoff(BIGRAMS, k=3, logprob=False), "k", 3),
            (StupidBackoff(BIGRAMS, alpha=0.3, logprob=False), "alpha", 0.3),
            (SimpleGoodTuring(UNIGRAMS, p_value=0.01, bins=50, logprob=False), "p_value", 0.01),
            (
                Interpolated(TRIGRAMS, LOW_ORDER, lambda_weight=0.4, logprob=False),
                "lambda_weight",
                0.4,
            ),
        ],
    )
    def test_parameter_is_readable(self, scorer: object, attribute: str, expected: object) -> None:
        """No ``# type: ignore`` needed to read it back."""
        assert getattr(scorer, attribute) == expected

    def test_the_shared_config_carries_only_shared_fields(self) -> None:
        """Three fields, each of which more than one method actually uses."""
        import dataclasses

        names = {field.name for field in dataclasses.fields(ScoringMethodConfig)}
        assert names == {"unobs_prob", "bins", "logprob"}

    def test_estimators_still_use_slots(self) -> None:
        """Moving parameters onto the estimator did not reintroduce ``__dict__``."""
        for scorer in (
            MLE(UNIGRAMS, logprob=False),
            Lidstone(UNIGRAMS, gamma=1.0, logprob=False),
            KneserNey(BIGRAMS, logprob=False),
            SimpleGoodTuring(UNIGRAMS, bins=50, logprob=False),
        ):
            assert not hasattr(scorer, "__dict__"), type(scorer).__name__


class TestParametersAreValidatedWhereTheyArrive:
    """Every method-specific range check raises, from the constructor."""

    @pytest.mark.parametrize(
        ("build", "message"),
        [
            (lambda: Lidstone(UNIGRAMS, gamma=-1.0), r"[Gg]amma"),
            (lambda: Bayesian(UNIGRAMS, alpha=0.0), r"[Aa]lpha"),
            (lambda: Bayesian(UNIGRAMS, alpha=-1.0), r"[Aa]lpha"),
            (lambda: KneserNey(BIGRAMS, discount=0.0), r"[Dd]iscount"),
            (lambda: KneserNey(BIGRAMS, discount=1.0), r"[Dd]iscount"),
            (lambda: AbsoluteDiscounting(BIGRAMS, discount=1.0), r"[Dd]iscount"),
            (lambda: PitmanYor(BIGRAMS, discount=-0.1), r"[Dd]iscount"),
            (lambda: KatzBackoff(BIGRAMS, k=0), r"k"),
            (lambda: StupidBackoff(BIGRAMS, alpha=0.0), r"[Aa]lpha"),
            (lambda: StupidBackoff(BIGRAMS, alpha=1.5), r"[Aa]lpha"),
            (lambda: SimpleGoodTuring(UNIGRAMS, p_value=0.0), r"p-value"),
            (lambda: SimpleGoodTuring(UNIGRAMS, p_value=1.0), r"p-value"),
            (lambda: Interpolated(TRIGRAMS, LOW_ORDER, lambda_weight=1.5), r"[Ll]ambda"),
            (lambda: Interpolated(TRIGRAMS, LOW_ORDER, lambda_weight=-0.1), r"[Ll]ambda"),
        ],
    )
    def test_invalid_parameter_raises(self, build: object, message: str) -> None:
        """The check runs at construction, not at first score."""
        with pytest.raises(ValueError, match=message):
            build()  # type: ignore[operator]

    @pytest.mark.parametrize(
        "build",
        [
            lambda: Uniform(UNIGRAMS, unobs_prob=1.5),
            lambda: MLE(UNIGRAMS, unobs_prob=-0.1),
            lambda: CertaintyDegree(UNIGRAMS, unobs_prob=2.0),
        ],
    )
    def test_shared_reserved_mass_is_validated(self, build: object) -> None:
        """The shared config still validates the fields it kept."""
        with pytest.raises(ValueError, match=r"reserved.*probability.*between"):
            build()  # type: ignore[operator]

    @pytest.mark.parametrize(
        "build",
        [
            lambda: Laplace(UNIGRAMS, bins=0),
            lambda: Lidstone(UNIGRAMS, gamma=1.0, bins=-5),
            lambda: CertaintyDegree(UNIGRAMS, bins=0),
        ],
    )
    def test_shared_bins_is_validated(self, build: object) -> None:
        """Likewise for ``bins``."""
        with pytest.raises(ValueError, match=r"bins.*positive"):
            build()  # type: ignore[operator]


class TestMemoizationStillDependsOnParameters:
    """The cache key is written down rather than reflected over the config."""

    def test_a_changed_parameter_is_a_different_entry(self) -> None:
        """It used to be derived from ``dir(self.config)``."""
        clear_all_caches()

        SimpleGoodTuring(UNIGRAMS, p_value=0.05, bins=50, logprob=False)
        after_first = get_cache_stats()["sgt_cache_size"]

        SimpleGoodTuring(UNIGRAMS, p_value=0.01, bins=50, logprob=False)
        after_second = get_cache_stats()["sgt_cache_size"]

        assert after_first == 1
        assert after_second == 2

    def test_the_same_parameters_reuse_one_entry(self) -> None:
        """And identical parameters still hit."""
        clear_all_caches()

        first = SimpleGoodTuring(UNIGRAMS, p_value=0.05, bins=50, logprob=False)
        second = SimpleGoodTuring(UNIGRAMS, p_value=0.05, bins=50, logprob=False)

        assert get_cache_stats()["sgt_cache_size"] == 1
        for element in first.support():
            assert second(element) == pytest.approx(first(element))

    def test_a_changed_parameter_changes_the_estimates(self) -> None:
        """The proof that the key matters: the numbers differ."""
        clear_all_caches()

        loose = SimpleGoodTuring(UNIGRAMS, p_value=0.5, bins=50, logprob=False)
        tight = SimpleGoodTuring(UNIGRAMS, p_value=0.001, bins=50, logprob=False)

        assert loose.total_unseen_mass == pytest.approx(tight.total_unseen_mass)
        # Both are valid distributions regardless of the confidence level.
        assert 0.0 < loose("a") <= 1.0
        assert 0.0 < tight("a") <= 1.0
