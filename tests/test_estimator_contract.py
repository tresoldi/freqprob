"""The contract every estimator satisfies, checked once over every estimator.

This suite replaces per-method copies of the same assertions. It had been
written out by hand three times as a ``getattr(freqprob, name)`` loop over a
short list, four times for log/linear agreement, and three times for pickle
round-trips -- with the result that half the estimators were missing from half
of those suites, and twenty-two functions in ``test_freqprob.py`` were
byte-for-byte duplicates of one another.

The registry lives in ``conftest.py``. Adding an estimator adds one row there
and earns every property below. What this suite deliberately does *not* check is
the mathematics: that stays in the golden-value tests, per method, because it is
the part no shared contract can express.
"""

import math
import pickle
from typing import ClassVar

import pytest
from conftest import UNIGRAMS, UNSEEN, EstimatorCase

from freqprob.base import ScoringMethod


class TestScoresAreWellFormed:
    """Whatever the method, a score is a probability or the log of one."""

    def test_plain_scores_are_probabilities(self, estimator_case: EstimatorCase) -> None:
        """Nothing in the plain output space escapes ``[0, 1]``."""
        scorer = estimator_case.make(False)

        for element in scorer.support():
            score = scorer(element)
            assert not math.isnan(score)
            assert 0.0 <= score <= 1.0

    def test_log_scores_are_finite_and_non_positive(self, estimator_case: EstimatorCase) -> None:
        """A log-probability is finite and at most zero."""
        scorer = estimator_case.make(True)

        for element in [*scorer.support(), UNSEEN]:
            score = scorer(element)
            assert math.isfinite(score)
            assert score <= 0.0

    def test_log_and_plain_scorers_agree(self, estimator_case: EstimatorCase) -> None:
        """``log`` of the plain score is the log score, for every element.

        The two used to be separate branches in each method, free to drift.
        """
        plain = estimator_case.make(False)
        logged = estimator_case.make(True)

        for element in [*plain.support(), UNSEEN]:
            expected = plain(element)
            if expected <= 0.0:
                continue  # MLE-style zero: the log scorer reports its floor
            assert logged(element) == pytest.approx(math.log(expected))

    def test_unseen_element_gets_the_reserved_score(self, estimator_case: EstimatorCase) -> None:
        """An element outside the support is scored by the reserved mass."""
        scorer = estimator_case.make(False)

        assert scorer(UNSEEN) == scorer.unobserved_score()

    def test_reserved_mass_is_positive_where_claimed(self, estimator_case: EstimatorCase) -> None:
        """A smoothing method gives unseen elements somewhere to go."""
        if not estimator_case.reserves_mass:
            pytest.skip(f"{estimator_case.name} reserves no mass by design")

        scorer = estimator_case.make(False)
        assert scorer.unobserved_score() > 0.0

    def test_more_frequent_elements_score_at_least_as_high(
        self, estimator_case: EstimatorCase
    ) -> None:
        """Where a method orders by count, it does so consistently."""
        if not estimator_case.ordered:
            pytest.skip(f"{estimator_case.name} does not order by raw count")

        scorer = estimator_case.make(False)
        # UNIGRAMS is a -> 8, b -> 5, c -> 3, d -> 2, e -> 1.
        assert scorer("a") >= scorer("b") >= scorer("c") >= scorer("d") >= scorer("e")


class TestMassAccounting:
    """Probability mass adds up, and is left over for what was not seen."""

    def test_observed_mass_does_not_exceed_one(self, estimator_case: EstimatorCase) -> None:
        """A smoothing method leaves room for the elements it did not see."""
        if not estimator_case.normalized:
            pytest.skip(f"{estimator_case.name} produces ranking scores, not a distribution")

        scorer = estimator_case.make(False)

        if estimator_case.conditional:
            # A conditional estimator normalizes per context, not globally.
            per_context: dict[object, float] = {}
            for element in scorer.support():
                context = element[0]  # type: ignore[index]
                per_context[context] = per_context.get(context, 0.0) + scorer(element)
            for total in per_context.values():
                assert total <= 1.0 + 1e-9
        else:
            total = sum(scorer(element) for element in scorer.support())
            assert total <= 1.0 + 1e-9

    def test_a_larger_bins_leaves_mass_over(self) -> None:
        """Modelling more bins than were observed takes mass from the observed ones.

        Whether a reserving method leaves mass *strictly* over is a property of
        ``bins``, not of smoothing in general: with ``bins`` defaulting to the
        support size, the observed elements can legitimately account for all of
        it while an unseen element still scores above zero.
        """
        from freqprob import Laplace

        exact = Laplace(UNIGRAMS, bins=len(UNIGRAMS), logprob=False)
        roomy = Laplace(UNIGRAMS, bins=100, logprob=False)

        assert sum(exact(element) for element in exact.support()) == pytest.approx(1.0)
        assert sum(roomy(element) for element in roomy.support()) < 1.0
        assert roomy.unobserved_score() > 0.0


class TestExtremeMagnitudes:
    """The contract holds on counts spanning six orders of magnitude."""

    EXTREME: ClassVar[dict[str, int]] = {
        "very_frequent": 10**6,
        "frequent": 10**4,
        "medium": 10**2,
        "rare": 1,
        "very_rare": 1,
    }

    def test_no_overflow_or_underflow(self, estimator_case: EstimatorCase) -> None:
        """Neither output space produces NaN or an infinity."""
        plain = estimator_case.make(False)
        logged = estimator_case.make(True)
        try:
            plain.fit(self.EXTREME)
            logged.fit(self.EXTREME)
        except ValueError as exc:  # pragma: no cover - method rejects the shape
            pytest.skip(f"{estimator_case.name} rejects this distribution: {exc}")

        for element in [*plain.support(), UNSEEN]:
            assert not math.isnan(plain(element))
            assert not math.isinf(plain(element))
            assert math.isfinite(logged(element))

    def test_ordering_survives_extreme_counts(self, estimator_case: EstimatorCase) -> None:
        """A million-to-one ratio does not invert the frequency ordering."""
        if not estimator_case.ordered:
            pytest.skip(f"{estimator_case.name} does not order by raw count")

        scorer = estimator_case.make(False)
        scorer.fit(self.EXTREME)

        assert scorer("very_frequent") >= scorer("frequent") >= scorer("medium")
        assert scorer("medium") >= scorer("rare")


class TestSupportAndRevision:
    """The interface an adapter indexes a scorer through."""

    def test_support_is_scored_without_the_reserved_value(
        self, estimator_case: EstimatorCase
    ) -> None:
        """Every element of the support has an estimate of its own."""
        scorer = estimator_case.make(False)

        support = list(scorer.support())
        assert support
        assert all(scorer(element) >= 0.0 for element in support)

    def test_predict_matches_calling_one_at_a_time(self, estimator_case: EstimatorCase) -> None:
        """The batch alias is the single-element path, repeated."""
        scorer = estimator_case.make(False)
        elements = [*scorer.support(), UNSEEN]

        assert scorer.predict(elements) == [scorer(element) for element in elements]

    def test_score_is_an_alias_for_calling(self, estimator_case: EstimatorCase) -> None:
        """The scikit-learn-style alias adds a name, not a behaviour."""
        scorer = estimator_case.make(False)

        for element in [*scorer.support(), UNSEEN]:
            assert scorer.score(element) == scorer(element)

    def test_revision_advances_on_refit(self, estimator_case: EstimatorCase) -> None:
        """Anything derived from the estimates can tell that they changed."""
        scorer = estimator_case.make(False)
        before = scorer.revision

        scorer.fit(estimator_case.fit_data)

        assert scorer.revision > before


class TestRefitting:
    """``fit`` resets state: a re-fit equals a fresh fit."""

    def test_refit_equals_a_fresh_fit(self, estimator_case: EstimatorCase) -> None:
        """Fitting to other data and back leaves no trace of the detour."""
        refitted = estimator_case.make(False)
        fresh = estimator_case.make(False)

        refitted.fit({"zzz": 4, "yyy": 2, "xxx": 1})
        refitted.fit(estimator_case.fit_data)

        assert sorted(map(str, refitted.support())) == sorted(map(str, fresh.support()))
        for element in fresh.support():
            assert refitted(element) == pytest.approx(fresh(element))
        assert refitted.unobserved_score() == pytest.approx(fresh.unobserved_score())

    def test_refit_returns_self(self, estimator_case: EstimatorCase) -> None:
        """The scikit-learn-style alias supports chaining."""
        scorer = estimator_case.make(False)

        assert scorer.fit(estimator_case.fit_data) is scorer


class TestPersistence:
    """A fitted scorer survives a round trip, whatever its internal state."""

    def test_pickle_round_trip_preserves_scores(self, estimator_case: EstimatorCase) -> None:
        """Including the auxiliary state the bigram methods carry."""
        scorer = estimator_case.make(False)
        restored = pickle.loads(pickle.dumps(scorer))

        assert isinstance(restored, ScoringMethod)
        for element in [*scorer.support(), UNSEEN]:
            assert restored(element) == pytest.approx(scorer(element))

    def test_save_and_load_preserve_scores(
        self, estimator_case: EstimatorCase, tmp_path: object
    ) -> None:
        """The public persistence path, not just raw pickling."""
        scorer = estimator_case.make(False)
        path = tmp_path / "model.pkl"  # type: ignore[operator]

        scorer.save(path)
        restored = type(scorer).load(path)

        for element in [*scorer.support(), UNSEEN]:
            assert restored(element) == pytest.approx(scorer(element))

    def test_load_rejects_a_different_estimator(
        self, estimator_case: EstimatorCase, tmp_path: object
    ) -> None:
        """``load`` is typed: it will not hand back someone else's model."""
        from freqprob import Uniform

        scorer = estimator_case.make(False)
        if isinstance(scorer, Uniform):
            pytest.skip("Uniform is the decoy this test loads through")

        path = tmp_path / "model.pkl"  # type: ignore[operator]
        scorer.save(path)

        with pytest.raises(TypeError, match="does not contain"):
            Uniform.load(path)


class TestDescription:
    """``__str__`` reports the method and how much it holds."""

    def test_str_names_the_method_and_its_size(self, estimator_case: EstimatorCase) -> None:
        """One assertion replacing a hand-written list of every method."""
        plain = estimator_case.make(False)
        logged = estimator_case.make(True)

        assert "scorer" in str(plain)
        assert "log-scorer" in str(logged)
        assert f"{len(list(plain.support()))} elements" in str(plain)
