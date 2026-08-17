"""The seam between an estimator and the adapters that wrap it.

``performance/`` used to bind to ``ScoringMethod``'s private ``_prob`` and
``_unobs`` attributes rather than to its interface. Four defects followed from
that, and each is pinned here:

1. a :class:`~freqprob.VectorizedScorer` kept serving scores from a snapshot
   taken at construction, so it disagreed with the estimator after a re-fit;
2. wrapping a lazy scorer produced all-zero scores, because a lazy scorer's
   table is empty until something asks for an estimate;
3. :class:`~freqprob.StreamingMLE` inherited ``fit()`` but overrode
   ``_compute_probabilities`` with an empty body, so re-fitting destroyed the
   model instead of replacing it;
4. :class:`~freqprob.StreamingLaplace` reached the MLE increment through
   ``update_single`` while ``update_batch`` used the Laplace formula, so the
   two entry points disagreed.

The interface these tests exercise is ``support()`` / ``unobserved_score()`` /
``revision``: the facts an adapter needs in order to index an estimator without
reaching past it.
"""

import pytest

from freqprob import (
    MLE,
    Laplace,
    StreamingLaplace,
    StreamingMLE,
    VectorizedScorer,
    create_lazy_laplace,
    create_lazy_mle,
)
from freqprob.base import Element, ScoringMethod


class TestEstimatorInterface:
    """``support()``, ``unobserved_score()`` and ``revision`` on the base."""

    def test_support_lists_the_scored_elements(self) -> None:
        """The support is the set of elements with an explicit estimate."""
        scorer = MLE({"a": 2, "b": 1}, logprob=False)
        assert sorted(scorer.support()) == ["a", "b"]

    def test_unobserved_score_matches_scoring_an_unseen_element(self) -> None:
        """The reserved score is what an unseen element actually gets."""
        scorer = Laplace({"a": 2, "b": 1}, bins=10, logprob=False)
        assert scorer.unobserved_score() == scorer("definitely-unseen")

    def test_revision_advances_on_refit(self) -> None:
        """Re-fitting changes the estimates, so the revision moves."""
        scorer = MLE({"a": 2}, logprob=False)
        before = scorer.revision
        scorer.fit({"b": 3})
        assert scorer.revision > before

    def test_revision_advances_on_streaming_update(self) -> None:
        """A streaming update changes the estimates, so the revision moves."""
        scorer = StreamingMLE(initial_freqdist={"a": 2}, logprob=False)
        before = scorer.revision
        scorer.update_single("a", 1)
        assert scorer.revision > before

    @pytest.mark.parametrize(
        ("factory", "kwargs"),
        [
            (MLE, {}),
            (Laplace, {"bins": 10}),
            (create_lazy_mle, {}),
            (create_lazy_laplace, {}),
        ],
    )
    def test_support_and_scores_agree(self, factory: object, kwargs: dict[str, object]) -> None:
        """Every element in the support scores as something other than the reserved value."""
        scorer = factory({"a": 3, "b": 2, "c": 1}, logprob=False, **kwargs)  # type: ignore[operator]
        support = list(scorer.support())

        assert sorted(support) == ["a", "b", "c"]
        assert scorer.predict(support) == [scorer(element) for element in support]


class TestVectorizedScorerTracksItsEstimator:
    """Defects 1 and 2: the vectorized adapter must not serve a stale snapshot."""

    def test_reflects_a_refit(self) -> None:
        """Scores follow the estimator instead of a construction-time snapshot."""
        scorer = MLE({"a": 3, "b": 1}, logprob=False)
        vectorized = VectorizedScorer(scorer)
        assert vectorized.score_batch(["a"])[0] == pytest.approx(0.75)

        scorer.fit({"a": 1, "b": 99})

        assert vectorized.score_batch(["a"])[0] == pytest.approx(scorer("a"))
        assert vectorized.score_batch(["a"])[0] == pytest.approx(0.01)

    def test_reflects_a_streaming_update(self) -> None:
        """Streaming updates reach the vectorized index too."""
        scorer = StreamingMLE(initial_freqdist={"a": 3, "b": 1}, logprob=False)
        vectorized = VectorizedScorer(scorer)
        assert vectorized.score_batch(["a"])[0] == pytest.approx(0.75)

        scorer.update_single("b", 4)

        assert vectorized.score_batch(["a"])[0] == pytest.approx(scorer("a"))

    def test_top_k_reflects_a_refit(self) -> None:
        """top_k re-indexes after the estimator changes."""
        scorer = MLE({"a": 9, "b": 1}, logprob=False)
        vectorized = VectorizedScorer(scorer)
        assert vectorized.top_k_elements(1)[0] == ["a"]

        scorer.fit({"a": 1, "b": 9})

        assert vectorized.top_k_elements(1)[0] == ["b"]

    @pytest.mark.parametrize("factory", [create_lazy_mle, create_lazy_laplace])
    def test_wrapping_a_lazy_scorer_forces_its_estimates(self, factory: object) -> None:
        """Defect 2: this returned ``[0.0, 0.0]`` while the scorer itself was fine."""
        lazy = factory({"a": 3, "b": 1}, logprob=False)  # type: ignore[operator]
        vectorized = VectorizedScorer(lazy)

        scores = vectorized.score_batch(["a", "b"])

        assert scores[0] == pytest.approx(lazy("a"))
        assert scores[1] == pytest.approx(lazy("b"))
        assert scores[0] > 0.0

    def test_lazy_support_is_the_whole_distribution(self) -> None:
        """Laziness is about when an estimate is produced, not which elements have one."""
        lazy = create_lazy_mle({"a": 3, "b": 1}, logprob=False)
        assert sorted(lazy.support()) == ["a", "b"]
        assert sorted(lazy.get_computed_elements()) == []


class TestStreamingHonorsTheEstimatorContract:
    """Defects 3 and 4: the streaming estimators are adapters, not a parallel hierarchy."""

    def test_refit_replaces_the_stream(self) -> None:
        """Defect 3: ``fit()`` used to leave the scorer with an empty table."""
        scorer = StreamingMLE(initial_freqdist={"a": 3, "b": 1}, logprob=False)
        assert scorer("a") == pytest.approx(0.75)

        scorer.fit({"q": 5})

        assert scorer("q") == pytest.approx(1.0)
        assert scorer("a") == scorer.unobserved_score()
        assert sorted(scorer.support()) == ["q"]

    def test_refit_behaves_like_a_fresh_fit(self) -> None:
        """The base contract: a re-fit equals a fresh fit."""
        refitted = StreamingMLE(initial_freqdist={"a": 3, "b": 1}, logprob=False)
        refitted.fit({"x": 2, "y": 2})
        fresh = StreamingMLE(initial_freqdist={"x": 2, "y": 2}, logprob=False)

        for element in ("x", "y", "unseen"):
            assert refitted(element) == pytest.approx(fresh(element))

    def test_streaming_updates_continue_after_a_refit(self) -> None:
        """A re-fit reseeds the stream rather than disabling it."""
        scorer = StreamingMLE(initial_freqdist={"a": 1}, logprob=False)
        scorer.fit({"x": 1})
        scorer.update_single("y", 1)

        assert scorer("x") == pytest.approx(0.5)
        assert scorer("y") == pytest.approx(0.5)

    @pytest.mark.parametrize("logprob", [False, True])
    def test_single_and_batch_updates_agree(self, logprob: bool) -> None:
        """Defect 4: ``update_single`` ran MLE math inside a Laplace estimator."""
        by_single = StreamingLaplace(initial_freqdist={"a": 3, "b": 1}, logprob=logprob)
        by_single.update_single("a", 1)

        by_batch = StreamingLaplace(initial_freqdist={"a": 3, "b": 1}, logprob=logprob)
        by_batch.update_batch(["a"], [1])

        fresh = StreamingLaplace(initial_freqdist={"a": 4, "b": 1}, logprob=logprob)

        for element in ("a", "b", "unseen"):
            assert by_single(element) == pytest.approx(fresh(element))
            assert by_batch(element) == pytest.approx(fresh(element))

    def test_streaming_mle_single_and_batch_updates_agree(self) -> None:
        """Both update paths run one definition of the mathematics."""
        by_single = StreamingMLE(initial_freqdist={"a": 3, "b": 1}, logprob=False)
        by_single.update_single("a", 1)

        by_batch = StreamingMLE(initial_freqdist={"a": 3, "b": 1}, logprob=False)
        by_batch.update_batch(["a"], [1])

        for element in ("a", "b"):
            assert by_single(element) == pytest.approx(by_batch(element))

    def test_probabilities_stay_normalized_after_updates(self) -> None:
        """Repeated single updates keep the estimates a distribution."""
        scorer = StreamingMLE(initial_freqdist={"a": 3, "b": 1}, logprob=False)
        for _ in range(5):
            scorer.update_single("c", 1)

        total = sum(scorer(element) for element in scorer.support())
        assert total == pytest.approx(1.0)


class TestAdaptersUseOnlyTheInterface:
    """No adapter should need the estimator's private attributes."""

    def test_vectorized_scorer_needs_no_private_state(self) -> None:
        """Any ScoringMethod can be wrapped, with no extra attributes."""

        class MinimalScorer(ScoringMethod):
            """An estimator exposing nothing beyond the interface."""

            def _compute_probabilities(self, freqdist: dict[Element, int]) -> None:
                total = sum(freqdist.values())
                self._prob = {element: count / total for element, count in freqdist.items()}
                self._unobs = 0.0

        scorer = MinimalScorer.__new__(MinimalScorer)
        ScoringMethod.__init__(scorer, MLE({"a": 1}, logprob=False).config)
        scorer.name = "Minimal"
        scorer.fit({"a": 3, "b": 1})

        vectorized = VectorizedScorer(scorer)
        assert vectorized.score_batch(["a", "b", "z"]) == pytest.approx([0.75, 0.25, 0.0])
