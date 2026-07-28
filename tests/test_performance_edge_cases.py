"""Edge-case coverage for the vectorized and lazy performance helpers.

``tests/test_efficiency.py`` exercises the common happy paths of
``freqprob.performance.vectorized`` and ``freqprob.performance.lazy``. This
module targets the branches those tests leave untouched: empty/degenerate
inputs, the log-probability code paths, unfitted-scorer guards, alternate
normalization methods, and the configured unobserved-probability behaviour of
the lazy computers.
"""

import math

import numpy as np
import pytest

from freqprob import MLE
from freqprob.base import ScoringMethodConfig
from freqprob.performance.lazy import (
    LazyBatchScorer,
    LazyLaplaceComputer,
    LazyMLEComputer,
    LazyScoringMethod,
    create_lazy_laplace,
    create_lazy_mle,
)
from freqprob.performance.vectorized import (
    BatchScorer,
    VectorizedScorer,
    create_vectorized_batch_scorer,
    elements_to_numpy,
    normalize_scores,
    scores_to_probabilities,
)

# These tests deliberately pass concrete dict/scorer types where the public
# signatures declare invariant Mapping/ScoringMethod parameters.
# mypy: disable-error-code=arg-type


class _EmptyScorer:
    """A scorer-like object with no usable ``_prob`` table."""

    def __call__(self, element: object) -> float:
        """Return a constant score for any element."""
        return 0.0


class TestVectorizedScorerEdgeCases:
    """Degenerate inputs to VectorizedScorer."""

    def test_scorer_without_prob_table(self) -> None:
        """A scorer lacking a _prob table produces default scores."""
        vectorized = VectorizedScorer(_EmptyScorer())
        scores = vectorized.score_batch(["a", "b"])
        assert np.array_equal(scores, np.array([0.0, 0.0]))

    def test_score_parallel(self) -> None:
        """score_parallel scores each batch independently."""
        scorer = MLE({"a": 3, "b": 2}, logprob=False)
        vectorized = VectorizedScorer(scorer)

        batches = [["a", "b"], ["a"], ["unknown"]]
        results = vectorized.score_parallel(batches)

        assert len(results) == 3
        assert results[0][0] == scorer("a")
        assert results[2][0] == scorer("unknown")

    def test_score_matrix_empty(self) -> None:
        """An empty 2D input returns an empty array."""
        scorer = MLE({"a": 1}, logprob=False)
        vectorized = VectorizedScorer(scorer)
        result = vectorized.score_matrix([])
        assert result.size == 0

    def test_score_matrix_with_empty_rows(self) -> None:
        """Empty rows are padded with the default probability."""
        scorer = MLE({"a": 3, "b": 2}, logprob=False)
        vectorized = VectorizedScorer(scorer)
        result = vectorized.score_matrix([["a", "b"], []])
        assert result.shape == (2, 2)
        assert result[0, 0] == scorer("a")

    def test_top_k_elements_empty(self) -> None:
        """top_k on an empty probability array returns empties."""
        vectorized = VectorizedScorer(_EmptyScorer())
        elements, scores = vectorized.top_k_elements(3)
        assert elements == []
        assert scores.size == 0

    def test_percentile_scores(self) -> None:
        """percentile_scores ranks elements against known scores."""
        scorer = MLE({"a": 5, "b": 3, "c": 2, "d": 1}, logprob=False)
        vectorized = VectorizedScorer(scorer)

        ranks = vectorized.percentile_scores(["a", "d"], [25.0, 50.0, 75.0])
        assert len(ranks) == 2
        # 'a' is the highest-scoring known element -> higher percentile than 'd'.
        assert ranks[0] > ranks[1]
        assert np.all((ranks >= 0) & (ranks <= 100))

    def test_percentile_scores_no_known_scores(self) -> None:
        """With no known scores, percentile ranks are all zero."""
        vectorized = VectorizedScorer(_EmptyScorer())
        ranks = vectorized.percentile_scores(["a", "b"], [50.0])
        assert np.array_equal(ranks, np.zeros(2))


class TestBatchScorerEdgeCases:
    """BatchScorer helpers not covered elsewhere."""

    def test_create_vectorized_batch_scorer(self) -> None:
        """The factory returns a configured BatchScorer."""
        scorers = {"mle": MLE({"a": 3, "b": 2}, logprob=False)}
        batch = create_vectorized_batch_scorer(scorers)
        assert isinstance(batch, BatchScorer)
        assert "mle" in batch.vectorized_scorers

    def test_benchmark_methods(self) -> None:
        """benchmark_methods returns non-negative timings per method."""
        scorers = {
            "mle": MLE({"a": 3, "b": 2}, logprob=False),
            "laplace": MLE({"a": 3, "b": 2}, logprob=False),
        }
        batch = BatchScorer(scorers)
        timings = batch.benchmark_methods(["a", "b", "unknown"], num_iterations=3)

        assert set(timings) == {"mle", "laplace"}
        assert all(t >= 0.0 for t in timings.values())


class TestNumpyConversionUtilities:
    """elements_to_numpy / normalize_scores branches."""

    def test_elements_to_numpy_empty(self) -> None:
        """An empty iterable yields an empty array."""
        assert elements_to_numpy([]).size == 0

    def test_elements_to_numpy_object_dtype(self) -> None:
        """Non str/int/float elements fall back to object dtype."""
        arr = elements_to_numpy([(1, 2), (3, 4)])  # type: ignore[list-item]
        assert arr.dtype == object

    def test_normalize_scores_softmax(self) -> None:
        """Softmax normalization produces a probability distribution."""
        result = normalize_scores(np.array([1.0, 2.0, 3.0]), "softmax")
        assert np.allclose(np.sum(result), 1.0)

    def test_normalize_scores_minmax_constant(self) -> None:
        """All-equal scores hit the min==max guard (uniform output)."""
        result = normalize_scores(np.array([5.0, 5.0, 5.0]), "minmax")
        assert np.allclose(result, np.ones(3) / 3)

    def test_normalize_scores_zscore_constant(self) -> None:
        """Zero standard deviation hits the std==0 guard (all zeros)."""
        result = normalize_scores(np.array([5.0, 5.0, 5.0]), "zscore")
        assert np.array_equal(result, np.zeros(3))

    def test_normalize_scores_unknown_method(self) -> None:
        """An unknown normalization method raises ValueError."""
        with pytest.raises(ValueError, match="Unknown normalization method"):
            normalize_scores(np.array([1.0, 2.0]), "nonsense")

    def test_scores_to_probabilities_stable(self) -> None:
        """Large magnitudes do not overflow (log-sum-exp)."""
        probs = scores_to_probabilities(np.array([1000.0, 1001.0, 1002.0]))
        assert np.allclose(np.sum(probs), 1.0)
        assert np.all(np.isfinite(probs))


class TestLazyComputers:
    """Direct tests of the lazy computation strategies."""

    def test_mle_zero_total_count(self) -> None:
        """An all-zero distribution yields probability 0 (no ZeroDivision)."""
        computer = LazyMLEComputer()
        config = ScoringMethodConfig(logprob=False)
        assert computer.compute_probability("a", {"a": 0}, config) == 0.0

    def test_mle_with_unobs_prob_observed(self) -> None:
        """Observed probabilities scale by (1 - unobs_prob) when configured."""
        computer = LazyMLEComputer()
        config = ScoringMethodConfig(logprob=False, unobs_prob=0.1)
        freqdist = {"a": 3, "b": 1}
        prob = computer.compute_probability("a", freqdist, config)
        assert prob == pytest.approx((3 / 4) * 0.9)

    def test_mle_with_unobs_prob_unobserved(self) -> None:
        """A zero-count element returns the configured unobserved mass."""
        computer = LazyMLEComputer()
        config = ScoringMethodConfig(logprob=False, unobs_prob=0.1)
        freqdist = {"a": 3}
        assert computer.compute_probability("missing", freqdist, config) == 0.1
        assert computer.compute_unobserved_probability(freqdist, config) == 0.1

    def test_laplace_unobserved_fresh_computer(self) -> None:
        """A fresh computer lazily populates totals in the unobserved path."""
        computer = LazyLaplaceComputer()
        config = ScoringMethodConfig(logprob=False)
        freqdist = {"a": 3, "b": 2}
        unobs = computer.compute_unobserved_probability(freqdist, config)
        assert unobs == pytest.approx(1.0 / (5 + 2))

    def test_laplace_respects_bins(self) -> None:
        """A configured bins value overrides the vocabulary size."""
        computer = LazyLaplaceComputer()
        config = ScoringMethodConfig(logprob=False, bins=10)
        freqdist = {"a": 3, "b": 2}
        prob = computer.compute_probability("a", freqdist, config)
        assert prob == pytest.approx((3 + 1) / (5 + 10))


class TestLazyScoringMethodEdgeCases:
    """Guards and log-probability paths of LazyScoringMethod."""

    def _unfitted(self) -> LazyScoringMethod:
        """Return an unfitted lazy MLE scorer."""
        return LazyScoringMethod(LazyMLEComputer(), ScoringMethodConfig(logprob=False), "Lazy MLE")

    def test_call_before_fit_raises(self) -> None:
        """Calling before fit raises a clear error."""
        with pytest.raises(ValueError, match="has not been fitted"):
            self._unfitted()("a")

    def test_precompute_before_fit_raises(self) -> None:
        """precompute_batch before fit raises a clear error."""
        with pytest.raises(ValueError, match="has not been fitted"):
            self._unfitted().precompute_batch({"a"})

    def test_force_full_before_fit_raises(self) -> None:
        """force_full_computation before fit raises a clear error."""
        with pytest.raises(ValueError, match="has not been fitted"):
            self._unfitted().force_full_computation()

    def test_unobserved_before_fit_raises(self) -> None:
        """_get_unobserved_probability before fit raises a clear error."""
        with pytest.raises(ValueError, match="has not been fitted"):
            self._unfitted()._get_unobserved_probability()

    def test_logprob_observed_element(self) -> None:
        """An observed element returns its log probability."""
        lazy = create_lazy_mle({"a": 3, "b": 1}, logprob=True)
        assert lazy("a") == pytest.approx(math.log(3 / 4))

    def test_logprob_zero_count_element(self) -> None:
        """A zero-count key has probability 0 -> log(1e-10) floor."""
        lazy = create_lazy_mle({"a": 3, "z": 0}, logprob=True)
        assert lazy("z") == pytest.approx(math.log(1e-10))

    def test_logprob_unobserved_floor(self) -> None:
        """MLE's zero unobserved mass floors at log(1e-10)."""
        lazy = create_lazy_mle({"a": 3, "b": 2}, logprob=True)
        assert lazy("unknown") == pytest.approx(math.log(1e-10))
        # A repeat access reuses the cached unobserved value.
        assert lazy("other") == pytest.approx(math.log(1e-10))

    def test_logprob_unobserved_positive(self) -> None:
        """Laplace's positive unobserved mass yields a real log value."""
        lazy = create_lazy_laplace({"a": 3, "b": 2}, logprob=True)
        expected = math.log(1.0 / (5 + 2))
        assert lazy("unknown") == pytest.approx(expected)

    def test_cached_element_returns_same(self) -> None:
        """A second call returns the cached computed value."""
        lazy = create_lazy_mle({"a": 3, "b": 2}, logprob=False)
        first = lazy("a")
        assert "a" in lazy.get_computed_elements()
        assert lazy("a") == first


class TestLazyBatchScorerEdgeCases:
    """LazyBatchScorer access-pattern branches."""

    def test_score_batch_single_unique_element(self) -> None:
        """A single unique element skips batch precomputation."""
        lazy = create_lazy_mle({"a": 3, "b": 2}, logprob=False)
        batch = LazyBatchScorer(lazy)
        scores = batch.score_batch(["a", "a", "a"])
        assert len(scores) == 3
        assert scores[0] == scores[1] == scores[2]

    def test_access_statistics_empty(self) -> None:
        """Access statistics on an unused scorer report zeros."""
        lazy = create_lazy_mle({"a": 3, "b": 2}, logprob=False)
        batch = LazyBatchScorer(lazy)
        stats = batch.get_access_statistics()
        assert stats == {"total_accesses": 0, "unique_elements": 0}

    def test_streaming_triggers_batch_precompute(self) -> None:
        """Crossing the streaming boundary precomputes seen elements."""
        freqdist = {f"w{i}": (i + 1) for i in range(120)}
        lazy = create_lazy_mle(freqdist, logprob=False)
        batch = LazyBatchScorer(lazy)

        stream = [f"w{i}" for i in range(100)]
        scores = list(batch.score_streaming(stream))

        assert len(scores) == 100
        assert len(lazy.get_computed_elements()) >= 100
