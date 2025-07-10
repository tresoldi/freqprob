"""Test computational efficiency features: caching, vectorization, and lazy evaluation."""

import time

import numpy as np

from freqprob import (
    MLE,
    BatchScorer,
    Laplace,
    LazyBatchScorer,
    SimpleGoodTuring,
    VectorizedScorer,
    WittenBell,
    clear_all_caches,
    create_lazy_laplace,
    create_lazy_mle,
    get_cache_stats,
)
from freqprob.cache import ComputationCache, MemoizedProperty
from freqprob.lazy import LazyLaplaceComputer, LazyMLEComputer
from freqprob.vectorized import elements_to_numpy, normalize_scores, scores_to_probabilities

# mypy: disable-error-code=arg-type


class TestCaching:
    """Test caching and memoization functionality."""

    def test_computation_cache_basic(self) -> None:
        """Test basic cache operations."""
        cache = ComputationCache(max_size=3)

        # Test empty cache
        assert cache.size() == 0
        assert cache.get({"a": 1}) is None

        # Test storing and retrieving
        cache.set({"a": 1}, "result1")
        assert cache.size() == 1
        assert cache.get({"a": 1}) == "result1"

        # Test different inputs
        cache.set({"b": 2}, "result2")
        assert cache.get({"b": 2}) == "result2"
        assert cache.get({"a": 1}) == "result1"

    def test_cache_size_limit(self) -> None:
        """Test cache size limiting and eviction."""
        cache = ComputationCache(max_size=2)

        cache.set({"a": 1}, "result1")
        cache.set({"b": 2}, "result2")
        assert cache.size() == 2

        # Adding third item should evict first
        cache.set({"c": 3}, "result3")
        assert cache.size() == 2
        assert cache.get({"a": 1}) is None  # Evicted
        assert cache.get({"b": 2}) == "result2"
        assert cache.get({"c": 3}) == "result3"

    def test_cache_key_generation(self) -> None:
        """Test that cache keys are generated correctly."""
        cache = ComputationCache()

        # Same data should generate same key
        cache.set({"a": 1, "b": 2}, "result1")
        cache.set({"b": 2, "a": 1}, "result2")  # Different order, should overwrite

        assert cache.get({"a": 1, "b": 2}) == "result2"
        assert cache.size() == 1

    def test_sgt_caching(self) -> None:
        """Test that Simple Good-Turing uses caching."""
        # Create a frequency distribution that SGT can handle
        freqdist = {f"word_{i}": i for i in range(1, 50)}

        clear_all_caches()
        initial_stats = get_cache_stats()

        # First computation should populate cache
        start_time = time.time()
        sgt1 = SimpleGoodTuring(freqdist, allow_fail=False)
        _ = time.time() - start_time  # first_time not used

        stats_after_first = get_cache_stats()
        assert stats_after_first["sgt_cache_size"] > initial_stats["sgt_cache_size"]

        # Second computation with same parameters should be faster (cached)
        start_time = time.time()
        sgt2 = SimpleGoodTuring(freqdist, allow_fail=False)
        _ = time.time() - start_time  # second_time not used

        # Results should be identical
        assert sgt1("word_1") == sgt2("word_1")
        assert sgt1("unseen") == sgt2("unseen")

        # Second computation should be faster (though this might be flaky in tests)
        # We'll just check that cache was used
        stats_after_second = get_cache_stats()
        assert stats_after_second["sgt_cache_size"] == stats_after_first["sgt_cache_size"]

    def test_general_method_caching(self) -> None:
        """Test caching for other methods like Witten-Bell."""
        freqdist = {"a": 5, "b": 3, "c": 2}

        clear_all_caches()
        initial_stats = get_cache_stats()

        # First computation
        wb1 = WittenBell(freqdist)
        stats_after_first = get_cache_stats()
        assert stats_after_first["general_cache_size"] > initial_stats["general_cache_size"]

        # Second computation with same parameters
        wb2 = WittenBell(freqdist)
        assert wb1("a") == wb2("a")

    def test_memoized_property(self) -> None:
        """Test memoized property decorator."""

        class TestClass:
            def __init__(self) -> None:
                self.computation_count = 0

            @MemoizedProperty
            def expensive_property(self) -> int:
                self.computation_count += 1
                return self.computation_count * 10

        obj = TestClass()

        # First access should compute
        result1 = obj.expensive_property
        assert result1 == 10
        assert obj.computation_count == 1

        # Second access should use cached value
        result2 = obj.expensive_property
        assert result2 == 10
        assert obj.computation_count == 1  # No additional computation


class TestVectorization:
    """Test vectorized operations functionality."""

    def test_vectorized_scorer_basic(self) -> None:
        """Test basic vectorized scoring."""
        scorer = MLE({"a": 3, "b": 2, "c": 1}, logprob=False)
        vectorized = VectorizedScorer(scorer)

        elements = ["a", "b", "c", "d"]
        scores = vectorized.score_batch(elements)

        assert isinstance(scores, np.ndarray)
        assert len(scores) == 4
        assert scores[0] == scorer("a")  # Known element
        assert scores[3] == scorer("d")  # Unknown element

    def test_vectorized_scorer_numpy_input(self) -> None:
        """Test vectorized scorer with numpy array input."""
        scorer = MLE({"a": 3, "b": 2}, logprob=False)
        vectorized = VectorizedScorer(scorer)

        elements = np.array(["a", "b", "unknown"])
        scores = vectorized.score_batch(elements)

        assert isinstance(scores, np.ndarray)
        assert len(scores) == 3

    def test_vectorized_scorer_matrix(self) -> None:
        """Test vectorized matrix scoring."""
        scorer = MLE({"a": 3, "b": 2}, logprob=False)
        vectorized = VectorizedScorer(scorer)

        elements_2d = [["a", "b"], ["a"], ["b", "unknown", "a"]]
        score_matrix = vectorized.score_matrix(elements_2d)

        assert score_matrix.shape == (3, 3)  # 3 rows, max 3 columns
        assert score_matrix[0, 0] == scorer("a")
        assert score_matrix[0, 1] == scorer("b")

    def test_vectorized_scorer_top_k(self) -> None:
        """Test top-k element retrieval."""
        scorer = MLE({"a": 5, "b": 3, "c": 2, "d": 1}, logprob=False)
        vectorized = VectorizedScorer(scorer)

        top_elements, top_scores = vectorized.top_k_elements(2)

        assert len(top_elements) == 2
        assert len(top_scores) == 2
        assert top_elements[0] == "a"  # Highest scoring
        assert top_scores[0] == scorer("a")

    def test_batch_scorer(self) -> None:
        """Test batch scorer with multiple methods."""
        scorers = {
            "mle": MLE({"a": 3, "b": 2}, logprob=False),
            "laplace": Laplace({"a": 3, "b": 2}, logprob=False),
        }
        batch_scorer = BatchScorer(scorers)

        elements = ["a", "b", "unknown"]
        results = batch_scorer.score_batch(elements)

        assert "mle" in results
        assert "laplace" in results
        assert isinstance(results["mle"], np.ndarray)
        assert len(results["mle"]) == 3

    def test_batch_scorer_comparison(self) -> None:
        """Test batch scorer comparison functionality."""
        scorers = {
            "mle": MLE({"a": 3, "b": 2}, logprob=False),
            "laplace": Laplace({"a": 3, "b": 2}, logprob=False),
        }
        batch_scorer = BatchScorer(scorers)

        elements = ["a", "b"]
        comparison = batch_scorer.score_and_compare(elements)

        assert "scores" in comparison
        assert "mean_scores" in comparison
        assert "std_scores" in comparison
        assert "mle_ranking" in comparison
        assert "laplace_ranking" in comparison

    def test_elements_to_numpy(self) -> None:
        """Test element conversion to numpy arrays."""
        # String elements
        str_elements = ["a", "b", "c"]
        str_array = elements_to_numpy(str_elements)
        assert str_array.dtype.kind == "U"  # Unicode string

        # Integer elements
        int_elements = [1, 2, 3]
        int_array = elements_to_numpy(int_elements)
        assert int_array.dtype == np.int64

        # Float elements
        float_elements = [1.0, 2.0, 3.0]
        float_array = elements_to_numpy(float_elements)
        assert float_array.dtype == np.float64

    def test_scores_to_probabilities(self) -> None:
        """Test log score to probability conversion."""
        log_scores = np.array([-1.0, -2.0, -3.0])
        probs = scores_to_probabilities(log_scores)

        assert np.allclose(np.sum(probs), 1.0)  # Should sum to 1
        assert probs[0] > probs[1] > probs[2]  # Should be decreasing

    def test_normalize_scores(self) -> None:
        """Test score normalization methods."""
        scores = np.array([1.0, 2.0, 3.0])

        # Softmax normalization
        softmax = normalize_scores(scores, "softmax")
        assert np.allclose(np.sum(softmax), 1.0)

        # Min-max normalization
        minmax = normalize_scores(scores, "minmax")
        assert minmax[0] == 0.0
        assert minmax[-1] == 1.0

        # Z-score normalization
        zscore = normalize_scores(scores, "zscore")
        assert np.allclose(np.mean(zscore), 0.0)


class TestLazyEvaluation:
    """Test lazy evaluation functionality."""

    def test_lazy_mle_computer(self) -> None:
        """Test lazy MLE computation."""
        freqdist = {"a": 3, "b": 2, "c": 1}
        computer = LazyMLEComputer()

        from freqprob.base import ScoringMethodConfig

        config = ScoringMethodConfig(logprob=False)

        # Test probability computation
        prob_a = computer.compute_probability("a", freqdist, config)
        expected_a = 3 / 6
        assert prob_a == expected_a

        # Test unobserved probability
        unobs_prob = computer.compute_unobserved_probability(freqdist, config)
        assert unobs_prob == 0.0  # MLE gives 0 for unobserved

    def test_lazy_laplace_computer(self) -> None:
        """Test lazy Laplace computation."""
        freqdist = {"a": 3, "b": 2}
        computer = LazyLaplaceComputer()

        from freqprob.base import ScoringMethodConfig

        config = ScoringMethodConfig(logprob=False)

        # Test probability computation
        prob_a = computer.compute_probability("a", freqdist, config)
        expected_a = (3 + 1) / (5 + 2)  # Laplace smoothing
        assert prob_a == expected_a

        # Test unobserved probability
        unobs_prob = computer.compute_unobserved_probability(freqdist, config)
        expected_unobs = 1 / (5 + 2)
        assert unobs_prob == expected_unobs

    def test_lazy_scoring_method(self) -> None:
        """Test lazy scoring method."""
        freqdist = {"a": 3, "b": 2, "c": 1}
        lazy_scorer = create_lazy_mle(freqdist, logprob=False)

        # Initially, no elements should be computed
        assert len(lazy_scorer.get_computed_elements()) == 0

        # Score one element
        score_a = lazy_scorer("a")
        assert len(lazy_scorer.get_computed_elements()) == 1
        assert "a" in lazy_scorer.get_computed_elements()

        # Score the same element again (should use cached value)
        score_a2 = lazy_scorer("a")
        assert score_a == score_a2
        assert len(lazy_scorer.get_computed_elements()) == 1

        # Score an unknown element
        score_unknown = lazy_scorer("unknown")
        assert score_unknown == 0.0  # MLE gives 0 for unobserved

    def test_lazy_precompute_batch(self) -> None:
        """Test lazy batch precomputation."""
        freqdist = {"a": 3, "b": 2, "c": 1, "d": 1}
        lazy_scorer = create_lazy_mle(freqdist, logprob=False)

        # Precompute a batch
        elements_to_precompute = {"a", "b", "c"}
        lazy_scorer.precompute_batch(elements_to_precompute)

        computed = lazy_scorer.get_computed_elements()
        assert "a" in computed
        assert "b" in computed
        assert "c" in computed
        assert "d" not in computed  # Not in the batch

    def test_lazy_force_full_computation(self) -> None:
        """Test forcing full computation in lazy scorer."""
        freqdist = {"a": 3, "b": 2, "c": 1}
        lazy_scorer = create_lazy_mle(freqdist, logprob=False)

        # Force full computation
        lazy_scorer.force_full_computation()

        computed = lazy_scorer.get_computed_elements()
        assert len(computed) == 3
        assert all(elem in computed for elem in freqdist)

    def test_lazy_batch_scorer(self) -> None:
        """Test lazy batch scorer."""
        freqdist = {"a": 3, "b": 2, "c": 1, "d": 1, "e": 1}
        lazy_scorer = create_lazy_mle(freqdist, logprob=False)
        batch_scorer = LazyBatchScorer(lazy_scorer)

        # Score a batch
        elements = ["a", "b", "unknown"]
        scores = batch_scorer.score_batch(elements)

        assert len(scores) == 3
        assert scores[0] > 0  # 'a' should have positive score
        assert scores[2] == 0  # 'unknown' should have zero score

        # Check access statistics
        stats = batch_scorer.get_access_statistics()
        assert stats["total_accesses"] == 3
        assert stats["unique_elements"] == 3

    def test_lazy_streaming(self) -> None:
        """Test lazy streaming scorer."""
        freqdist = {"a": 3, "b": 2, "c": 1}
        lazy_scorer = create_lazy_mle(freqdist, logprob=False)
        batch_scorer = LazyBatchScorer(lazy_scorer)

        # Stream elements
        element_stream = ["a", "b", "a", "c", "b", "a"]
        scores = list(batch_scorer.score_streaming(element_stream))

        assert len(scores) == 6
        assert scores[0] == scores[2] == scores[5]  # All 'a' scores should be equal

    def test_create_lazy_laplace(self) -> None:
        """Test lazy Laplace scorer creation."""
        freqdist = {"a": 3, "b": 2}
        lazy_scorer = create_lazy_laplace(freqdist, logprob=False)

        # Test that it behaves like Laplace smoothing
        score_a = lazy_scorer("a")
        expected_a = (3 + 1) / (5 + 2)
        assert score_a == expected_a

        score_unknown = lazy_scorer("unknown")
        expected_unknown = 1 / (5 + 2)
        assert score_unknown == expected_unknown


class TestEfficiencyIntegration:
    """Test integration of efficiency features."""

    def test_vectorized_vs_regular_scoring(self) -> None:
        """Test that vectorized scoring gives same results as regular scoring."""
        freqdist = {"a": 5, "b": 3, "c": 2, "d": 1}
        scorer = MLE(freqdist, logprob=False)
        vectorized = VectorizedScorer(scorer)

        elements = ["a", "b", "c", "d", "unknown"]

        # Regular scoring
        regular_scores = [scorer(elem) for elem in elements]

        # Vectorized scoring
        vectorized_scores = vectorized.score_batch(elements)

        # Should be identical
        np.testing.assert_array_almost_equal(regular_scores, vectorized_scores)

    def test_lazy_vs_regular_scoring(self) -> None:
        """Test that lazy scoring gives same results as regular scoring."""
        freqdist = {"a": 5, "b": 3, "c": 2}

        # Regular MLE
        regular_scorer = MLE(freqdist, logprob=False)

        # Lazy MLE
        lazy_scorer = create_lazy_mle(freqdist, logprob=False)

        elements = ["a", "b", "c", "unknown"]

        for elem in elements:
            regular_score = regular_scorer(elem)
            lazy_score = lazy_scorer(elem)
            assert abs(regular_score - lazy_score) < 1e-10

    def test_performance_improvement_demo(self) -> None:
        """Demonstrate performance improvements (not a strict test)."""
        # Create a large frequency distribution
        freqdist = {f"word_{i}": max(1, 100 - i) for i in range(100)}

        # Test elements (only subset of total)
        test_elements = [f"word_{i}" for i in range(0, 100, 10)]  # Every 10th element

        # Regular MLE
        regular_scorer = MLE(freqdist, logprob=False)

        # Lazy MLE
        lazy_scorer = create_lazy_mle(freqdist, logprob=False)

        # Time regular scoring
        start_time = time.time()
        regular_scores = [regular_scorer(elem) for elem in test_elements]
        regular_time = time.time() - start_time

        # Time lazy scoring
        start_time = time.time()
        lazy_scores = [lazy_scorer(elem) for elem in test_elements]
        lazy_time = time.time() - start_time

        # Results should be the same
        np.testing.assert_array_almost_equal(regular_scores, lazy_scores)

        # Lazy should compute fewer elements
        computed_elements = lazy_scorer.get_computed_elements()
        assert len(computed_elements) == len(test_elements)  # Only computed what was needed

        # Print timing information (for manual inspection, not assertion)
        print(f"Regular scoring time: {regular_time:.6f}s")
        print(f"Lazy scoring time: {lazy_time:.6f}s")
        print(f"Elements computed lazily: {len(computed_elements)}/{len(freqdist)}")


def test_clear_caches() -> None:
    """Test cache clearing functionality."""
    # Create some cached computations
    freqdist = {"a": 3, "b": 2}
    WittenBell(freqdist)  # Should populate cache

    stats_before = get_cache_stats()
    assert stats_before["general_cache_size"] > 0

    # Clear caches
    clear_all_caches()

    stats_after = get_cache_stats()
    assert stats_after["general_cache_size"] == 0
    assert stats_after["sgt_cache_size"] == 0
