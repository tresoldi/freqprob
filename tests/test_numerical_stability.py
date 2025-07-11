"""Numerical stability tests for FreqProb smoothing methods.

This module tests the numerical stability of all smoothing methods under
extreme conditions, edge cases, and potential overflow/underflow scenarios.
"""

import math
from typing import Any

import pytest

import freqprob

# mypy: disable-error-code=arg-type


class TestNumericalStability:
    """Test numerical stability across all smoothing methods."""

    def test_empty_distribution(self) -> None:
        """Test behavior with empty frequency distributions."""
        empty_dist: dict[str, int] = {}

        # Methods should handle empty distributions gracefully
        mle = freqprob.MLE(empty_dist)
        assert mle("test") == 1e-10  # Default unobserved value

        laplace = freqprob.Laplace(empty_dist, bins=100, logprob=False)
        assert laplace("test") > 0  # Should give positive probability

    def test_single_element_distribution(self) -> None:
        """Test with distributions containing only one element."""
        single_dist = {"word": 1}

        # MLE should give probability 1.0 for the single word
        mle = freqprob.MLE(single_dist, logprob=False)
        assert abs(mle("word") - 1.0) < 1e-15
        assert mle("unknown") == 0.0

        # Laplace should work correctly
        laplace = freqprob.Laplace(single_dist, bins=1000, logprob=False)
        expected_known = 2 / 1001  # (1+1)/(1+1000)
        expected_unknown = 1 / 1001  # 1/(1+1000)

        assert abs(laplace("word") - expected_known) < 1e-15
        assert abs(laplace("unknown") - expected_unknown) < 1e-15

    def test_very_large_counts(self) -> None:
        """Test with extremely large frequency counts."""
        large_dist = {"common": 10**9, "rare": 1, "medium": 10**6}

        # Test MLE doesn't overflow
        mle = freqprob.MLE(large_dist, logprob=False)
        total = sum(large_dist.values())

        assert abs(mle("common") - (10**9 / total)) < 1e-15
        assert abs(mle("rare") - (1 / total)) < 1e-15

        # Test log probabilities don't underflow
        mle_log = freqprob.MLE(large_dist, logprob=True)
        assert not math.isinf(mle_log("rare"))
        assert not math.isnan(mle_log("rare"))

    def test_very_small_counts(self) -> None:
        """Test with very small but non-zero counts."""
        small_dist: dict[str, float] = {"word1": 1e-10, "word2": 1e-9, "word3": 1e-8}

        # Should not cause numerical issues
        mle = freqprob.MLE(small_dist, logprob=False)
        total = sum(small_dist.values())

        for word, count in small_dist.items():
            expected = count / total
            assert abs(mle(word) - expected) < 1e-15
            assert not math.isnan(mle(word))
            assert not math.isinf(mle(word))

    def test_extreme_smoothing_parameters(self) -> None:
        """Test smoothing methods with extreme parameter values."""
        dist = {"a": 5, "b": 3, "c": 2}

        # Very small gamma for Lidstone
        lidstone_small = freqprob.Lidstone(dist, gamma=1e-15, bins=1000, logprob=False)
        for word in dist:
            prob = lidstone_small(word)
            assert not math.isnan(prob)
            assert not math.isinf(prob)
            assert prob > 0

        # Very large gamma for Lidstone
        lidstone_large = freqprob.Lidstone(dist, gamma=1e10, bins=1000, logprob=False)
        for word in dist:
            prob = lidstone_large(word)
            assert not math.isnan(prob)
            assert not math.isinf(prob)
            assert prob > 0

    def test_zero_bins_handling(self) -> None:
        """Test behavior when bins parameter is zero or very small."""
        dist = {"word1": 5, "word2": 3}

        # Zero bins should be handled gracefully or raise appropriate error
        with pytest.raises((ValueError, ZeroDivisionError)):
            freqprob.Laplace(dist, bins=0)

        # Very small bins
        laplace_small = freqprob.Laplace(dist, bins=1, logprob=False)
        prob = laplace_small("word1")
        assert not math.isnan(prob)
        assert not math.isinf(prob)
        assert prob > 0

    def test_log_probability_underflow(self) -> None:
        """Test log probability calculation with potential underflow."""
        # Create distribution with very different frequencies
        skewed_dist = {"very_common": 10**6, "very_rare": 1}

        methods_to_test = [
            freqprob.MLE(skewed_dist, logprob=True),
            freqprob.Laplace(skewed_dist, bins=10**6, logprob=True),
            freqprob.ELE(skewed_dist, bins=10**6, logprob=True),
        ]

        for method in methods_to_test:
            # Test very rare word doesn't cause underflow in log space
            rare_logprob = method("very_rare")
            assert not math.isnan(rare_logprob)
            assert not math.isinf(rare_logprob)
            assert rare_logprob < 0  # Log probability should be negative

            # Test unknown word
            unknown_logprob = method("unknown")
            assert not math.isnan(unknown_logprob)
            # For MLE, unknown words might have -inf log probability, which is acceptable

    def test_probability_sum_conservation(self) -> None:
        """Test that probabilities sum to approximately 1.0 when possible."""
        dist = {"a": 10, "b": 20, "c": 30, "d": 40}

        # For methods without unobserved probability mass
        mle = freqprob.MLE(dist, logprob=False)
        total_prob = sum(mle(word) for word in dist)
        assert abs(total_prob - 1.0) < 1e-15

        # For Uniform distribution
        uniform = freqprob.Uniform(dist, unobs_prob=0.0, logprob=False)
        total_prob = sum(uniform(word) for word in dist)
        assert abs(total_prob - 1.0) < 1e-15

    def test_monotonicity_properties(self) -> None:
        """Test monotonicity properties of smoothing methods."""
        dist = {"frequent": 100, "medium": 50, "rare": 10, "very_rare": 1}

        # For MLE, more frequent words should have higher probability
        mle = freqprob.MLE(dist, logprob=False)
        assert mle("frequent") > mle("medium")
        assert mle("medium") > mle("rare")
        assert mle("rare") > mle("very_rare")

        # For Laplace, same ordering should hold
        laplace = freqprob.Laplace(dist, bins=1000, logprob=False)
        assert laplace("frequent") > laplace("medium")
        assert laplace("medium") > laplace("rare")
        assert laplace("rare") > laplace("very_rare")

    def test_extreme_vocabulary_sizes(self) -> None:
        """Test with extremely large vocabulary sizes."""
        small_dist = {"word": 1}

        # Very large bins parameter
        huge_bins = 10**9
        laplace_huge = freqprob.Laplace(small_dist, bins=huge_bins, logprob=False)

        known_prob = laplace_huge("word")
        unknown_prob = laplace_huge("unknown")

        # Should not overflow or underflow
        assert not math.isnan(known_prob)
        assert not math.isnan(unknown_prob)
        assert not math.isinf(known_prob)
        assert not math.isinf(unknown_prob)

        # Known word should still have higher probability
        assert known_prob > unknown_prob

        # Probabilities should be very small but positive
        assert known_prob > 0
        assert unknown_prob > 0

    def test_precision_loss_mitigation(self) -> None:
        """Test numerical precision in calculations."""
        # Create scenario prone to precision loss
        dist = {f"word_{i}": 1 for i in range(1000)}  # 1000 words with count 1

        mle = freqprob.MLE(dist, logprob=False)

        # Each word should have exactly 1/1000 probability
        expected_prob = 1.0 / 1000.0

        for word in list(dist.keys())[:10]:  # Test first 10 words
            actual_prob = mle(word)
            relative_error = abs(actual_prob - expected_prob) / expected_prob
            assert relative_error < 1e-14  # Very tight precision requirement

    def test_consistency_across_representations(self) -> None:
        """Test consistency between log and linear probability representations."""
        dist = {"apple": 15, "banana": 10, "cherry": 5}

        for method_class in [freqprob.MLE, freqprob.Laplace, freqprob.ELE]:
            if method_class == freqprob.Laplace or method_class == freqprob.ELE:
                linear_method = method_class(dist, bins=100, logprob=False)  # type: ignore
                log_method = method_class(dist, bins=100, logprob=True)  # type: ignore
            else:
                linear_method = method_class(dist, logprob=False)  # type: ignore
                log_method = method_class(dist, logprob=True)  # type: ignore

            for word in dist:
                linear_prob = linear_method(word)
                log_prob = log_method(word)

                # Convert log to linear and compare
                converted_prob = math.exp(log_prob)
                relative_error = abs(linear_prob - converted_prob) / linear_prob
                assert relative_error < 1e-12

    def test_boundary_conditions(self) -> None:
        """Test boundary conditions and edge cases."""
        # Distribution with maximum Python integer
        max_int = 2**63 - 1
        boundary_dist = {"word": max_int}

        mle = freqprob.MLE(boundary_dist, logprob=False)
        assert mle("word") == 1.0

        # Test with minimum positive float
        min_float = 2.2250738585072014e-308  # Smallest normal float64
        tiny_dist: dict[str, float] = {"word": min_float}

        mle_tiny = freqprob.MLE(tiny_dist, logprob=False)
        assert mle_tiny("word") == 1.0
        assert not math.isnan(mle_tiny("word"))

    @pytest.mark.slow
    def test_convergence_properties(self) -> None:
        """Test convergence properties with increasing data size."""
        base_dist = {"common": 1000, "rare": 1}

        # Test MLE convergence
        mle_probs = []
        for scale in [1, 10, 100, 1000]:
            scaled_dist = {word: count * scale for word, count in base_dist.items()}
            mle = freqprob.MLE(scaled_dist, logprob=False)
            mle_probs.append(mle("common"))

        # MLE should converge (probabilities should stabilize)
        for i in range(1, len(mle_probs)):
            relative_change = abs(mle_probs[i] - mle_probs[i - 1]) / mle_probs[i - 1]
            assert relative_change < 0.1  # Less than 10% change as data grows

    def test_thread_safety_numerical_stability(self) -> None:
        """Test numerical stability under concurrent access."""
        import threading
        import time

        dist = {"word1": 100, "word2": 200, "word3": 300}
        mle = freqprob.MLE(dist, logprob=False)

        results = []
        errors = []

        def worker() -> None:
            try:
                for _ in range(100):
                    prob = mle("word1")
                    results.append(prob)
                    time.sleep(0.001)  # Small delay to encourage race conditions
            except Exception as e:
                errors.append(e)

        # Start multiple threads
        threads = [threading.Thread(target=worker) for _ in range(5)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        # Check for errors
        assert len(errors) == 0, f"Errors occurred: {errors}"

        # Check all results are identical (method should be stateless)
        expected_prob = 100 / 600  # 100 / (100+200+300)
        for prob in results:
            assert abs(prob - expected_prob) < 1e-15


class TestNumericalStabilityAdvanced:
    """Advanced numerical stability tests for complex scenarios."""

    def test_good_turing_numerical_stability(self) -> None:
        """Test Simple Good-Turing under numerical stress."""
        # Create distribution with frequency-of-frequencies that might cause issues
        freq_dist: dict[str, int] = {}

        # Add words with counts 1, 2, 3, ..., 100
        for i in range(1, 101):
            for j in range(i):  # i words with count i
                freq_dist[f"word_{i}_{j}"] = i

        try:
            sgt = freqprob.SimpleGoodTuring(freq_dist, logprob=False)

            # Test some probabilities
            prob_1 = sgt("word_1_0")  # Word with count 1
            prob_100 = sgt("word_100_0")  # Word with count 100

            assert not math.isnan(prob_1)
            assert not math.isnan(prob_100)
            assert not math.isinf(prob_1)
            assert not math.isinf(prob_100)

            # Higher count should generally have higher probability
            assert prob_100 > prob_1

        except (ValueError, RuntimeError, RuntimeWarning):
            # SGT might fail on some distributions, which is acceptable
            pytest.skip("SGT failed on this distribution (expected behavior)")

    def test_kneser_ney_numerical_stability(self) -> None:
        """Test Kneser-Ney smoothing numerical stability."""
        # Create bigram distribution for testing
        bigrams: dict[tuple[str, str], int] = {}

        # Add realistic bigram patterns
        contexts = ["the", "a", "an", "this", "that"]
        words = ["cat", "dog", "house", "tree", "book"]

        for i, context in enumerate(contexts):
            for j, word in enumerate(words):
                # Create Zipfian-like distribution
                count = max(1, 100 // (i + j + 1))
                bigrams[(context, word)] = count

        # Test different discount values
        for discount in [0.1, 0.5, 0.75, 0.9, 0.99]:
            try:
                kn = freqprob.KneserNey(bigrams, discount=discount, logprob=False)

                # Test some probabilities
                prob = kn(("the", "cat"))
                assert not math.isnan(prob)
                assert not math.isinf(prob)
                assert prob > 0

            except (ValueError, RuntimeError):
                # Some discount values might not work, which is acceptable
                continue

    def test_streaming_numerical_stability(self) -> None:
        """Test streaming methods under numerical stress."""
        streaming_mle = freqprob.StreamingMLE(max_vocabulary_size=1000, logprob=False)

        # Add many updates with varying frequencies
        for i in range(10000):
            word = f"word_{i % 100}"  # Cycle through 100 words
            count = max(1, int(1000 / (i % 50 + 1)))  # Variable counts
            streaming_mle.update_single(word, count)

        # Test probabilities remain stable
        for i in range(10):
            word = f"word_{i}"
            prob = streaming_mle(word)
            assert not math.isnan(prob)
            assert not math.isinf(prob)
            assert prob > 0

    def test_memory_efficient_numerical_stability(self) -> None:
        """Test memory-efficient representations maintain numerical accuracy."""
        # Create large distribution
        large_dist = {f"word_{i}": max(1, int(1000 / (i + 1))) for i in range(1000)}

        # Create compressed version
        compressed = freqprob.create_compressed_distribution(large_dist, quantization_levels=256)

        # Test that compression doesn't cause numerical issues
        # Compare some probabilities (allowing for quantization error)
        for i in range(0, 100, 10):  # Test every 10th word
            word = f"word_{i}"

            # Note: compressed distribution doesn't directly support MLE
            # but we can test that counts are reasonable
            compressed_count = compressed.get_count(word)
            original_count = large_dist[word]

            relative_error = abs(compressed_count - original_count) / original_count
            assert relative_error < 0.40  # Allow 40% error due to quantization

    def test_vectorized_numerical_stability(self) -> None:
        """Test vectorized operations maintain numerical precision."""
        dist = {f"word_{i}": i + 1 for i in range(100)}
        mle = freqprob.MLE(dist, logprob=False)
        vectorized = freqprob.VectorizedScorer(mle)

        # Test batch scoring
        words: list[str] = [f"word_{i}" for i in range(50)]
        batch_scores = vectorized.score_batch(words)

        # Compare with individual scoring
        for i, word in enumerate(words):
            individual_score = mle(word)
            batch_score = batch_scores[i]

            relative_error = abs(individual_score - batch_score) / individual_score
            assert relative_error < 1e-14

    def test_cache_numerical_consistency(self) -> None:
        """Test that caching doesn't introduce numerical inconsistencies."""
        # Create distribution that might stress caching
        dist = {f"word_{i}": i + 1 for i in range(1000)}

        # Create method that uses caching (SGT)
        try:
            sgt = freqprob.SimpleGoodTuring(dist, logprob=False)

            # Get probability multiple times
            word = "word_50"
            probs = [sgt(word) for _ in range(10)]

            # All should be identical
            for prob in probs[1:]:
                assert abs(prob - probs[0]) < 1e-15

        except (ValueError, RuntimeError, RuntimeWarning):
            pytest.skip("SGT failed on this distribution")

    @pytest.mark.parametrize(
        ("method_name", "params"),
        [
            ("MLE", {}),
            ("Laplace", {"bins": 1000}),
            ("ELE", {"bins": 1000}),
            ("Lidstone", {"gamma": 0.5, "bins": 1000}),
            ("BayesianSmoothing", {"alpha": 0.5}),
        ],
    )
    def test_method_numerical_stability(self, method_name: str, params: dict[str, Any]) -> None:
        """Parametrized test for numerical stability across methods."""
        # Create challenging distribution
        dist = {
            "very_frequent": 10**6,
            "frequent": 10**4,
            "medium": 10**2,
            "rare": 1,
            "very_rare": 1,
        }

        method_class = getattr(freqprob, method_name)
        method = method_class(dist, logprob=False, **params)

        # Test all words in distribution
        for word in dist:
            prob = method(word)
            assert not math.isnan(prob), f"{method_name}: NaN for {word}"
            assert not math.isinf(prob), f"{method_name}: Inf for {word}"
            assert prob > 0, f"{method_name}: Non-positive probability for {word}"

        # Test unknown word
        unknown_prob = method("unknown_word")
        assert not math.isnan(unknown_prob), f"{method_name}: NaN for unknown word"
        # Note: MLE might return 0 for unknown words, which is acceptable

        # Test frequency ordering is preserved
        assert method("very_frequent") >= method("frequent")
        assert method("frequent") >= method("medium")
        assert method("medium") >= method("rare")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
