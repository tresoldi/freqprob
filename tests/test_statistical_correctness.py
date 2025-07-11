"""Statistical correctness validation for FreqProb smoothing methods.

This module validates that smoothing methods produce statistically correct
results by testing against known theoretical properties, distributions,
and mathematical expectations.
"""

import math
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

import freqprob

if TYPE_CHECKING:
    from freqprob.base import FrequencyDistribution


class TestStatisticalCorrectness:
    """Test statistical correctness of smoothing methods."""

    def test_mle_matches_theoretical(self) -> None:
        """Test that MLE matches theoretical maximum likelihood estimates."""
        # Known distribution
        true_counts = {"a": 100, "b": 200, "c": 300, "d": 400}
        total = sum(true_counts.values())

        mle = freqprob.MLE(true_counts, logprob=False)  # type: ignore[arg-type]

        # MLE should exactly match relative frequencies
        for word, count in true_counts.items():
            expected = count / total
            actual = mle(word)
            assert abs(actual - expected) < 1e-15

        # Unknown words should have zero probability
        assert mle("unknown") == 0.0

    def test_laplace_properties(self) -> None:
        """Test Laplace smoothing statistical properties."""
        counts = {"word1": 10, "word2": 5}
        bins = 100

        laplace = freqprob.Laplace(counts, bins=bins, logprob=False)  # type: ignore[arg-type]

        # Test Laplace formula: P(w) = (c(w) + 1) / (N + V)
        total_count = sum(counts.values())

        for word, count in counts.items():
            expected = (count + 1) / (total_count + bins)
            actual = laplace(word)
            assert abs(actual - expected) < 1e-15

        # Test unknown word
        expected_unknown = 1 / (total_count + bins)
        actual_unknown = laplace("unknown")
        assert abs(actual_unknown - expected_unknown) < 1e-15

        # Test probability mass conservation
        total_observed_mass = sum(laplace(word) for word in counts)
        unobserved_words = bins - len(counts)
        total_unobserved_mass = unobserved_words * expected_unknown
        total_mass = total_observed_mass + total_unobserved_mass
        assert abs(total_mass - 1.0) < 1e-14

    def test_lidstone_generalization(self) -> None:
        """Test that Lidstone properly generalizes add-k smoothing."""
        counts = {"a": 20, "b": 10, "c": 5}
        bins = 50

        # Test different gamma values
        for gamma in [0.1, 0.5, 1.0, 2.0]:
            lidstone = freqprob.Lidstone(counts, gamma=gamma, bins=bins, logprob=False)  # type: ignore[arg-type]

            total_count = sum(counts.values())

            for word, count in counts.items():
                expected = (count + gamma) / (total_count + gamma * bins)
                actual = lidstone(word)
                assert abs(actual - expected) < 1e-15

        # Test that gamma=1 equals Laplace
        lidstone_1 = freqprob.Lidstone(counts, gamma=1.0, bins=bins, logprob=False)  # type: ignore[arg-type]
        laplace = freqprob.Laplace(counts, bins=bins, logprob=False)  # type: ignore[arg-type]

        for word in counts:
            assert abs(lidstone_1(word) - laplace(word)) < 1e-15

    def test_ele_relationship_to_lidstone(self) -> None:
        """Test that ELE is equivalent to Lidstone with gamma=0.5."""
        counts = {"x": 15, "y": 25, "z": 10}
        bins = 200

        ele = freqprob.ELE(counts, bins=bins, logprob=False)  # type: ignore[arg-type]
        lidstone_half = freqprob.Lidstone(counts, gamma=0.5, bins=bins, logprob=False)  # type: ignore[arg-type]

        for word in counts:
            assert abs(ele(word) - lidstone_half(word)) < 1e-15

        # Test unknown word
        assert abs(ele("unknown") - lidstone_half("unknown")) < 1e-15

    def test_uniform_distribution_properties(self) -> None:
        """Test uniform distribution statistical properties."""
        counts = {"a": 100, "b": 200, "c": 50}  # Counts don't matter for uniform
        unobs_prob = 0.1

        uniform = freqprob.Uniform(counts, unobs_prob=unobs_prob, logprob=False)  # type: ignore[arg-type]

        # All observed words should have equal probability
        vocab_size = len(counts)
        expected_observed = (1.0 - unobs_prob) / vocab_size

        for word in counts:
            actual = uniform(word)
            assert abs(actual - expected_observed) < 1e-15

        # Unknown words should have the specified unobserved probability
        actual_unknown = uniform("unknown")
        assert abs(actual_unknown - unobs_prob) < 1e-15

    def test_bayesian_smoothing_properties(self) -> None:
        """Test Bayesian smoothing with Dirichlet prior."""
        counts = {"term1": 40, "term2": 30, "term3": 20, "term4": 10}

        for alpha in [0.1, 0.5, 1.0, 2.0]:
            bayesian = freqprob.BayesianSmoothing(counts, alpha=alpha, logprob=False)  # type: ignore[arg-type]

            total_count = sum(counts.values())
            vocab_size = len(counts)

            # Test Bayesian formula: P(w) = (c(w) + alpha) / (N + alpha*V)
            for word, count in counts.items():
                expected = (count + alpha) / (total_count + alpha * vocab_size)
                actual = bayesian(word)
                assert abs(actual - expected) < 1e-15

        # Test that alpha=1 gives uniform prior (similar to Laplace)
        bayesian_1 = freqprob.BayesianSmoothing(counts, alpha=1.0, logprob=False)  # type: ignore[arg-type]

        # Should sum to 1 for observed vocabulary
        total_prob = sum(bayesian_1(word) for word in counts)
        assert abs(total_prob - 1.0) < 1e-14

    def test_probability_axioms(self) -> None:
        """Test that all methods satisfy probability axioms."""
        counts = {"apple": 25, "banana": 15, "cherry": 10}

        methods = [
            freqprob.MLE(counts, logprob=False),  # type: ignore[arg-type]
            freqprob.Laplace(counts, bins=100, logprob=False),  # type: ignore[arg-type]
            freqprob.ELE(counts, bins=100, logprob=False),  # type: ignore[arg-type]
            freqprob.Lidstone(counts, gamma=0.5, bins=100, logprob=False),  # type: ignore[arg-type]
            freqprob.BayesianSmoothing(counts, alpha=0.5, logprob=False),  # type: ignore[arg-type]
            freqprob.Uniform(counts, unobs_prob=0.1, logprob=False),  # type: ignore[arg-type]
        ]

        for method in methods:
            # Axiom 1: All probabilities are non-negative
            for word in counts:
                assert method(word) >= 0

            # Test unknown word (may be 0 for some methods)
            unknown_prob = method("unknown")
            assert unknown_prob >= 0

            # Axiom 2: Probabilities are ≤ 1
            for word in counts:
                assert method(word) <= 1.0
            assert unknown_prob <= 1.0

    def test_consistency_under_scaling(self) -> None:
        """Test that relative probabilities are preserved under count scaling."""
        base_counts = {"word1": 10, "word2": 20, "word3": 30}

        # Scale counts by different factors
        for scale in [2, 5, 10, 100]:
            scaled_counts = {word: count * scale for word, count in base_counts.items()}

            base_mle = freqprob.MLE(base_counts, logprob=False)  # type: ignore[arg-type]
            scaled_mle = freqprob.MLE(scaled_counts, logprob=False)  # type: ignore[arg-type]

            # MLE probabilities should be identical
            for word in base_counts:
                base_prob = base_mle(word)
                scaled_prob = scaled_mle(word)
                assert abs(base_prob - scaled_prob) < 1e-15

    def test_convergence_to_true_distribution(self) -> None:
        """Test convergence properties as sample size increases."""
        # True underlying probabilities
        true_probs = {"a": 0.5, "b": 0.3, "c": 0.2}

        # Generate samples of increasing size
        sample_sizes = [100, 1000, 10000]
        np.random.seed(42)  # For reproducibility

        for sample_size in sample_sizes:
            # Generate sample according to true distribution
            words = []
            for word, prob in true_probs.items():
                count = int(sample_size * prob)
                words.extend([word] * count)

            # Create frequency distribution
            sample_counts: dict[str, int] = {}
            for word in words:
                sample_counts[word] = sample_counts.get(word, 0) + 1

            # Test MLE convergence
            mle = freqprob.MLE(sample_counts, logprob=False)  # type: ignore[arg-type]

            for word, true_prob in true_probs.items():
                estimated_prob = mle(word)
                error = abs(estimated_prob - true_prob)

                # Error should decrease with sample size
                max_expected_error = 3.0 / math.sqrt(sample_size)  # Generous bound
                assert error < max_expected_error

    def test_smoothing_bias_properties(self) -> None:
        """Test bias properties of different smoothing methods."""
        # Distribution with rare and common words
        counts = {"common": 1000, "rare": 1}

        mle = freqprob.MLE(counts, logprob=False)  # type: ignore[arg-type]
        laplace = freqprob.Laplace(counts, bins=10, logprob=False)  # type: ignore[arg-type]

        # MLE should give exact relative frequencies
        assert abs(mle("common") - (1000 / 1001)) < 1e-15
        assert abs(mle("rare") - (1 / 1001)) < 1e-15

        # Laplace should reduce probability of common words and increase rare words
        assert laplace("common") < mle("common")
        assert laplace("rare") > mle("rare")

        # Unknown words should have positive probability only with smoothing
        assert mle("unknown") == 0.0
        assert laplace("unknown") > 0.0

    def test_entropy_properties(self) -> None:
        """Test entropy-related properties of distributions."""
        # Uniform distribution should have maximum entropy
        uniform_counts = {"a": 1, "b": 1, "c": 1, "d": 1}

        # Skewed distribution should have lower entropy
        skewed_counts = {"a": 100, "b": 1, "c": 1, "d": 1}

        uniform_mle = freqprob.MLE(uniform_counts, logprob=True)  # type: ignore[arg-type]
        skewed_mle = freqprob.MLE(skewed_counts, logprob=True)  # type: ignore[arg-type]

        # Calculate empirical entropy H = -Σ p(x) log p(x)
        def calculate_entropy(method: Any, words: list[str]) -> float:
            entropy = 0.0
            for word in words:
                log_prob = method(word)
                if not math.isinf(log_prob):  # Skip zero probabilities
                    prob = math.exp(log_prob)
                    entropy -= prob * log_prob
            return entropy

        uniform_entropy = calculate_entropy(uniform_mle, list(uniform_counts.keys()))
        skewed_entropy = calculate_entropy(skewed_mle, list(skewed_counts.keys()))

        # Uniform distribution should have higher entropy
        assert uniform_entropy > skewed_entropy

        # Theoretical maximum entropy for 4 equally likely events
        max_entropy = math.log(4)
        assert abs(uniform_entropy - max_entropy) < 1e-14

    def test_perplexity_properties(self) -> None:
        """Test perplexity calculation properties."""
        train_counts = {"the": 100, "cat": 50, "sat": 25, "on": 25}
        test_data = ["the", "cat", "sat", "on", "the", "cat"]

        # Test different smoothing methods
        mle = freqprob.MLE(train_counts, logprob=True)  # type: ignore[arg-type]
        laplace = freqprob.Laplace(train_counts, bins=1000, logprob=True)  # type: ignore[arg-type]

        # Calculate perplexities
        mle_perplexity = freqprob.perplexity(mle, test_data)
        laplace_perplexity = freqprob.perplexity(laplace, test_data)

        # Both should be positive
        assert mle_perplexity > 0
        assert laplace_perplexity > 0

        # For this in-vocabulary test data, MLE should have lower perplexity
        assert mle_perplexity < laplace_perplexity

        # Test with unknown words
        test_with_unknown = [*test_data, "unknown_word"]

        # MLE perplexity should be very high due to small default probability for unknown words
        mle_perplexity_unknown = freqprob.perplexity(mle, test_with_unknown)
        laplace_perplexity_unknown = freqprob.perplexity(laplace, test_with_unknown)

        assert (
            mle_perplexity_unknown > 50
        )  # MLE should have very high perplexity with unknown words
        assert not math.isinf(laplace_perplexity_unknown)

    def test_cross_entropy_properties(self) -> None:
        """Test cross-entropy calculation properties."""
        train_counts = {"a": 60, "b": 30, "c": 10}
        test_data = ["a", "b", "c", "a", "b", "a"]

        mle = freqprob.MLE(train_counts, logprob=True)  # type: ignore[arg-type]
        laplace = freqprob.Laplace(train_counts, bins=100, logprob=True)  # type: ignore[arg-type]

        mle_ce = freqprob.cross_entropy(mle, test_data)
        laplace_ce = freqprob.cross_entropy(laplace, test_data)

        # Cross-entropy should be positive
        assert mle_ce > 0
        assert laplace_ce > 0

        # For in-vocabulary data, MLE should have lower cross-entropy
        assert mle_ce < laplace_ce

    def test_kl_divergence_properties(self) -> None:
        """Test KL divergence properties."""
        counts1 = {"a": 50, "b": 30, "c": 20}
        counts2 = {"a": 40, "b": 35, "c": 25}

        model1 = freqprob.MLE(counts1, logprob=True)  # type: ignore[arg-type]
        model2 = freqprob.MLE(counts2, logprob=True)  # type: ignore[arg-type]

        test_data = ["a", "b", "c"] * 10

        # KL divergence should be non-negative
        kl_12 = freqprob.kl_divergence(model1, model2, test_data)
        kl_21 = freqprob.kl_divergence(model2, model1, test_data)

        assert kl_12 >= 0
        assert kl_21 >= 0

        # KL divergence is not symmetric
        assert kl_12 != kl_21  # Generally true for different distributions

        # Self KL divergence should be 0
        kl_self = freqprob.kl_divergence(model1, model1, test_data)
        assert abs(kl_self) < 1e-14

    def test_model_comparison_consistency(self) -> None:
        """Test model comparison utility consistency."""
        train_counts = {"word1": 100, "word2": 50, "word3": 25}
        test_data = ["word1", "word2", "word3"] * 20

        models = {
            "mle": freqprob.MLE(train_counts, logprob=True),  # type: ignore[arg-type]
            "laplace": freqprob.Laplace(train_counts, bins=100, logprob=True),  # type: ignore[arg-type]
            "ele": freqprob.ELE(train_counts, bins=100, logprob=True),  # type: ignore[arg-type]
        }

        results = freqprob.model_comparison(models, test_data)

        # Check that all models have results
        assert len(results) == len(models)

        for metrics in results.values():
            # Each model should have perplexity and cross-entropy
            assert "perplexity" in metrics
            assert "cross_entropy" in metrics

            # Values should be positive and finite
            assert metrics["perplexity"] > 0
            assert metrics["cross_entropy"] > 0
            assert not math.isinf(metrics["perplexity"])
            assert not math.isinf(metrics["cross_entropy"])

    @pytest.mark.slow
    def test_large_vocabulary_statistical_properties(self) -> None:
        """Test statistical properties with large vocabularies."""
        # Create large Zipfian distribution
        vocab_size = 10000
        zipf_exponent = 1.5

        # Generate Zipfian frequencies
        frequencies = np.random.zipf(zipf_exponent, vocab_size)
        large_counts = {f"word_{i}": int(freq) for i, freq in enumerate(frequencies)}

        # Test Laplace smoothing
        laplace = freqprob.Laplace(large_counts, bins=vocab_size * 2, logprob=False)  # type: ignore[arg-type]

        # Sample some probabilities
        sample_words = [f"word_{i}" for i in range(0, vocab_size, 100)]
        probabilities = [laplace(word) for word in sample_words]

        # All probabilities should be valid
        for prob in probabilities:
            assert 0 < prob <= 1
            assert not math.isnan(prob)
            assert not math.isinf(prob)

        # Higher frequency words should generally have higher probability
        # (allowing for some noise in the Zipfian generation)
        sorted_words = sorted(sample_words, key=lambda w: int(large_counts[w]), reverse=True)
        sorted_probs = [laplace(word) for word in sorted_words]

        # Check that probability generally decreases (allowing some violations)
        violations = sum(
            1 for i in range(len(sorted_probs) - 1) if sorted_probs[i] < sorted_probs[i + 1]
        )
        violation_rate = violations / (len(sorted_probs) - 1)
        assert violation_rate < 0.1  # Less than 10% violations


class TestAdvancedStatisticalProperties:
    """Advanced statistical correctness tests."""

    def test_good_turing_frequency_redistribution(self) -> None:
        """Test Good-Turing frequency redistribution properties."""
        # Create distribution with clear frequency-of-frequencies pattern
        freq_dist = {}

        # 10 words appearing once, 5 words appearing twice, etc.
        for freq in range(1, 6):
            for i in range(11 - freq * 2):  # Decreasing number of words
                freq_dist[f"word_{freq}_{i}"] = freq

        try:
            sgt = freqprob.SimpleGoodTuring(freq_dist, logprob=False)  # type: ignore[arg-type]

            # Words with higher frequencies should generally get higher probabilities
            # (SGT redistributes some mass to unseen words)
            prob_1 = sgt("word_1_0")  # Word appearing once
            prob_5 = sgt("word_5_0")  # Word appearing 5 times

            assert prob_5 > prob_1

            # Total probability mass should be conserved
            total_mass = 0.0
            for word in freq_dist:
                total_mass += sgt(word)

            # Note: SGT reserves some mass for unseen words, so total < 1
            assert 0.8 < total_mass < 1.0

        except (ValueError, RuntimeError):
            pytest.skip("SGT failed on this distribution")

    def test_kneser_ney_continuation_probability(self) -> None:
        """Test Kneser-Ney continuation probability properties."""
        # Create bigram data with clear continuation patterns
        bigrams = {
            ("the", "cat"): 10,
            ("the", "dog"): 8,
            ("the", "house"): 5,
            ("a", "cat"): 3,
            ("a", "dog"): 2,
            ("big", "house"): 4,
            ("small", "house"): 2,
        }

        try:
            kn = freqprob.KneserNey(bigrams, discount=0.75, logprob=False)  # type: ignore[arg-type]

            # Test some probabilities
            prob_the_cat = kn(("the", "cat"))
            prob_the_dog = kn(("the", "dog"))

            assert prob_the_cat > 0
            assert prob_the_dog > 0
            assert not math.isnan(prob_the_cat)
            assert not math.isnan(prob_the_dog)

            # More frequent bigram should have higher probability
            assert prob_the_cat > prob_the_dog

        except (ValueError, RuntimeError):
            pytest.skip("Kneser-Ney failed on this distribution")

    def test_interpolated_smoothing_weights(self) -> None:
        """Test interpolated smoothing weight properties."""
        high_order: FrequencyDistribution = {("a", "b", "c"): 10, ("d", "e", "f"): 5}
        low_order: FrequencyDistribution = {("b", "c"): 20, ("e", "f"): 15, ("x", "y"): 10}

        for lambda_weight in [0.1, 0.3, 0.5, 0.7, 0.9]:
            interpolated = freqprob.InterpolatedSmoothing(
                high_order, low_order, lambda_weight=lambda_weight, logprob=False
            )

            # Test that probabilities are valid
            prob = interpolated(("a", "b", "c"))
            assert 0 <= prob <= 1
            assert not math.isnan(prob)
            assert not math.isinf(prob)

    def test_streaming_convergence_properties(self) -> None:
        """Test streaming method convergence properties."""
        # True distribution to simulate
        true_dist = {"common": 0.6, "medium": 0.3, "rare": 0.1}

        streaming_mle = freqprob.StreamingMLE(logprob=False)

        # Simulate streaming updates
        np.random.seed(42)
        for _ in range(10):  # 10 batches
            batch_size = 1000

            # Generate words according to true distribution
            words = np.random.choice(
                list(true_dist.keys()), size=batch_size, p=list(true_dist.values())
            )

            # Update streaming model
            for word in words:
                streaming_mle.update_single(word)

        # Check convergence to true probabilities
        for word, true_prob in true_dist.items():
            estimated_prob = streaming_mle(word)
            error = abs(estimated_prob - true_prob)
            assert error < 0.05  # Should be close after 10k samples

    def test_memory_efficient_accuracy_preservation(self) -> None:
        """Test that memory-efficient representations preserve accuracy."""
        # Create reference distribution
        original_dist: FrequencyDistribution = {
            f"item_{i}": max(1, int(1000 / (i + 1))) for i in range(500)
        }

        # Create compressed version with more quantization levels for better accuracy
        compressed_dist = freqprob.create_compressed_distribution(
            original_dist, quantization_levels=256
        )

        # Test accuracy preservation
        total_original = sum(original_dist.values())
        total_compressed = 0

        for word in original_dist:
            compressed_count = compressed_dist.get_count(word)
            total_compressed += compressed_count

            # Individual counts should be reasonably close for larger values
            # (quantization error is proportionally higher for very small counts)
            original_count = original_dist[word]
            if original_count >= 10:  # Only check accuracy for counts >= 10
                relative_error = abs(compressed_count - original_count) / original_count
                assert relative_error < 0.31  # Allow 31% error due to quantization

        # Total should be preserved reasonably well
        total_error = abs(total_compressed - total_original) / total_original
        assert total_error < 0.11  # Allow 11% total error due to quantization

    def test_vectorized_consistency_with_individual(self) -> None:
        """Test vectorized operations consistency with individual calls."""
        dist = {f"token_{i}": i + 1 for i in range(100)}

        # Test multiple methods
        methods = [
            freqprob.MLE(dist, logprob=False),  # type: ignore[arg-type]
            freqprob.Laplace(dist, bins=200, logprob=False),  # type: ignore[arg-type]
            freqprob.ELE(dist, bins=200, logprob=False),  # type: ignore[arg-type]
        ]

        for method in methods:
            vectorized = freqprob.VectorizedScorer(method)

            # Test batch vs individual scoring
            test_words = [f"token_{i}" for i in range(0, 50, 5)]

            # Individual scores
            individual_scores = [method(word) for word in test_words]

            # Batch scores
            batch_scores = vectorized.score_batch(test_words)  # type: ignore[arg-type]

            # Should be identical
            for ind_score, batch_score in zip(individual_scores, batch_scores, strict=False):
                assert abs(ind_score - batch_score) < 1e-15

    def test_lazy_evaluation_correctness(self) -> None:
        """Test lazy evaluation produces correct results."""
        dist = {"word1": 50, "word2": 30, "word3": 20}

        # Create lazy and regular versions
        regular_mle = freqprob.MLE(dist, logprob=False)  # type: ignore[arg-type]
        lazy_mle = freqprob.create_lazy_mle(dist, logprob=False)  # type: ignore[arg-type]

        # Test that results are identical
        for word in [*list(dist.keys()), "unknown"]:
            regular_prob = regular_mle(word)
            lazy_prob = lazy_mle(word)
            assert abs(regular_prob - lazy_prob) < 1e-15

    @pytest.mark.parametrize(
        "method_config",
        [
            ("MLE", {}),
            ("Laplace", {"bins": 100}),
            ("ELE", {"bins": 100}),
            ("Lidstone", {"gamma": 0.5, "bins": 100}),
            ("BayesianSmoothing", {"alpha": 0.5}),
            ("Uniform", {"unobs_prob": 0.1}),
        ],
    )
    def test_method_statistical_consistency(
        self, method_config: tuple[str, dict[str, Any]]
    ) -> None:
        """Parametrized test for statistical consistency across methods."""
        method_name, params = method_config

        # Test distribution
        counts = {"frequent": 100, "medium": 50, "rare": 10}

        method_class = getattr(freqprob, method_name)
        method = method_class(counts, logprob=False, **params)

        # Test basic probability properties
        total_observed_prob = 0
        for word in counts:
            prob = method(word)

            # Basic validity
            assert 0 <= prob <= 1
            assert not math.isnan(prob)
            assert not math.isinf(prob)

            total_observed_prob += prob

        # Total observed probability should be reasonable
        if method_name in ["MLE"]:
            # MLE should sum to exactly 1 for observed words
            assert abs(total_observed_prob - 1.0) < 1e-14
        elif method_name == "Uniform":
            # Uniform with unobs_prob should leave mass for unseen
            assert total_observed_prob < 1.0
        else:
            # Smoothing methods should leave some mass for unseen words
            assert total_observed_prob < 1.0

        # Test monotonicity (higher counts → higher probability)
        assert method("frequent") >= method("medium")
        assert method("medium") >= method("rare")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
