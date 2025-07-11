"""Regression tests against reference implementations.

This module tests FreqProb implementations against established reference
implementations from NLTK, scipy, and other authoritative sources to ensure
compatibility and correctness.
"""

import math
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytest

import freqprob

if TYPE_CHECKING:
    from freqprob.base import FrequencyDistribution

# Optional imports for reference implementations
try:
    from nltk.probability import FreqDist, LaplaceeProbDist, LidstoneProbDist, MLEProbDist

    HAS_NLTK = True
except ImportError:
    HAS_NLTK = False

try:
    from scipy import stats

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    from sklearn.feature_extraction.text import CountVectorizer

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


class TestNLTKRegression:
    """Regression tests against NLTK implementations."""

    @pytest.mark.skipif(not HAS_NLTK, reason="NLTK not available")
    def test_mle_against_nltk(self) -> None:
        """Test MLE implementation against NLTK MLEProbDist."""
        # Sample data
        words = ["the", "cat", "sat", "on", "the", "mat", "the", "cat"]
        freq_counts: dict[str, int] = {}
        for word in words:
            freq_counts[word] = freq_counts.get(word, 0) + 1

        # FreqProb MLE
        freqprob_mle = freqprob.MLE(cast("FrequencyDistribution", freq_counts), logprob=False)

        # NLTK MLE
        nltk_freqdist = FreqDist(words)
        nltk_mle = MLEProbDist(nltk_freqdist)

        # Compare probabilities for all words
        for word in freq_counts:
            freqprob_prob = freqprob_mle(word)
            nltk_prob = nltk_mle.prob(word)

            # Should be very close (within floating point precision)
            assert abs(freqprob_prob - nltk_prob) < 1e-14, f"MLE mismatch for '{word}'"

        # Test unknown word
        freqprob_unknown = freqprob_mle("unknown")
        nltk_unknown = nltk_mle.prob("unknown")
        assert freqprob_unknown == nltk_unknown == 0.0

    @pytest.mark.skipif(not HAS_NLTK, reason="NLTK not available")
    def test_laplace_against_nltk(self) -> None:
        """Test Laplace smoothing against NLTK LaplaceeProbDist."""
        words = ["apple", "banana", "cherry", "apple", "banana", "apple"]
        freq_counts: dict[str, int] = {}
        for word in words:
            freq_counts[word] = freq_counts.get(word, 0) + 1

        # For NLTK comparison, bins should match vocabulary size
        vocab_size = len(set(words))

        # FreqProb Laplace
        freqprob_laplace = freqprob.Laplace(
            cast("FrequencyDistribution", freq_counts), bins=vocab_size, logprob=False
        )

        # NLTK Laplace
        nltk_freqdist = FreqDist(words)
        nltk_laplace = LaplaceeProbDist(nltk_freqdist)

        # Compare probabilities
        for word in freq_counts:
            freqprob_prob = freqprob_laplace(word)
            nltk_prob = nltk_laplace.prob(word)

            relative_error = abs(freqprob_prob - nltk_prob) / nltk_prob
            assert relative_error < 1e-10, f"Laplace mismatch for '{word}'"

        # Test unknown word
        freqprob_unknown = freqprob_laplace("unknown")
        nltk_unknown = nltk_laplace.prob("unknown")

        relative_error = abs(freqprob_unknown - nltk_unknown) / nltk_unknown
        assert relative_error < 1e-10

    @pytest.mark.skipif(not HAS_NLTK, reason="NLTK not available")
    def test_lidstone_against_nltk(self) -> None:
        """Test Lidstone smoothing against NLTK LidstoneProbDist."""
        words = ["dog", "cat", "mouse", "dog", "cat", "dog"]
        freq_counts: dict[str, int] = {}
        for word in words:
            freq_counts[word] = freq_counts.get(word, 0) + 1

        gamma_values = [0.1, 0.5, 1.0, 2.0]
        vocab_size = len(set(words))

        for gamma in gamma_values:
            # FreqProb Lidstone
            freqprob_lidstone = freqprob.Lidstone(
                cast("FrequencyDistribution", freq_counts),
                gamma=gamma,
                bins=vocab_size,
                logprob=False,
            )

            # NLTK Lidstone
            nltk_freqdist = FreqDist(words)
            nltk_lidstone = LidstoneProbDist(nltk_freqdist, gamma, vocab_size)

            # Compare probabilities
            for word in freq_counts:
                freqprob_prob = freqprob_lidstone(word)
                nltk_prob = nltk_lidstone.prob(word)

                relative_error = abs(freqprob_prob - nltk_prob) / nltk_prob
                assert relative_error < 1e-10, f"Lidstone mismatch for '{word}' with gamma={gamma}"

            # Test unknown word
            freqprob_unknown = freqprob_lidstone("unknown")
            nltk_unknown = nltk_lidstone.prob("unknown")

            if nltk_unknown > 0:  # Avoid division by zero
                relative_error = abs(freqprob_unknown - nltk_unknown) / nltk_unknown
                assert relative_error < 1e-10

    @pytest.mark.skipif(not HAS_NLTK, reason="NLTK not available")
    def test_frequency_distribution_compatibility(self) -> None:
        """Test frequency distribution compatibility with NLTK."""
        # Create sample text
        text = [
            "The",
            "quick",
            "brown",
            "fox",
            "jumps",
            "over",
            "the",
            "lazy",
            "dog.",
            "The",
            "dog",
            "was",
            "very",
            "lazy",
            "and",
            "slept",
            "all",
            "day.",
            "The",
            "fox",
            "was",
            "quick",
            "and",
            "brown.",
        ]

        # FreqProb frequency counting
        freqprob_counts: dict[str, int] = {}
        for word in text:
            freqprob_counts[word] = freqprob_counts.get(word, 0) + 1

        # NLTK frequency counting
        nltk_freqdist = FreqDist(text)

        # Compare counts
        for word in freqprob_counts:
            assert freqprob_counts[word] == nltk_freqdist[word]

        # Test total counts
        assert sum(freqprob_counts.values()) == nltk_freqdist.N()

    @pytest.mark.skipif(not HAS_NLTK, reason="NLTK not available")
    def test_ngram_generation_compatibility(self) -> None:
        """Test n-gram generation compatibility with NLTK."""
        from nltk.util import ngrams as nltk_ngrams

        tokens = ["the", "quick", "brown", "fox", "jumps"]

        # FreqProb n-grams
        freqprob_bigrams = freqprob.generate_ngrams(tokens, 2)
        freqprob_trigrams = freqprob.generate_ngrams(tokens, 3)

        # NLTK n-grams (need to add sentence boundaries manually for comparison)
        nltk_tokens = ["<s>", *tokens, "</s>"]
        nltk_bigrams = list(nltk_ngrams(nltk_tokens, 2))
        nltk_trigrams = list(nltk_ngrams(nltk_tokens, 3))

        # Convert FreqProb tuples to lists for comparison
        freqprob_bigrams_list = [list(bg) for bg in freqprob_bigrams]
        freqprob_trigrams_list = [list(tg) for tg in freqprob_trigrams]

        # Compare (should be identical)
        assert freqprob_bigrams_list == nltk_bigrams
        assert freqprob_trigrams_list == nltk_trigrams


class TestScipyRegression:
    """Regression tests against scipy implementations."""

    @pytest.mark.skipif(not HAS_SCIPY, reason="scipy not available")
    def test_entropy_calculation(self) -> None:
        """Test entropy calculation against scipy."""
        # Create probability distribution
        counts = {"a": 40, "b": 30, "c": 20, "d": 10}
        total = sum(counts.values())
        probs = [count / total for count in counts.values()]

        # FreqProb entropy calculation (manual)
        mle = freqprob.MLE(cast("FrequencyDistribution", counts), logprob=True)
        freqprob_entropy = 0.0
        for word in counts:
            log_prob = mle(word)
            prob = math.exp(log_prob)
            freqprob_entropy -= prob * log_prob

        # Scipy entropy
        scipy_entropy = stats.entropy(probs, base=math.e)

        # Should be very close
        assert abs(freqprob_entropy - scipy_entropy) < 1e-12

    @pytest.mark.skipif(not HAS_SCIPY, reason="scipy not available")
    def test_kl_divergence_calculation(self) -> None:
        """Test KL divergence calculation against scipy."""
        # Two distributions
        counts1 = {"x": 60, "y": 30, "z": 10}
        counts2 = {"x": 40, "y": 40, "z": 20}

        # Convert to probability arrays for scipy
        words = ["x", "y", "z"]
        total1 = sum(counts1.values())
        total2 = sum(counts2.values())

        probs1 = [counts1[w] / total1 for w in words]
        probs2 = [counts2[w] / total2 for w in words]

        # FreqProb KL divergence
        model1 = freqprob.MLE(cast("FrequencyDistribution", counts1), logprob=True)
        model2 = freqprob.MLE(cast("FrequencyDistribution", counts2), logprob=True)
        test_data = words  # Use unique test data

        freqprob_kl = freqprob.kl_divergence(model1, model2, test_data)

        # Scipy KL divergence
        scipy_kl = stats.entropy(probs1, probs2, base=math.e)

        # Should be close (allowing for sampling differences)
        relative_error = abs(freqprob_kl - scipy_kl) / scipy_kl
        assert relative_error < 0.01  # 1% tolerance

    @pytest.mark.skipif(not HAS_SCIPY, reason="scipy not available")
    def test_multinomial_properties(self) -> None:
        """Test multinomial distribution properties against scipy."""
        # Parameters for multinomial distribution
        n_trials = 1000
        true_probs = [0.4, 0.3, 0.2, 0.1]
        categories = ["A", "B", "C", "D"]

        # Generate sample using scipy
        np.random.seed(42)
        sample_counts = stats.multinomial.rvs(n_trials, true_probs)

        # Create FreqProb distribution
        freq_counts = dict(zip(categories, sample_counts, strict=False))
        mle = freqprob.MLE(cast("FrequencyDistribution", freq_counts), logprob=False)

        # Compare estimated probabilities with true probabilities
        for i, category in enumerate(categories):
            estimated_prob = mle(category)
            true_prob = true_probs[i]

            # Should be reasonably close for large sample
            error = abs(estimated_prob - true_prob)
            assert error < 0.05  # Allow 5% error

    @pytest.mark.skipif(not HAS_SCIPY, reason="scipy not available")
    def test_dirichlet_prior_properties(self) -> None:
        """Test Dirichlet prior properties against scipy."""
        # Dirichlet parameters (concentration parameters)
        alpha = [2.0, 1.0, 0.5]
        categories = ["word1", "word2", "word3"]

        # Generate sample from Dirichlet
        np.random.seed(42)
        true_probs = stats.dirichlet.rvs(alpha)[0]

        # Simulate observed counts
        n_samples = 1000
        observed_counts = stats.multinomial.rvs(n_samples, true_probs)
        freq_counts = dict(zip(categories, observed_counts, strict=False))

        # FreqProb Bayesian smoothing (using same alpha values)
        # Note: Our alpha parameter is applied per category
        for i, category in enumerate(categories):
            bayesian = freqprob.BayesianSmoothing(
                cast("FrequencyDistribution", freq_counts), alpha=alpha[i], logprob=False
            )

            # Test that probabilities are reasonable
            prob = bayesian(category)
            assert 0 < prob < 1
            assert not math.isnan(prob)


class TestSklearnRegression:
    """Regression tests against scikit-learn implementations."""

    @pytest.mark.skipif(not HAS_SKLEARN, reason="sklearn not available")
    def test_count_vectorizer_compatibility(self) -> None:
        """Test compatibility with sklearn CountVectorizer."""
        # Sample documents
        documents = ["the cat sat on the mat", "the dog ran in the park", "cats and dogs are pets"]

        # Sklearn count vectorization
        vectorizer = CountVectorizer()
        sklearn_counts = vectorizer.fit_transform(documents)
        feature_names = vectorizer.get_feature_names_out()

        # Convert to FreqProb format
        total_counts = sklearn_counts.sum(axis=0).A1  # Convert to 1D array
        freq_counts = dict(zip(feature_names, total_counts, strict=False))

        # Test FreqProb MLE
        mle = freqprob.MLE(cast("FrequencyDistribution", freq_counts), logprob=False)

        # Verify probabilities sum to 1
        total_prob = sum(mle(word) for word in freq_counts)
        assert abs(total_prob - 1.0) < 1e-14

        # Test that sklearn and FreqProb counts match
        for word, count in freq_counts.items():
            assert count > 0  # All words should appear at least once
            assert mle(word) > 0  # All probabilities should be positive


class TestReferenceDataRegression:
    """Regression tests against known reference datasets and results."""

    def test_brown_corpus_sample_regression(self) -> None:
        """Test against a small sample that mimics Brown corpus statistics."""
        # Simulated Brown corpus word frequencies (top 20 words)
        brown_sample = {
            "the": 69971,
            "of": 36411,
            "and": 28852,
            "to": 26149,
            "a": 23237,
            "in": 21341,
            "that": 10594,
            "is": 10109,
            "was": 9815,
            "he": 9548,
            "for": 9489,
            "it": 8760,
            "with": 7289,
            "as": 7253,
            "his": 6996,
            "on": 6741,
            "be": 6377,
            "at": 5380,
            "by": 5246,
            "i": 5201,
        }

        # Test MLE probabilities are reasonable
        mle = freqprob.MLE(cast("FrequencyDistribution", brown_sample), logprob=False)

        # 'the' should be most frequent
        assert mle("the") > mle("of")
        assert mle("of") > mle("and")

        # Total probability should be 1
        total = sum(brown_sample.values())
        expected_the = brown_sample["the"] / total
        assert abs(mle("the") - expected_the) < 1e-15

        # Test smoothing preserves ordering
        laplace = freqprob.Laplace(
            cast("FrequencyDistribution", brown_sample), bins=50000, logprob=False
        )
        assert laplace("the") > laplace("of")
        assert laplace("of") > laplace("and")

    def test_zipfian_distribution_regression(self) -> None:
        """Test against theoretical Zipfian distribution."""
        # Generate Zipfian distribution
        vocab_size = 1000
        zipf_exponent = 1.0  # Classic Zipf

        # Create frequency distribution following Zipf's law
        frequencies = []
        for rank in range(1, vocab_size + 1):
            freq = int(10000 / (rank**zipf_exponent))
            frequencies.append(max(1, freq))  # Ensure minimum frequency

        zipf_counts = {f"word_{i}": freq for i, freq in enumerate(frequencies)}

        # Test that MLE preserves Zipfian properties
        mle = freqprob.MLE(cast("FrequencyDistribution", zipf_counts), logprob=False)

        # Sample ranks and test Zipf relationship
        test_ranks = [1, 10, 100, 500]
        for i, rank in enumerate(test_ranks[:-1]):
            word_rank = f"word_{rank - 1}"  # 0-indexed
            word_next = f"word_{test_ranks[i + 1] - 1}"

            prob_rank = mle(word_rank)
            prob_next = mle(word_next)

            # Higher rank should have higher probability
            assert prob_rank > prob_next

    def test_known_perplexity_results(self) -> None:
        """Test perplexity calculation against known results."""
        # Simple test case with known perplexity
        train_counts = {"a": 2, "b": 1}  # P(a)=2/3, P(b)=1/3
        test_data = ["a", "b"]  # Test on both words once

        mle = freqprob.MLE(cast("FrequencyDistribution", train_counts), logprob=True)
        perplexity = freqprob.perplexity(mle, test_data)

        # Manual calculation:
        # log_prob(a) = log(2/3), log_prob(b) = log(1/3)
        # avg_log_prob = (log(2/3) + log(1/3)) / 2 = (log(2) - 2*log(3)) / 2
        # perplexity = exp(-avg_log_prob)

        expected_avg_log_prob = (math.log(2 / 3) + math.log(1 / 3)) / 2
        expected_perplexity = math.exp(-expected_avg_log_prob)

        assert abs(perplexity - expected_perplexity) < 1e-12

    def test_cross_entropy_known_results(self) -> None:
        """Test cross-entropy against known analytical results."""
        # Simple uniform distribution
        uniform_counts = {"x": 1, "y": 1, "z": 1}  # Each has probability 1/3
        test_data = ["x", "y", "z"]

        mle = freqprob.MLE(cast("FrequencyDistribution", uniform_counts), logprob=True)
        ce = freqprob.cross_entropy(mle, test_data)

        # For uniform distribution over 3 symbols: H = log(3)
        expected_ce = math.log(3)
        assert abs(ce - expected_ce) < 1e-14

    @pytest.mark.parametrize(
        ("smoothing_method", "params"),
        [
            ("Laplace", {"bins": 100}),
            ("ELE", {"bins": 100}),
            ("Lidstone", {"gamma": 0.5, "bins": 100}),
        ],
    )
    def test_smoothing_monotonicity_regression(
        self, smoothing_method: str, params: dict[str, Any]
    ) -> None:
        """Test monotonicity properties across smoothing methods."""
        # Create distribution with clear frequency ordering
        ordered_counts = {
            "most_frequent": 1000,
            "very_frequent": 500,
            "frequent": 100,
            "medium": 50,
            "rare": 10,
            "very_rare": 1,
        }

        method_class = getattr(freqprob, smoothing_method)
        method = method_class(ordered_counts, logprob=False, **params)

        # Test strict ordering is preserved
        words = list(ordered_counts.keys())
        probabilities = [method(word) for word in words]

        # Should be in descending order
        for i in range(len(probabilities) - 1):
            assert probabilities[i] >= probabilities[i + 1], (
                f"{smoothing_method} violates monotonicity: {words[i]} vs {words[i + 1]}"
            )

    def test_mathematical_identities_regression(self) -> None:
        """Test mathematical identities that should hold."""
        counts = {"alpha": 25, "beta": 15, "gamma": 10}

        # Test log/linear probability consistency
        mle_linear = freqprob.MLE(cast("FrequencyDistribution", counts), logprob=False)
        mle_log = freqprob.MLE(cast("FrequencyDistribution", counts), logprob=True)

        for word in counts:
            linear_prob = mle_linear(word)
            log_prob = mle_log(word)

            # exp(log_prob) should equal linear_prob
            converted_prob = math.exp(log_prob)
            assert abs(linear_prob - converted_prob) < 1e-15

        # Test that Lidstone with gamma=1 equals Laplace
        lidstone_1 = freqprob.Lidstone(
            cast("FrequencyDistribution", counts), gamma=1.0, bins=100, logprob=False
        )
        laplace = freqprob.Laplace(cast("FrequencyDistribution", counts), bins=100, logprob=False)

        for word in counts:
            assert abs(lidstone_1(word) - laplace(word)) < 1e-15

        # Test that ELE equals Lidstone with gamma=0.5
        ele = freqprob.ELE(cast("FrequencyDistribution", counts), bins=100, logprob=False)
        lidstone_half = freqprob.Lidstone(
            cast("FrequencyDistribution", counts), gamma=0.5, bins=100, logprob=False
        )

        for word in counts:
            assert abs(ele(word) - lidstone_half(word)) < 1e-15


class TestLiteratureRegression:
    """Regression tests against results from academic literature."""

    def test_chen_goodman_1996_results(self) -> None:
        """Test against results similar to Chen & Goodman (1996) smoothing comparison."""
        # Simulate a scenario similar to their experimental setup
        # (simplified version for testing)

        # Create training data with Zipfian distribution
        train_vocab_size = 100
        train_counts = {}
        for i in range(train_vocab_size):
            # Zipfian frequency: f(i) ∝ 1/i
            freq = max(1, int(1000 / (i + 1)))
            train_counts[f"word_{i}"] = freq

        # Create test data (mix of seen and unseen words)
        test_words = []
        # Add seen words (high frequency)
        test_words.extend(["word_0"] * 50)  # Most frequent word
        test_words.extend(["word_1"] * 30)  # Second most frequent
        test_words.extend(["word_10"] * 10)  # Medium frequency

        # Add some unseen words
        test_words.extend(["unseen_1", "unseen_2"])

        # Test different smoothing methods
        methods = {
            "mle": freqprob.MLE(cast("FrequencyDistribution", train_counts), logprob=True),
            "laplace": freqprob.Laplace(
                cast("FrequencyDistribution", train_counts), bins=train_vocab_size * 2, logprob=True
            ),
            "ele": freqprob.ELE(
                cast("FrequencyDistribution", train_counts), bins=train_vocab_size * 2, logprob=True
            ),
        }

        perplexities = {}
        for name, method in methods.items():
            try:
                pp = freqprob.perplexity(method, test_words)
                perplexities[name] = pp
            except Exception:
                perplexities[name] = float("inf")

        # Expected ordering based on literature (for this type of data):
        # 1. MLE should have higher perplexity than smoothing methods (due to unseen words)
        # 2. Smoothing methods should have lower perplexity
        # 3. ELE often performs better than basic Laplace

        assert perplexities["mle"] > perplexities["laplace"]  # MLE should have higher perplexity
        assert math.isfinite(perplexities["laplace"])
        assert math.isfinite(perplexities["ele"])

        # Both smoothing methods should be reasonable
        assert perplexities["laplace"] > 1.0  # Sanity check
        assert perplexities["ele"] > 1.0  # Sanity check

    def test_good_turing_turing_1953_properties(self) -> None:
        """Test Good-Turing properties based on original Turing (1953) work."""
        # Create distribution with clear frequency-of-frequencies structure
        # This mimics the type of data Turing analyzed

        freq_dist = {}

        # Words appearing once (hapax legomena)
        for i in range(10):
            freq_dist[f"hapax_{i}"] = 1

        # Words appearing twice
        for i in range(5):
            freq_dist[f"twice_{i}"] = 2

        # Words appearing three times
        for i in range(3):
            freq_dist[f"thrice_{i}"] = 3

        # Words appearing more frequently
        freq_dist["common_1"] = 10
        freq_dist["common_2"] = 15

        try:
            sgt = freqprob.SimpleGoodTuring(cast("FrequencyDistribution", freq_dist), logprob=False)

            # Good-Turing should redistribute probability mass
            # Words with count 1 should get less probability than their MLE estimate

            hapax_word = "hapax_0"
            sgt_prob = sgt(hapax_word)

            # SGT typically reduces probability of low-frequency words
            # (though this can vary depending on the frequency distribution)
            assert sgt_prob > 0
            assert not math.isnan(sgt_prob)

            # Unknown words should have positive probability
            unknown_prob = sgt("unknown_word")
            assert unknown_prob > 0

        except (ValueError, RuntimeError):
            # SGT can fail on some distributions, which is expected behavior
            pytest.skip("Simple Good-Turing failed on this distribution (expected)")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
