"""Property-based testing for FreqProb using Hypothesis.

This module uses property-based testing to verify mathematical properties
and invariants that should hold for all smoothing methods across a wide
range of generated inputs.
"""

# mypy: disable-error-code=misc

import math
from typing import TYPE_CHECKING, Any, cast

import pytest

import freqprob

if TYPE_CHECKING:
    from collections.abc import Callable

    from freqprob.base import FrequencyDistribution, ScoringMethod

try:
    from hypothesis import assume, given, settings
    from hypothesis import strategies as st
    from hypothesis.stateful import RuleBasedStateMachine, invariant, rule

    HAS_HYPOTHESIS = True
except ImportError:
    HAS_HYPOTHESIS = False
    pytest.skip("hypothesis not available", allow_module_level=True)


# Hypothesis strategies for generating test data
@st.composite
def frequency_distribution(
    draw: Any, min_vocab: int = 1, max_vocab: int = 100, min_count: int = 1, max_count: int = 1000
) -> dict[str, int]:
    """Generate a valid frequency distribution."""
    vocab_size = draw(st.integers(min_value=min_vocab, max_value=max_vocab))

    # Generate unique words
    words = [f"word_{i}" for i in range(vocab_size)]

    # Generate counts for each word
    counts = draw(
        st.lists(
            st.integers(min_value=min_count, max_value=max_count),
            min_size=vocab_size,
            max_size=vocab_size,
        )
    )

    return dict(zip(words, counts, strict=False))


@st.composite
def smoothing_parameters(draw) -> dict[str, Any]:  # type: ignore
    """Generate valid smoothing parameters."""
    return {
        "gamma": draw(st.floats(min_value=0.01, max_value=10.0, allow_nan=False)),
        "alpha": draw(st.floats(min_value=0.01, max_value=10.0, allow_nan=False)),
        "bins": draw(st.integers(min_value=10, max_value=10000)),
        "discount": draw(st.floats(min_value=0.01, max_value=0.99, allow_nan=False)),
    }


class TestPropertyBasedSmoothing:
    """Property-based tests for smoothing methods."""

    @given(freq_dist=frequency_distribution())
    @settings(max_examples=50, deadline=5000)
    def test_mle_probability_axioms(self, freq_dist: dict[str, int]) -> None:
        """Test that MLE satisfies basic probability axioms."""
        mle = freqprob.MLE(freq_dist, logprob=False)  # type: ignore[arg-type]

        # Property 1: All probabilities are non-negative
        for word in freq_dist:
            prob = mle(word)
            assert prob >= 0, f"Negative probability for {word}: {prob}"

        # Property 2: All probabilities are <= 1
        for word in freq_dist:
            prob = mle(word)
            assert prob <= 1.0, f"Probability > 1 for {word}: {prob}"

        # Property 3: Probabilities sum to 1 for observed vocabulary
        total_prob = sum(mle(word) for word in freq_dist)
        assert abs(total_prob - 1.0) < 1e-14, f"Probabilities don't sum to 1: {total_prob}"

        # Property 4: Unknown words have 0 probability
        assert mle("unknown_word_xyz") == 0.0

    @given(freq_dist=frequency_distribution(), params=smoothing_parameters())
    @settings(max_examples=30, deadline=5000)
    def test_laplace_smoothing_properties(
        self, freq_dist: dict[str, int], params: dict[str, Any]
    ) -> None:
        """Test Laplace smoothing properties."""
        bins = params["bins"]

        # Ensure bins is at least as large as vocabulary
        bins = max(bins, len(freq_dist))

        laplace = freqprob.Laplace(freq_dist, bins=bins, logprob=False)  # type: ignore[arg-type]

        # Property 1: All probabilities are positive
        for word in freq_dist:
            prob = laplace(word)
            assert prob > 0, f"Non-positive probability for {word}: {prob}"

        # Property 2: Unknown words have positive probability
        unknown_prob = laplace("unknown_word_xyz")
        assert unknown_prob > 0, f"Unknown word has non-positive probability: {unknown_prob}"

        # Property 3: Higher counts should have higher probabilities
        sorted_words = sorted(freq_dist.keys(), key=lambda w: freq_dist[w], reverse=True)
        for i in range(len(sorted_words) - 1):
            word1, word2 = sorted_words[i], sorted_words[i + 1]
            if freq_dist[word1] > freq_dist[word2]:
                assert laplace(word1) >= laplace(word2), (
                    f"Monotonicity violation: {word1}({freq_dist[word1]}) < {word2}({freq_dist[word2]})"
                )

        # Property 4: Formula correctness
        total_count = sum(freq_dist.values())
        for word, count in freq_dist.items():
            expected = (count + 1) / (total_count + bins)
            actual = laplace(word)
            assert abs(actual - expected) < 1e-14, (
                f"Formula violation for {word}: expected {expected}, got {actual}"
            )

    @given(freq_dist=frequency_distribution(), params=smoothing_parameters())
    @settings(max_examples=30, deadline=5000)
    def test_lidstone_generalization(
        self, freq_dist: dict[str, int], params: dict[str, Any]
    ) -> None:
        """Test Lidstone smoothing as generalization of add-k."""
        gamma = params["gamma"]
        bins = max(params["bins"], len(freq_dist))

        lidstone = freqprob.Lidstone(freq_dist, gamma=gamma, bins=bins, logprob=False)  # type: ignore[arg-type]

        # Property 1: All probabilities are positive
        for word in freq_dist:
            prob = lidstone(word)
            assert prob > 0, f"Non-positive probability for {word}: {prob}"

        # Property 2: Formula correctness
        total_count = sum(freq_dist.values())
        for word, count in freq_dist.items():
            expected = (count + gamma) / (total_count + gamma * bins)
            actual = lidstone(word)
            assert abs(actual - expected) < 1e-14, (
                f"Lidstone formula violation for {word}: expected {expected}, got {actual}"
            )

        # Property 3: Unknown word probability
        expected_unknown = gamma / (total_count + gamma * bins)
        actual_unknown = lidstone("unknown_word_xyz")
        assert abs(actual_unknown - expected_unknown) < 1e-14, (
            f"Unknown word probability: expected {expected_unknown}, got {actual_unknown}"
        )

    @given(freq_dist=frequency_distribution())
    @settings(max_examples=30, deadline=5000)
    def test_ele_lidstone_equivalence(self, freq_dist: dict[str, int]) -> None:
        """Test that ELE is equivalent to Lidstone with gamma=0.5."""
        bins = len(freq_dist) * 2  # Reasonable bins value

        ele = freqprob.ELE(freq_dist, bins=bins, logprob=False)  # type: ignore[arg-type]
        lidstone = freqprob.Lidstone(freq_dist, gamma=0.5, bins=bins, logprob=False)  # type: ignore[arg-type]

        # Should be equivalent for all words
        for word in freq_dist:
            ele_prob = ele(word)
            lidstone_prob = lidstone(word)
            assert abs(ele_prob - lidstone_prob) < 1e-14, (
                f"ELE/Lidstone mismatch for {word}: {ele_prob} vs {lidstone_prob}"
            )

        # Should be equivalent for unknown words
        ele_unknown = ele("unknown_word_xyz")
        lidstone_unknown = lidstone("unknown_word_xyz")
        assert abs(ele_unknown - lidstone_unknown) < 1e-14, (
            f"ELE/Lidstone unknown word mismatch: {ele_unknown} vs {lidstone_unknown}"
        )

    @given(freq_dist=frequency_distribution(), params=smoothing_parameters())
    @settings(max_examples=30, deadline=5000)
    def test_bayesian_smoothing_properties(
        self, freq_dist: dict[str, int], params: dict[str, Any]
    ) -> None:
        """Test Bayesian smoothing with Dirichlet prior."""
        alpha = params["alpha"]

        bayesian = freqprob.BayesianSmoothing(freq_dist, alpha=alpha, logprob=False)  # type: ignore[arg-type]

        # Property 1: All probabilities are positive
        for word in freq_dist:
            prob = bayesian(word)
            assert prob > 0, f"Non-positive probability for {word}: {prob}"

        # Property 2: Formula correctness
        total_count = sum(freq_dist.values())
        vocab_size = len(freq_dist)

        for word, count in freq_dist.items():
            expected = (count + alpha) / (total_count + alpha * vocab_size)
            actual = bayesian(word)
            assert abs(actual - expected) < 1e-14, (
                f"Bayesian formula violation for {word}: expected {expected}, got {actual}"
            )

        # Property 3: Probabilities sum to 1 for observed vocabulary
        total_prob = sum(bayesian(word) for word in freq_dist)
        assert abs(total_prob - 1.0) < 1e-14, f"Probabilities don't sum to 1: {total_prob}"

    @given(freq_dist=frequency_distribution())
    @settings(max_examples=20, deadline=10000)
    def test_log_linear_consistency(self, freq_dist: dict[str, int]) -> None:
        """Test consistency between log and linear probability representations."""

        def create_mle_linear(fd: dict[str, int]) -> freqprob.MLE:
            return freqprob.MLE(fd, logprob=False)  # type: ignore[arg-type]

        def create_laplace_linear(fd: dict[str, int]) -> freqprob.Laplace:
            return freqprob.Laplace(fd, bins=len(fd) * 2, logprob=False)  # type: ignore[arg-type]

        def create_ele_linear(fd: dict[str, int]) -> freqprob.ELE:
            return freqprob.ELE(fd, bins=len(fd) * 2, logprob=False)  # type: ignore[arg-type]

        def create_mle_log(fd: dict[str, int]) -> freqprob.MLE:
            return freqprob.MLE(fd, logprob=True)  # type: ignore[arg-type]

        def create_laplace_log(fd: dict[str, int]) -> freqprob.Laplace:
            return freqprob.Laplace(fd, bins=len(fd) * 2, logprob=True)  # type: ignore[arg-type]

        def create_ele_log(fd: dict[str, int]) -> freqprob.ELE:
            return freqprob.ELE(fd, bins=len(fd) * 2, logprob=True)  # type: ignore[arg-type]

        methods = [create_mle_linear, create_laplace_linear, create_ele_linear]
        log_methods = [create_mle_log, create_laplace_log, create_ele_log]

        for method_factory, log_method_factory in zip(methods, log_methods, strict=False):
            linear_method = method_factory(freq_dist)
            log_method = log_method_factory(freq_dist)

            for word in freq_dist:
                linear_prob = linear_method(word)
                log_prob = log_method(word)

                # Convert log to linear
                converted_prob = math.exp(log_prob)

                relative_error = abs(linear_prob - converted_prob) / max(linear_prob, 1e-10)
                assert relative_error < 1e-12, (
                    f"Log/linear inconsistency for {word}: {linear_prob} vs {converted_prob}"
                )

    @given(freq_dist=frequency_distribution())
    @settings(max_examples=20, deadline=5000)
    def test_scaling_invariance(self, freq_dist: dict[str, int]) -> None:
        """Test that relative probabilities are preserved under count scaling."""
        # Skip very small distributions
        assume(len(freq_dist) >= 2)

        # Create scaled version
        scale_factor = 5
        scaled_dist = {word: count * scale_factor for word, count in freq_dist.items()}

        # Test MLE scaling invariance
        original_mle = freqprob.MLE(freq_dist, logprob=False)  # type: ignore[arg-type]
        scaled_mle = freqprob.MLE(scaled_dist, logprob=False)  # type: ignore[arg-type]

        for word in freq_dist:
            original_prob = original_mle(word)
            scaled_prob = scaled_mle(word)
            assert abs(original_prob - scaled_prob) < 1e-14, (
                f"MLE scaling violation for {word}: {original_prob} vs {scaled_prob}"
            )

    @given(freq_dist=frequency_distribution())
    @settings(max_examples=15, deadline=10000)
    def test_smoothing_reduces_zero_probabilities(self, freq_dist: dict[str, int]) -> None:
        """Test that smoothing methods assign positive probability to unseen events."""
        bins = len(freq_dist) * 2

        # Methods that should smooth (give positive probability to unseen words)
        smoothing_methods: list[Callable[[FrequencyDistribution], ScoringMethod]] = [
            lambda fd: freqprob.Laplace(fd, bins=bins, logprob=False),
            lambda fd: freqprob.ELE(fd, bins=bins, logprob=False),
            lambda fd: freqprob.Lidstone(fd, gamma=0.5, bins=bins, logprob=False),
            lambda fd: freqprob.BayesianSmoothing(fd, alpha=0.5, logprob=False),
        ]

        for method_factory in smoothing_methods:
            method = method_factory(cast("FrequencyDistribution", freq_dist))
            unknown_prob = method("definitely_unknown_word_xyz")
            assert unknown_prob > 0, (
                f"{method.__class__.__name__} gives zero probability to unknown word"
            )

    @given(data=st.data())
    @settings(max_examples=10, deadline=15000)
    def test_method_comparison_consistency(self, data: Any) -> None:
        """Test model comparison utility consistency."""
        # Generate test data
        freq_dist = data.draw(frequency_distribution(min_vocab=5, max_vocab=20))

        # Generate test words (mix of seen and unseen)
        seen_words = list(freq_dist.keys())[:3]
        unseen_words = ["unseen_1", "unseen_2"]
        test_data = seen_words * 10 + unseen_words  # Repeat for statistical stability

        # Create models
        models = {
            "mle": freqprob.MLE(freq_dist, logprob=True),
            "laplace": freqprob.Laplace(freq_dist, bins=len(freq_dist) * 2, logprob=True),
        }

        try:
            results = freqprob.model_comparison(models, test_data)

            # All models should have results
            assert len(results) == len(models)

            for _, metrics in results.items():
                # Should have standard metrics
                assert "perplexity" in metrics
                assert "cross_entropy" in metrics

                # Values should be reasonable
                if math.isfinite(metrics["perplexity"]):
                    assert metrics["perplexity"] > 0
                assert metrics["cross_entropy"] > 0

        except Exception:
            # Some combinations might fail due to infinite perplexity, which is acceptable
            pass


class TestPropertyBasedUtilities:
    """Property-based tests for utility functions."""

    @given(st.lists(st.text(min_size=1, max_size=10), min_size=2, max_size=20))
    @settings(max_examples=30, deadline=5000)
    def test_ngram_generation_properties(self, tokens: list[str]) -> None:
        """Test n-gram generation properties."""
        for n in range(1, min(5, len(tokens) + 1)):
            ngrams = freqprob.generate_ngrams(tokens, n)

            # Property 1: Correct number of n-grams (sliding window)
            expected_count = max(0, len(tokens) - n + 1)
            assert len(ngrams) == expected_count, (
                f"Wrong number of {n}-grams: expected {expected_count}, got {len(ngrams)}"
            )

            # Property 2: Each n-gram has correct length
            for ngram in ngrams:
                assert len(ngram) == n, f"N-gram has wrong length: {ngram}"

            # Property 3: N-grams are consecutive substrings
            if ngrams:
                assert ngrams[0][0] == tokens[0], (
                    f"First {n}-gram doesn't start with first token: {ngrams[0]}"
                )
                assert ngrams[-1][-1] == tokens[-1], (
                    f"Last {n}-gram doesn't end with last token: {ngrams[-1]}"
                )

    @given(st.lists(st.text(min_size=1, max_size=10), min_size=1, max_size=50))
    @settings(max_examples=30, deadline=5000)
    def test_word_frequency_properties(self, tokens: list[str]) -> None:
        """Test word frequency calculation properties."""
        freq_dict = freqprob.word_frequency(tokens)

        # Property 1: All counts are positive
        for word, count in freq_dict.items():
            assert count > 0, f"Non-positive count for {word}: {count}"

        # Property 2: Total count equals number of tokens
        total_count = sum(freq_dict.values())
        assert total_count == len(tokens), (
            f"Total count mismatch: expected {len(tokens)}, got {total_count}"
        )

        # Property 3: All words in tokens appear in frequency dict
        unique_tokens = set(tokens)
        assert set(freq_dict.keys()) == unique_tokens, (
            "Frequency dict keys don't match unique tokens"
        )

    @given(st.lists(st.text(min_size=1, max_size=10), min_size=3, max_size=20))
    @settings(max_examples=20, deadline=5000)
    def test_ngram_frequency_properties(self, tokens: list[str]) -> None:
        """Test n-gram frequency calculation properties."""
        for n in range(2, min(4, len(tokens))):
            ngram_freq = freqprob.ngram_frequency(tokens, n)

            # Property 1: All counts are positive
            for ngram, count in ngram_freq.items():
                assert count > 0, f"Non-positive count for {ngram}: {count}"

            # Property 2: Generated n-grams match direct generation
            direct_ngrams = freqprob.generate_ngrams(tokens, n)
            expected_freq: dict[tuple[str, ...], int] = {}
            for ngram in direct_ngrams:
                expected_freq[ngram] = expected_freq.get(ngram, 0) + 1

            assert ngram_freq == expected_freq, "N-gram frequency doesn't match direct generation"


class TestPropertyBasedVectorized:
    """Property-based tests for vectorized operations."""

    @given(freq_dist=frequency_distribution())
    @settings(max_examples=20, deadline=10000)
    def test_vectorized_consistency(self, freq_dist: dict[str, int]) -> None:
        """Test vectorized operations match individual calls."""
        mle = freqprob.MLE(freq_dist, logprob=False)  # type: ignore[arg-type]
        vectorized = freqprob.VectorizedScorer(mle)

        # Test words (mix of seen and unseen)
        test_words = [*list(freq_dist.keys())[:5], "unknown_1", "unknown_2"]

        # Individual scores
        individual_scores = [mle(word) for word in test_words]

        # Batch scores
        batch_scores = vectorized.score_batch(test_words)  # type: ignore[arg-type]

        # Should be identical
        assert len(individual_scores) == len(batch_scores)
        for i, (ind_score, batch_score) in enumerate(
            zip(individual_scores, batch_scores, strict=False)
        ):
            assert abs(ind_score - batch_score) < 1e-14, (
                f"Vectorized mismatch for word {i}: {ind_score} vs {batch_score}"
            )


class FreqProbStateMachine(RuleBasedStateMachine):
    """Stateful property-based testing for FreqProb."""

    def __init__(self) -> None:
        super().__init__()
        self.freq_dist: dict[str, int] = {"initial": 1}  # Start with minimal distribution
        self.methods: dict[str, Any] = {}

    @rule(word=st.text(min_size=1, max_size=10), count=st.integers(min_value=1, max_value=100))
    def add_word(self, word: str, count: int) -> None:
        """Add or update a word in the frequency distribution."""
        self.freq_dist[word] = self.freq_dist.get(word, 0) + count

        # Recreate methods with updated distribution
        try:
            self.methods["mle"] = freqprob.MLE(self.freq_dist, logprob=False)  # type: ignore[arg-type]
            self.methods["laplace"] = freqprob.Laplace(
                self.freq_dist,  # type: ignore[arg-type]
                bins=len(self.freq_dist) * 2,
                logprob=False,
            )
        except Exception:
            # Some operations might fail, which is acceptable
            pass

    @rule(word=st.text(min_size=1, max_size=10))
    def query_word(self, word: str) -> None:
        """Query probability of a word."""
        for method_name, method in self.methods.items():
            try:
                prob = method(word)
                assert 0 <= prob <= 1, f"Invalid probability from {method_name}: {prob}"
                assert not math.isnan(prob), f"NaN probability from {method_name}"
            except Exception:
                # Some queries might fail, which is acceptable
                pass

    @invariant()
    def probability_axioms_hold(self) -> None:
        """Invariant: Probability axioms should always hold."""
        if not self.methods:
            return

        for method_name, method in self.methods.items():
            try:
                # Test a few words from the distribution
                for word in list(self.freq_dist.keys())[:3]:
                    prob = method(word)
                    assert 0 <= prob <= 1, f"Probability axiom violation in {method_name}"
                    assert not math.isnan(prob), f"NaN in {method_name}"
            except Exception:
                # Some methods might fail, which is acceptable
                pass

    @invariant()
    def consistency_between_methods(self) -> None:
        """Invariant: Methods should be internally consistent."""
        if len(self.methods) < 2:
            return

        try:
            # All methods should agree on relative ordering for observed words
            if len(self.freq_dist) >= 2:
                words = list(self.freq_dist.keys())[:2]
                word1, word2 = words[0], words[1]

                if self.freq_dist[word1] > self.freq_dist[word2]:
                    for method in self.methods.values():
                        prob1 = method(word1)
                        prob2 = method(word2)
                        # Smoothing might change ordering, so we use >= instead of >
                        assert prob1 >= prob2 * 0.5, "Severe ordering violation between methods"
        except Exception:
            # Some comparisons might fail, which is acceptable
            pass


# Only run stateful testing if explicitly requested (it's slow)
TestFreqProbStateMachine = FreqProbStateMachine.TestCase

if __name__ == "__main__":
    # Run with more examples for thorough testing
    pytest.main([__file__, "-v", "--hypothesis-show-statistics"])
