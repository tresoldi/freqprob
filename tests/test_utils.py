"""Test the utility functions for NLP tasks and model comparison."""

import pytest

from freqprob import (
    MLE,
    Laplace,
    cross_entropy,
    generate_ngrams,
    kl_divergence,
    model_comparison,
    ngram_frequency,
    perplexity,
    word_frequency,
)


def test_generate_ngrams_string() -> None:
    """Test n-gram generation from string."""
    text = "hello"

    # Test unigrams
    unigrams = generate_ngrams(text, 1)
    expected = [("h",), ("e",), ("l",), ("l",), ("o",)]
    assert unigrams == expected

    # Test bigrams
    bigrams = generate_ngrams(text, 2)
    expected = [("h", "e"), ("e", "l"), ("l", "l"), ("l", "o")]  # type: ignore[list-item]
    assert bigrams == expected

    # Test trigrams
    trigrams = generate_ngrams(text, 3)
    expected = [("h", "e", "l"), ("e", "l", "l"), ("l", "l", "o")]  # type: ignore[list-item]
    assert trigrams == expected


def test_generate_ngrams_list() -> None:
    """Test n-gram generation from token list."""
    tokens = ["hello", "world", "test"]

    # Test unigrams
    unigrams = generate_ngrams(tokens, 1)
    expected = [("hello",), ("world",), ("test",)]
    assert unigrams == expected

    # Test bigrams
    bigrams = generate_ngrams(tokens, 2)
    expected = [("hello", "world"), ("world", "test")]  # type: ignore[list-item]
    assert bigrams == expected

    # Test trigrams
    trigrams = generate_ngrams(tokens, 3)
    expected = [("hello", "world", "test")]  # type: ignore[list-item]
    assert trigrams == expected


def test_generate_ngrams_edge_cases() -> None:
    """Test edge cases for n-gram generation."""
    # Empty input
    assert generate_ngrams("", 1) == []
    assert generate_ngrams([], 1) == []

    # n larger than input
    assert generate_ngrams("hi", 5) == []
    assert generate_ngrams(["hi"], 5) == []

    # n = 0 should raise error
    with pytest.raises(ValueError, match="n must be positive"):
        generate_ngrams("test", 0)

    # Negative n should raise error
    with pytest.raises(ValueError, match="n must be positive"):
        generate_ngrams("test", -1)


def test_word_frequency_string() -> None:
    """Test word frequency from string."""
    text = "hello world hello"

    # Test counts
    freq = word_frequency(text)
    expected = {"hello": 2, "world": 1}
    assert freq == expected

    # Test normalized frequencies
    freq_norm = word_frequency(text, normalize=True)
    expected_norm = {"hello": 2 / 3, "world": 1 / 3}
    assert freq_norm == expected_norm


def test_word_frequency_list() -> None:
    """Test word frequency from token list."""
    tokens = ["hello", "world", "hello"]

    # Test counts
    freq = word_frequency(tokens)
    expected = {"hello": 2, "world": 1}
    assert freq == expected

    # Test normalized frequencies
    freq_norm = word_frequency(tokens, normalize=True)
    expected_norm = {"hello": 2 / 3, "world": 1 / 3}
    assert freq_norm == expected_norm


def test_ngram_frequency() -> None:
    """Test n-gram frequency computation."""
    text = "hello"

    # Test bigram frequencies
    freq = ngram_frequency(text, 2)
    expected = {("h", "e"): 1, ("e", "l"): 1, ("l", "l"): 1, ("l", "o"): 1}
    assert freq == expected

    # Test normalized frequencies
    freq_norm = ngram_frequency(text, 2, normalize=True)
    expected_norm = {("h", "e"): 0.25, ("e", "l"): 0.25, ("l", "l"): 0.25, ("l", "o"): 0.25}
    assert freq_norm == expected_norm


def test_perplexity() -> None:
    """Test perplexity calculation."""
    # Create a simple model
    model = MLE({"a": 2, "b": 1}, logprob=True)

    # Test data
    test_data = ["a", "b", "a"]

    # Calculate perplexity
    pp = perplexity(model, test_data)

    # Perplexity should be positive
    assert pp > 0

    # Test with model not using log probabilities
    model_no_log = MLE({"a": 2, "b": 1}, logprob=False)
    with pytest.raises(ValueError, match="Model must be configured for log probabilities"):
        perplexity(model_no_log, test_data)


def test_cross_entropy() -> None:
    """Test cross-entropy calculation."""
    # Create a simple model
    model = MLE({"a": 2, "b": 1}, logprob=True)

    # Test data
    test_data = ["a", "b", "a"]

    # Calculate cross-entropy
    ce = cross_entropy(model, test_data)

    # Cross-entropy should be positive
    assert ce > 0

    # Test with model not using log probabilities
    model_no_log = MLE({"a": 2, "b": 1}, logprob=False)
    with pytest.raises(ValueError, match="Model must be configured for log probabilities"):
        cross_entropy(model_no_log, test_data)


def test_kl_divergence() -> None:
    """Test KL divergence calculation."""
    # Create two simple models
    p_model = MLE({"a": 2, "b": 1}, logprob=True)
    q_model = Laplace({"a": 2, "b": 1}, logprob=True)

    # Test data
    test_data = ["a", "b", "a"]

    # Calculate KL divergence
    kl_div = kl_divergence(p_model, q_model, test_data)

    # KL divergence should be non-negative
    assert kl_div >= 0

    # Test with models not using log probabilities
    p_model_no_log = MLE({"a": 2, "b": 1}, logprob=False)
    q_model_no_log = Laplace({"a": 2, "b": 1}, logprob=False)

    with pytest.raises(ValueError, match="Both models must be configured for log probabilities"):
        kl_divergence(p_model_no_log, q_model, test_data)

    with pytest.raises(ValueError, match="Both models must be configured for log probabilities"):
        kl_divergence(p_model, q_model_no_log, test_data)


def test_model_comparison() -> None:
    """Test model comparison functionality."""
    # Create multiple models
    models = {
        "mle": MLE({"a": 2, "b": 1}, logprob=True),
        "laplace": Laplace({"a": 2, "b": 1}, logprob=True),
    }

    # Test data
    test_data = ["a", "b", "a"]

    # Compare models
    results = model_comparison(models, test_data)

    # Check structure
    assert "mle" in results
    assert "laplace" in results

    for metrics in results.values():
        assert "perplexity" in metrics
        assert "cross_entropy" in metrics
        assert metrics["perplexity"] > 0
        assert metrics["cross_entropy"] > 0

    # Test with model not using log probabilities
    models_no_log = {
        "mle": MLE({"a": 2, "b": 1}, logprob=False),
        "laplace": Laplace({"a": 2, "b": 1}, logprob=True),
    }

    with pytest.raises(ValueError, match="Model.*must be configured for log probabilities"):
        model_comparison(models_no_log, test_data)


def test_different_data_types() -> None:
    """Test that the library works with different data types."""
    # Test with integers
    int_data = {1: 5, 2: 3, 3: 2}
    model_int = MLE(int_data, logprob=True)  # type: ignore[arg-type]
    assert model_int(1) > model_int(2)
    assert model_int(2) > model_int(3)

    # Test with tuples
    tuple_data = {("a", "b"): 3, ("b", "c"): 2, ("c", "d"): 1}
    model_tuple = MLE(tuple_data, logprob=True)  # type: ignore[arg-type]
    assert model_tuple(("a", "b")) > model_tuple(("b", "c"))
    assert model_tuple(("b", "c")) > model_tuple(("c", "d"))

    # Test with frozensets
    frozenset_data = {frozenset(["a", "b"]): 2, frozenset(["b", "c"]): 1}
    model_frozenset = MLE(frozenset_data, logprob=True)  # type: ignore[arg-type]
    assert model_frozenset(frozenset(["a", "b"])) > model_frozenset(frozenset(["b", "c"]))

    # Test with mixed types
    mixed_data = {"hello": 3, 42: 2, (1, 2): 1}
    model_mixed = MLE(mixed_data, logprob=True)  # type: ignore[arg-type]
    assert model_mixed("hello") > model_mixed(42)
    assert model_mixed(42) > model_mixed((1, 2))


def test_utils_with_ngrams() -> None:
    """Test utility functions with n-gram data."""
    text = "hello world hello"

    # Create frequency distribution
    freq_dist = ngram_frequency(text.split(), 2)

    # Create model from n-grams
    model = MLE(freq_dist, logprob=True)  # type: ignore[arg-type]

    # Test perplexity with n-grams
    pp = perplexity(model, [("hello", "world"), ("world", "hello")])
    assert pp > 0

    # Test cross-entropy with n-grams
    ce = cross_entropy(model, [("hello", "world"), ("world", "hello")])
    assert ce > 0
