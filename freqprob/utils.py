"""Utility functions for common NLP tasks and model comparison.

This module provides convenience functions for generating n-grams, computing
word frequencies, and comparing probability models.
"""

import math
from collections import Counter
from collections.abc import Iterable

from .base import Element, ScoringMethod


def generate_ngrams(text: str | list[str], n: int) -> list[tuple[str, ...]]:
    """Generate n-grams from text.

    Parameters
    ----------
    text : Union[str, List[str]]
        Input text as string or list of tokens
    n : int
        Size of n-grams to generate

    Returns:
    -------
    List[Tuple[str, ...]]
        List of n-gram tuples

    Examples:
    --------
    >>> generate_ngrams("hello world", 2)
    [('h', 'e'), ('e', 'l'), ('l', 'l'), ('l', 'o'), ('o', ' '), (' ', 'w'), ('w', 'o'), ('o', 'r'), ('r', 'l'), ('l', 'd')]

    >>> generate_ngrams(["hello", "world", "test"], 2)
    [('hello', 'world'), ('world', 'test')]
    """
    tokens = list(text) if isinstance(text, str) else text

    if n <= 0:
        raise ValueError("n must be positive")

    if len(tokens) < n:
        return []

    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def word_frequency(text: str | list[str], normalize: bool = False) -> dict[str, int | float]:
    """Compute word frequency from text.

    Parameters
    ----------
    text : Union[str, List[str]]
        Input text as string or list of tokens
    normalize : bool, default=False
        If True, return relative frequencies instead of counts

    Returns:
    -------
    Dict[str, Union[int, float]]
        Dictionary mapping words to their frequencies

    Examples:
    --------
    >>> word_frequency("hello world hello")
    {'hello': 2, 'world': 1}

    >>> word_frequency(["hello", "world", "hello"], normalize=True)
    {'hello': 0.6666666666666666, 'world': 0.3333333333333333}
    """
    tokens = text.split() if isinstance(text, str) else text

    freq_dict = Counter(tokens)

    if normalize:
        total = sum(freq_dict.values())
        return {word: count / total for word, count in freq_dict.items()}

    return dict(freq_dict)


def ngram_frequency(
    text: str | list[str], n: int, normalize: bool = False
) -> dict[tuple[str, ...], int | float]:
    """Compute n-gram frequency from text.

    Parameters
    ----------
    text : Union[str, List[str]]
        Input text as string or list of tokens
    n : int
        Size of n-grams to generate
    normalize : bool, default=False
        If True, return relative frequencies instead of counts

    Returns:
    -------
    Dict[Tuple[str, ...], Union[int, float]]
        Dictionary mapping n-grams to their frequencies

    Examples:
    --------
    >>> ngram_frequency("hello world", 2)
    {('h', 'e'): 1, ('e', 'l'): 1, ('l', 'l'): 1, ('l', 'o'): 1, ('o', ' '): 1, (' ', 'w'): 1, ('w', 'o'): 1, ('o', 'r'): 1, ('r', 'l'): 1, ('l', 'd'): 1}
    """
    ngrams = generate_ngrams(text, n)
    freq_dict = Counter(ngrams)

    if normalize:
        total = sum(freq_dict.values())
        return {ngram: count / total for ngram, count in freq_dict.items()}

    return dict(freq_dict)


def perplexity(model: ScoringMethod, test_data: Iterable[Element]) -> float:
    """Calculate perplexity of a model on test data.

    Perplexity is defined as exp(H(p)) where H(p) is the cross-entropy.
    Lower perplexity indicates better model performance.

    Parameters
    ----------
    model : ScoringMethod
        Fitted probability model
    test_data : Iterable[Element]
        Test data elements

    Returns:
    -------
    float
        Perplexity value

    Examples:
    --------
    >>> from freqprob import MLE
    >>> model = MLE({'a': 2, 'b': 1}, logprob=True)
    >>> perplexity(model, ['a', 'b', 'a'])
    1.8171205928321397
    """
    if not model.logprob:
        raise ValueError("Model must be configured for log probabilities")

    log_probs = [model(element) for element in test_data]
    cross_entropy = -sum(log_probs) / len(log_probs)

    return math.exp(cross_entropy)


def cross_entropy(model: ScoringMethod, test_data: Iterable[Element]) -> float:
    """Calculate cross-entropy of a model on test data.

    Cross-entropy measures the average number of bits needed to encode
    test data using the model's probability distribution.

    Parameters
    ----------
    model : ScoringMethod
        Fitted probability model
    test_data : Iterable[Element]
        Test data elements

    Returns:
    -------
    float
        Cross-entropy value

    Examples:
    --------
    >>> from freqprob import MLE
    >>> model = MLE({'a': 2, 'b': 1}, logprob=True)
    >>> cross_entropy(model, ['a', 'b', 'a'])
    0.5943761750071414
    """
    if not model.logprob:
        raise ValueError("Model must be configured for log probabilities")

    log_probs = [model(element) for element in test_data]
    return -sum(log_probs) / len(log_probs)


def kl_divergence(
    p_model: ScoringMethod, q_model: ScoringMethod, test_data: Iterable[Element]
) -> float:
    """Calculate Kullback-Leibler divergence between two models.

    KL divergence measures how much one probability distribution differs
    from another. It's not symmetric: KL(P||Q) ≠ KL(Q||P).

    Parameters
    ----------
    p_model : ScoringMethod
        First probability model (reference)
    q_model : ScoringMethod
        Second probability model (approximate)
    test_data : Iterable[Element]
        Test data elements

    Returns:
    -------
    float
        KL divergence value

    Examples:
    --------
    >>> from freqprob import MLE, Laplace
    >>> p_model = MLE({'a': 2, 'b': 1}, logprob=True)
    >>> q_model = Laplace({'a': 2, 'b': 1}, logprob=True)
    >>> kl_divergence(p_model, q_model, ['a', 'b', 'a'])
    0.0
    """
    if not p_model.logprob or not q_model.logprob:
        raise ValueError("Both models must be configured for log probabilities")

    kl_div = 0.0
    for element in test_data:
        p_log_prob = p_model(element)
        q_log_prob = q_model(element)

        # Convert to regular probabilities for KL calculation
        p_prob = math.exp(p_log_prob)
        q_prob = math.exp(q_log_prob)

        if p_prob > 0 and q_prob > 0:
            kl_div += p_prob * math.log(p_prob / q_prob)

    return kl_div


def model_comparison(
    models: dict[str, ScoringMethod], test_data: Iterable[Element]
) -> dict[str, dict[str, float]]:
    """Compare multiple models using various metrics.

    Parameters
    ----------
    models : Dict[str, ScoringMethod]
        Dictionary mapping model names to fitted models
    test_data : Iterable[Element]
        Test data elements

    Returns:
    -------
    Dict[str, Dict[str, float]]
        Dictionary with model names as keys and metrics as values

    Examples:
    --------
    >>> from freqprob import MLE, Laplace
    >>> models = {
    ...     'mle': MLE({'a': 2, 'b': 1}, logprob=True),
    ...     'laplace': Laplace({'a': 2, 'b': 1}, logprob=True)
    ... }
    >>> model_comparison(models, ['a', 'b', 'a'])
    {'mle': {'perplexity': 1.8171205928321397, 'cross_entropy': 0.5943761750071414}, 'laplace': {'perplexity': 1.9659482062417916, 'cross_entropy': 0.6754887502163469}}
    """
    test_data_list = list(test_data)
    results = {}

    for name, model in models.items():
        if not model.logprob:
            raise ValueError(f"Model '{name}' must be configured for log probabilities")

        results[name] = {
            "perplexity": perplexity(model, test_data_list),
            "cross_entropy": cross_entropy(model, test_data_list),
        }

    return results
