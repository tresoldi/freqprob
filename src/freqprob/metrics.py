"""Model-evaluation metrics: perplexity, cross-entropy, KL divergence."""

import math
from collections.abc import Iterable

from .base import Element, ScoringMethod


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
