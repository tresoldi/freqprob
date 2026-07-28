"""Model-evaluation metrics: perplexity, cross-entropy, KL divergence."""

import math
from collections.abc import Iterable

from .base import Element, ScoringMethod


def perplexity(model: ScoringMethod, test_data: Iterable[Element]) -> float:
    """Calculate the perplexity of a model on test data.

    Perplexity is ``exp(H(p))`` where ``H(p)`` is the cross-entropy. Lower
    perplexity indicates a model that better predicts the test data.

    Args:
        model: A fitted probability model. Must be configured with
            ``logprob=True``.
        test_data: Iterable of test elements to score.

    Returns:
        The perplexity value.

    Raises:
        ValueError: If ``model`` is not configured for log-probabilities.

    Examples:
        >>> from freqprob import MLE, perplexity
        >>> model = MLE({"a": 3, "b": 2, "c": 1}, logprob=True)
        >>> round(perplexity(model, ["a", "b", "a", "c"]), 4)
        2.913
    """
    if not model.logprob:
        raise ValueError("Model must be configured for log probabilities")

    log_probs = [model(element) for element in test_data]
    cross_entropy = -sum(log_probs) / len(log_probs)

    return math.exp(cross_entropy)


def cross_entropy(model: ScoringMethod, test_data: Iterable[Element]) -> float:
    """Calculate the cross-entropy of a model on test data.

    Cross-entropy is the average negative log-probability the model assigns to
    the test data. Lower values indicate a better fit.

    Args:
        model: A fitted probability model. Must be configured with
            ``logprob=True``.
        test_data: Iterable of test elements to score.

    Returns:
        The cross-entropy value (in nats, since natural logs are used).

    Raises:
        ValueError: If ``model`` is not configured for log-probabilities.

    Examples:
        >>> from freqprob import MLE, cross_entropy
        >>> model = MLE({"a": 3, "b": 2, "c": 1}, logprob=True)
        >>> round(cross_entropy(model, ["a", "b", "a", "c"]), 4)
        1.0692
    """
    if not model.logprob:
        raise ValueError("Model must be configured for log probabilities")

    log_probs = [model(element) for element in test_data]
    return -sum(log_probs) / len(log_probs)


def kl_divergence(
    p_model: ScoringMethod, q_model: ScoringMethod, test_data: Iterable[Element]
) -> float:
    """Calculate the Kullback-Leibler divergence between two models.

    KL divergence measures how much the approximate distribution ``Q`` diverges
    from the reference distribution ``P``. It is not symmetric:
    ``KL(P||Q) != KL(Q||P)``.

    Args:
        p_model: The reference probability model ``P``. Must use ``logprob=True``.
        q_model: The approximate probability model ``Q``. Must use ``logprob=True``.
        test_data: Iterable of test elements over which to accumulate the
            divergence.

    Returns:
        The KL divergence value.

    Raises:
        ValueError: If either model is not configured for log-probabilities.

    Examples:
        >>> from freqprob import MLE, Laplace, kl_divergence
        >>> p_model = MLE({"a": 3, "b": 2, "c": 1}, logprob=True)
        >>> q_model = Laplace({"a": 3, "b": 2, "c": 1}, logprob=True)
        >>> round(kl_divergence(p_model, q_model, ["a", "b", "a", "c"]), 4)
        0.0698
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
    """Compare multiple models using perplexity and cross-entropy.

    Each model is evaluated on the same test data, producing a nested mapping of
    model name to its metric values.

    Args:
        models: Mapping of model names to fitted models. Every model must use
            ``logprob=True``.
        test_data: Iterable of test elements. It is materialized once and reused
            across all models.

    Returns:
        A mapping of each model name to a ``{"perplexity": ..., "cross_entropy":
        ...}`` dictionary.

    Raises:
        ValueError: If any model is not configured for log-probabilities.

    Examples:
        >>> from freqprob import MLE, Laplace, model_comparison
        >>> models = {
        ...     "mle": MLE({"a": 3, "b": 2, "c": 1}, logprob=True),
        ...     "laplace": Laplace({"a": 3, "b": 2, "c": 1}, logprob=True),
        ... }
        >>> results = model_comparison(models, ["a", "b", "a", "c"])
        >>> sorted(results)
        ['laplace', 'mle']
        >>> round(results["mle"]["perplexity"], 4)
        2.913
        >>> round(results["laplace"]["cross_entropy"], 4)
        1.0561
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
