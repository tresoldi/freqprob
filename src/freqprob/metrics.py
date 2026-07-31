"""Model-evaluation metrics: perplexity, cross-entropy, KL divergence, entropy.

Alongside the model-scoring metrics, this module provides sample-based estimators
that work directly on a frequency distribution: :func:`sample_coverage` (the
Good-Turing estimate of observed mass) and the bias-corrected entropy estimators
:func:`chao_shen_entropy` and :func:`nsb_entropy`, which correct the downward bias
of the naive plug-in entropy when many types are rare or unobserved.
"""

import math
from collections.abc import Iterable

import numpy as np
from scipy.special import gammaln, polygamma, psi

from .base import Element, FrequencyDistribution, ScoringMethod


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


def sample_coverage(freqdist: FrequencyDistribution) -> float:
    """Estimate the proportion of total probability mass that was observed.

    Uses the Good-Turing estimator of the *missing mass*: the share of
    probability belonging to unseen types is approximately ``N1 / N``, where
    ``N1`` is the number of types seen exactly once (singletons) and ``N`` the
    total number of observations. Coverage is the complement, ``1 - N1 / N``.

    Args:
        freqdist: Mapping of elements to observed counts.

    Returns:
        The estimated observed mass, in ``[0, 1]``. Returns ``0.0`` for an empty
        distribution.

    Examples:
        >>> from freqprob import sample_coverage
        >>> sample_coverage({"a": 10, "b": 5, "c": 3})  # no singletons
        1.0
        >>> round(sample_coverage({"a": 8, "b": 1, "c": 1}), 3)  # two singletons of 10
        0.8
    """
    total = sum(freqdist.values())
    if total == 0:
        return 0.0
    n1 = sum(1 for count in freqdist.values() if count == 1)
    return 1.0 - n1 / total


def chao_shen_entropy(freqdist: FrequencyDistribution) -> float:
    """Estimate Shannon entropy with the Chao-Shen bias correction.

    The naive plug-in entropy underestimates the true entropy when the sample
    misses probability mass. The Chao-Shen estimator combines Good-Turing
    coverage adjustment with a Horvitz-Thompson correction for unseen types,
    which reduces this bias:

    ``H = - sum_i (C p_i) log(C p_i) / (1 - (1 - C p_i) ** N)``

    where ``p_i`` is the relative frequency of type ``i``, ``C`` the sample
    coverage (see :func:`sample_coverage`), and ``N`` the total count.

    Args:
        freqdist: Mapping of elements to observed counts.

    Returns:
        The estimated entropy in nats (natural log). Returns ``0.0`` for an
        empty distribution or when coverage is zero (all singletons).

    Examples:
        >>> from freqprob import chao_shen_entropy
        >>> import math
        >>> # Approaches the true entropy of a uniform distribution when well sampled.
        >>> round(chao_shen_entropy({"a": 1000, "b": 1000, "c": 1000}), 3)
        1.099
        >>> round(math.log(3), 3)
        1.099
    """
    total = sum(freqdist.values())
    if total == 0:
        return 0.0
    n1 = sum(1 for count in freqdist.values() if count == 1)
    coverage = 1.0 - n1 / total
    if coverage <= 0.0:
        return 0.0

    entropy = 0.0
    for count in freqdist.values():
        if count <= 0:
            continue
        p = coverage * (count / total)
        if p <= 0.0:
            continue
        inclusion = 1.0 - (1.0 - p) ** total
        if inclusion <= 0.0:
            continue
        entropy -= p * math.log(p) / inclusion
    return entropy


def nsb_entropy(freqdist: FrequencyDistribution, bins: int | None = None) -> float:
    """Estimate Shannon entropy with the Nemenman-Shafee-Bialek estimator.

    The NSB estimator is a Bayesian entropy estimator that averages the posterior
    mean entropy of a symmetric Dirichlet model over its concentration parameter,
    using a prior chosen so the *a priori* entropy is (nearly) uniform. It is
    well suited to the undersampled regime, where the number of possible types is
    comparable to or larger than the number of observations.

    Args:
        freqdist: Mapping of elements to observed counts.
        bins: The size of the assumed alphabet (number of possible types),
            reserving entropy for types that could occur but were not seen. If
            ``None`` (the default), the number of observed types is used, so no
            unseen types are assumed. Must be at least the number of observed
            types.

    Returns:
        The estimated entropy in nats (natural log), in ``[0, log(bins)]``.
        Returns ``0.0`` for an empty distribution or a single possible type.

    Raises:
        ValueError: If ``bins`` is smaller than the number of observed types.

    Examples:
        >>> from freqprob import nsb_entropy
        >>> import math
        >>> # Well-sampled uniform distribution approaches log(K).
        >>> round(nsb_entropy({"a": 500, "b": 500, "c": 500, "d": 500}), 2)
        1.39
        >>> round(math.log(4), 2)
        1.39
    """
    counts = [int(count) for count in freqdist.values() if count > 0]
    n_observed_types = len(counts)

    if bins is None:
        bins = n_observed_types
    if bins < n_observed_types:
        raise ValueError("bins must be at least the number of observed types")

    total = sum(counts)
    if total == 0 or bins <= 1:
        return 0.0

    k = float(bins)
    n = float(total)
    observed = np.asarray(counts, dtype=float)

    # Integrate over the per-bin Dirichlet concentration parameter ``a`` on a
    # geometric grid, weighting each value by the evidence and the prior that
    # makes the a-priori mean entropy uniform.
    a = np.geomspace(1e-8, 1e8, 6000)
    ka = k * a

    # Log evidence (unnormalized): unobserved bins contribute nothing in log.
    log_evidence = gammaln(ka) - gammaln(n + ka)
    for count in observed:
        log_evidence = log_evidence + gammaln(count + a) - gammaln(a)

    # Prior Jacobian d(xi)/da, where xi(a) = psi(k a + 1) - psi(a + 1) is the
    # a-priori mean entropy; a flat prior on xi makes this the density in ``a``.
    jacobian = k * polygamma(1, ka + 1.0) - polygamma(1, a + 1.0)
    jacobian = np.clip(jacobian, 0.0, None)

    # Posterior mean entropy given ``a`` (Wolpert & Wolf, 1995).
    denom = n + ka
    observed_term = np.zeros_like(a)
    for count in observed:
        observed_term = observed_term + (count + a) * psi(count + a + 1.0)
    unobserved_term = (k - n_observed_types) * a * psi(a + 1.0)
    mean_entropy = psi(denom + 1.0) - (observed_term + unobserved_term) / denom

    # Combine in log space; weight = evidence * jacobian * grid spacing.
    spacing = np.gradient(a)
    with np.errstate(divide="ignore"):
        log_weight = log_evidence + np.log(np.where(jacobian > 0, jacobian, 1.0)) + np.log(spacing)
    log_weight = np.where(jacobian > 0, log_weight, -np.inf)

    finite = np.isfinite(log_weight)
    if not finite.any():
        return 0.0
    log_weight = log_weight - np.max(log_weight[finite])
    weight = np.where(finite, np.exp(log_weight), 0.0)

    normalizer = float(np.sum(weight))
    if normalizer <= 0.0:
        return 0.0
    return float(np.sum(weight * mean_entropy) / normalizer)
