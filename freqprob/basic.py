"""Basic probability scoring methods.

This module implements fundamental probability scoring methods including
Uniform, Random, and Maximum Likelihood Estimation (MLE). These methods
serve as building blocks and baselines for more sophisticated smoothing
techniques.
"""

import math
import random
from dataclasses import dataclass

from .base import FrequencyDistribution, Probability, ScoringMethod, ScoringMethodConfig


@dataclass
class UniformConfig(ScoringMethodConfig):
    """Configuration for Uniform distribution.

    Attributes:
    ----------
    unobs_prob : Probability
        Reserved probability mass for unobserved elements (default: 0.0)
    logprob : bool
        Whether to return log-probabilities (default: True)
    """

    unobs_prob: Probability = 0.0

    logprob: bool = True


@dataclass
class RandomConfig(ScoringMethodConfig):
    """Configuration for Random distribution.

    Attributes:
    ----------
    unobs_prob : Probability
        Reserved probability mass for unobserved elements (default: 0.0)
    logprob : bool
        Whether to return log-probabilities (default: True)
    seed : Optional[int]
        Random seed for reproducible results (default: None)
    """

    unobs_prob: Probability = 0.0

    logprob: bool = True
    seed: int | None = None


@dataclass
class MLEConfig(ScoringMethodConfig):
    """Configuration for Maximum Likelihood Estimation.

    Attributes:
    ----------
    unobs_prob : Probability
        Reserved probability mass for unobserved elements (default: 0.0)
    logprob : bool
        Whether to return log-probabilities (default: True)
    """

    unobs_prob: Probability = 0.0

    logprob: bool = True


class Uniform(ScoringMethod):
    """Uniform probability distribution.

    The simplest smoothing method that assigns equal probability to all
    observed elements, ignoring their frequency counts. This serves as
    a baseline method and can be useful when no frequency information
    should influence the probability estimates.

    Mathematical Formulation
    ------------------------
    For a vocabulary of size |V| with reserved mass p₀:

    P(wᵢ) = (1 - p₀) / |V|  for observed words wᵢ ∈ V
    P(w)  = p₀              for unobserved words w ∉ V

    Parameters
    ----------
    freqdist : FrequencyDistribution
        Frequency distribution mapping elements to their observed counts.
        Note: counts are ignored, only vocabulary size matters.
    unobs_prob : Probability, default=0.0
        Reserved probability mass for unobserved elements (0.0 ≤ p₀ ≤ 1.0)
    logprob : bool, default=True
        Whether to return log-probabilities or probabilities

    Examples:
    --------
    >>> freqdist = {'apple': 10, 'banana': 1}  # counts ignored
    >>> uniform = Uniform(freqdist, unobs_prob=0.1, logprob=False)
    >>> uniform('apple')   # Same as banana despite higher count
    0.45
    >>> uniform('banana')  # Same probability
    0.45
    >>> uniform('cherry')  # Unobserved element
    0.1

    With log-probabilities:
    >>> uniform_log = Uniform(freqdist, logprob=True)
    >>> uniform_log('apple')
    -0.6931471805599453
    >>> uniform_log('banana')
    -0.6931471805599453

    Notes:
    -----
    This method completely ignores frequency information, making it
    useful as a non-informative prior or baseline. For methods that
    utilize frequency counts, see MLE, Lidstone, or other smoothing methods.
    """

    __slots__ = ()

    def __init__(
        self, freqdist: FrequencyDistribution, unobs_prob: Probability = 0.0, logprob: bool = True
    ) -> None:
        """Initialize Uniform distribution."""
        config = UniformConfig(unobs_prob=unobs_prob, logprob=logprob)
        super().__init__(config)
        self.name = "Uniform"
        self.fit(freqdist)

    def _compute_probabilities(self, freqdist: FrequencyDistribution) -> None:
        """Compute uniform probabilities for all elements.

        Parameters
        ----------
        freqdist : FrequencyDistribution
            Frequency distribution (counts are ignored, only vocabulary size used)
        """
        unobs_prob = self.config.unobs_prob or 0.0

        vocab_size = len(freqdist)

        if self.logprob:
            # Avoid domain errors by ensuring unobs_prob >= machine epsilon
            unobs_prob = max(unobs_prob, self._unobs)
            uniform_prob = (1.0 - unobs_prob) / vocab_size
            log_uniform_prob = math.log(uniform_prob)
            self._prob = dict.fromkeys(freqdist, log_uniform_prob)
            self._unobs = math.log(unobs_prob)
        else:
            uniform_prob = (1.0 - unobs_prob) / vocab_size
            self._prob = dict.fromkeys(freqdist, uniform_prob)
            self._unobs = unobs_prob


class Random(ScoringMethod):
    """Random probability distribution.

    Assigns random probabilities to elements, useful for testing and as a
    baseline that provides non-deterministic probability estimates. The random
    probabilities are generated by creating random "counts" within the range
    of observed counts and then applying MLE normalization.

    Mathematical Formulation
    ------------------------
    1. Generate random counts: c'ᵢ ~ Uniform(min(cᵢ), max(cᵢ))
    2. Normalize: P(wᵢ) = (1 - p₀) * c'ᵢ / Σⱼc'ⱼ
    3. Unobserved: P(w) = p₀ for w ∉ V

    Parameters
    ----------
    freqdist : FrequencyDistribution
        Frequency distribution mapping elements to observed counts
    unobs_prob : Probability, default=0.0
        Reserved probability mass for unobserved elements
    logprob : bool, default=True
        Whether to return log-probabilities or probabilities
    seed : Optional[int], default=None
        Random seed for reproducible results

    Examples:
    --------
    >>> freqdist = {'apple': 5, 'banana': 2, 'cherry': 8}
    >>> random_dist = Random(freqdist, seed=42, logprob=False)
    >>> random_dist('apple')  # Random probability
    0.3157894736842105
    >>> random_dist('banana')
    0.21052631578947367

    Reproducible results with seed:
    >>> random1 = Random(freqdist, seed=123, logprob=False)
    >>> random2 = Random(freqdist, seed=123, logprob=False)
    >>> random1('apple') == random2('apple')
    True

    Notes:
    -----
    This method is primarily useful for:
    - Testing and debugging other components
    - Providing a randomized baseline for comparison
    - Simulating noisy probability estimates

    The random counts are constrained to the range [min_count, max_count]
    from the original distribution to maintain some relationship to the data.
    """

    __slots__ = ()

    def __init__(
        self,
        freqdist: FrequencyDistribution,
        unobs_prob: Probability = 0.0,
        logprob: bool = True,
        seed: int | None = None,
    ) -> None:
        """Initialize Random distribution."""
        config = RandomConfig(unobs_prob=unobs_prob, logprob=logprob, seed=seed)
        super().__init__(config)
        self.name = "Random"
        self.fit(freqdist)

    def _compute_probabilities(self, freqdist: FrequencyDistribution) -> None:
        """Compute random probabilities based on randomized counts.

        Parameters
        ----------
        freqdist : FrequencyDistribution
            Original frequency distribution used to determine count range
        """
        unobs_prob = self.config.unobs_prob or 0.0

        # Generate random counts within the observed range
        if not freqdist:
            return  # Handle empty distribution

        min_count, max_count = min(freqdist.values()), max(freqdist.values())
        random.seed(self.config.seed)  # type: ignore

        # Ensure minimum count is at least 1 to avoid zero probabilities
        min_count = max(min_count, 1)
        random_counts = {elem: random.randint(min_count, max_count) for elem in freqdist}
        total_random_counts = sum(random_counts.values())

        if self.logprob:
            unobs_prob = max(unobs_prob, self._unobs)  # Avoid domain errors
            available_mass = 1.0 - unobs_prob
            self._prob = {
                elem: math.log((count / total_random_counts) * available_mass)
                for elem, count in random_counts.items()
            }
            self._unobs = math.log(unobs_prob)
        else:
            available_mass = 1.0 - unobs_prob
            self._prob = {
                elem: (count / total_random_counts) * available_mass
                for elem, count in random_counts.items()
            }
            self._unobs = unobs_prob


class MLE(ScoringMethod):
    """Maximum Likelihood Estimation probability distribution.

    The most intuitive probability estimation method that directly uses
    observed frequencies as probability estimates. This is the natural
    baseline that estimates probability as the relative frequency of
    each element in the observed data.

    Mathematical Formulation
    ------------------------
    For elements with counts cᵢ and total count N = Σⱼcⱼ:

    P(wᵢ) = (1 - p₀) * cᵢ / N  for observed elements wᵢ ∈ V
    P(w)  = p₀                 for unobserved elements w ∉ V

    This is the maximum likelihood estimate under a multinomial model.

    Parameters
    ----------
    freqdist : FrequencyDistribution
        Frequency distribution mapping elements to their observed counts
    unobs_prob : Probability, default=0.0
        Reserved probability mass for unobserved elements (0.0 ≤ p₀ ≤ 1.0)
    logprob : bool, default=True
        Whether to return log-probabilities or probabilities

    Examples:
    --------
    Basic MLE without unobserved mass:
    >>> freqdist = {'apple': 6, 'banana': 3, 'cherry': 1}
    >>> mle = MLE(freqdist, unobs_prob=0.0, logprob=False)
    >>> mle('apple')      # Most frequent
    0.6
    >>> mle('banana')     # Medium frequency
    0.3
    >>> mle('cherry')     # Least frequent
    0.1
    >>> mle('unknown')    # Unobserved
    0.0

    With reserved mass for unobserved elements:
    >>> mle_smooth = MLE(freqdist, unobs_prob=0.1, logprob=False)
    >>> mle_smooth('apple')    # Scaled down by (1-0.1)
    0.54
    >>> mle_smooth('unknown')  # Gets reserved mass
    0.1

    Log-probability mode:
    >>> mle_log = MLE(freqdist, logprob=True)
    >>> mle_log('apple')
    -0.5108256237659907
    >>> mle_log('banana')
    -1.2039728043259361

    Properties
    ----------
    - Intuitive and widely understood
    - Optimal for large datasets when true distribution matches observed
    - Assigns zero probability to unobserved events (unless unobs_prob > 0)
    - Prone to overfitting on small datasets
    - Forms the basis for many smoothing methods

    Notes:
    -----
    MLE is the foundation for most other smoothing methods. While simple,
    it can be problematic for sparse data due to zero probabilities for
    unobserved events. Consider Lidstone/Laplace smoothing for small datasets.
    """

    __slots__ = ()

    def __init__(
        self, freqdist: FrequencyDistribution, unobs_prob: Probability = 0.0, logprob: bool = True
    ) -> None:
        """Initialize MLE distribution."""
        config = MLEConfig(unobs_prob=unobs_prob, logprob=logprob)
        super().__init__(config)
        self.name = "MLE"
        self.fit(freqdist)

    def _compute_probabilities(self, freqdist: FrequencyDistribution) -> None:
        """Compute Maximum Likelihood probability estimates.

        Parameters
        ----------
        freqdist : FrequencyDistribution
            Frequency distribution with element counts
        """
        unobs_prob = self.config.unobs_prob or 0.0

        # Calculate total count for normalization
        total_count = sum(freqdist.values())

        if total_count == 0:
            # Handle empty distribution edge case
            return

        available_mass = 1.0 - unobs_prob

        if self.logprob:
            unobs_prob = max(unobs_prob, self._unobs)  # Avoid domain errors
            self._prob = {
                elem: math.log((count / total_count) * available_mass)
                for elem, count in freqdist.items()
            }
            self._unobs = math.log(unobs_prob)
        else:
            self._prob = {
                elem: (count / total_count) * available_mass for elem, count in freqdist.items()
            }
            self._unobs = unobs_prob
