"""Basic probability scoring methods.

This module implements fundamental probability scoring methods including
Uniform, Random, and Maximum Likelihood Estimation (MLE). These methods
serve as building blocks and baselines for more sophisticated smoothing
techniques.
"""

import math
import random
from dataclasses import dataclass

from freqprob.base import (
    MIN_PROBABILITY,
    FrequencyDistribution,
    Probability,
    ScoringMethod,
    ScoringMethodConfig,
)


@dataclass
class RandomConfig(ScoringMethodConfig):
    """Configuration for the Random distribution.

    Attributes:
        unobs_prob: Reserved probability mass for unobserved elements
            (default: ``0.0``).
        logprob: Whether to return log-probabilities (default: ``True``).
        seed: Random seed for reproducible results (default: ``None``).
    """

    seed: int | None = None


class Uniform(ScoringMethod):
    """Uniform probability distribution.

    The simplest baseline: assign equal probability to every observed element,
    ignoring the counts entirely. Useful as a non-informative prior or as a
    reference point when comparing smoothing methods.

    For a vocabulary of size ``V`` with reserved mass ``p0``, each observed
    element gets ``(1 - p0) / V`` and any unobserved element gets ``p0``.

    Args:
        freqdist: Mapping of elements to observed counts. The counts are
            ignored; only the number of distinct elements matters.
        unobs_prob: Reserved probability mass ``p0`` for unobserved elements
            (``0.0 <= p0 <= 1.0``). Defaults to ``0.0``.
        logprob: Return log-probabilities if ``True`` (the default), otherwise
            plain probabilities.

    Examples:
        Counts are ignored, so both observed elements score the same:

        >>> uniform = Uniform({"apple": 10, "banana": 1}, unobs_prob=0.1, logprob=False)
        >>> uniform("apple")
        0.45
        >>> uniform("banana")
        0.45
        >>> uniform("cherry")  # unobserved
        0.1

    Note:
        Because it discards frequency information, ``Uniform`` is meant as a
        baseline. Use `MLE`, `Lidstone`, or `Laplace` when counts should matter.
    """

    __slots__ = ()

    def __init__(
        self,
        freqdist: FrequencyDistribution,
        unobs_prob: Probability = 0.0,
        logprob: bool = True,
    ) -> None:
        """Initialize Uniform distribution."""
        config = ScoringMethodConfig(unobs_prob=unobs_prob, logprob=logprob)
        super().__init__(config)
        self.name = "Uniform"
        self.fit(freqdist)

    def _compute_probabilities(self, freqdist: FrequencyDistribution) -> None:
        """Compute uniform probabilities for all elements.

        Parameters
        ----------
        freqdist : FrequencyDistribution
            Frequency distribution (counts are ignored, only support size used)
        """
        unobs_prob = self.config.unobs_prob or 0.0

        vocab_size = len(freqdist)

        if self.logprob:
            # Avoid domain errors by ensuring unobs_prob >= machine epsilon
            unobs_prob = max(unobs_prob, MIN_PROBABILITY)
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

    Assigns pseudo-random probabilities, useful for testing and as a randomized
    baseline. Each element is given a random count drawn from the observed count
    range, and those counts are then normalized like `MLE`. A fixed ``seed``
    makes the result reproducible.

    Args:
        freqdist: Mapping of elements to observed counts. Only the min/max
            counts are used, to bound the random counts.
        unobs_prob: Reserved probability mass for unobserved elements.
            Defaults to ``0.0``.
        logprob: Return log-probabilities if ``True`` (the default), otherwise
            plain probabilities.
        seed: Seed for the random number generator, for reproducible results.
            Defaults to ``None``.

    Examples:
        The probabilities are random but form a valid distribution, and a fixed
        seed is reproducible:

        >>> freqdist = {"apple": 5, "banana": 2, "cherry": 8}
        >>> a = Random(freqdist, seed=123, logprob=False)
        >>> b = Random(freqdist, seed=123, logprob=False)
        >>> a("apple") == b("apple")
        True
        >>> round(sum(a(e) for e in freqdist), 6)
        1.0

    Note:
        Primarily for testing, debugging, and providing a randomized baseline
        for comparison. It is not a smoothing method in any meaningful sense.
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
            unobs_prob = max(unobs_prob, MIN_PROBABILITY)  # Avoid domain errors
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
    """Maximum Likelihood Estimation.

    Estimates each element's probability as its relative frequency in the
    observed data — the most intuitive baseline. For counts ``c_i`` with total
    ``N``, each observed element gets ``(1 - p0) * c_i / N`` and any unobserved
    element gets the reserved mass ``p0`` (``0.0`` by default, i.e. unseen
    elements score zero).

    Args:
        freqdist: Mapping of elements to observed counts.
        unobs_prob: Reserved probability mass ``p0`` for unobserved elements
            (``0.0 <= p0 <= 1.0``). Defaults to ``0.0``.
        logprob: Return log-probabilities if ``True`` (the default), otherwise
            plain probabilities.

    Examples:
        Probabilities are relative frequencies; unseen elements score zero:

        >>> mle = MLE({"apple": 6, "banana": 3, "cherry": 1}, logprob=False)
        >>> mle("apple")
        0.6
        >>> mle("banana")
        0.3
        >>> mle("unknown")
        0.0

        Reserving mass for unseen elements rescales the observed ones:

        >>> smoothed = MLE({"apple": 6, "banana": 3, "cherry": 1},
        ...                unobs_prob=0.1, logprob=False)
        >>> round(smoothed("apple"), 3)
        0.54
        >>> smoothed("unknown")
        0.1

    Note:
        MLE underlies most smoothing methods but assigns zero probability to
        unseen events, which is problematic for sparse data. Prefer `Laplace`,
        `Lidstone`, or a more advanced method when unseen elements are likely.
    """

    __slots__ = ()

    def __init__(
        self,
        freqdist: FrequencyDistribution,
        unobs_prob: Probability = 0.0,
        logprob: bool = True,
    ) -> None:
        """Initialize MLE distribution."""
        config = ScoringMethodConfig(unobs_prob=unobs_prob, logprob=logprob)
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
            unobs_prob = max(unobs_prob, MIN_PROBABILITY)  # Avoid domain errors
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
