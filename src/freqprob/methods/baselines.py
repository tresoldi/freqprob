"""Basic probability scoring methods.

This module implements fundamental probability scoring methods including
Uniform, Random, and Maximum Likelihood Estimation (MLE). These methods
serve as building blocks and baselines for more sophisticated smoothing
techniques.
"""

import random
from dataclasses import dataclass

from freqprob.base import (
    Estimates,
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

    def _estimate(self, freqdist: FrequencyDistribution) -> Estimates:
        """Spread the available mass evenly over the observed elements.

        Args:
            freqdist: Frequency distribution; counts are ignored, only the
                support size is used.

        Returns:
            The uniform probability for each element, and the reserved mass.
        """
        unobs_prob = self.config.unobs_prob or 0.0
        uniform_prob = (1.0 - unobs_prob) / len(freqdist)

        return dict.fromkeys(freqdist, uniform_prob), unobs_prob


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

    def _estimate(self, freqdist: FrequencyDistribution) -> Estimates:
        """Draw a random count per element, then normalize.

        Args:
            freqdist: Frequency distribution used to determine the count range.

        Returns:
            The randomized probability of each element, and the reserved mass.
        """
        unobs_prob = self.config.unobs_prob or 0.0

        if not freqdist:
            return {}, unobs_prob

        min_count, max_count = min(freqdist.values()), max(freqdist.values())
        rng = random.Random(self.config.seed)  # type: ignore[attr-defined]

        # Ensure minimum count is at least 1 to avoid zero probabilities
        min_count = max(min_count, 1)
        random_counts = {elem: rng.randint(min_count, max_count) for elem in freqdist}
        total_random_counts = sum(random_counts.values())

        available_mass = 1.0 - unobs_prob
        table = {
            elem: (count / total_random_counts) * available_mass
            for elem, count in random_counts.items()
        }

        return table, unobs_prob


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

    def _estimate(self, freqdist: FrequencyDistribution) -> Estimates:
        """Divide each count by the total, scaled by the available mass.

        Args:
            freqdist: Frequency distribution with element counts.

        Returns:
            The relative frequency of each element, and the reserved mass.
        """
        unobs_prob = self.config.unobs_prob or 0.0
        total_count = sum(freqdist.values())

        if total_count == 0:
            # Handle empty distribution edge case: there is no mass anywhere.
            return {}, 0.0

        available_mass = 1.0 - unobs_prob
        table = {elem: (count / total_count) * available_mass for elem, count in freqdist.items()}

        return table, unobs_prob
