"""Base classes for frequency probability scoring methods.

This module provides the abstract base class and common functionality
for all smoothing methods in the freqprob library.
"""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, TypeVar

# Type aliases for clarity
Element = str | int | float | tuple[Any, ...] | frozenset[Any]
Count = int
Probability = float
LogProbability = float
FrequencyDistribution = Mapping[Element, Count]

# Generic type variable for method chaining
T = TypeVar("T", bound="ScoringMethod")


@dataclass
class ScoringMethodConfig:
    """Configuration for scoring methods.

    This dataclass encapsulates all configuration parameters that can be
    used across different scoring methods, providing type safety and validation.

    Attributes:
    ----------
    unobs_prob : Probability | None
        Reserved probability mass for unobserved elements (0.0 ≤ p ≤ 1.0)
    gamma : float | None
        Smoothing parameter for additive methods (gamma >= 0)
    bins : int | None
        Total number of possible bins/elements (B ≥ 1)
    logprob : bool
        Whether to return log-probabilities instead of probabilities

    Examples:
    --------
    >>> config = ScoringMethodConfig(unobs_prob=0.1, logprob=True)
    >>> config.unobs_prob
    0.1

    >>> config = ScoringMethodConfig(gamma=1.5, bins=1000)
    >>> config.gamma
    1.5

    Raises:
    ------
    ValueError
        If any parameter is outside its valid range
    """

    unobs_prob: Probability | None = None

    gamma: float | None = None
    bins: int | None = None
    logprob: bool = True

    def __post_init__(self) -> None:
        """Validate configuration parameters after initialization.

        Raises:
        ------
        ValueError
            If unobs_prob is not in [0.0, 1.0], gamma is negative,
            or bins is not positive
        """
        if self.unobs_prob is not None and not 0.0 <= self.unobs_prob <= 1.0:
            raise ValueError("The reserved mass probability must be between 0.0 and 1.0")

        if self.gamma is not None and self.gamma < 0:
            raise ValueError("Gamma must be a non-negative real number.")

        if self.bins is not None and self.bins < 1:
            raise ValueError("Number of bins must be a positive integer.")


class ScoringMethod(ABC):
    """Abstract base class for frequency-based probability smoothing methods.

    This class provides a unified interface for all probability estimation
    methods, supporting both regular probabilities and log-probabilities.

    The general workflow is:
    1. Initialize with a configuration
    2. Fit to a frequency distribution
    3. Score individual elements

    Mathematical Foundation
    -----------------------
    Given a frequency distribution D = {(w₁, c₁), (w₂, c₂), ..., (wₙ, cₙ)}
    where wᵢ are elements and cᵢ are their counts, smoothing methods estimate:

    P(w) = probability of element w

    For unobserved elements (w ∉ D), methods reserve probability mass
    to avoid zero probabilities.

    Attributes:
    ----------
    config : ScoringMethodConfig
        Configuration parameters for the method
    name : str | None
        Human-readable name of the method
    logprob : bool | None
        Whether this instance returns log-probabilities

    Examples:
    --------
    >>> from freqprob import MLE
    >>> freqdist = {'apple': 3, 'banana': 2, 'cherry': 1}
    >>> scorer = MLE(freqdist, logprob=False)
    >>> scorer('apple')  # Most frequent item
    0.5
    >>> scorer('unknown')  # Unobserved item
    0.0
    """

    __slots__ = ("_prob", "_unobs", "config", "logprob", "name")

    def __init__(self, config: ScoringMethodConfig) -> None:
        """Initialize the scoring method.

        Parameters
        ----------
        config : ScoringMethodConfig
            Configuration object containing method parameters

        Note:
        ----
        This constructor should typically be called by subclass constructors,
        not directly by users.
        """
        self.config: ScoringMethodConfig = config

        self._unobs: Probability | LogProbability = 1e-10  # Default value to avoid domain errors
        self._prob: dict[Element, Probability | LogProbability] = {}
        self.logprob: bool | None = config.logprob
        self.name: str | None = None

    def __call__(self, element: Element) -> Probability | LogProbability:
        """Score a single element.

        Parameters
        ----------
        element : Element
            Element to be scored

        Returns:
        -------
        Probability | LogProbability
            The probability (if logprob=False) or log-probability (if logprob=True)
            of the element. Returns probability for unobserved elements based
            on the method's smoothing strategy.

        Examples:
        --------
        >>> scorer = MLE({'a': 2, 'b': 1}, logprob=False)
        >>> scorer('a')
        0.6666666666666666
        >>> scorer('c')  # unobserved
        0.0
        """
        return self._prob.get(element, self._unobs)

    def __str__(self) -> str:
        """Return a string representation of the smoothing method.

        Returns:
        -------
        str
            Human-readable description of the method

        Raises:
        ------
        ValueError
            If the method has not been properly initialized

        Examples:
        --------
        >>> str(MLE({'a': 1}, logprob=True))
        'MLE log-scorer, 1 elements.'
        """
        if self.name is None:
            raise ValueError("The smoothing method has not been (properly) initialized.")

        buffer = []
        if self.logprob:
            buffer.append(f"{self.name} log-scorer")
        else:
            buffer.append(f"{self.name} scorer")

        buffer.append(f"{len(self._prob)} elements.")

        return ", ".join(buffer)

    @abstractmethod
    def _compute_probabilities(self, freqdist: FrequencyDistribution) -> None:
        """Compute probabilities for the given frequency distribution.

        This method must be implemented by subclasses to compute the
        actual probability values according to their specific smoothing strategy.

        Parameters
        ----------
        freqdist : FrequencyDistribution
            Frequency distribution mapping elements to their observed counts

        Note:
        ----
        Implementations should populate self._prob and self._unobs.
        """

    def fit(self, freqdist: FrequencyDistribution) -> "ScoringMethod":
        """Fit the scoring method to a frequency distribution.

        This method trains the scorer on the provided frequency data,
        computing probability estimates for all observed elements.

        Parameters
        ----------
        freqdist : FrequencyDistribution
            Frequency distribution mapping elements to their observed counts

        Returns:
        -------
        ScoringMethod
            Self, for method chaining

        Examples:
        --------
        >>> scorer = MLE({}).fit({'a': 2, 'b': 1})
        >>> scorer('a')
        0.6666666666666666
        """
        self._compute_probabilities(freqdist)

        return self
