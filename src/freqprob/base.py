"""Base classes for frequency probability scoring methods.

This module provides the abstract base class and common functionality
for all smoothing methods in the freqprob library.
"""

import pickle
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

# Type aliases for clarity
Element = str | int | float | tuple[Any, ...] | frozenset[Any]
Count = int
Probability = float
LogProbability = float
FrequencyDistribution = Mapping[Element, Count]

# Smallest probability used as a floor to avoid log(0) domain errors. Kept as a
# module constant so methods can use it directly rather than reading `_unobs`
# (which holds the *output* value and would be wrong on a second fit()).
MIN_PROBABILITY: Probability = 1e-10

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

    __slots__ = ("_prob", "_total_unseen_mass", "_unobs", "config", "logprob", "name")

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

        self._unobs: Probability | LogProbability = MIN_PROBABILITY  # avoid domain errors
        self._prob: dict[Element, Probability | LogProbability] = {}
        self._total_unseen_mass: float | None = (
            None  # For methods that track total unseen mass (e.g., SGT)
        )
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
        # Reset state so re-fitting an already-fitted scorer behaves like a
        # fresh fit: methods that assign into self._prob incrementally would
        # otherwise leave stale keys, and self._unobs (which holds the previous
        # output) would otherwise pollute epsilon floors on a second fit.
        self._prob = {}
        self._unobs = MIN_PROBABILITY
        self._total_unseen_mass = None
        self._compute_probabilities(freqdist)

        return self

    def score(self, element: Element) -> Probability | LogProbability:
        """Score a single element (scikit-learn-style alias for ``__call__``).

        Parameters
        ----------
        element : Element
            Element to be scored

        Returns:
        -------
        Probability | LogProbability
            The probability (or log-probability) of the element

        Examples:
        --------
        >>> scorer = MLE({'a': 2, 'b': 1}, logprob=False)
        >>> scorer.score('a')
        0.6666666666666666
        """
        return self(element)

    def predict(self, elements: Iterable[Element]) -> list[Probability | LogProbability]:
        """Score many elements at once (scikit-learn-style batch alias).

        Parameters
        ----------
        elements : Iterable[Element]
            Elements to be scored

        Returns:
        -------
        list[Probability | LogProbability]
            One probability (or log-probability) per input element, in order

        Examples:
        --------
        >>> scorer = MLE({'a': 2, 'b': 1}, logprob=False)
        >>> scorer.predict(['a', 'b', 'c'])
        [0.6666666666666666, 0.3333333333333333, 0.0]
        """
        return [self(element) for element in elements]

    def __getstate__(self) -> dict[str, Any]:
        """Collect all ``__slots__`` values across the class hierarchy for pickling."""
        state: dict[str, Any] = {}
        for klass in type(self).__mro__:
            for slot in getattr(klass, "__slots__", ()):
                if hasattr(self, slot):
                    state[slot] = getattr(self, slot)
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore ``__slots__`` values from a pickled state."""
        for slot, value in state.items():
            setattr(self, slot, value)

    def save(self, path: str | Path) -> None:
        """Serialize this fitted scorer to a file.

        The scorer can be restored later with :meth:`load`, avoiding the cost of
        re-fitting. Serialization uses :mod:`pickle`.

        Parameters
        ----------
        path : str | Path
            Destination file path

        Examples:
        --------
        >>> scorer = MLE({'a': 2, 'b': 1}, logprob=False)
        >>> scorer.save('model.pkl')                 # doctest: +SKIP
        >>> restored = MLE.load('model.pkl')         # doctest: +SKIP
        >>> restored('a') == scorer('a')             # doctest: +SKIP
        True
        """
        with Path(path).open("wb") as fh:
            pickle.dump(self, fh, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls: type[T], path: str | Path) -> T:  # noqa: PYI019  (Self needs typing_extensions on 3.10)
        """Load a scorer previously written with :meth:`save`.

        Parameters
        ----------
        path : str | Path
            Path to a file created by :meth:`save`

        Returns:
        -------
        ScoringMethod
            The restored scorer

        Raises:
        ------
        TypeError
            If the file does not contain a compatible scorer

        Warning:
        -------
        Only load files you trust: unpickling executes arbitrary code from the
        file, so never load a scorer from an untrusted source.
        """
        with Path(path).open("rb") as fh:
            obj = pickle.load(fh)
        if not isinstance(obj, cls):
            raise TypeError(f"File does not contain a {cls.__name__}: got {type(obj).__name__}")
        return obj
