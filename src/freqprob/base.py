"""Base classes for frequency probability scoring methods.

This module provides the abstract base class and common functionality
for all smoothing methods in the freqprob library.
"""

import math
import pickle
from abc import ABC, abstractmethod
from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

# Type aliases for clarity
Element = str | int | float | tuple[Any, ...] | frozenset[Any]
Count = int
Probability = float
LogProbability = float
FrequencyDistribution = Mapping[Element, Count]

# What a smoothing method returns from `_estimate`: the probability of each
# observed element, and the probability reserved for everything unobserved.
# Always in plain probability space -- `ScoringMethod.fit` handles the floor and
# the conversion to log-probabilities.
Estimates = tuple[Mapping[Element, Probability], Probability]

# Smallest probability used as a floor to avoid log(0) domain errors. Kept as a
# module constant so methods can use it directly rather than reading `_unobs`
# (which holds the *output* value and would be wrong on a second fit()).
MIN_PROBABILITY: Probability = 1e-10

# Generic type variable for method chaining
T = TypeVar("T", bound="ScoringMethod")


@dataclass
class ScoringMethodConfig:
    """Configuration for scoring methods.

    This dataclass encapsulates the configuration parameters shared across
    scoring methods, providing type safety and validation of their ranges.

    Attributes:
        unobs_prob: Reserved probability mass for unobserved elements
            (``0.0 <= p <= 1.0``).
        gamma: Smoothing parameter for additive methods (``gamma >= 0``).
        bins: Total number of possible bins/elements (``bins >= 1``).
        logprob: Whether to return log-probabilities instead of probabilities.

    Raises:
        ValueError: If any parameter is outside its valid range.

    Examples:
        >>> config = ScoringMethodConfig(unobs_prob=0.1, logprob=True)
        >>> config.unobs_prob
        0.1
        >>> config = ScoringMethodConfig(gamma=1.5, bins=1000)
        >>> config.gamma
        1.5
    """

    unobs_prob: Probability | None = None

    gamma: float | None = None
    bins: int | None = None
    logprob: bool = True

    def __post_init__(self) -> None:
        """Validate configuration parameters after initialization.

        Raises:
            ValueError: If ``unobs_prob`` is not in ``[0.0, 1.0]``, ``gamma`` is
                negative, or ``bins`` is not positive.
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

    The general workflow is to initialize a method with a configuration, fit it
    to a frequency distribution, and then score individual elements. Given a
    frequency distribution mapping elements to their observed counts, each method
    estimates a probability ``P(w)`` for every element ``w``. For unobserved
    elements, methods reserve probability mass to avoid returning zero.

    Subclasses must implement :meth:`_compute_probabilities`, which populates the
    internal probability table and the reserved mass for unobserved elements.

    Attributes:
        config: Configuration parameters for the method.
        name: Human-readable name of the method, or ``None`` before it is set.
        logprob: Whether this instance returns log-probabilities.

    Examples:
        >>> from freqprob import MLE
        >>> freqdist = {"apple": 3, "banana": 2, "cherry": 1}
        >>> scorer = MLE(freqdist, logprob=False)
        >>> scorer("apple")  # most frequent item
        0.5
        >>> scorer("unknown")  # unobserved item
        0.0
    """

    __slots__ = (
        "_prob",
        "_revision",
        "_total_unseen_mass",
        "_unobs",
        "config",
        "logprob",
        "name",
    )

    def __init__(self, config: ScoringMethodConfig) -> None:
        """Initialize the scoring method.

        Args:
            config: Configuration object containing the method parameters.

        Note:
            This constructor should typically be called by subclass
            constructors, not directly by users.
        """
        self.config: ScoringMethodConfig = config

        self._unobs: Probability | LogProbability = MIN_PROBABILITY  # avoid domain errors
        self._prob: dict[Element, Probability | LogProbability] = {}
        self._total_unseen_mass: float | None = (
            None  # For methods that track total unseen mass (e.g., SGT)
        )
        self._revision: int = 0  # bumped whenever the estimates change
        self.logprob: bool | None = config.logprob
        self.name: str | None = None

    def __call__(self, element: Element) -> Probability | LogProbability:
        """Score a single element.

        Args:
            element: The element to be scored.

        Returns:
            The probability (when ``logprob=False``) or log-probability (when
            ``logprob=True``) of the element. Unobserved elements are scored
            using the method's reserved probability mass.

        Examples:
            >>> from freqprob import MLE
            >>> scorer = MLE({"a": 2, "b": 1}, logprob=False)
            >>> scorer("a")
            0.6666666666666666
            >>> scorer("c")  # unobserved
            0.0
        """
        return self._prob.get(element, self._unobs)

    def support(self) -> Collection[Element]:
        """Return the elements this scorer holds an explicit estimate for.

        Every other element is scored with :meth:`unobserved_score`. Together
        with :attr:`revision`, this is what lets a caller build its own index
        over the scorer without reaching into its internals.

        Returns:
            The elements with an explicit estimate. The result is a read-only
            view: it must not be mutated, and it may become stale once
            :attr:`revision` changes.

        Examples:
            >>> from freqprob import MLE
            >>> scorer = MLE({"a": 2, "b": 1}, logprob=False)
            >>> sorted(scorer.support())
            ['a', 'b']
        """
        return self._prob.keys()

    def unobserved_score(self) -> Probability | LogProbability:
        """Return the score given to every element outside :meth:`support`.

        Returns:
            The reserved probability (or log-probability) for unobserved
            elements.

        Examples:
            >>> from freqprob import MLE
            >>> scorer = MLE({"a": 2, "b": 1}, logprob=False)
            >>> scorer.unobserved_score()
            0.0
        """
        return self._unobs

    @property
    def revision(self) -> int:
        """Count of how many times this scorer's estimates have changed.

        Starts at ``0`` and increases on every :meth:`fit` and on every
        streaming update. A caller that caches anything derived from
        :meth:`support`, :meth:`unobserved_score`, or the scores themselves
        should record this value alongside the cache and rebuild when it
        changes.

        Examples:
            >>> from freqprob import MLE
            >>> scorer = MLE({"a": 2, "b": 1}, logprob=False)
            >>> before = scorer.revision
            >>> _ = scorer.fit({"a": 1})
            >>> scorer.revision > before
            True
        """
        return self._revision

    def _mark_changed(self) -> None:
        """Record that the estimates changed, invalidating callers' caches."""
        self._revision += 1

    def _score_of(self, prob: Probability) -> Probability | LogProbability:
        """Convert a plain probability into this scorer's output space.

        This is where the log/plain duality lives. Estimators compute in plain
        probability space and hand the result here.

        A log-scorer substitutes :data:`MIN_PROBABILITY` for a probability of
        zero, which keeps :func:`math.log` in its domain without disturbing
        small-but-genuine estimates. A plain scorer returns the probability
        untouched, so it can still report an exact ``0.0`` for an element it
        gives no mass to.

        Args:
            prob: A probability in ``[0.0, 1.0]``.

        Returns:
            The log of the probability for a log-scorer, otherwise the
            probability unchanged.
        """
        if self.logprob:
            return math.log(prob) if prob > 0.0 else math.log(MIN_PROBABILITY)
        return prob

    def __str__(self) -> str:
        """Return a string representation of the smoothing method.

        Returns:
            A human-readable description of the method, including its name,
            whether it is a log-scorer, and the number of observed elements.

        Raises:
            ValueError: If the method has not been properly initialized.

        Examples:
            >>> from freqprob import MLE
            >>> str(MLE({"a": 1}, logprob=True))
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
    def _estimate(self, freqdist: FrequencyDistribution) -> "Estimates":
        """Estimate probabilities for the given frequency distribution.

        This is the one method a smoothing method must implement, and it should
        contain nothing but that method's mathematics. Work in **plain
        probability space** and return the result; :meth:`fit` applies the
        :data:`MIN_PROBABILITY` floor, converts to log-probabilities when the
        scorer is a log-scorer, and installs the result. An implementation
        therefore never writes ``self._prob``, never branches on
        ``self.logprob``, and never calls :func:`math.log`.

        Args:
            freqdist: Frequency distribution mapping elements to their observed
                counts.

        Returns:
            The probability of each observed element, and the probability
            reserved for everything else.
        """

    def fit(self, freqdist: FrequencyDistribution) -> "ScoringMethod":
        """Fit the scoring method to a frequency distribution.

        This method trains the scorer on the provided frequency data, computing
        probability estimates for all observed elements. It resets any prior
        state so that re-fitting behaves like a fresh fit.

        Args:
            freqdist: Frequency distribution mapping elements to their observed
                counts.

        Returns:
            This scorer, to allow method chaining.

        Examples:
            >>> from freqprob import MLE
            >>> scorer = MLE({}, logprob=False).fit({"a": 2, "b": 1})
            >>> scorer("a")
            0.6666666666666666
        """
        # Reset state so re-fitting an already-fitted scorer behaves like a
        # fresh fit: an estimator that carries auxiliary state would otherwise
        # leave stale keys behind.
        self._prob = {}
        self._unobs = MIN_PROBABILITY
        self._total_unseen_mass = None

        self._install(self._estimate(freqdist))

        return self

    def _install(self, estimates: "Estimates") -> None:
        """Convert estimates into this scorer's output space and adopt them.

        The single place where a probability becomes a score. Called by
        :meth:`fit`, and by any estimator that revises its estimates outside a
        fit (the streaming scorers do this on every update).

        Args:
            estimates: A probability table and a reserved mass, both in plain
                probability space.
        """
        table, reserved = estimates

        self._prob = {element: self._score_of(prob) for element, prob in table.items()}
        self._unobs = self._score_of(reserved)
        self._mark_changed()

    def score(self, element: Element) -> Probability | LogProbability:
        """Score a single element (scikit-learn-style alias for ``__call__``).

        Args:
            element: The element to be scored.

        Returns:
            The probability (or log-probability) of the element.

        Examples:
            >>> from freqprob import MLE
            >>> scorer = MLE({"a": 2, "b": 1}, logprob=False)
            >>> scorer.score("a")
            0.6666666666666666
        """
        return self(element)

    def predict(self, elements: Iterable[Element]) -> list[Probability | LogProbability]:
        """Score many elements at once (scikit-learn-style batch alias).

        Args:
            elements: The elements to be scored.

        Returns:
            One probability (or log-probability) per input element, in order.

        Examples:
            >>> from freqprob import MLE
            >>> scorer = MLE({"a": 2, "b": 1}, logprob=False)
            >>> scorer.predict(["a", "b", "c"])
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
        # Pickles written before `_revision` existed carry no value for it.
        if "_revision" not in state:
            self._revision = 0

    def save(self, path: str | Path) -> None:
        """Serialize this fitted scorer to a file.

        The scorer can be restored later with :meth:`load`, avoiding the cost of
        re-fitting. Serialization uses :mod:`pickle`.

        Args:
            path: Destination file path.

        Examples:
            >>> from freqprob import MLE
            >>> scorer = MLE({"a": 2, "b": 1}, logprob=False)
            >>> scorer.save("model.pkl")             # doctest: +SKIP
            >>> restored = MLE.load("model.pkl")     # doctest: +SKIP
            >>> restored("a") == scorer("a")         # doctest: +SKIP
            True
        """
        with Path(path).open("wb") as fh:
            pickle.dump(self, fh, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls: type[T], path: str | Path) -> T:  # noqa: PYI019  (Self needs typing_extensions on 3.10)
        """Load a scorer previously written with :meth:`save`.

        Args:
            path: Path to a file created by :meth:`save`.

        Returns:
            The restored scorer.

        Raises:
            TypeError: If the file does not contain a compatible scorer.

        Note:
            Only load files you trust: unpickling executes arbitrary code from
            the file, so never load a scorer from an untrusted source.

        Examples:
            >>> from freqprob import MLE
            >>> restored = MLE.load("model.pkl")     # doctest: +SKIP
            >>> restored("a")                        # doctest: +SKIP
            0.6666666666666666
        """
        with Path(path).open("rb") as fh:
            obj = pickle.load(fh)
        if not isinstance(obj, cls):
            raise TypeError(f"File does not contain a {cls.__name__}: got {type(obj).__name__}")
        return obj
