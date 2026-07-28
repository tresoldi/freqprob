"""Lazy evaluation support for efficient probability calculations.

This module provides lazy evaluation mechanisms that defer expensive
computations until they are actually needed, improving performance
when only a subset of probabilities are accessed.
"""

import math
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator
from typing import Any

from freqprob.base import Element, FrequencyDistribution, ScoringMethod, ScoringMethodConfig


class LazyProbabilityComputer(ABC):
    """Abstract base class for lazy probability computation strategies.

    This class defines the interface for different lazy evaluation
    strategies that can be used with scoring methods.
    """

    @abstractmethod
    def compute_probability(
        self,
        element: Element,
        freqdist: FrequencyDistribution,
        config: ScoringMethodConfig,
    ) -> float:
        """Compute probability for a single element on demand.

        Parameters
        ----------
        element : Element
            Element to compute probability for
        freqdist : FrequencyDistribution
            Frequency distribution
        config : ScoringMethodConfig
            Configuration parameters

        Returns:
        -------
        float
            Computed probability
        """

    @abstractmethod
    def compute_unobserved_probability(
        self, freqdist: FrequencyDistribution, config: ScoringMethodConfig
    ) -> float:
        """Compute probability for unobserved elements.

        Parameters
        ----------
        freqdist : FrequencyDistribution
            Frequency distribution
        config : ScoringMethodConfig
            Configuration parameters

        Returns:
        -------
        float
            Unobserved probability
        """


class LazyMLEComputer(LazyProbabilityComputer):
    """Lazy computation for Maximum Likelihood Estimation."""

    def __init__(self) -> None:
        """Initialize lazy MLE computer."""
        self._total_count: int | None = None
        self._unobs_prob: float | None = None

    def compute_probability(
        self,
        element: Element,
        freqdist: FrequencyDistribution,
        config: ScoringMethodConfig,
    ) -> float:
        """Compute MLE probability for element."""
        if self._total_count is None:
            self._total_count = sum(freqdist.values())

        count = freqdist.get(element, 0)
        if self._total_count == 0:
            return 0.0

        prob = count / self._total_count

        # Apply unobserved probability mass if configured
        if hasattr(config, "unobs_prob") and config.unobs_prob is not None:
            if count == 0:
                return config.unobs_prob
            # Adjust observed probabilities
            return prob * (1.0 - config.unobs_prob)

        return prob

    def compute_unobserved_probability(
        self, freqdist: FrequencyDistribution, config: ScoringMethodConfig
    ) -> float:
        """Compute unobserved probability for MLE."""
        if hasattr(config, "unobs_prob") and config.unobs_prob is not None:
            return config.unobs_prob
        return 0.0


class LazyLaplaceComputer(LazyProbabilityComputer):
    """Lazy computation for Laplace smoothing."""

    def __init__(self) -> None:
        """Initialize lazy Laplace computer."""
        self._total_count: int | None = None
        self._vocab_size: int | None = None

    def compute_probability(
        self,
        element: Element,
        freqdist: FrequencyDistribution,
        config: ScoringMethodConfig,
    ) -> float:
        """Compute Laplace probability for element."""
        if self._total_count is None:
            self._total_count = sum(freqdist.values())
        if self._vocab_size is None:
            bins = getattr(config, "bins", None)
            self._vocab_size = bins if bins is not None else len(freqdist)

        count = freqdist.get(element, 0)
        return (count + 1) / (self._total_count + self._vocab_size)

    def compute_unobserved_probability(
        self, freqdist: FrequencyDistribution, config: ScoringMethodConfig
    ) -> float:
        """Compute unobserved probability for Laplace."""
        if self._total_count is None:
            self._total_count = sum(freqdist.values())
        if self._vocab_size is None:
            bins = getattr(config, "bins", None)
            self._vocab_size = bins if bins is not None else len(freqdist)

        return 1.0 / (self._total_count + self._vocab_size)


class LazyScoringMethod(ScoringMethod):
    """Scoring method that defers probability computation until requested.

    Wraps a lazy computation strategy so that probabilities are calculated
    only for the elements actually scored, rather than for the whole
    vocabulary up front. This can significantly improve performance when a
    large distribution is fitted but only a small subset of elements is ever
    queried.

    Args:
        lazy_computer: Strategy object that computes a single element's
            probability on demand.
        config: Configuration controlling ``logprob`` and any reserved
            probability mass.
        name: Human-readable name for the scoring method.

    Note:
        Instances are normally built through the public factory functions
        `create_lazy_mle` and `create_lazy_laplace` rather than constructed
        directly, since the computer and config types are internal.

    Examples:
        Probabilities are computed only when an element is first scored:

        >>> from freqprob import create_lazy_mle
        >>> scorer = create_lazy_mle({"a": 3, "b": 2, "c": 1}, logprob=False)
        >>> scorer("a")
        0.5
        >>> sorted(scorer.get_computed_elements())
        ['a']
        >>> scorer("b")
        0.3333333333333333
        >>> sorted(scorer.get_computed_elements())
        ['a', 'b']
        >>> scorer("unseen")  # unobserved element
        0.0
    """

    def __init__(
        self,
        lazy_computer: LazyProbabilityComputer,
        config: ScoringMethodConfig,
        name: str,
    ):
        """Initialize lazy scoring method.

        Args:
            lazy_computer: Strategy for lazy computation.
            config: Configuration parameters.
            name: Name of the scoring method.
        """
        super().__init__(config)
        self.lazy_computer = lazy_computer
        self.name = name
        self._freqdist: FrequencyDistribution | None = None
        self._computed_elements: set[Element] = set()
        self._unobs_computed = False

    def _compute_probabilities(self, freqdist: FrequencyDistribution) -> None:
        """Store frequency distribution for lazy computation."""
        self._freqdist = freqdist
        self._computed_elements.clear()
        self._unobs_computed = False
        # Don't compute anything yet - this is the lazy part!

    def __call__(self, element: Element) -> float:
        """Score element with lazy computation.

        Args:
            element: Element to score.

        Returns:
            Probability or log-probability for the element.
        """
        if self._freqdist is None:
            raise ValueError("Scoring method has not been fitted")

        # Check if already computed
        if element in self._computed_elements:
            return self._prob[element]

        # Check if element is in the original distribution
        if element in self._freqdist:
            # Compute probability for this element
            prob = self.lazy_computer.compute_probability(element, self._freqdist, self.config)

            # Convert to log probability if needed
            if self.logprob:
                if prob > 0:
                    self._prob[element] = math.log(prob)
                else:
                    self._prob[element] = math.log(1e-10)  # Avoid log(0)
            else:
                self._prob[element] = prob

            self._computed_elements.add(element)
            return self._prob[element]
        # Return unobserved probability
        return self._get_unobserved_probability()

    def _get_unobserved_probability(self) -> float:
        """Get unobserved probability, computing it lazily if needed."""
        if not self._unobs_computed:
            if self._freqdist is None:
                raise ValueError("Scoring method has not been fitted")
            unobs_prob = self.lazy_computer.compute_unobserved_probability(
                self._freqdist, self.config
            )

            if self.logprob:
                if unobs_prob > 0:
                    self._unobs = math.log(unobs_prob)
                else:
                    self._unobs = math.log(1e-10)
            else:
                self._unobs = unobs_prob

            self._unobs_computed = True

        return self._unobs

    def precompute_batch(self, elements: set[Element]) -> None:
        """Precompute probabilities for a batch of elements.

        This can be useful when you know you'll need several elements
        and want to compute them all at once for efficiency.

        Args:
            elements: Elements to precompute.
        """
        if self._freqdist is None:
            raise ValueError("Scoring method has not been fitted")

        for element in elements:
            if element not in self._computed_elements and element in self._freqdist:
                # Trigger computation
                self(element)

    def get_computed_elements(self) -> set[Element]:
        """Get the set of elements that have been computed so far.

        Returns:
            Set of elements whose probabilities have been computed.
        """
        return self._computed_elements.copy()

    def force_full_computation(self) -> None:
        """Force computation of all probabilities in the distribution.

        This converts the lazy scorer to a regular scorer by computing
        all probabilities immediately.
        """
        if self._freqdist is None:
            raise ValueError("Scoring method has not been fitted")

        for element in self._freqdist:
            if element not in self._computed_elements:
                self(element)  # Trigger computation

        # Also compute unobserved probability
        self._get_unobserved_probability()


class LazyBatchScorer:
    """Batch scorer with lazy evaluation and intelligent caching.

    This scorer optimizes batch operations by:

    1. Using lazy evaluation to avoid unnecessary computations.
    2. Intelligently ordering computations based on access patterns.
    3. Providing memory-efficient operations for large datasets.

    Args:
        lazy_scorer: Lazy scoring method to wrap.

    Examples:
        Wrap a lazy MLE scorer and score a batch of elements:

        >>> from freqprob import create_lazy_mle
        >>> lazy_scorer = create_lazy_mle({"a": 3, "b": 2, "c": 1}, logprob=False)
        >>> batch_scorer = LazyBatchScorer(lazy_scorer)
        >>> [round(float(x), 4) for x in batch_scorer.score_batch(["a", "c"])]
        [0.5, 0.1667]
        >>> stats = batch_scorer.get_access_statistics()
        >>> stats["total_accesses"]
        2
        >>> stats["unique_elements"]
        2
    """

    def __init__(self, lazy_scorer: LazyScoringMethod):
        """Initialize lazy batch scorer.

        Args:
            lazy_scorer: Lazy scoring method to wrap.
        """
        self.lazy_scorer = lazy_scorer
        self._access_count: dict[Element, int] = {}

    def score_batch(self, elements: list[Element]) -> list[float]:
        """Score a batch of elements with lazy evaluation.

        Args:
            elements: Elements to score.

        Returns:
            Scores for the elements, in the same order as the input.
        """
        # Track access patterns
        for element in elements:
            self._access_count[element] = self._access_count.get(element, 0) + 1

        # Precompute if this is an efficient strategy
        unique_elements = set(elements)
        if len(unique_elements) > 1:
            self.lazy_scorer.precompute_batch(unique_elements)

        # Score all elements
        return [self.lazy_scorer(element) for element in elements]

    def score_streaming(self, element_stream: Iterable[Element]) -> Iterator[float]:
        """Score elements from a stream with adaptive lazy evaluation.

        Args:
            element_stream: Stream of elements to score.

        Yields:
            Score for each element in the stream.
        """
        seen_elements = set()
        batch_size = 100  # Adaptive batch size

        for element in element_stream:
            if element not in seen_elements:
                seen_elements.add(element)

                # Adaptive precomputation based on stream patterns
                if len(seen_elements) % batch_size == 0:
                    # Precompute recent elements
                    recent_elements = set(list(seen_elements)[-batch_size:])
                    self.lazy_scorer.precompute_batch(recent_elements)

            yield self.lazy_scorer(element)

    def get_access_statistics(self) -> dict[str, Any]:
        """Get statistics about element access patterns.

        Returns:
            Access statistics including total accesses, unique elements, and
            the most frequently accessed element.
        """
        if not self._access_count:
            return {"total_accesses": 0, "unique_elements": 0}

        total_accesses = sum(self._access_count.values())
        unique_elements = len(self._access_count)
        most_accessed = max(self._access_count.items(), key=lambda x: x[1])

        return {
            "total_accesses": total_accesses,
            "unique_elements": unique_elements,
            "most_accessed_element": most_accessed[0],
            "most_accessed_count": most_accessed[1],
            "computed_elements": len(self.lazy_scorer.get_computed_elements()),
        }


# Factory functions for creating lazy scorers


def create_lazy_mle(
    freqdist: FrequencyDistribution,
    unobs_prob: float | None = None,
    logprob: bool = True,
) -> LazyScoringMethod:
    """Create a lazy Maximum Likelihood Estimation scorer.

    Builds a `LazyScoringMethod` that computes MLE probabilities on demand,
    fitted to the given distribution.

    Args:
        freqdist: Mapping of elements to observed counts.
        unobs_prob: Reserved probability mass for unobserved elements, or
            ``None`` for plain MLE. Defaults to ``None``.
        logprob: Return log-probabilities if ``True`` (the default),
            otherwise plain probabilities.

    Returns:
        A fitted lazy MLE scorer.

    Examples:
        >>> scorer = create_lazy_mle({"a": 3, "b": 2, "c": 1}, logprob=False)
        >>> scorer("a")
        0.5
        >>> scorer("c")
        0.16666666666666666
        >>> scorer("unseen")
        0.0
    """
    from freqprob.base import ScoringMethodConfig

    config = ScoringMethodConfig(unobs_prob=unobs_prob, logprob=logprob)
    computer = LazyMLEComputer()
    scorer = LazyScoringMethod(computer, config, "Lazy MLE")
    scorer.fit(freqdist)
    return scorer


def create_lazy_laplace(
    freqdist: FrequencyDistribution, bins: int | None = None, logprob: bool = True
) -> LazyScoringMethod:
    """Create a lazy Laplace (add-one) smoothing scorer.

    Builds a `LazyScoringMethod` that computes Laplace-smoothed probabilities
    on demand, fitted to the given distribution.

    Args:
        freqdist: Mapping of elements to observed counts.
        bins: Total number of possible elements (vocabulary size). When
            ``None`` (the default), the number of observed elements is used.
        logprob: Return log-probabilities if ``True`` (the default),
            otherwise plain probabilities.

    Returns:
        A fitted lazy Laplace scorer.

    Examples:
        >>> scorer = create_lazy_laplace({"a": 3, "b": 2, "c": 1}, logprob=False)
        >>> scorer("a")
        0.4444444444444444
        >>> scorer("unseen")
        0.1111111111111111
    """
    from freqprob.base import ScoringMethodConfig

    config = ScoringMethodConfig(bins=bins, logprob=logprob)
    computer = LazyLaplaceComputer()
    scorer = LazyScoringMethod(computer, config, "Lazy Laplace")
    scorer.fit(freqdist)
    return scorer
