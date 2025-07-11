"""Streaming and incremental frequency distribution updates.

This module provides efficient streaming support for frequency distributions,
allowing incremental updates and online learning scenarios with minimal
memory overhead.
"""

import math
import pickle
import threading
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Iterator
from typing import Any

from .base import Element, FrequencyDistribution, ScoringMethod, ScoringMethodConfig


class StreamingFrequencyDistribution:
    """Memory-efficient streaming frequency distribution with incremental updates.

    This class supports real-time updates to frequency distributions while
    maintaining memory efficiency through various optimization strategies.

    Parameters
    ----------
    max_vocabulary_size : Optional[int]
        Maximum vocabulary size to maintain (None for unlimited)
    min_count_threshold : int, default=1
        Minimum count for elements to be retained
    decay_factor : Optional[float]
        Exponential decay factor for aging old observations (None for no decay)
    compression_threshold : int, default=10000
        Threshold for triggering compression operations

    Examples:
    --------
    >>> stream_dist = StreamingFrequencyDistribution(max_vocabulary_size=1000)
    >>> stream_dist.update('word1')
    >>> stream_dist.update('word1', count=5)
    >>> stream_dist.update_batch(['word2', 'word3', 'word1'])
    >>> stream_dist.get_count('word1')
    7
    >>> len(stream_dist)
    3
    """

    def __init__(
        self,
        max_vocabulary_size: int | None = None,
        min_count_threshold: int = 1,
        decay_factor: float | None = None,
        compression_threshold: int = 10000,
    ):
        """Initialize streaming frequency distribution.

        Parameters
        ----------
        max_vocabulary_size : Optional[int]
            Maximum vocabulary size (None for unlimited)
        min_count_threshold : int, default=1
            Minimum count threshold for retention
        decay_factor : Optional[float]
            Decay factor for time-based forgetting (0 < decay < 1)
        compression_threshold : int, default=10000
            Number of updates before triggering compression
        """
        self._counts: dict[Element, float] = defaultdict(float)

        self._total_count: float = 0.0
        self._update_count: int = 0
        self._max_vocab_size = max_vocabulary_size
        self._min_count_threshold = min_count_threshold
        self._decay_factor = decay_factor
        self._compression_threshold = compression_threshold
        self._lock = threading.RLock()

        # Statistics tracking
        self._creation_order: dict[Element, int] = {}
        self._last_access: dict[Element, int] = {}

    def update(self, element: Element, count: int = 1) -> None:
        """Update count for a single element.

        Parameters
        ----------
        element : Element
            Element to update
        count : int, default=1
            Count increment
        """
        with self._lock:
            self._counts[element] += count
            self._total_count += count
            self._update_count += 1

            # Track creation order and access patterns
            if element not in self._creation_order:
                self._creation_order[element] = self._update_count
            self._last_access[element] = self._update_count

            # Apply decay if configured
            if self._decay_factor is not None:
                self._apply_decay()

            # Trigger compression if needed
            if self._update_count % self._compression_threshold == 0 or (
                self._max_vocab_size and len(self._counts) > self._max_vocab_size
            ):
                self._compress()

    def update_batch(self, elements: list[Element], counts: list[int] | None = None) -> None:
        """Update counts for multiple elements efficiently.

        Parameters
        ----------
        elements : List[Element]
            Elements to update
        counts : Optional[List[int]]
            Counts for each element (defaults to 1 for all)
        """
        if counts is None:
            counts = [1] * len(elements)
        elif len(counts) != len(elements):
            raise ValueError("Length of counts must match length of elements")

        with self._lock:
            for element, count in zip(elements, counts, strict=False):
                self._counts[element] += count
                self._total_count += count

                if element not in self._creation_order:
                    self._creation_order[element] = self._update_count
                self._last_access[element] = self._update_count

            self._update_count += len(elements)

            # Apply decay and compression
            if self._decay_factor is not None:
                self._apply_decay()

            if self._update_count % self._compression_threshold == 0 or (
                self._max_vocab_size and len(self._counts) > self._max_vocab_size
            ):
                self._compress()

    def _apply_decay(self) -> None:
        """Apply exponential decay to all counts."""
        if self._decay_factor is None:
            return

        _ = 1.0 - self._decay_factor  # decay_amount calculated but not used
        total_decay = 0.0

        for element in list(self._counts.keys()):
            old_count = self._counts[element]
            new_count = old_count * self._decay_factor
            self._counts[element] = new_count
            total_decay += old_count - new_count

        self._total_count -= total_decay

        # Remove elements that have decayed below threshold
        self._remove_low_count_elements()

    def _compress(self) -> None:
        """Compress the distribution by removing low-frequency elements."""
        # Remove elements below minimum threshold
        self._remove_low_count_elements()

        # If still over vocabulary limit, remove least frequent/oldest elements
        if self._max_vocab_size and len(self._counts) > self._max_vocab_size:
            self._enforce_vocabulary_limit()

    def _remove_low_count_elements(self) -> None:
        """Remove elements with counts below the minimum threshold."""
        to_remove = []

        removed_count = 0.0

        for element, count in self._counts.items():
            if count < self._min_count_threshold:
                to_remove.append(element)
                removed_count += count

        for element in to_remove:
            del self._counts[element]
            self._creation_order.pop(element, None)
            self._last_access.pop(element, None)

        self._total_count -= removed_count

    def _enforce_vocabulary_limit(self) -> None:
        """Enforce maximum vocabulary size by removing least important elements."""
        if not self._max_vocab_size or len(self._counts) <= self._max_vocab_size:
            return

        # Sort by importance (combination of frequency and recency)
        elements_by_importance = []
        for element, count in self._counts.items():
            # Combine frequency and recency for importance score
            frequency_score = count / self._total_count
            recency_score = self._last_access.get(element, 0) / self._update_count
            importance = frequency_score * 0.7 + recency_score * 0.3
            elements_by_importance.append((importance, element))

        # Sort by importance (lowest first)
        elements_by_importance.sort()

        # Remove least important elements
        num_to_remove = len(self._counts) - self._max_vocab_size
        removed_count = 0.0

        for _, element in elements_by_importance[:num_to_remove]:
            removed_count += self._counts[element]
            del self._counts[element]
            self._creation_order.pop(element, None)
            self._last_access.pop(element, None)

        self._total_count -= removed_count

    def get_count(self, element: Element) -> float:
        """Get count for an element.

        Parameters
        ----------
        element : Element
            Element to query

        Returns:
        -------
        float
            Count for the element
        """
        with self._lock:
            return self._counts.get(element, 0.0)

    def get_frequency(self, element: Element) -> float:
        """Get relative frequency for an element.

        Parameters
        ----------
        element : Element
            Element to query

        Returns:
        -------
        float
            Relative frequency (count / total_count)
        """
        with self._lock:
            if self._total_count == 0:
                return 0.0
            return self._counts.get(element, 0.0) / self._total_count

    def get_total_count(self) -> float:
        """Get total count across all elements."""
        with self._lock:
            return self._total_count

    def get_vocabulary_size(self) -> int:
        """Get current vocabulary size."""
        with self._lock:
            return len(self._counts)

    def items(self) -> Iterator[tuple[Element, float]]:
        """Iterate over (element, count) pairs."""
        with self._lock:
            # Create a snapshot to avoid modification during iteration
            return iter(dict(self._counts).items())

    def keys(self) -> Iterator[Element]:
        """Iterate over elements."""
        with self._lock:
            return iter(list(self._counts.keys()))

    def values(self) -> Iterator[float]:
        """Iterate over counts."""
        with self._lock:
            return iter(list(self._counts.values()))

    def __len__(self) -> int:
        """Return vocabulary size."""
        return self.get_vocabulary_size()

    def __contains__(self, element: Element) -> bool:
        """Check if element is in distribution."""
        with self._lock:
            return element in self._counts

    def __iter__(self) -> Iterator[Element]:
        """Iterate over elements."""
        return self.keys()

    def __getitem__(self, element: Element) -> float:
        """Get count for element (dict-like interface)."""
        return self.get_count(element)

    def to_dict(self) -> dict[Element, int]:
        """Convert to regular dictionary with integer counts.

        Returns:
        -------
        Dict[Element, int]
            Dictionary representation
        """
        with self._lock:
            return {element: int(count) for element, count in self._counts.items()}

    def get_statistics(self) -> dict[str, Any]:
        """Get statistics about the streaming distribution.

        Returns:
        -------
        Dict[str, Any]
            Statistics dictionary
        """
        with self._lock:
            return {
                "vocabulary_size": len(self._counts),
                "total_count": self._total_count,
                "update_count": self._update_count,
                "max_vocab_size": self._max_vocab_size,
                "min_count_threshold": self._min_count_threshold,
                "decay_factor": self._decay_factor,
                "most_frequent": (
                    max(self._counts.items(), key=lambda x: x[1]) if self._counts else None
                ),
                "average_count": self._total_count / len(self._counts) if self._counts else 0,
            }


class IncrementalScoringMethod(ABC):
    """Abstract base class for scoring methods that support incremental updates.

    This class defines the interface for scoring methods that can be updated
    incrementally as new data arrives, without recomputing everything from scratch.
    """

    @abstractmethod
    def update_single(self, element: Element, count: int = 1) -> None:
        """Update the model with a single element observation.

        Parameters
        ----------
        element : Element
            Observed element
        count : int, default=1
            Count of observations
        """

    @abstractmethod
    def update_batch(self, elements: list[Element], counts: list[int] | None = None) -> None:
        """Update the model with multiple element observations.

        Parameters
        ----------
        elements : List[Element]
            Observed elements
        counts : Optional[List[int]]
            Counts for each element
        """

    @abstractmethod
    def get_update_count(self) -> int:
        """Get the number of updates performed."""


class StreamingMLE(ScoringMethod, IncrementalScoringMethod):
    """Streaming Maximum Likelihood Estimation with incremental updates.

    This implementation maintains a streaming frequency distribution and
    updates probability estimates incrementally as new data arrives.

    Parameters
    ----------
    initial_freqdist : Optional[Dict[Element, int]]
        Initial frequency distribution
    max_vocabulary_size : Optional[int]
        Maximum vocabulary size to maintain
    unobs_prob : Optional[float]
        Probability mass for unobserved elements
    logprob : bool, default=True
        Whether to return log-probabilities

    Examples:
    --------
    >>> streaming_mle = StreamingMLE(max_vocabulary_size=1000, logprob=False)
    >>> streaming_mle.update_single('word1', 5)
    >>> streaming_mle.update_batch(['word2', 'word3'])
    >>> streaming_mle('word1')  # Get probability
    0.7142857142857143
    """

    def __init__(
        self,
        initial_freqdist: dict[Element, int] | None = None,
        max_vocabulary_size: int | None = None,
        unobs_prob: float | None = None,
        logprob: bool = True,
    ):
        """Initialize streaming MLE scorer."""
        config = ScoringMethodConfig(unobs_prob=unobs_prob, logprob=logprob)

        super().__init__(config)
        self.name = "Streaming MLE"

        # Initialize streaming distribution
        self._stream_dist = StreamingFrequencyDistribution(max_vocabulary_size=max_vocabulary_size)

        # Add initial data if provided
        if initial_freqdist:
            for element, count in initial_freqdist.items():
                self._stream_dist.update(element, count)

        self._recompute_probabilities()

    def _compute_probabilities(self, freqdist: FrequencyDistribution) -> None:
        """Override parent method - not used in streaming mode."""
        # Overridden by _recompute_probabilities

    def _recompute_probabilities(self) -> None:
        """Recompute probabilities from streaming distribution."""
        total_count = self._stream_dist.get_total_count()

        unobs_prob = getattr(self.config, "unobs_prob", None)

        if total_count == 0:
            self._unobs = 0.0 if not self.logprob else math.log(1e-10)
            return

        self._prob.clear()

        # Compute probabilities for observed elements
        for element in self._stream_dist:
            count = self._stream_dist.get_count(element)
            prob = count / total_count

            # Apply unobserved probability mass if configured
            if unobs_prob is not None and prob > 0:
                prob *= 1.0 - unobs_prob

            if self.logprob:
                self._prob[element] = math.log(prob) if prob > 0 else math.log(1e-10)
            else:
                self._prob[element] = prob

        # Set unobserved probability
        unobs_val = unobs_prob if unobs_prob is not None else 0.0

        if self.logprob:
            self._unobs = math.log(unobs_val) if unobs_val > 0 else math.log(1e-10)
        else:
            self._unobs = unobs_val

    def update_single(self, element: Element, count: int = 1) -> None:
        """Update with a single element observation."""
        self._stream_dist.update(element, count)

        # Efficient incremental update instead of full recomputation
        self._incremental_update(element, count)

    def update_batch(self, elements: list[Element], counts: list[int] | None = None) -> None:
        """Update with multiple element observations."""
        if counts is None:
            counts = [1] * len(elements)

        self._stream_dist.update_batch(elements, counts)

        # Recompute probabilities after batch update
        self._recompute_probabilities()

    def _incremental_update(self, element: Element, count: int) -> None:
        """Efficiently update probabilities for a single element change."""
        total_count = self._stream_dist.get_total_count()

        element_count = self._stream_dist.get_count(element)

        if total_count == 0:
            return

        # Update probability for this element
        prob = element_count / total_count
        unobs_prob = getattr(self.config, "unobs_prob", None)

        if unobs_prob is not None:
            prob *= 1.0 - unobs_prob

        if self.logprob:
            self._prob[element] = math.log(prob) if prob > 0 else math.log(1e-10)
        else:
            self._prob[element] = prob

        # Update probabilities for all other elements (they've been diluted)
        if count > 0:  # Only if we actually added count
            for other_element in self._prob:
                if other_element != element:
                    old_prob = (
                        math.exp(self._prob[other_element])
                        if self.logprob
                        else self._prob[other_element]
                    )
                    new_prob = old_prob * (total_count - count) / total_count

                    if unobs_prob is not None:
                        new_prob *= 1.0 - unobs_prob

                    if self.logprob:
                        self._prob[other_element] = (
                            math.log(new_prob) if new_prob > 0 else math.log(1e-10)
                        )
                    else:
                        self._prob[other_element] = new_prob

    def get_update_count(self) -> int:
        """Get the number of updates performed."""
        return self._stream_dist._update_count

    def get_streaming_statistics(self) -> dict[str, Any]:
        """Get statistics about the streaming distribution."""
        return self._stream_dist.get_statistics()

    def save_state(self, filepath: str) -> None:
        """Save the current state to disk.

        Parameters
        ----------
        filepath : str
            Path to save the state
        """
        # Create a copy of stream_dist without the lock
        stream_dist_data = {
            "_counts": dict(self._stream_dist._counts),
            "_total_count": self._stream_dist._total_count,
            "_update_count": self._stream_dist._update_count,
            "_max_vocab_size": self._stream_dist._max_vocab_size,
            "_min_count_threshold": self._stream_dist._min_count_threshold,
            "_decay_factor": self._stream_dist._decay_factor,
            "_compression_threshold": self._stream_dist._compression_threshold,
            "_creation_order": dict(self._stream_dist._creation_order),
            "_last_access": dict(self._stream_dist._last_access),
        }

        state = {"stream_dist_data": stream_dist_data, "config": self.config, "name": self.name}

        with open(filepath, "wb") as f:
            pickle.dump(state, f)

    @classmethod
    def load_state(cls, filepath: str) -> "StreamingMLE":
        """Load state from disk.

        Parameters
        ----------
        filepath : str
            Path to load the state from

        Returns:
        -------
        StreamingMLE
            Loaded streaming MLE instance
        """
        with open(filepath, "rb") as f:
            state = pickle.load(f)

        instance = cls.__new__(cls)
        instance.config = state["config"]
        instance.name = state["name"]

        # Reconstruct the streaming distribution
        stream_dist_data = state["stream_dist_data"]
        instance._stream_dist = StreamingFrequencyDistribution(
            max_vocabulary_size=stream_dist_data["_max_vocab_size"],
            min_count_threshold=stream_dist_data["_min_count_threshold"],
            decay_factor=stream_dist_data["_decay_factor"],
            compression_threshold=stream_dist_data["_compression_threshold"],
        )

        # Restore the state
        instance._stream_dist._counts = stream_dist_data["_counts"]
        instance._stream_dist._total_count = stream_dist_data["_total_count"]
        instance._stream_dist._update_count = stream_dist_data["_update_count"]
        instance._stream_dist._creation_order = stream_dist_data["_creation_order"]
        instance._stream_dist._last_access = stream_dist_data["_last_access"]

        instance._prob = {}
        instance._unobs = 0.0
        instance.logprob = state["config"].logprob

        instance._recompute_probabilities()
        return instance


class StreamingLaplace(StreamingMLE):
    """Streaming Laplace smoothing with incremental updates.

    Extends StreamingMLE to provide Laplace (add-one) smoothing
    in a streaming context.

    Parameters
    ----------
    initial_freqdist : Optional[Dict[Element, int]]
        Initial frequency distribution
    max_vocabulary_size : Optional[int]
        Maximum vocabulary size to maintain
    bins : Optional[int]
        Total number of possible bins
    logprob : bool, default=True
        Whether to return log-probabilities
    """

    def __init__(
        self,
        initial_freqdist: dict[Element, int] | None = None,
        max_vocabulary_size: int | None = None,
        bins: int | None = None,
        logprob: bool = True,
    ):
        """Initialize streaming Laplace scorer."""
        # Initialize as MLE first
        super().__init__(initial_freqdist, max_vocabulary_size, None, logprob)
        self.name = "Streaming Laplace"

        # Update config for Laplace-specific parameters
        self.config.bins = bins
        self._recompute_probabilities()

    def _recompute_probabilities(self) -> None:
        """Recompute probabilities with Laplace smoothing."""
        total_count = self._stream_dist.get_total_count()

        vocab_size = self._stream_dist.get_vocabulary_size()
        bins = getattr(self.config, "bins", None)

        # For Laplace smoothing, if bins is not specified, we need to consider
        # the original vocabulary size, not just the current one
        if bins is None:
            bins = vocab_size

        if bins == 0:
            bins = 1  # Avoid division by zero

        self._prob.clear()

        # Laplace smoothing: (count + 1) / (total + bins)
        denominator = total_count + bins

        # Compute probabilities for observed elements
        for element in self._stream_dist:
            count = self._stream_dist.get_count(element)
            prob = (count + 1.0) / denominator

            if self.logprob:
                self._prob[element] = math.log(prob)
            else:
                self._prob[element] = prob

        # Unobserved probability
        unobs_prob = 1.0 / denominator
        if self.logprob:
            self._unobs = math.log(unobs_prob)
        else:
            self._unobs = unobs_prob


class StreamingDataProcessor:
    """High-level processor for streaming text data.

    This class provides utilities for processing streaming text data
    and maintaining multiple frequency distributions efficiently.

    Parameters
    ----------
    scoring_methods : Dict[str, IncrementalScoringMethod]
        Dictionary of scoring methods to maintain
    batch_size : int, default=1000
        Batch size for processing
    auto_save_interval : Optional[int]
        Interval for automatic state saving (None to disable)

    Examples:
    --------
    >>> methods = {
    ...     'mle': StreamingMLE(max_vocabulary_size=10000),
    ...     'laplace': StreamingLaplace(max_vocabulary_size=10000)
    ... }
    >>> processor = StreamingDataProcessor(methods)
    >>> processor.process_text_stream(["word1", "word2", "word1"])
    >>> processor.get_score('mle', 'word1')
    """

    def __init__(
        self,
        scoring_methods: dict[str, IncrementalScoringMethod],
        batch_size: int = 1000,
        auto_save_interval: int | None = None,
    ):
        """Initialize streaming data processor."""
        self.scoring_methods = scoring_methods

        self.batch_size = batch_size
        self.auto_save_interval = auto_save_interval
        self._processed_count = 0
        self._batch_buffer: list[Element] = []

    def process_element(self, element: Element, count: int = 1) -> None:
        """Process a single element.

        Parameters
        ----------
        element : Element
            Element to process
        count : int, default=1
            Count for the element
        """
        for method in self.scoring_methods.values():
            method.update_single(element, count)

        self._processed_count += count
        self._check_auto_save()

    def process_batch(self, elements: list[Element], counts: list[int] | None = None) -> None:
        """Process a batch of elements.

        Parameters
        ----------
        elements : List[Element]
            Elements to process
        counts : Optional[List[int]]
            Counts for each element
        """
        for method in self.scoring_methods.values():
            method.update_batch(elements, counts)

        self._processed_count += len(elements)
        self._check_auto_save()

    def process_text_stream(self, text_stream: Iterator[str]) -> None:
        """Process a stream of text tokens.

        Parameters
        ----------
        text_stream : Iterator[str]
            Stream of text tokens
        """
        for token in text_stream:
            self._batch_buffer.append(token)

            if len(self._batch_buffer) >= self.batch_size:
                self.process_batch(self._batch_buffer)
                self._batch_buffer.clear()

        # Process remaining elements in buffer
        if self._batch_buffer:
            self.process_batch(self._batch_buffer)
            self._batch_buffer.clear()

    def get_score(self, method_name: str, element: Element) -> float:
        """Get score for an element from a specific method.

        Parameters
        ----------
        method_name : str
            Name of the scoring method
        element : Element
            Element to score

        Returns:
        -------
        float
            Score for the element
        """
        if method_name not in self.scoring_methods:
            raise ValueError(f"Unknown scoring method: {method_name}")

        method = self.scoring_methods[method_name]
        # IncrementalScoringMethod implementations inherit __call__ from ScoringMethod
        return float(method(element))  # type: ignore[operator]

    def get_statistics(self) -> dict[str, Any]:
        """Get statistics for all scoring methods.

        Returns:
        -------
        Dict[str, Any]
            Statistics for each method
        """
        stats = {"processed_count": self._processed_count, "methods": {}}

        for name, method in self.scoring_methods.items():
            method_stats = {
                "update_count": method.get_update_count(),
                "name": getattr(method, "name", name),
            }

            # Add method-specific statistics if available
            if hasattr(method, "get_streaming_statistics"):
                method_stats.update(method.get_streaming_statistics())

            stats["methods"][name] = method_stats  # type: ignore[index]

        return stats

    def _check_auto_save(self) -> None:
        """Check if automatic saving should be triggered."""
        if self.auto_save_interval and self._processed_count % self.auto_save_interval == 0:
            self.save_all_states(f"auto_save_{self._processed_count}")

    def save_all_states(self, base_filename: str) -> None:
        """Save states of all scoring methods.

        Parameters
        ----------
        base_filename : str
            Base filename for saving states
        """
        for name, method in self.scoring_methods.items():
            if hasattr(method, "save_state"):
                method.save_state(f"{base_filename}_{name}.pkl")

    def clear_buffers(self) -> None:
        """Clear all internal buffers."""
        self._batch_buffer.clear()
