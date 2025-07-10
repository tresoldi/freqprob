"""Memory-efficient representations for large vocabularies.

This module provides compressed and memory-optimized data structures
for handling large-scale frequency distributions and probability models.
"""

import array
import gzip
import math
import pickle
import sys
from collections import defaultdict
from collections.abc import Iterator

from .base import Element, FrequencyDistribution


class CompressedFrequencyDistribution:
    """Memory-efficient compressed frequency distribution.

    This class uses various compression techniques to reduce memory usage
    for large vocabularies while maintaining fast access patterns.

    Techniques used:
    - Integer compression for counts
    - String interning for element storage
    - Sparse representation for zero counts
    - Optional quantization for approximate counts

    Parameters
    ----------
    quantization_levels : Optional[int]
        Number of quantization levels for count compression (None for exact)
    use_compression : bool, default=True
        Whether to use data compression
    intern_strings : bool, default=True
        Whether to intern string elements for memory efficiency

    Examples:
    --------
    >>> compressed_dist = CompressedFrequencyDistribution()
    >>> compressed_dist.update({'word1': 1000, 'word2': 500, 'word3': 1})
    >>> compressed_dist.get_count('word1')
    1000
    >>> compressed_dist.get_memory_usage()
    """

    def __init__(
        self,
        quantization_levels: int | None = None,
        use_compression: bool = True,
        intern_strings: bool = True,
    ):
        """Initialize compressed frequency distribution."""
        self.quantization_levels = quantization_levels
        self.use_compression = use_compression
        self.intern_strings = intern_strings

        # Core data structures
        self._element_to_id: dict[Element, int] = {}
        self._id_to_element: dict[int, Element] = {}
        self._next_id = 0

        # Compressed count storage
        if quantization_levels:
            # Use smaller integer type for quantized counts
            self._counts = array.array("H")  # unsigned short (16-bit)
            self._max_quantized_value = quantization_levels - 1
        else:
            # Use 32-bit integers for exact counts
            self._counts = array.array("I")  # unsigned int (32-bit)

        # Statistics
        self._total_count = 0
        self._original_count_range = (0, 0)  # For quantization

        # String interning cache
        if intern_strings:
            self._string_cache: dict[str, str] = {}

    def _intern_element(self, element: Element) -> Element:
        """Intern element to save memory if it's a string."""
        if self.intern_strings and isinstance(element, str):
            if element not in self._string_cache:
                self._string_cache[element] = sys.intern(element)
            return self._string_cache[element]
        return element

    def _get_element_id(self, element: Element) -> int:
        """Get or create ID for element."""
        element = self._intern_element(element)
        if element not in self._element_to_id:
            element_id = self._next_id
            self._element_to_id[element] = element_id
            self._id_to_element[element_id] = element
            self._next_id += 1

            # Extend counts array if needed
            while len(self._counts) <= element_id:
                self._counts.append(0)

            return element_id
        return self._element_to_id[element]

    def _quantize_count(self, count: int) -> int:
        """Quantize count if quantization is enabled."""
        if not self.quantization_levels:
            return count

        if self._original_count_range[1] == 0:
            return 0  # No range established yet

        # Linear quantization
        range_size = self._original_count_range[1] - self._original_count_range[0]
        if range_size == 0:
            return 0

        normalized = (count - self._original_count_range[0]) / range_size
        quantized = int(normalized * self._max_quantized_value)
        return max(0, min(quantized, self._max_quantized_value))

    def _dequantize_count(self, quantized_count: int) -> int:
        """Convert quantized count back to approximate original count."""
        if not self.quantization_levels:
            return quantized_count

        if self._original_count_range[1] == 0:
            return 0

        # Linear dequantization
        range_size = self._original_count_range[1] - self._original_count_range[0]
        normalized = quantized_count / self._max_quantized_value
        return int(self._original_count_range[0] + normalized * range_size)

    def update(self, freqdist: FrequencyDistribution) -> None:
        """Update with a frequency distribution.

        Parameters
        ----------
        freqdist : FrequencyDistribution
            Frequency distribution to add
        """
        # Update count range for quantization
        if freqdist and self.quantization_levels:
            counts = list(freqdist.values())
            min_count, max_count = min(counts), max(counts)

            if self._original_count_range[1] == 0:
                self._original_count_range = (min_count, max_count)
            else:
                self._original_count_range = (
                    min(self._original_count_range[0], min_count),
                    max(self._original_count_range[1], max_count),
                )

        # Add elements and counts
        for element, count in freqdist.items():
            element_id = self._get_element_id(element)
            quantized_count = self._quantize_count(count)

            if element_id < len(self._counts):
                self._counts[element_id] += quantized_count
            else:
                # This shouldn't happen due to array extension in _get_element_id
                while len(self._counts) <= element_id:
                    self._counts.append(0)
                self._counts[element_id] = quantized_count

            self._total_count += count  # Keep exact total

    def get_count(self, element: Element) -> int:
        """Get count for an element.

        Parameters
        ----------
        element : Element
            Element to query

        Returns:
        -------
        int
            Count for the element (dequantized if quantization is used)
        """
        element = self._intern_element(element)
        if element not in self._element_to_id:
            return 0

        element_id = self._element_to_id[element]
        if element_id >= len(self._counts):
            return 0

        quantized_count = self._counts[element_id]
        return self._dequantize_count(quantized_count)

    def get_total_count(self) -> int:
        """Get total count across all elements."""
        return self._total_count

    def get_vocabulary_size(self) -> int:
        """Get vocabulary size."""
        return len(self._element_to_id)

    def items(self) -> Iterator[tuple[Element, int]]:
        """Iterate over (element, count) pairs."""
        for element_id, element in self._id_to_element.items():
            if element_id < len(self._counts) and self._counts[element_id] > 0:
                count = self._dequantize_count(self._counts[element_id])
                yield element, count

    def keys(self) -> Iterator[Element]:
        """Iterate over elements with non-zero counts."""
        for element_id, element in self._id_to_element.items():
            if element_id < len(self._counts) and self._counts[element_id] > 0:
                yield element

    def values(self) -> Iterator[int]:
        """Iterate over non-zero counts."""
        for element_id in range(len(self._counts)):
            if self._counts[element_id] > 0:
                yield self._dequantize_count(self._counts[element_id])

    def to_dict(self) -> dict[Element, int]:
        """Convert to regular dictionary."""
        return dict(self.items())

    def get_memory_usage(self) -> dict[str, int]:
        """Get detailed memory usage information.

        Returns:
        -------
        Dict[str, int]
            Memory usage breakdown in bytes
        """
        element_to_id_size = sys.getsizeof(self._element_to_id)
        for k, v in self._element_to_id.items():
            element_to_id_size += sys.getsizeof(k) + sys.getsizeof(v)

        id_to_element_size = sys.getsizeof(self._id_to_element)
        for elem_id, element in self._id_to_element.items():
            id_to_element_size += sys.getsizeof(elem_id) + sys.getsizeof(element)

        counts_size = self._counts.buffer_info()[1] * self._counts.itemsize

        string_cache_size = 0
        if self.intern_strings:
            string_cache_size = sys.getsizeof(self._string_cache)
            for key, value in self._string_cache.items():
                string_cache_size += sys.getsizeof(key) + sys.getsizeof(value)

        return {
            "element_to_id_mapping": element_to_id_size,
            "id_to_element_mapping": id_to_element_size,
            "counts_array": counts_size,
            "string_cache": string_cache_size,
            "total": element_to_id_size + id_to_element_size + counts_size + string_cache_size,
        }

    def compress_to_bytes(self) -> bytes:
        """Compress the entire distribution to bytes.

        Returns:
        -------
        bytes
            Compressed representation
        """
        data = {
            "element_to_id": self._element_to_id,
            "id_to_element": self._id_to_element,
            "counts": self._counts.tolist(),
            "total_count": self._total_count,
            "quantization_levels": self.quantization_levels,
            "original_count_range": self._original_count_range,
            "next_id": self._next_id,
        }

        serialized = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)

        if self.use_compression:
            return gzip.compress(serialized)
        return serialized

    @classmethod
    def decompress_from_bytes(
        cls, compressed_data: bytes, use_compression: bool = True
    ) -> "CompressedFrequencyDistribution":
        """Decompress distribution from bytes.

        Parameters
        ----------
        compressed_data : bytes
            Compressed data
        use_compression : bool, default=True
            Whether the data is compressed

        Returns:
        -------
        CompressedFrequencyDistribution
            Decompressed distribution
        """
        serialized = gzip.decompress(compressed_data) if use_compression else compressed_data

        data = pickle.loads(serialized)

        instance = cls(
            quantization_levels=data["quantization_levels"], use_compression=use_compression
        )

        instance._element_to_id = data["element_to_id"]
        instance._id_to_element = data["id_to_element"]
        instance._total_count = data["total_count"]
        instance._original_count_range = data["original_count_range"]
        instance._next_id = data["next_id"]

        # Reconstruct counts array
        if data["quantization_levels"]:
            instance._counts = array.array("H", data["counts"])
        else:
            instance._counts = array.array("I", data["counts"])

        return instance


class SparseFrequencyDistribution:
    """Sparse representation for frequency distributions with many zero counts.

    This class is optimized for distributions where most elements have zero
    counts, using sparse data structures for memory efficiency.

    Parameters
    ----------
    default_count : int, default=0
        Default count for unobserved elements
    use_sorted_storage : bool, default=True
        Whether to keep elements sorted by frequency for faster access

    Examples:
    --------
    >>> sparse_dist = SparseFrequencyDistribution()
    >>> sparse_dist.update({'rare_word': 1, 'common_word': 1000})
    >>> sparse_dist.get_count('rare_word')
    1
    >>> sparse_dist.get_top_k(5)  # Get top 5 most frequent
    """

    def __init__(self, default_count: int = 0, use_sorted_storage: bool = True):
        """Initialize sparse frequency distribution."""
        self.default_count = default_count
        self.use_sorted_storage = use_sorted_storage

        # Core sparse storage: only store non-zero (or non-default) counts
        self._sparse_counts: dict[Element, int] = {}
        self._total_count = 0

        # Optional sorted storage for fast frequency-based queries
        if use_sorted_storage:
            self._sorted_elements: list[tuple[int, Element]] = []  # (count, element)
            self._needs_resort = False

    def update(self, freqdist: FrequencyDistribution) -> None:
        """Update with a frequency distribution.

        Parameters
        ----------
        freqdist : FrequencyDistribution
            Frequency distribution to add
        """
        for element, count in freqdist.items():
            if count != self.default_count:
                if element in self._sparse_counts:
                    self._sparse_counts[element] += count
                else:
                    self._sparse_counts[element] = count

                self._total_count += count

                if self.use_sorted_storage:
                    self._needs_resort = True

    def increment(self, element: Element, count: int = 1) -> None:
        """Increment count for a single element.

        Parameters
        ----------
        element : Element
            Element to increment
        count : int, default=1
            Count increment
        """
        if element in self._sparse_counts:
            self._sparse_counts[element] += count
        else:
            self._sparse_counts[element] = self.default_count + count

        self._total_count += count

        if self.use_sorted_storage:
            self._needs_resort = True

    def get_count(self, element: Element) -> int:
        """Get count for an element.

        Parameters
        ----------
        element : Element
            Element to query

        Returns:
        -------
        int
            Count for the element
        """
        return self._sparse_counts.get(element, self.default_count)

    def get_total_count(self) -> int:
        """Get total count across all elements."""
        return self._total_count

    def get_vocabulary_size(self) -> int:
        """Get number of elements with non-default counts."""
        return len(self._sparse_counts)

    def _ensure_sorted(self) -> None:
        """Ensure sorted storage is up to date."""
        if self.use_sorted_storage and self._needs_resort:
            self._sorted_elements = [
                (count, element) for element, count in self._sparse_counts.items()
            ]
            self._sorted_elements.sort(reverse=True)  # Highest count first
            self._needs_resort = False

    def get_top_k(self, k: int) -> list[tuple[Element, int]]:
        """Get top-k most frequent elements.

        Parameters
        ----------
        k : int
            Number of top elements to return

        Returns:
        -------
        List[Tuple[Element, int]]
            List of (element, count) pairs
        """
        if not self.use_sorted_storage:
            # Sort on demand
            sorted_items = sorted(self._sparse_counts.items(), key=lambda x: x[1], reverse=True)
            return sorted_items[:k]

        self._ensure_sorted()
        return [(element, count) for count, element in self._sorted_elements[:k]]

    def get_elements_with_count_range(self, min_count: int, max_count: int) -> list[Element]:
        """Get elements with counts in a specific range.

        Parameters
        ----------
        min_count : int
            Minimum count (inclusive)
        max_count : int
            Maximum count (inclusive)

        Returns:
        -------
        List[Element]
            Elements with counts in the specified range
        """
        result = []
        for element, count in self._sparse_counts.items():
            if min_count <= count <= max_count:
                result.append(element)
        return result

    def get_count_histogram(self) -> dict[int, int]:
        """Get histogram of counts (count -> frequency of that count).

        Returns:
        -------
        Dict[int, int]
            Histogram mapping counts to their frequencies
        """
        histogram: dict[int, int] = defaultdict(int)
        for count in self._sparse_counts.values():
            histogram[count] += 1
        return dict(histogram)

    def items(self) -> Iterator[tuple[Element, int]]:
        """Iterate over (element, count) pairs for non-default elements."""
        return iter(self._sparse_counts.items())

    def keys(self) -> Iterator[Element]:
        """Iterate over elements with non-default counts."""
        return iter(self._sparse_counts.keys())

    def values(self) -> Iterator[int]:
        """Iterate over non-default counts."""
        return iter(self._sparse_counts.values())

    def to_dict(self) -> dict[Element, int]:
        """Convert to regular dictionary."""
        return dict(self._sparse_counts)

    def get_memory_usage(self) -> dict[str, int]:
        """Get memory usage information.

        Returns:
        -------
        Dict[str, int]
            Memory usage breakdown in bytes
        """
        sparse_counts_size = sys.getsizeof(self._sparse_counts)
        for k, v in self._sparse_counts.items():
            sparse_counts_size += sys.getsizeof(k) + sys.getsizeof(v)

        sorted_elements_size = 0
        if self.use_sorted_storage:
            sorted_elements_size = sys.getsizeof(self._sorted_elements)
            for item in self._sorted_elements:
                sorted_elements_size += sys.getsizeof(item)

        return {
            "sparse_counts": sparse_counts_size,
            "sorted_elements": sorted_elements_size,
            "total": sparse_counts_size + sorted_elements_size,
        }


class QuantizedProbabilityTable:
    """Quantized probability table for memory-efficient probability storage.

    This class stores probabilities using quantization to reduce memory usage
    while maintaining reasonable precision for most applications.

    Parameters
    ----------
    num_quantization_levels : int, default=65536
        Number of quantization levels (determines precision)
    log_space : bool, default=True
        Whether to quantize in log space for better precision

    Examples:
    --------
    >>> prob_table = QuantizedProbabilityTable(num_quantization_levels=1024)
    >>> prob_table.set_probabilities({'word1': 0.5, 'word2': 0.3, 'word3': 0.2})
    >>> prob_table.get_probability('word1')
    0.5004884004884005
    """

    def __init__(self, num_quantization_levels: int = 65536, log_space: bool = True):
        """Initialize quantized probability table."""
        self.num_quantization_levels = num_quantization_levels
        self.log_space = log_space

        # Use 16-bit integers for quantized probabilities
        self._quantized_probs: dict[Element, int] = {}

        # Quantization parameters
        if log_space:
            # Log space: quantize log probabilities
            self._min_log_prob = -20.0  # Very small probability
            self._max_log_prob = 0.0  # Maximum probability (log(1) = 0)
            self._log_range = self._max_log_prob - self._min_log_prob
        else:
            # Linear space: quantize probabilities directly
            self._min_prob = 0.0
            self._max_prob = 1.0
            self._prob_range = self._max_prob - self._min_prob

        # Default probability for unobserved elements
        self._default_quantized_prob = 0  # Minimum quantization level

    def _quantize_probability(self, prob: float) -> int:
        """Quantize a probability value.

        Parameters
        ----------
        prob : float
            Probability to quantize

        Returns:
        -------
        int
            Quantized probability
        """
        if prob <= 0:
            return 0

        if self.log_space:
            log_prob = math.log(prob)
            log_prob = max(self._min_log_prob, min(self._max_log_prob, log_prob))
            normalized = (log_prob - self._min_log_prob) / self._log_range
        else:
            prob = max(self._min_prob, min(self._max_prob, prob))
            normalized = (prob - self._min_prob) / self._prob_range

        quantized = int(normalized * (self.num_quantization_levels - 1))
        return max(0, min(quantized, self.num_quantization_levels - 1))

    def _dequantize_probability(self, quantized_prob: int) -> float:
        """Dequantize a probability value.

        Parameters
        ----------
        quantized_prob : int
            Quantized probability

        Returns:
        -------
        float
            Dequantized probability
        """
        if quantized_prob <= 0:
            return math.exp(self._min_log_prob) if self.log_space else self._min_prob

        normalized = quantized_prob / (self.num_quantization_levels - 1)

        if self.log_space:
            log_prob = self._min_log_prob + normalized * self._log_range
            return math.exp(log_prob)
        return self._min_prob + normalized * self._prob_range

    def set_probabilities(self, probabilities: dict[Element, float]) -> None:
        """Set probabilities for multiple elements.

        Parameters
        ----------
        probabilities : Dict[Element, float]
            Dictionary mapping elements to their probabilities
        """
        self._quantized_probs.clear()
        for element, prob in probabilities.items():
            self._quantized_probs[element] = self._quantize_probability(prob)

    def set_probability(self, element: Element, probability: float) -> None:
        """Set probability for a single element.

        Parameters
        ----------
        element : Element
            Element to set probability for
        probability : float
            Probability value
        """
        self._quantized_probs[element] = self._quantize_probability(probability)

    def get_probability(self, element: Element) -> float:
        """Get probability for an element.

        Parameters
        ----------
        element : Element
            Element to query

        Returns:
        -------
        float
            Dequantized probability
        """
        quantized = self._quantized_probs.get(element, self._default_quantized_prob)
        return self._dequantize_probability(quantized)

    def set_default_probability(self, default_prob: float) -> None:
        """Set default probability for unobserved elements.

        Parameters
        ----------
        default_prob : float
            Default probability
        """
        self._default_quantized_prob = self._quantize_probability(default_prob)

    def get_elements(self) -> list[Element]:
        """Get all elements with stored probabilities."""
        return list(self._quantized_probs.keys())

    def get_memory_usage(self) -> dict[str, int]:
        """Get memory usage information.

        Returns:
        -------
        Dict[str, int]
            Memory usage breakdown in bytes
        """
        quantized_probs_size = sys.getsizeof(self._quantized_probs)
        for k, v in self._quantized_probs.items():
            quantized_probs_size += sys.getsizeof(k) + sys.getsizeof(v)

        return {"quantized_probabilities": quantized_probs_size, "total": quantized_probs_size}

    def get_quantization_error_stats(
        self, original_probs: dict[Element, float]
    ) -> dict[str, float]:
        """Analyze quantization error compared to original probabilities.

        Parameters
        ----------
        original_probs : Dict[Element, float]
            Original probabilities for comparison

        Returns:
        -------
        Dict[str, float]
            Error statistics
        """
        errors = []
        relative_errors = []

        for element, original_prob in original_probs.items():
            if element in self._quantized_probs:
                quantized_prob = self.get_probability(element)
                error = abs(original_prob - quantized_prob)
                errors.append(error)

                if original_prob > 0:
                    relative_error = error / original_prob
                    relative_errors.append(relative_error)

        if not errors:
            return {
                "mean_absolute_error": 0.0,
                "max_absolute_error": 0.0,
                "mean_relative_error": 0.0,
                "max_relative_error": 0.0,
            }

        return {
            "mean_absolute_error": sum(errors) / len(errors),
            "max_absolute_error": max(errors),
            "mean_relative_error": (
                sum(relative_errors) / len(relative_errors) if relative_errors else 0.0
            ),
            "max_relative_error": max(relative_errors) if relative_errors else 0.0,
            "num_elements": len(errors),
        }


def memory_usage_comparison(
    original_dict: dict[Element, int],
    compressed_dist: CompressedFrequencyDistribution,
    sparse_dist: SparseFrequencyDistribution,
) -> dict[str, dict[str, int]]:
    """Compare memory usage between different representations.

    Parameters
    ----------
    original_dict : Dict[Element, int]
        Original dictionary representation
    compressed_dist : CompressedFrequencyDistribution
        Compressed distribution
    sparse_dist : SparseFrequencyDistribution
        Sparse distribution

    Returns:
    -------
    Dict[str, Dict[str, int]]
        Memory usage comparison
    """
    # Calculate original dictionary size
    original_size = sys.getsizeof(original_dict)
    for k, v in original_dict.items():
        original_size += sys.getsizeof(k) + sys.getsizeof(v)

    return {
        "original_dict": {"total": original_size},
        "compressed": compressed_dist.get_memory_usage(),
        "sparse": sparse_dist.get_memory_usage(),
    }


# Utility functions for creating memory-efficient representations


def create_compressed_distribution(
    freqdist: FrequencyDistribution,
    quantization_levels: int | None = None,
    use_compression: bool = True,
) -> CompressedFrequencyDistribution:
    """Create a compressed frequency distribution from a regular one.

    Parameters
    ----------
    freqdist : FrequencyDistribution
        Original frequency distribution
    quantization_levels : Optional[int]
        Number of quantization levels for compression
    use_compression : bool, default=True
        Whether to use data compression

    Returns:
    -------
    CompressedFrequencyDistribution
        Compressed distribution
    """
    compressed = CompressedFrequencyDistribution(
        quantization_levels=quantization_levels, use_compression=use_compression
    )
    compressed.update(freqdist)
    return compressed


def create_sparse_distribution(
    freqdist: FrequencyDistribution, default_count: int = 0
) -> SparseFrequencyDistribution:
    """Create a sparse frequency distribution from a regular one.

    Parameters
    ----------
    freqdist : FrequencyDistribution
        Original frequency distribution
    default_count : int, default=0
        Default count for unobserved elements

    Returns:
    -------
    SparseFrequencyDistribution
        Sparse distribution
    """
    sparse = SparseFrequencyDistribution(default_count=default_count)
    sparse.update(freqdist)
    return sparse
