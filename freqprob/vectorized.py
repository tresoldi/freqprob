"""Vectorized operations for efficient batch processing.

This module provides numpy-based implementations for batch scoring operations,
allowing efficient processing of large datasets without Python loops.
"""

from collections.abc import Iterable
from typing import Any

import numpy as np

from .base import Element, ScoringMethod


class VectorizedScorer:
    """Wrapper class that adds vectorized operations to any scoring method.

    This class wraps existing scoring methods to provide efficient batch
    operations using numpy arrays and vectorized computations.

    Parameters
    ----------
    scorer : ScoringMethod
        The underlying scoring method to wrap

    Examples:
    --------
    >>> from freqprob import MLE
    >>> scorer = MLE({'a': 3, 'b': 2, 'c': 1}, logprob=False)
    >>> vectorized = VectorizedScorer(scorer)
    >>> elements = ['a', 'b', 'c', 'd']
    >>> scores = vectorized.score_batch(elements)
    >>> scores
    array([0.5, 0.33333333, 0.16666667, 0.0])
    """

    def __init__(self, scorer: ScoringMethod):
        """Initialize vectorized scorer wrapper.

        Parameters
        ----------
        scorer : ScoringMethod
            Scoring method to wrap with vectorized operations
        """
        self.scorer = scorer
        self._element_cache: dict[Any, int] = {}
        self._prob_array: np.ndarray[Any, Any] = np.array([])
        self._default_prob: float = 0.0
        self._build_lookup_arrays()

    def _build_lookup_arrays(self) -> None:
        """Build lookup arrays for fast vectorized access."""
        if hasattr(self.scorer, "_prob") and self.scorer._prob:
            # Create element to index mapping
            elements = list(self.scorer._prob.keys())
            self._element_cache = {elem: idx for idx, elem in enumerate(elements)}

            # Create probability array
            probs = [self.scorer._prob[elem] for elem in elements]
            self._prob_array = np.array(probs, dtype=np.float64)

            # Store default probability for unseen elements
            self._default_prob = float(self.scorer._unobs)
        else:
            self._element_cache = {}
            self._prob_array = np.array([])
            self._default_prob = 0.0

    def score_batch(self, elements: list[Element] | np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Score a batch of elements efficiently using vectorized operations.

        Parameters
        ----------
        elements : Union[List[Element], np.ndarray]
            Elements to score

        Returns:
        -------
        np.ndarray
            Array of scores for the elements

        Examples:
        --------
        >>> scorer = MLE({'a': 3, 'b': 2}, logprob=False)
        >>> vectorized = VectorizedScorer(scorer)
        >>> vectorized.score_batch(['a', 'b', 'unknown'])
        array([0.6, 0.4, 0.0])
        """
        if isinstance(elements, np.ndarray):
            elements = elements.tolist()

        # Convert elements to indices, using -1 for unknown elements
        indices = np.array([self._element_cache.get(elem, -1) for elem in elements], dtype=np.int32)

        # Create result array
        result = np.full(len(elements), self._default_prob, dtype=np.float64)

        # Find known elements (indices >= 0)
        known_mask = indices >= 0
        if np.any(known_mask):
            known_indices = indices[known_mask]
            result[known_mask] = self._prob_array[known_indices]

        return result

    def score_parallel(self, element_batches: list[list[Element]]) -> list[np.ndarray[Any, Any]]:
        """Score multiple batches in parallel (when available).

        Parameters
        ----------
        element_batches : List[List[Element]]
            List of element batches to score

        Returns:
        -------
        List[np.ndarray]
            List of score arrays for each batch
        """
        # For now, this is a sequential implementation
        # In the future, could use multiprocessing or threading
        return [self.score_batch(batch) for batch in element_batches]

    def score_matrix(self, elements_2d: list[list[Element]]) -> np.ndarray[Any, Any]:
        """Score a 2D array of elements, returning a matrix of scores.

        Parameters
        ----------
        elements_2d : List[List[Element]]
            2D list of elements to score

        Returns:
        -------
        np.ndarray
            2D array of scores with shape (len(elements_2d), max_row_length)
        """
        if not elements_2d:
            return np.array([])

        # Find maximum row length for padding
        max_length = max(len(row) for row in elements_2d)

        # Create padded matrix
        result = np.full((len(elements_2d), max_length), self._default_prob, dtype=np.float64)

        for i, row in enumerate(elements_2d):
            if row:  # Skip empty rows
                scores = self.score_batch(row)
                result[i, : len(scores)] = scores

        return result

    def top_k_elements(self, k: int) -> tuple[list[Element], np.ndarray[Any, Any]]:
        """Get the top-k highest scoring elements.

        Parameters
        ----------
        k : int
            Number of top elements to return

        Returns:
        -------
        Tuple[List[Element], np.ndarray]
            Tuple of (elements, scores) for the top-k elements
        """
        if len(self._prob_array) == 0:
            return [], np.array([])

        # Get top-k indices
        top_indices = np.argpartition(self._prob_array, -k)[-k:]
        top_indices = top_indices[np.argsort(self._prob_array[top_indices])][::-1]

        # Get corresponding elements and scores
        elements = list(self._element_cache.keys())
        top_elements = [elements[i] for i in top_indices]
        top_scores = self._prob_array[top_indices]

        return top_elements, top_scores

    def percentile_scores(
        self, elements: list[Element], percentiles: list[float]
    ) -> np.ndarray[Any, Any]:
        """Compute percentile ranks for element scores.

        Parameters
        ----------
        elements : List[Element]
            Elements to compute percentiles for
        percentiles : List[float]
            Percentiles to compute (0-100)

        Returns:
        -------
        np.ndarray
            Percentile ranks for each element
        """
        scores = self.score_batch(elements)

        # Get all known scores for percentile calculation
        all_scores = self._prob_array
        if len(all_scores) == 0:
            return np.zeros(len(elements))

        # Compute percentile ranks
        percentile_ranks = np.zeros(len(scores))
        for i, score in enumerate(scores):
            rank = (all_scores < score).sum() / len(all_scores) * 100
            percentile_ranks[i] = rank

        return percentile_ranks


def create_vectorized_batch_scorer(scorers: dict[str, ScoringMethod]) -> "BatchScorer":
    """Create a batch scorer that can handle multiple scoring methods efficiently.

    Parameters
    ----------
    scorers : Dict[str, ScoringMethod]
        Dictionary mapping scorer names to scoring methods

    Returns:
    -------
    BatchScorer
        Batch scorer instance
    """
    return BatchScorer(scorers)


class BatchScorer:
    """Efficient batch scoring using multiple scoring methods.

    This class allows scoring elements using multiple methods simultaneously,
    with optimized operations for processing large batches of data.

    Parameters
    ----------
    scorers : Dict[str, ScoringMethod]
        Dictionary mapping scorer names to scoring methods

    Examples:
    --------
    >>> from freqprob import MLE, Laplace
    >>> scorers = {
    ...     'mle': MLE({'a': 3, 'b': 2}, logprob=False),
    ...     'laplace': Laplace({'a': 3, 'b': 2}, logprob=False)
    ... }
    >>> batch_scorer = BatchScorer(scorers)
    >>> results = batch_scorer.score_batch(['a', 'b', 'c'])
    >>> results['mle']  # MLE scores
    array([0.6, 0.4, 0.0])
    >>> results['laplace']  # Laplace scores
    array([0.5, 0.375, 0.125])
    """

    def __init__(self, scorers: dict[str, ScoringMethod]):
        """Initialize batch scorer with multiple scoring methods.

        Parameters
        ----------
        scorers : Dict[str, ScoringMethod]
            Dictionary mapping scorer names to scoring methods
        """
        self.scorers = scorers
        self.vectorized_scorers = {
            name: VectorizedScorer(scorer) for name, scorer in scorers.items()
        }

    def score_batch(self, elements: list[Element]) -> dict[str, np.ndarray[Any, Any]]:
        """Score elements using all configured scoring methods.

        Parameters
        ----------
        elements : List[Element]
            Elements to score

        Returns:
        -------
        Dict[str, np.ndarray]
            Dictionary mapping scorer names to score arrays
        """
        return {
            name: vectorized.score_batch(elements)
            for name, vectorized in self.vectorized_scorers.items()
        }

    def score_and_compare(self, elements: list[Element]) -> dict[str, Any]:
        """Score elements and provide comparison statistics.

        Parameters
        ----------
        elements : List[Element]
            Elements to score

        Returns:
        -------
        Dict[str, Any]
            Dictionary with scores and comparison statistics
        """
        scores = self.score_batch(elements)

        # Compute comparison statistics
        score_matrix = np.stack(list(scores.values()), axis=0)

        result = {
            "scores": scores,
            "mean_scores": np.mean(score_matrix, axis=0),
            "std_scores": np.std(score_matrix, axis=0),
            "min_scores": np.min(score_matrix, axis=0),
            "max_scores": np.max(score_matrix, axis=0),
        }

        # Add ranking information
        for name, score_array in scores.items():
            ranked_indices = np.argsort(score_array)[::-1]
            result[f"{name}_ranking"] = ranked_indices

        return result

    def benchmark_methods(
        self, elements: list[Element], num_iterations: int = 100
    ) -> dict[str, float]:
        """Benchmark the performance of different scoring methods.

        Parameters
        ----------
        elements : List[Element]
            Elements to use for benchmarking
        num_iterations : int, default=100
            Number of iterations for timing

        Returns:
        -------
        Dict[str, float]
            Dictionary mapping method names to average execution times
        """
        import time

        results = {}

        for name, vectorized in self.vectorized_scorers.items():
            start_time = time.time()
            for _ in range(num_iterations):
                vectorized.score_batch(elements)
            end_time = time.time()

            avg_time = (end_time - start_time) / num_iterations
            results[name] = avg_time

        return results


# Numpy array conversion utilities


def elements_to_numpy(elements: Iterable[str | int | float]) -> np.ndarray[Any, Any]:
    """Convert elements to numpy array with appropriate dtype.

    Parameters
    ----------
    elements : Iterable[Union[str, int, float]]
        Elements to convert

    Returns:
    -------
    np.ndarray
        Numpy array of elements
    """
    elem_list = list(elements)
    if not elem_list:
        return np.array([])

    # Try to determine appropriate dtype
    first_elem = elem_list[0]
    if isinstance(first_elem, str):
        return np.array(elem_list, dtype="U")  # Unicode string
    if isinstance(first_elem, int):
        return np.array(elem_list, dtype=np.int64)
    if isinstance(first_elem, float):
        return np.array(elem_list, dtype=np.float64)
    return np.array(elem_list, dtype=object)  # type: ignore[unreachable]


def scores_to_probabilities(log_scores: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Convert log scores to probabilities using numerically stable computation.

    Parameters
    ----------
    log_scores : np.ndarray
        Array of log scores

    Returns:
    -------
    np.ndarray
        Array of probabilities
    """
    # Use log-sum-exp trick for numerical stability
    max_score = np.max(log_scores)
    exp_scores = np.exp(log_scores - max_score)
    result: np.ndarray[Any, Any] = exp_scores / np.sum(exp_scores)
    return result


def normalize_scores(scores: np.ndarray[Any, Any], method: str = "softmax") -> np.ndarray[Any, Any]:
    """Normalize scores using various methods.

    Parameters
    ----------
    scores : np.ndarray
        Array of scores to normalize
    method : str, default='softmax'
        Normalization method ('softmax', 'minmax', 'zscore')

    Returns:
    -------
    np.ndarray
        Normalized scores
    """
    if method == "softmax":
        return scores_to_probabilities(scores)
    if method == "minmax":
        min_score, max_score = np.min(scores), np.max(scores)
        if max_score == min_score:
            result: np.ndarray[Any, Any] = np.ones_like(scores) / len(scores)
            return result
        result = (scores - min_score) / (max_score - min_score)
        return result
    if method == "zscore":
        mean_score, std_score = np.mean(scores), np.std(scores)
        if std_score == 0:
            return np.zeros_like(scores)
        result = (scores - mean_score) / std_score
        return result
    raise ValueError(f"Unknown normalization method: {method}")
