"""Caching and memoization utilities for computational efficiency.

This module provides caching mechanisms to speed up expensive calculations
in probability smoothing methods, particularly for computationally intensive
methods like Simple Good-Turing.
"""

import hashlib
import pickle
from collections.abc import Callable
from functools import wraps
from typing import Any

from .base import FrequencyDistribution


class ComputationCache:
    """Cache for expensive computations in scoring methods.

    This cache stores the results of computationally intensive operations
    to avoid redundant calculations when the same parameters are used.

    Attributes:
    ----------
    _cache : Dict[str, Any]
        Internal cache storage mapping cache keys to computed results
    max_size : int | None
        Maximum number of entries to keep in cache (None for unlimited)
    """

    def __init__(self, max_size: int | None = 1000):
        """Initialize the computation cache.

        Parameters
        ----------
        max_size : int | None, default=1000
            Maximum number of cache entries. If None, cache has no size limit.
        """
        self._cache: dict[str, Any] = {}
        self.max_size = max_size

    def _generate_key(self, freqdist: FrequencyDistribution, **kwargs: Any) -> str:
        """Generate a unique cache key for the given parameters.

        Parameters
        ----------
        freqdist : FrequencyDistribution
            Frequency distribution to hash
        **kwargs
            Additional parameters to include in the hash

        Returns:
        -------
        str
            Unique cache key for the parameters
        """
        # Create a deterministic representation of the input
        sorted_items = sorted(freqdist.items())
        sorted_kwargs = sorted(kwargs.items())

        # Create a hash of the parameters
        hash_input = pickle.dumps((sorted_items, sorted_kwargs), protocol=pickle.HIGHEST_PROTOCOL)
        return hashlib.sha256(hash_input).hexdigest()

    def get(self, freqdist: FrequencyDistribution, **kwargs: Any) -> Any | None:
        """Retrieve cached result for the given parameters.

        Parameters
        ----------
        freqdist : FrequencyDistribution
            Frequency distribution
        **kwargs
            Additional parameters

        Returns:
        -------
        Optional[Any]
            Cached result if available, None otherwise
        """
        key = self._generate_key(freqdist, **kwargs)
        return self._cache.get(key)

    def set(self, freqdist: FrequencyDistribution, result: Any, **kwargs: Any) -> None:
        """Store result in cache for the given parameters.

        Parameters
        ----------
        freqdist : FrequencyDistribution
            Frequency distribution
        result : Any
            Result to cache
        **kwargs
            Additional parameters
        """
        key = self._generate_key(freqdist, **kwargs)

        # Implement LRU-style eviction if cache is full
        if self.max_size is not None and len(self._cache) >= self.max_size:
            # Remove oldest entry (first in dict in Python 3.7+)
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]

        self._cache[key] = result

    def clear(self) -> None:
        """Clear all cached entries."""
        self._cache.clear()

    def size(self) -> int:
        """Return the current number of cached entries."""
        return len(self._cache)


# Global cache instances for different types of computations
_sgt_cache = ComputationCache(max_size=100)  # Smaller cache for SGT due to large results
_general_cache = ComputationCache(max_size=1000)


def cached_sgt_computation(
    func: Callable[[Any, FrequencyDistribution], None],
) -> Callable[[Any, FrequencyDistribution], None]:
    """Cache Simple Good-Turing computations.

    Parameters
    ----------
    func : Callable
        Function to cache (should be _compute_probabilities method)

    Returns:
    -------
    Callable
        Wrapped function with caching
    """

    @wraps(func)
    def wrapper(self: Any, freqdist: FrequencyDistribution) -> None:
        # Create cache key using configuration parameters
        config_params = {
            "p_value": getattr(self.config, "p_value", None),
            "default_p0": getattr(self.config, "default_p0", None),
            "allow_fail": getattr(self.config, "allow_fail", None),
            "logprob": self.logprob,
        }

        # Try to get cached result
        cached_result = _sgt_cache.get(freqdist, **config_params)
        if cached_result is not None:
            # Restore cached probability distributions
            self._prob, self._unobs = cached_result
            return

        # Compute if not cached
        func(self, freqdist)

        # Cache the result (deep copy to avoid reference issues)
        result = (dict(self._prob), self._unobs)
        _sgt_cache.set(freqdist, result, **config_params)

    return wrapper


def cached_computation(
    cache_instance: ComputationCache | None = None,
) -> Callable[
    [Callable[[Any, FrequencyDistribution], None]], Callable[[Any, FrequencyDistribution], None]
]:
    """Cache expensive computations.

    Parameters
    ----------
    cache_instance : ComputationCache, optional
        Cache instance to use. If None, uses global general cache.

    Returns:
    -------
    Callable
        Decorator function
    """
    if cache_instance is None:
        cache_instance = _general_cache

    def decorator(
        func: Callable[[Any, FrequencyDistribution], None],
    ) -> Callable[[Any, FrequencyDistribution], None]:
        @wraps(func)
        def wrapper(self: Any, freqdist: FrequencyDistribution) -> None:
            # Create cache key using all configuration parameters
            config_params = {}
            if hasattr(self, "config"):
                config_params = {
                    attr: getattr(self.config, attr, None)
                    for attr in dir(self.config)
                    if not attr.startswith("_")
                }
            config_params["logprob"] = self.logprob

            # Try to get cached result
            cached_result = cache_instance.get(freqdist, **config_params)
            if cached_result is not None:
                # Restore cached probability distributions
                self._prob, self._unobs = cached_result
                return

            # Compute if not cached
            func(self, freqdist)

            # Cache the result
            result = (dict(self._prob), self._unobs)
            cache_instance.set(freqdist, result, **config_params)

        return wrapper

    return decorator


def clear_all_caches() -> None:
    """Clear all global caches."""
    _sgt_cache.clear()
    _general_cache.clear()


def get_cache_stats() -> dict[str, int]:
    """Get statistics about cache usage.

    Returns:
    -------
    Dict[str, int]
        Dictionary with cache statistics
    """
    return {"sgt_cache_size": _sgt_cache.size(), "general_cache_size": _general_cache.size()}


class MemoizedProperty:
    """Property decorator that caches the result of expensive property calculations.

    This is useful for properties that perform expensive computations but
    should behave like normal attributes.
    """

    def __init__(self, func: Callable[[Any], Any]):
        """Initialize memoized property.

        Parameters
        ----------
        func : Callable
            Function to memoize
        """
        self.func = func
        self.attrname: str | None = None
        self.__doc__ = func.__doc__

    def __set_name__(self, owner: Any, name: str) -> None:
        """Set the name when used as a descriptor."""
        if self.attrname is None:
            self.attrname = name
        elif name != self.attrname:
            raise RuntimeError(
                f"Cannot assign the same memoized_property to two different names "
                f"({self.attrname!r} and {name!r})."
            )

    def __get__(self, instance: Any, owner: Any = None) -> Any:
        """Get the cached property value."""
        if instance is None:
            return self
        if self.attrname is None:
            raise TypeError(
                "Cannot use memoized_property instance without calling __set_name__ on it."
            )
        try:
            cache = instance.__dict__
        except AttributeError:
            msg = (
                f"No '__dict__' attribute on {type(instance).__name__!r} "
                f"instance to cache {self.attrname!r} property."
            )
            raise TypeError(msg) from None
        val = cache.get(self.attrname, None)
        if val is None:
            with _get_cache_lock(instance):
                # Check again after acquiring lock
                val = cache.get(self.attrname, None)
                if val is None:
                    val = self.func(instance)
                    cache[self.attrname] = val
        return val


def _get_cache_lock(instance: Any) -> Any:
    """Get a lock for thread-safe caching (simplified implementation)."""
    # For simplicity, we'll use a basic approach
    # In production, you might want a more sophisticated locking mechanism
    if not hasattr(instance, "_cache_lock"):
        import threading

        instance._cache_lock = threading.RLock()
    return instance._cache_lock
