"""Memory profiling and monitoring utilities.

This module provides tools for monitoring memory usage, profiling performance,
and analyzing memory efficiency of different frequency distribution representations.
"""

import gc
import sys
import time
import tracemalloc
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
from typing import Any, Protocol, cast

try:
    import psutil

    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

from .base import FrequencyDistribution, ScoringMethod


class ProfiledFunction(Protocol):
    """Protocol for functions decorated with profile_memory_usage."""

    _profiler: Any

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Call the profiled function."""
        ...

    def get_profiler(self) -> Any:
        """Get the memory profiler instance."""
        ...


@dataclass
class MemorySnapshot:
    """Memory usage snapshot.

    Attributes:
    ----------
    timestamp : float
        Time when snapshot was taken
    rss_mb : float
        Resident Set Size in MB
    vms_mb : float
        Virtual Memory Size in MB
    python_objects_mb : float
        Memory used by Python objects in MB
    peak_mb : Optional[float]
        Peak memory usage since last reset (if available)
    """

    timestamp: float

    rss_mb: float
    vms_mb: float
    python_objects_mb: float
    peak_mb: float | None = None


@dataclass
class PerformanceMetrics:
    """Performance measurement results.

    Attributes:
    ----------
    operation_name : str
        Name of the operation
    execution_time : float
        Execution time in seconds
    memory_before : MemorySnapshot
        Memory usage before operation
    memory_after : MemorySnapshot
        Memory usage after operation
    memory_peak : Optional[float]
        Peak memory usage during operation (if available)
    """

    operation_name: str

    execution_time: float
    memory_before: MemorySnapshot
    memory_after: MemorySnapshot
    memory_peak: float | None = None

    @property
    def memory_delta_mb(self) -> float:
        """Memory change in MB."""
        return self.memory_after.rss_mb - self.memory_before.rss_mb

    @property
    def python_objects_delta_mb(self) -> float:
        """Python objects memory change in MB."""
        return self.memory_after.python_objects_mb - self.memory_before.python_objects_mb


class MemoryProfiler:
    """Memory profiler for analyzing memory usage patterns.

    This class provides utilities for monitoring memory usage during
    various operations and analyzing memory efficiency.

    Parameters
    ----------
    enable_tracemalloc : bool, default=True
        Whether to enable detailed Python memory tracing
    snapshot_interval : float, default=1.0
        Interval in seconds for automatic snapshots

    Examples:
    --------
    >>> profiler = MemoryProfiler()
    >>> with profiler.profile_operation("test_operation"):
    ...     # Your operation here
    ...     pass
    >>> metrics = profiler.get_latest_metrics()
    >>> print(f"Memory used: {metrics.memory_delta_mb:.2f} MB")
    """

    def __init__(self, enable_tracemalloc: bool = True, snapshot_interval: float = 1.0):
        """Initialize memory profiler."""
        self.enable_tracemalloc = enable_tracemalloc

        self.snapshot_interval = snapshot_interval
        self._process = psutil.Process() if HAS_PSUTIL else None
        self._snapshots: list[MemorySnapshot] = []
        self._metrics: list[PerformanceMetrics] = []

        if enable_tracemalloc and not tracemalloc.is_tracing():
            tracemalloc.start()

    def take_snapshot(self) -> MemorySnapshot:
        """Take a memory usage snapshot.

        Returns:
        -------
        MemorySnapshot
            Current memory usage snapshot
        """
        # Get process memory info
        if self._process is not None:
            memory_info = self._process.memory_info()
            rss_mb = memory_info.rss / 1024 / 1024
            vms_mb = memory_info.vms / 1024 / 1024
        else:
            rss_mb = 0.0
            vms_mb = 0.0

        # Get Python object memory usage
        python_objects_mb = 0.0
        peak_mb = None

        if self.enable_tracemalloc and tracemalloc.is_tracing():
            current, peak = tracemalloc.get_traced_memory()
            python_objects_mb = current / 1024 / 1024
            peak_mb = peak / 1024 / 1024

        snapshot = MemorySnapshot(
            timestamp=time.time(),
            rss_mb=rss_mb,
            vms_mb=vms_mb,
            python_objects_mb=python_objects_mb,
            peak_mb=peak_mb,
        )

        self._snapshots.append(snapshot)
        return snapshot

    @contextmanager
    def profile_operation(self, operation_name: str) -> Iterator[None]:
        """Context manager for profiling an operation.

        Parameters
        ----------
        operation_name : str
            Name of the operation being profiled

        Yields:
        ------
        None
        """
        # Reset tracemalloc peak if enabled
        if self.enable_tracemalloc and tracemalloc.is_tracing():
            tracemalloc.reset_peak()

        # Take before snapshot
        memory_before = self.take_snapshot()
        start_time = time.time()

        try:
            yield
        finally:
            # Take after snapshot
            end_time = time.time()
            memory_after = self.take_snapshot()

            # Get peak memory if available
            memory_peak = None
            if self.enable_tracemalloc and tracemalloc.is_tracing():
                _, peak = tracemalloc.get_traced_memory()
                memory_peak = peak / 1024 / 1024

            # Create metrics
            metrics = PerformanceMetrics(
                operation_name=operation_name,
                execution_time=end_time - start_time,
                memory_before=memory_before,
                memory_after=memory_after,
                memory_peak=memory_peak,
            )

            self._metrics.append(metrics)

    def get_latest_metrics(self) -> PerformanceMetrics | None:
        """Get the latest performance metrics.

        Returns:
        -------
        Optional[PerformanceMetrics]
            Latest metrics or None if no operations have been profiled
        """
        return self._metrics[-1] if self._metrics else None

    def get_all_metrics(self) -> list[PerformanceMetrics]:
        """Get all performance metrics.

        Returns:
        -------
        List[PerformanceMetrics]
            All recorded metrics
        """
        return self._metrics.copy()

    def get_snapshots(self) -> list[MemorySnapshot]:
        """Get all memory snapshots.

        Returns:
        -------
        List[MemorySnapshot]
            All recorded snapshots
        """
        return self._snapshots.copy()

    def clear_history(self) -> None:
        """Clear all recorded metrics and snapshots."""
        self._metrics.clear()

        self._snapshots.clear()

    def get_memory_summary(self) -> dict[str, Any]:
        """Get a summary of memory usage patterns.

        Returns:
        -------
        Dict[str, Any]
            Memory usage summary
        """
        if not self._snapshots:
            return {"error": "No snapshots available"}

        rss_values = [s.rss_mb for s in self._snapshots]
        python_values = [s.python_objects_mb for s in self._snapshots]

        return {
            "total_snapshots": len(self._snapshots),
            "time_range": {
                "start": self._snapshots[0].timestamp,
                "end": self._snapshots[-1].timestamp,
                "duration": self._snapshots[-1].timestamp - self._snapshots[0].timestamp,
            },
            "rss_memory": {
                "current_mb": rss_values[-1],
                "min_mb": min(rss_values),
                "max_mb": max(rss_values),
                "avg_mb": sum(rss_values) / len(rss_values),
            },
            "python_objects": {
                "current_mb": python_values[-1],
                "min_mb": min(python_values),
                "max_mb": max(python_values),
                "avg_mb": sum(python_values) / len(python_values),
            },
        }


def profile_memory_usage(
    operation_name: str | None = None,
) -> Callable[[Callable[..., Any]], ProfiledFunction]:
    """Decorate functions to profile memory usage.

    Parameters
    ----------
    operation_name : str, optional
        Name for the operation (defaults to function name)

    Returns:
    -------
    function
        Decorated function

    Examples:
    --------
    >>> @profile_memory_usage("test_function")
    ... def my_function():
    ...     return [i**2 for i in range(1000000)]
    >>> result = my_function()
    """

    def decorator(func: Callable[..., Any]) -> ProfiledFunction:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if typed_wrapper._profiler is None:
                typed_wrapper._profiler = MemoryProfiler()

            op_name = operation_name or func.__name__
            with typed_wrapper._profiler.profile_operation(op_name):
                return func(*args, **kwargs)

        # Initialize attributes for mypy
        typed_wrapper: ProfiledFunction = cast("ProfiledFunction", wrapper)
        typed_wrapper._profiler = None

        def get_profiler() -> Any:
            return typed_wrapper._profiler

        # Use setattr to assign the method to avoid mypy method assignment error
        typed_wrapper.get_profiler = get_profiler  # type: ignore[method-assign]

        return typed_wrapper

    return decorator


class DistributionMemoryAnalyzer:
    """Analyzer for comparing memory usage of different distribution representations.

    This class helps analyze and compare memory usage between regular dictionaries,
    compressed distributions, sparse distributions, and other representations.

    Examples:
    --------
    >>> analyzer = DistributionMemoryAnalyzer()
    >>> freqdist = {'word1': 1000, 'word2': 500, 'word3': 1}
    >>> comparison = analyzer.compare_representations(freqdist)
    >>> print(comparison['memory_savings'])
    """

    def __init__(self) -> None:
        """Initialize distribution memory analyzer."""
        self.profiler = MemoryProfiler()

    def measure_distribution_memory(self, freqdist: FrequencyDistribution) -> dict[str, float]:
        """Measure memory usage of a frequency distribution.

        Parameters
        ----------
        freqdist : FrequencyDistribution
            Frequency distribution to measure

        Returns:
        -------
        Dict[str, float]
            Memory usage measurements in MB
        """
        # Force garbage collection for accurate measurement
        gc.collect()

        # Measure dictionary size
        dict_size = sys.getsizeof(freqdist)
        for k, v in freqdist.items():
            dict_size += sys.getsizeof(k) + sys.getsizeof(v)

        return {
            "total_mb": dict_size / 1024 / 1024,
            "num_elements": len(freqdist),
            "bytes_per_element": dict_size / len(freqdist) if freqdist else 0,
        }

    def compare_representations(self, freqdist: FrequencyDistribution) -> dict[str, Any]:
        """Compare memory usage of different distribution representations.

        Parameters
        ----------
        freqdist : FrequencyDistribution
            Original frequency distribution

        Returns:
        -------
        Dict[str, Any]
            Comparison results
        """
        from .memory_efficient import create_compressed_distribution, create_sparse_distribution

        results: dict[str, Any] = {}

        # Measure original dictionary
        with self.profiler.profile_operation("measure_original"):
            original_memory = self.measure_distribution_memory(freqdist)
        results["original"] = original_memory

        # Test compressed distribution
        with self.profiler.profile_operation("create_compressed"):
            compressed_dist = create_compressed_distribution(freqdist)
            compressed_memory = compressed_dist.get_memory_usage()
        results["compressed"] = compressed_memory

        # Test quantized compressed distribution
        with self.profiler.profile_operation("create_quantized"):
            quantized_dist = create_compressed_distribution(freqdist, quantization_levels=1024)
            quantized_memory = quantized_dist.get_memory_usage()
        results["quantized"] = quantized_memory

        # Test sparse distribution
        with self.profiler.profile_operation("create_sparse"):
            sparse_dist = create_sparse_distribution(freqdist)
            sparse_memory = sparse_dist.get_memory_usage()
        results["sparse"] = sparse_memory

        # Calculate savings
        original_total = original_memory["total_mb"] * 1024 * 1024  # Convert to bytes

        savings: dict[str, dict[str, float]] = {}
        for name, memory_info in [
            ("compressed", compressed_memory),
            ("quantized", quantized_memory),
            ("sparse", sparse_memory),
        ]:
            total_bytes = memory_info["total"]
            savings[name] = {
                "absolute_savings_mb": (original_total - total_bytes) / 1024 / 1024,
                "percentage_savings": ((original_total - total_bytes) / original_total) * 100,
                "compression_ratio": original_total / total_bytes,
            }

        results["memory_savings"] = savings
        results["profiling_metrics"] = [m.__dict__ for m in self.profiler.get_all_metrics()]

        return results

    def benchmark_scoring_methods(
        self,
        freqdist: FrequencyDistribution,
        test_elements: list[str],
        methods_to_test: list[str] | None = None,
    ) -> dict[str, Any]:
        """Benchmark memory usage and performance of different scoring methods.

        Parameters
        ----------
        freqdist : FrequencyDistribution
            Frequency distribution for testing
        test_elements : List[str]
            Elements to score for performance testing
        methods_to_test : Optional[List[str]]
            Scoring methods to test (defaults to common methods)

        Returns:
        -------
        Dict[str, Any]
            Benchmark results
        """
        if methods_to_test is None:
            methods_to_test = ["MLE", "Laplace", "StreamingMLE"]

        results: dict[str, Any] = {}

        for method_name in methods_to_test:
            method_results: dict[str, Any] = {}

            try:
                # Create scorer
                scorer: ScoringMethod
                with self.profiler.profile_operation(f"create_{method_name}"):
                    if method_name == "MLE":
                        from .basic import MLE

                        scorer = MLE(freqdist)
                    elif method_name == "Laplace":
                        from .lidstone import Laplace

                        scorer = Laplace(freqdist)
                    elif method_name == "StreamingMLE":
                        from .streaming import StreamingMLE

                        scorer = StreamingMLE(freqdist)  # type: ignore[arg-type]
                    else:
                        continue

                creation_metrics = self.profiler.get_latest_metrics()
                method_results["creation"] = creation_metrics.__dict__

                # Test scoring performance
                with self.profiler.profile_operation(f"score_{method_name}"):
                    scores = [scorer(element) for element in test_elements]

                scoring_metrics = self.profiler.get_latest_metrics()
                method_results["scoring"] = scoring_metrics.__dict__
                method_results["num_scores"] = len(scores)

                # Get method-specific memory usage if available
                if hasattr(scorer, "get_memory_usage"):
                    method_results["method_memory"] = scorer.get_memory_usage()
                elif hasattr(scorer, "get_streaming_statistics"):
                    method_results["streaming_stats"] = scorer.get_streaming_statistics()

                results[method_name] = method_results

            except Exception as e:
                results[method_name] = {"error": str(e)}

        return results


class MemoryMonitor:
    """Continuous memory monitor for long-running processes.

    This class provides continuous monitoring of memory usage patterns
    and can trigger alerts when memory usage exceeds thresholds.

    Parameters
    ----------
    memory_threshold_mb : float, default=1000.0
        Memory threshold in MB for triggering alerts
    monitoring_interval : float, default=5.0
        Monitoring interval in seconds

    Examples:
    --------
    >>> monitor = MemoryMonitor(memory_threshold_mb=500.0)
    >>> monitor.start_monitoring()
    >>> # Your long-running process
    >>> monitor.stop_monitoring()
    >>> report = monitor.get_monitoring_report()
    """

    def __init__(self, memory_threshold_mb: float = 1000.0, monitoring_interval: float = 5.0):
        """Initialize memory monitor."""
        self.memory_threshold_mb = memory_threshold_mb

        self.monitoring_interval = monitoring_interval
        self._process = psutil.Process() if HAS_PSUTIL else None
        self._monitoring = False
        self._snapshots: list[MemorySnapshot] = []
        self._alerts: list[dict[str, Any]] = []
        self._profiler = MemoryProfiler()

    def start_monitoring(self) -> None:
        """Start continuous memory monitoring."""
        self._monitoring = True

        print(f"Started memory monitoring (threshold: {self.memory_threshold_mb} MB)")

    def stop_monitoring(self) -> None:
        """Stop memory monitoring."""
        self._monitoring = False

        print("Stopped memory monitoring")

    def check_memory(self) -> dict[str, Any] | None:
        """Check current memory usage and trigger alerts if needed.

        Returns:
        -------
        Optional[Dict[str, Any]]
            Alert information if threshold exceeded, None otherwise
        """
        snapshot = self._profiler.take_snapshot()

        self._snapshots.append(snapshot)

        if snapshot.rss_mb > self.memory_threshold_mb:
            alert = {
                "timestamp": snapshot.timestamp,
                "memory_mb": snapshot.rss_mb,
                "threshold_mb": self.memory_threshold_mb,
                "excess_mb": snapshot.rss_mb - self.memory_threshold_mb,
            }
            self._alerts.append(alert)
            print(
                f"MEMORY ALERT: {snapshot.rss_mb:.1f} MB (threshold: {self.memory_threshold_mb} MB)"
            )
            return alert

        return None

    def get_monitoring_report(self) -> dict[str, Any]:
        """Get a comprehensive monitoring report.

        Returns:
        -------
        Dict[str, Any]
            Monitoring report
        """
        if not self._snapshots:
            return {"error": "No monitoring data available"}

        memory_values = [s.rss_mb for s in self._snapshots]

        return {
            "monitoring_duration": self._snapshots[-1].timestamp - self._snapshots[0].timestamp,
            "total_snapshots": len(self._snapshots),
            "memory_statistics": {
                "min_mb": min(memory_values),
                "max_mb": max(memory_values),
                "avg_mb": sum(memory_values) / len(memory_values),
                "current_mb": memory_values[-1],
            },
            "threshold_violations": len(self._alerts),
            "alerts": self._alerts,
            "memory_trend": self._calculate_memory_trend(),
        }

    def _calculate_memory_trend(self) -> str:
        """Calculate overall memory trend."""
        if len(self._snapshots) < 2:
            return "insufficient_data"

        first_half = self._snapshots[: len(self._snapshots) // 2]
        second_half = self._snapshots[len(self._snapshots) // 2 :]

        first_avg = sum(s.rss_mb for s in first_half) / len(first_half)
        second_avg = sum(s.rss_mb for s in second_half) / len(second_half)

        if second_avg > first_avg * 1.1:
            return "increasing"
        if second_avg < first_avg * 0.9:
            return "decreasing"
        return "stable"


# Utility functions


def get_object_memory_usage(obj: Any) -> dict[str, int | float | str]:
    """Get detailed memory usage information for an object.

    Parameters
    ----------
    obj : Any
        Object to analyze

    Returns:
    -------
    Dict[str, int | float | str]
        Memory usage breakdown
    """
    # Basic object size
    basic_size = sys.getsizeof(obj)

    # Additional analysis based on object type
    if isinstance(obj, dict):
        total_size = basic_size
        for k, v in obj.items():
            total_size += sys.getsizeof(k) + sys.getsizeof(v)
        return {
            "basic_size": basic_size,
            "total_size": total_size,
            "num_items": len(obj),
            "avg_item_size": (total_size - basic_size) / len(obj) if obj else 0,
        }
    if isinstance(obj, list | tuple):
        total_size = basic_size
        for item in obj:
            total_size += sys.getsizeof(item)
        return {
            "basic_size": basic_size,
            "total_size": total_size,
            "num_items": len(obj),
            "avg_item_size": (total_size - basic_size) / len(obj) if obj else 0,
        }
    return {"basic_size": basic_size, "total_size": basic_size, "type": type(obj).__name__}


def force_garbage_collection() -> dict[str, int]:
    """Force garbage collection and return statistics.

    Returns:
    -------
    Dict[str, int]
        Garbage collection statistics
    """
    # Get stats before collection
    objects_before = len(gc.get_objects())

    # Force collection
    collected = gc.collect()

    # Get stats after collection
    stats_after = gc.get_stats()
    objects_after = len(gc.get_objects())

    return {
        "objects_collected": collected,
        "objects_before": objects_before,
        "objects_after": objects_after,
        "objects_freed": objects_before - objects_after,
        "gc_generations": len(stats_after),
    }
