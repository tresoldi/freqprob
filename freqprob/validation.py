"""Performance profiling and validation tools for FreqProb.

This module provides comprehensive profiling tools for analyzing performance,
memory usage, and computational efficiency of smoothing methods.
"""

import json
import math
import threading
import time
import tracemalloc
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

try:
    import psutil

    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

HAS_PLOTTING = False


@dataclass
class PerformanceMetrics:
    """Container for performance measurement results."""

    operation_name: str
    duration_seconds: float
    memory_peak_mb: float
    memory_delta_mb: float
    cpu_percent: float = 0.0
    iterations: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "operation_name": self.operation_name,
            "duration_seconds": self.duration_seconds,
            "memory_peak_mb": self.memory_peak_mb,
            "memory_delta_mb": self.memory_delta_mb,
            "cpu_percent": self.cpu_percent,
            "iterations": self.iterations,
            "throughput_per_sec": (
                self.iterations / self.duration_seconds if self.duration_seconds > 0 else 0
            ),
            "metadata": self.metadata,
            "timestamp": self.timestamp,
        }


@dataclass
class ValidationResult:
    """Container for validation test results."""

    test_name: str
    passed: bool
    error_message: str | None = None
    metrics: PerformanceMetrics | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "test_name": self.test_name,
            "passed": self.passed,
            "error_message": self.error_message,
            "details": self.details,
        }
        if self.metrics:
            result["metrics"] = self.metrics.to_dict()
        return result


class PerformanceProfiler:
    """Advanced performance profiler for FreqProb methods."""

    def __init__(self, enable_detailed_tracking: bool = True):
        """Initialize the performance profiler.

        Args:
            enable_detailed_tracking: Whether to enable detailed memory/CPU tracking
        """
        self.enable_detailed_tracking = enable_detailed_tracking
        self.results: list[PerformanceMetrics] = []
        self._lock = threading.Lock()

    @contextmanager
    def profile_operation(
        self, operation_name: str, iterations: int = 1, **metadata: Any
    ) -> Generator[None, None, None]:
        """Context manager for profiling operations.

        Args:
            operation_name: Name of the operation being profiled
            iterations: Number of iterations performed
            **metadata: Additional metadata to store
        """
        # Start measurements
        start_time = time.perf_counter()
        start_memory = 0
        process = None

        if self.enable_detailed_tracking:
            tracemalloc.start()
            if HAS_PSUTIL:
                process = psutil.Process()
                start_memory = process.memory_info().rss / 1024 / 1024  # MB

        try:
            yield
        finally:
            # End measurements
            end_time = time.perf_counter()
            duration = end_time - start_time

            memory_peak = 0.0
            memory_delta = 0.0
            cpu_percent = 0.0

            if self.enable_detailed_tracking:
                if tracemalloc.is_tracing():
                    current, peak = tracemalloc.get_traced_memory()
                    memory_peak = float(peak) / 1024 / 1024  # MB
                    tracemalloc.stop()

                if HAS_PSUTIL and process:
                    end_memory = process.memory_info().rss / 1024 / 1024  # MB
                    memory_delta = end_memory - start_memory
                    cpu_percent = process.cpu_percent()

            # Create metrics
            metrics = PerformanceMetrics(
                operation_name=operation_name,
                duration_seconds=duration,
                memory_peak_mb=memory_peak,
                memory_delta_mb=memory_delta,
                cpu_percent=cpu_percent,
                iterations=iterations,
                metadata=metadata,
            )

            # Store results thread-safely
            with self._lock:
                self.results.append(metrics)

    def profile_method_creation(
        self, method_class: type, freq_dist: dict[str, int], iterations: int = 10, **kwargs: Any
    ) -> PerformanceMetrics:
        """Profile the creation time of a smoothing method."""
        with self.profile_operation(
            f"{method_class.__name__}_creation",
            iterations=iterations,
            vocab_size=len(freq_dist),
            total_count=sum(freq_dist.values()),
            **kwargs,
        ):
            for _ in range(iterations):
                _ = method_class(freq_dist, **kwargs)

        return self.results[-1]

    def profile_query_performance(
        self, method: Any, test_words: list[str], iterations: int = 1000
    ) -> PerformanceMetrics:
        """Profile query performance for a method."""
        with self.profile_operation(
            f"{method.__class__.__name__}_query",
            iterations=iterations * len(test_words),
            num_test_words=len(test_words),
        ):
            for _ in range(iterations):
                for word in test_words:
                    _ = method(word)

        return self.results[-1]

    def profile_batch_operations(
        self, method: Any, test_words: list[str], batch_sizes: list[int]
    ) -> list[PerformanceMetrics]:
        """Profile batch operations with different batch sizes."""
        from . import VectorizedScorer

        vectorized = VectorizedScorer(method)
        results = []

        for batch_size in batch_sizes:
            test_batch = test_words[:batch_size]

            with self.profile_operation(
                f"{method.__class__.__name__}_batch",
                iterations=len(test_batch),
                batch_size=batch_size,
            ):
                _ = vectorized.score_batch(test_batch)  # type: ignore[arg-type]

            results.append(self.results[-1])

        return results

    def profile_memory_scaling(
        self, method_class: type, base_dist: dict[str, int], scale_factors: list[int], **kwargs: Any
    ) -> list[PerformanceMetrics]:
        """Profile memory usage scaling with dataset size."""
        results = []

        for scale in scale_factors:
            scaled_dist = {word: count * scale for word, count in base_dist.items()}

            with self.profile_operation(
                f"{method_class.__name__}_scaling",
                iterations=1,
                scale_factor=scale,
                vocab_size=len(scaled_dist),
                total_count=sum(scaled_dist.values()),
            ):
                method = method_class(scaled_dist, **kwargs)
                # Force some computation to ensure memory is allocated
                _ = method(next(iter(scaled_dist.keys())))

            results.append(self.results[-1])

        return results

    def profile_concurrent_access(
        self,
        method: Any,
        test_words: list[str],
        num_threads: int = 4,
        queries_per_thread: int = 100,
    ) -> PerformanceMetrics:
        """Profile concurrent access to methods."""
        errors = []

        def worker() -> None:
            try:
                for _ in range(queries_per_thread):
                    for word in test_words[:10]:  # Limit words per thread
                        _ = method(word)
            except Exception as e:
                errors.append(e)

        with self.profile_operation(
            f"{method.__class__.__name__}_concurrent",
            iterations=num_threads * queries_per_thread * 10,
            num_threads=num_threads,
            queries_per_thread=queries_per_thread,
        ):
            threads = [threading.Thread(target=worker) for _ in range(num_threads)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

        # Add error information to metadata
        self.results[-1].metadata["errors"] = len(errors)
        self.results[-1].metadata["error_details"] = [str(e) for e in errors[:5]]

        return self.results[-1]

    def get_summary_statistics(self) -> dict[str, Any]:
        """Get summary statistics for all profiled operations."""
        if not self.results:
            return {}

        summary = {"total_operations": len(self.results), "by_operation": {}, "overall_stats": {}}

        # Group by operation name
        by_operation: dict[str, list[PerformanceMetrics]] = {}
        for result in self.results:
            op_name = result.operation_name
            if op_name not in by_operation:
                by_operation[op_name] = []
            by_operation[op_name].append(result)

        # Calculate statistics per operation
        for op_name, results in by_operation.items():
            durations = [r.duration_seconds for r in results]
            memory_deltas = [r.memory_delta_mb for r in results]
            memory_peaks = [r.memory_peak_mb for r in results]

            summary["by_operation"][op_name] = {  # type: ignore[index]
                "count": len(results),
                "duration": {
                    "mean": np.mean(durations),
                    "std": np.std(durations),
                    "min": np.min(durations),
                    "max": np.max(durations),
                    "median": np.median(durations),
                },
                "memory_delta": {
                    "mean": np.mean(memory_deltas),
                    "std": np.std(memory_deltas),
                    "min": np.min(memory_deltas),
                    "max": np.max(memory_deltas),
                },
                "memory_peak": {
                    "mean": np.mean(memory_peaks),
                    "std": np.std(memory_peaks),
                    "max": np.max(memory_peaks),
                },
            }

        # Overall statistics
        all_durations = [r.duration_seconds for r in self.results]
        all_memory = [r.memory_delta_mb for r in self.results]

        summary["overall_stats"] = {
            "total_duration": sum(all_durations),
            "avg_duration": np.mean(all_durations),
            "total_memory_delta": sum(all_memory),
            "max_memory_delta": max(all_memory) if all_memory else 0,
        }

        return summary

    def export_results(self, filepath: Path, format: str = "json") -> None:
        """Export profiling results to file."""
        filepath = Path(filepath)

        if format.lower() == "json":
            data = {
                "summary": self.get_summary_statistics(),
                "detailed_results": [r.to_dict() for r in self.results],
            }
            with open(filepath, "w") as f:
                json.dump(data, f, indent=2)

        elif format.lower() == "csv":
            import csv

            with open(filepath, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "operation_name",
                        "duration_seconds",
                        "memory_peak_mb",
                        "memory_delta_mb",
                        "cpu_percent",
                        "iterations",
                        "throughput_per_sec",
                    ]
                )
                for result in self.results:
                    writer.writerow(
                        [
                            result.operation_name,
                            result.duration_seconds,
                            result.memory_peak_mb,
                            result.memory_delta_mb,
                            result.cpu_percent,
                            result.iterations,
                            (
                                result.iterations / result.duration_seconds
                                if result.duration_seconds > 0
                                else 0
                            ),
                        ]
                    )

    def clear_results(self) -> None:
        """Clear all stored results."""
        with self._lock:
            self.results.clear()


class ValidationSuite:
    """Comprehensive validation suite for FreqProb methods."""

    def __init__(self, profiler: PerformanceProfiler | None = None):
        """Initialize validation suite.

        Args:
            profiler: Optional performance profiler to use
        """
        self.profiler = profiler or PerformanceProfiler()
        self.results: list[ValidationResult] = []

    def validate_numerical_stability(
        self, method_class: type, test_distributions: list[dict[str, int]], **kwargs: Any
    ) -> list[ValidationResult]:
        """Validate numerical stability across different distributions."""
        results = []

        for i, dist in enumerate(test_distributions):
            test_name = f"{method_class.__name__}_stability_{i}"

            try:
                with self.profiler.profile_operation(test_name):
                    method = method_class(dist, **kwargs)

                    # Test probabilities for all words
                    for word in dist:
                        prob = method(word)
                        if math.isnan(prob) or math.isinf(prob) or prob < 0:
                            raise ValueError(f"Invalid probability for '{word}': {prob}")

                    # Test unknown word
                    unknown_prob = method("unknown_test_word")
                    if math.isnan(unknown_prob) or math.isinf(unknown_prob) or unknown_prob < 0:
                        raise ValueError(f"Invalid probability for unknown word: {unknown_prob}")

                results.append(
                    ValidationResult(
                        test_name=test_name,
                        passed=True,
                        metrics=self.profiler.results[-1] if self.profiler.results else None,
                        details={"distribution_size": len(dist), "total_count": sum(dist.values())},
                    )
                )

            except Exception as e:
                results.append(
                    ValidationResult(
                        test_name=test_name,
                        passed=False,
                        error_message=str(e),
                        details={"distribution_size": len(dist), "total_count": sum(dist.values())},
                    )
                )

        self.results.extend(results)
        return results

    def validate_statistical_correctness(
        self,
        method_class: type,
        reference_dist: dict[str, int],
        tolerance: float = 1e-10,
        **kwargs: Any,
    ) -> ValidationResult:
        """Validate statistical correctness against known theoretical results."""
        test_name = f"{method_class.__name__}_statistical_correctness"

        try:
            with self.profiler.profile_operation(test_name):
                method = method_class(reference_dist, logprob=False, **kwargs)

                # Test that probabilities sum to expected value
                if method_class.__name__ == "MLE":
                    # MLE should sum to exactly 1 for observed vocabulary
                    total_prob = sum(method(word) for word in reference_dist)
                    if abs(total_prob - 1.0) > tolerance:
                        raise ValueError(f"MLE probabilities don't sum to 1: {total_prob}")

                # Test monotonicity (higher counts should have higher/equal probability)
                sorted_items = sorted(reference_dist.items(), key=lambda x: x[1], reverse=True)
                for i in range(len(sorted_items) - 1):
                    word1, count1 = sorted_items[i]
                    word2, count2 = sorted_items[i + 1]

                    prob1 = method(word1)
                    prob2 = method(word2)

                    if count1 > count2 and prob1 < prob2:
                        raise ValueError(
                            f"Monotonicity violation: {word1}({count1})={prob1} < {word2}({count2})={prob2}"
                        )

            result = ValidationResult(
                test_name=test_name,
                passed=True,
                metrics=self.profiler.results[-1] if self.profiler.results else None,
                details={"method": method_class.__name__, "tolerance": tolerance},
            )

        except Exception as e:
            result = ValidationResult(
                test_name=test_name,
                passed=False,
                error_message=str(e),
                details={"method": method_class.__name__, "tolerance": tolerance},
            )

        self.results.append(result)
        return result

    def validate_performance_regression(
        self,
        method_class: type,
        reference_dist: dict[str, int],
        max_duration_seconds: float = 10.0,
        max_memory_mb: float = 1000.0,
        **kwargs: Any,
    ) -> ValidationResult:
        """Validate that performance hasn't regressed beyond acceptable limits."""
        test_name = f"{method_class.__name__}_performance_regression"

        try:
            metrics = self.profiler.profile_method_creation(
                method_class, reference_dist, iterations=1, **kwargs
            )

            # Check duration
            if metrics.duration_seconds > max_duration_seconds:
                raise ValueError(
                    f"Creation took too long: {metrics.duration_seconds:.2f}s > {max_duration_seconds}s"
                )

            # Check memory usage
            if metrics.memory_delta_mb > max_memory_mb:
                raise ValueError(
                    f"Memory usage too high: {metrics.memory_delta_mb:.2f}MB > {max_memory_mb}MB"
                )

            result = ValidationResult(
                test_name=test_name,
                passed=True,
                metrics=metrics,
                details={
                    "max_duration_seconds": max_duration_seconds,
                    "max_memory_mb": max_memory_mb,
                    "actual_duration": metrics.duration_seconds,
                    "actual_memory": metrics.memory_delta_mb,
                },
            )

        except Exception as e:
            result = ValidationResult(
                test_name=test_name,
                passed=False,
                error_message=str(e),
                details={
                    "max_duration_seconds": max_duration_seconds,
                    "max_memory_mb": max_memory_mb,
                },
            )

        self.results.append(result)
        return result

    def validate_thread_safety(
        self, method: Any, test_words: list[str], num_threads: int = 4
    ) -> ValidationResult:
        """Validate thread safety of method implementations."""
        test_name = f"{method.__class__.__name__}_thread_safety"

        try:
            metrics = self.profiler.profile_concurrent_access(
                method, test_words, num_threads=num_threads
            )

            # Check for errors
            if metrics.metadata.get("errors", 0) > 0:
                raise ValueError(f"Thread safety errors: {metrics.metadata['error_details']}")

            result = ValidationResult(
                test_name=test_name,
                passed=True,
                metrics=metrics,
                details={"num_threads": num_threads, "test_words_count": len(test_words)},
            )

        except Exception as e:
            result = ValidationResult(
                test_name=test_name,
                passed=False,
                error_message=str(e),
                details={"num_threads": num_threads, "test_words_count": len(test_words)},
            )

        self.results.append(result)
        return result

    def run_comprehensive_validation(
        self, method_classes: list[type], test_distributions: list[dict[str, int]], **kwargs: Any
    ) -> dict[str, list[ValidationResult]]:
        """Run comprehensive validation across all methods and test cases."""
        all_results = {}

        for method_class in method_classes:
            method_results = []

            # Test numerical stability
            stability_results = self.validate_numerical_stability(
                method_class, test_distributions, **kwargs
            )
            method_results.extend(stability_results)

            # Test statistical correctness on first distribution
            if test_distributions:
                correctness_result = self.validate_statistical_correctness(
                    method_class, test_distributions[0], **kwargs
                )
                method_results.append(correctness_result)

                # Test performance regression
                performance_result = self.validate_performance_regression(
                    method_class, test_distributions[0], **kwargs
                )
                method_results.append(performance_result)

                # Test thread safety (create method instance first)
                try:
                    method_instance = method_class(test_distributions[0], **kwargs)
                    test_words = list(test_distributions[0].keys())[:10]
                    thread_safety_result = self.validate_thread_safety(method_instance, test_words)
                    method_results.append(thread_safety_result)
                except Exception:
                    # Some methods might not support the parameters
                    pass

            all_results[method_class.__name__] = method_results

        return all_results

    def generate_validation_report(self, output_path: Path) -> None:
        """Generate comprehensive validation report."""
        output_path = Path(output_path)

        # Calculate summary statistics
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.passed)
        failed_tests = total_tests - passed_tests

        # Group results by test type
        by_test_type: dict[str, list[ValidationResult]] = {}
        for result in self.results:
            test_type = result.test_name.split("_")[-1]  # Get last part of test name
            if test_type not in by_test_type:
                by_test_type[test_type] = []
            by_test_type[test_type].append(result)

        # Create report data
        report_data = {
            "summary": {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": failed_tests,
                "success_rate": passed_tests / total_tests if total_tests > 0 else 0,
                "timestamp": time.time(),
            },
            "by_test_type": {},
            "detailed_results": [r.to_dict() for r in self.results],
            "profiler_summary": self.profiler.get_summary_statistics(),
        }

        # Add test type summaries
        for test_type, results in by_test_type.items():
            type_passed = sum(1 for r in results if r.passed)
            report_data["by_test_type"][test_type] = {  # type: ignore[index]
                "total": len(results),
                "passed": type_passed,
                "failed": len(results) - type_passed,
                "success_rate": type_passed / len(results) if results else 0,
            }

        # Save report
        with open(output_path, "w") as f:
            json.dump(report_data, f, indent=2)

    def clear_results(self) -> None:
        """Clear all validation results."""
        self.results.clear()
        self.profiler.clear_results()


class BenchmarkSuite:
    """Comprehensive benchmarking suite with comparison capabilities."""

    def __init__(self, profiler: PerformanceProfiler | None = None):
        """Initialize benchmark suite."""
        self.profiler = profiler or PerformanceProfiler()
        self.benchmark_results: dict[str, list[PerformanceMetrics]] = {}

    def benchmark_creation_scaling(
        self, method_classes: list[type], vocab_sizes: list[int], **kwargs: Any
    ) -> dict[str, list[PerformanceMetrics]]:
        """Benchmark creation time scaling with vocabulary size."""
        results = {}

        for method_class in method_classes:
            method_results = []

            for vocab_size in vocab_sizes:
                # Create test distribution
                test_dist = {f"word_{i}": max(1, int(1000 / (i + 1))) for i in range(vocab_size)}

                try:
                    metrics = self.profiler.profile_method_creation(
                        method_class, test_dist, iterations=5, **kwargs
                    )
                    method_results.append(metrics)
                except Exception as e:
                    # Create dummy metrics for failed cases
                    metrics = PerformanceMetrics(
                        operation_name=f"{method_class.__name__}_creation",
                        duration_seconds=float("inf"),
                        memory_peak_mb=float("inf"),
                        memory_delta_mb=float("inf"),
                        metadata={"error": str(e), "vocab_size": vocab_size},
                    )
                    method_results.append(metrics)

            results[method_class.__name__] = method_results

        self.benchmark_results.update(results)
        return results

    def benchmark_query_scaling(
        self, methods: dict[str, Any], query_counts: list[int]
    ) -> dict[str, list[PerformanceMetrics]]:
        """Benchmark query performance scaling."""
        results = {}

        # Create test words
        all_test_words = [f"test_word_{i}" for i in range(max(query_counts))]

        for method_name, method in methods.items():
            method_results = []

            for query_count in query_counts:
                test_words = all_test_words[:query_count]

                try:
                    metrics = self.profiler.profile_query_performance(
                        method, test_words, iterations=100
                    )
                    method_results.append(metrics)
                except Exception as e:
                    metrics = PerformanceMetrics(
                        operation_name=f"{method_name}_query",
                        duration_seconds=float("inf"),
                        memory_peak_mb=float("inf"),
                        memory_delta_mb=float("inf"),
                        metadata={"error": str(e), "query_count": query_count},
                    )
                    method_results.append(metrics)

            results[method_name] = method_results

        self.benchmark_results.update(results)
        return results

    def compare_methods(
        self, method_configs: list[tuple[type, dict[str, Any]]], test_distribution: dict[str, int]
    ) -> dict[str, PerformanceMetrics]:
        """Compare multiple methods on the same distribution."""
        results = {}

        for method_class, kwargs in method_configs:
            method_name = f"{method_class.__name__}_{hash(str(sorted(kwargs.items()))) % 10000}"

            try:
                metrics = self.profiler.profile_method_creation(
                    method_class, test_distribution, iterations=3, **kwargs
                )
                results[method_name] = metrics
            except Exception as e:
                metrics = PerformanceMetrics(
                    operation_name=f"{method_name}_creation",
                    duration_seconds=float("inf"),
                    memory_peak_mb=float("inf"),
                    memory_delta_mb=float("inf"),
                    metadata={"error": str(e), "config": kwargs},
                )
                results[method_name] = metrics

        return results

    def export_benchmark_report(self, output_path: Path) -> None:
        """Export comprehensive benchmark report."""
        output_path = Path(output_path)

        # Compile all results
        all_results = []
        for method_name, metrics_list in self.benchmark_results.items():
            for metrics in metrics_list:
                result_dict = metrics.to_dict()
                result_dict["method_name"] = method_name
                all_results.append(result_dict)

        # Create summary
        summary = self.profiler.get_summary_statistics()

        report = {
            "benchmark_summary": summary,
            "detailed_results": all_results,
            "metadata": {
                "total_benchmarks": len(all_results),
                "methods_tested": list(self.benchmark_results.keys()),
                "timestamp": time.time(),
            },
        }

        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)


# Convenience functions for common validation tasks
def quick_validate_method(method_class: type, test_dist: dict[str, int], **kwargs: Any) -> bool:
    """Quick validation check for a method."""
    validator = ValidationSuite()
    result = validator.validate_statistical_correctness(method_class, test_dist, **kwargs)
    return result.passed


def profile_method_performance(
    method_class: type, test_dist: dict[str, int], **kwargs: Any
) -> PerformanceMetrics:
    """Quick performance profiling for a method."""
    profiler = PerformanceProfiler()
    return profiler.profile_method_creation(method_class, test_dist, **kwargs)


def compare_method_performance(
    method_configs: list[tuple[type, dict[str, Any]]], test_dist: dict[str, int]
) -> dict[str, float]:
    """Compare creation times of multiple methods."""
    benchmarker = BenchmarkSuite()
    results = benchmarker.compare_methods(method_configs, test_dist)

    return {name: metrics.duration_seconds for name, metrics in results.items()}
