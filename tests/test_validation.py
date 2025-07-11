"""Test validation and performance profiling functionality."""

import time

import pytest

from freqprob import MLE, Laplace
from freqprob.validation import (
    HAS_PSUTIL,
    BenchmarkSuite,
    PerformanceMetrics,
    PerformanceProfiler,
    ValidationResult,
    ValidationSuite,
    compare_method_performance,
    profile_method_performance,
    quick_validate_method,
)


class TestPerformanceMetrics:
    """Test PerformanceMetrics dataclass."""

    def test_performance_metrics_creation(self) -> None:
        """Test creating PerformanceMetrics."""
        metrics = PerformanceMetrics(
            operation_name="test_op",
            duration_seconds=1.5,
            memory_peak_mb=10.0,
            memory_delta_mb=5.0,
            cpu_percent=50.0,
            iterations=100,
        )

        assert metrics.operation_name == "test_op"
        assert metrics.duration_seconds == 1.5
        assert metrics.memory_peak_mb == 10.0
        assert metrics.memory_delta_mb == 5.0
        assert metrics.cpu_percent == 50.0
        assert metrics.iterations == 100
        assert metrics.timestamp > 0
        assert isinstance(metrics.metadata, dict)

    def test_performance_metrics_to_dict(self) -> None:
        """Test converting PerformanceMetrics to dictionary."""
        metrics = PerformanceMetrics(
            operation_name="test_op",
            duration_seconds=1.5,
            memory_peak_mb=10.0,
            memory_delta_mb=5.0,
        )

        result = metrics.to_dict()

        assert result["operation_name"] == "test_op"
        assert result["duration_seconds"] == 1.5
        assert result["memory_peak_mb"] == 10.0
        assert result["memory_delta_mb"] == 5.0
        assert "timestamp" in result
        assert "throughput_per_sec" in result


class TestValidationResult:
    """Test ValidationResult dataclass."""

    def test_validation_result_creation(self) -> None:
        """Test creating ValidationResult."""
        result = ValidationResult(
            test_name="test_validation",
            passed=True,
            error_message=None,
        )

        assert result.test_name == "test_validation"
        assert result.passed is True
        assert result.error_message is None
        assert result.metrics is None
        assert isinstance(result.details, dict)

    def test_validation_result_with_metrics(self) -> None:
        """Test ValidationResult with metrics."""
        metrics = PerformanceMetrics(
            operation_name="test_op",
            duration_seconds=1.0,
            memory_peak_mb=5.0,
            memory_delta_mb=2.0,
        )

        result = ValidationResult(
            test_name="test_validation",
            passed=True,
            metrics=metrics,
        )

        assert result.metrics == metrics

    def test_validation_result_to_dict(self) -> None:
        """Test converting ValidationResult to dictionary."""
        result = ValidationResult(
            test_name="test_validation",
            passed=False,
            error_message="Test error",
            details={"key": "value"},
        )

        result_dict = result.to_dict()

        assert result_dict["test_name"] == "test_validation"
        assert result_dict["passed"] is False
        assert result_dict["error_message"] == "Test error"
        assert result_dict["details"] == {"key": "value"}


class TestPerformanceProfiler:
    """Test PerformanceProfiler functionality."""

    def test_profiler_creation(self) -> None:
        """Test creating PerformanceProfiler."""
        profiler = PerformanceProfiler()

        assert profiler.results == []
        assert profiler.enable_detailed_tracking is True

    def test_profiler_basic_profiling(self) -> None:
        """Test basic profiling functionality."""
        profiler = PerformanceProfiler()

        with profiler.profile_operation("test_operation"):
            # Simple operation
            _ = [i**2 for i in range(1000)]

        assert len(profiler.results) == 1
        metrics = profiler.results[0]
        assert metrics.operation_name == "test_operation"
        assert metrics.duration_seconds > 0
        assert metrics.memory_peak_mb >= 0

    def test_profiler_method_creation(self) -> None:
        """Test profiling method creation."""
        profiler = PerformanceProfiler()

        freq_dist = {"a": 10, "b": 20, "c": 30}
        metrics = profiler.profile_method_creation(MLE, freq_dist, iterations=3)

        assert metrics.operation_name == "MLE_creation"
        assert metrics.duration_seconds > 0
        assert metrics.iterations == 3
        assert metrics.metadata["vocab_size"] == 3
        assert metrics.metadata["total_count"] == 60

    def test_profiler_query_performance(self) -> None:
        """Test profiling query performance."""
        profiler = PerformanceProfiler()

        freq_dist = {"a": 10, "b": 20, "c": 30}
        method = MLE(freq_dist, logprob=False)  # type: ignore[arg-type]
        test_words = ["a", "b", "c", "unknown"]

        metrics = profiler.profile_query_performance(method, test_words, iterations=10)

        assert metrics.operation_name == "MLE_query"
        assert metrics.duration_seconds > 0
        assert metrics.iterations == 10 * len(test_words)
        assert metrics.metadata["num_test_words"] == len(test_words)

    def test_profiler_batch_operations(self) -> None:
        """Test profiling batch operations."""
        profiler = PerformanceProfiler()

        freq_dist = {"a": 10, "b": 20, "c": 30}
        method = MLE(freq_dist, logprob=False)  # type: ignore[arg-type]
        test_words = ["a", "b", "c", "unknown"]
        batch_sizes = [2, 4]

        results = profiler.profile_batch_operations(method, test_words, batch_sizes)

        assert len(results) == 2
        for i, result in enumerate(results):
            assert result.operation_name == "MLE_batch"
            assert result.metadata["batch_size"] == batch_sizes[i]

    def test_profiler_memory_scaling(self) -> None:
        """Test profiling memory scaling."""
        profiler = PerformanceProfiler()

        base_dist = {"a": 10, "b": 20}
        scale_factors = [1, 2]

        results = profiler.profile_memory_scaling(MLE, base_dist, scale_factors)

        assert len(results) == 2
        for i, result in enumerate(results):
            assert result.operation_name == "MLE_scaling"
            assert result.metadata["scale_factor"] == scale_factors[i]
            assert result.metadata["vocab_size"] == 2

    def test_profiler_concurrent_access(self) -> None:
        """Test profiling concurrent access."""
        profiler = PerformanceProfiler()

        freq_dist = {"a": 10, "b": 20, "c": 30}
        method = MLE(freq_dist, logprob=False)  # type: ignore[arg-type]
        test_words = ["a", "b", "c"]

        metrics = profiler.profile_concurrent_access(
            method, test_words, num_threads=2, queries_per_thread=5
        )

        assert metrics.operation_name == "MLE_concurrent"
        assert metrics.metadata["num_threads"] == 2
        assert metrics.metadata["queries_per_thread"] == 5
        assert "errors" in metrics.metadata

    def test_profiler_get_summary_statistics(self) -> None:
        """Test getting summary statistics."""
        profiler = PerformanceProfiler()

        with profiler.profile_operation("test_op1"):
            time.sleep(0.01)

        with profiler.profile_operation("test_op2"):
            time.sleep(0.01)

        summary = profiler.get_summary_statistics()

        assert summary["total_operations"] == 2
        assert "by_operation" in summary
        assert "overall_stats" in summary


class TestValidationSuite:
    """Test ValidationSuite functionality."""

    def test_validation_suite_creation(self) -> None:
        """Test creating ValidationSuite."""
        suite = ValidationSuite()

        assert suite.results == []

    def test_validation_suite_numerical_stability(self) -> None:
        """Test numerical stability validation."""
        suite = ValidationSuite()

        freq_dist = {"a": 10, "b": 20, "c": 30}
        test_distributions = [freq_dist, {"x": 5, "y": 15}]
        results = suite.validate_numerical_stability(MLE, test_distributions)

        assert isinstance(results, list)
        assert len(results) == 2
        assert all(isinstance(result, ValidationResult) for result in results)
        assert all("stability" in result.test_name for result in results)

    def test_validation_suite_statistical_correctness(self) -> None:
        """Test statistical correctness validation."""
        suite = ValidationSuite()

        freq_dist = {"a": 10, "b": 20, "c": 30}
        result = suite.validate_statistical_correctness(MLE, freq_dist)

        assert isinstance(result, ValidationResult)
        assert result.test_name == "MLE_statistical_correctness"

    def test_validation_suite_performance_regression(self) -> None:
        """Test performance regression validation."""
        suite = ValidationSuite()

        freq_dist = {"a": 10, "b": 20, "c": 30}
        result = suite.validate_performance_regression(MLE, freq_dist)

        assert isinstance(result, ValidationResult)
        assert result.test_name == "MLE_performance_regression"

    def test_validation_suite_thread_safety(self) -> None:
        """Test thread safety validation."""
        suite = ValidationSuite()

        freq_dist = {"a": 10, "b": 20, "c": 30}
        mle_instance = MLE(freq_dist, logprob=False)  # type: ignore[arg-type]
        result = suite.validate_thread_safety(mle_instance, freq_dist)  # type: ignore[arg-type]

        assert isinstance(result, ValidationResult)
        assert "thread_safety" in result.test_name

    def test_validation_suite_run_comprehensive_validation(self) -> None:
        """Test running comprehensive validation."""
        suite = ValidationSuite()

        freq_dist = {"a": 10, "b": 20, "c": 30}
        method_classes = [MLE, Laplace]
        test_distributions = [freq_dist, {"x": 5, "y": 15}]

        results = suite.run_comprehensive_validation(method_classes, test_distributions)  # type: ignore[arg-type]

        assert len(results) > 0
        assert isinstance(results, dict)
        assert "MLE" in results
        assert "Laplace" in results


class TestBenchmarkSuite:
    """Test BenchmarkSuite functionality."""

    def test_benchmark_suite_creation(self) -> None:
        """Test creating BenchmarkSuite."""
        suite = BenchmarkSuite()

        assert suite.benchmark_results == {}

    def test_benchmark_suite_compare_methods(self) -> None:
        """Test comparing methods."""
        suite = BenchmarkSuite()

        freq_dist = {"a": 10, "b": 20, "c": 30}
        method_configs = [
            (MLE, {}),
            (Laplace, {"bins": 100}),
        ]

        results = suite.compare_methods(method_configs, freq_dist)  # type: ignore[arg-type]

        assert len(results) == 2
        assert all(isinstance(metrics, PerformanceMetrics) for metrics in results.values())

    def test_benchmark_suite_benchmark_creation_scaling(self) -> None:
        """Test benchmarking creation scaling."""
        suite = BenchmarkSuite()

        method_classes = [MLE, Laplace]
        vocab_sizes = [5, 10]

        results = suite.benchmark_creation_scaling(method_classes, vocab_sizes)  # type: ignore[arg-type]

        assert len(results) == 2
        assert "MLE" in results
        assert "Laplace" in results
        assert all(isinstance(metrics_list, list) for metrics_list in results.values())

    def test_benchmark_suite_benchmark_query_scaling(self) -> None:
        """Test benchmarking query scaling."""
        suite = BenchmarkSuite()

        freq_dist = {"a": 10, "b": 20, "c": 30}
        methods = {
            "MLE": MLE(freq_dist, logprob=False),  # type: ignore[arg-type]
            "Laplace": Laplace(freq_dist, logprob=False),  # type: ignore[arg-type]
        }
        query_counts = [2, 4]

        results = suite.benchmark_query_scaling(methods, query_counts)

        assert len(results) == 2
        assert "MLE" in results
        assert "Laplace" in results


class TestUtilityFunctions:
    """Test utility functions."""

    def test_quick_validate_method(self) -> None:
        """Test quick method validation."""
        freq_dist = {"a": 10, "b": 20, "c": 30}

        result = quick_validate_method(MLE, freq_dist)

        assert isinstance(result, bool)
        assert result is True

    def test_profile_method_performance(self) -> None:
        """Test profiling method performance."""
        freq_dist = {"a": 10, "b": 20, "c": 30}

        metrics = profile_method_performance(MLE, freq_dist, iterations=3)

        assert isinstance(metrics, PerformanceMetrics)
        assert metrics.operation_name == "MLE_creation"
        assert metrics.iterations == 3

    def test_compare_method_performance(self) -> None:
        """Test comparing method performance."""
        freq_dist = {"a": 10, "b": 20, "c": 30}
        method_configs = [
            (MLE, {}),
            (Laplace, {"bins": 100}),
        ]

        results = compare_method_performance(method_configs, freq_dist)  # type: ignore[arg-type]

        assert isinstance(results, dict)
        assert len(results) == 2
        # The method names include hash suffixes, so we check that we have 2 results
        assert all(isinstance(time_val, float) for time_val in results.values())
        # Check that method names contain the expected class names
        method_names = list(results.keys())
        assert any("MLE" in name for name in method_names)
        assert any("Laplace" in name for name in method_names)


@pytest.mark.skipif(not HAS_PSUTIL, reason="psutil not available")
class TestPsutilFeatures:
    """Test features that require psutil."""

    def test_profiler_with_psutil(self) -> None:
        """Test profiler functionality with psutil."""
        profiler = PerformanceProfiler(enable_detailed_tracking=True)

        with profiler.profile_operation("test_operation"):
            # Simple operation
            _ = [i**2 for i in range(1000)]

        assert len(profiler.results) == 1
        metrics = profiler.results[0]
        assert metrics.cpu_percent >= 0
        assert metrics.memory_delta_mb >= 0
