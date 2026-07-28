"""Tests for the memory-profiling and monitoring utilities.

These cover ``freqprob.performance.profiling``: the ``MemoryProfiler`` context
manager and its bookkeeping, the ``DistributionMemoryAnalyzer``
comparison/benchmark helpers, the ``MemoryMonitor`` alert/report logic, and the
standalone memory-inspection utilities.

The environment may or may not have ``psutil`` installed. When it is absent the
RSS/VMS figures read back as ``0.0`` (the documented fallback), so tests here
assert on structure and on the fields that do not depend on process memory, and
force the psutil-independent code paths (e.g. alerts) with explicit thresholds
or injected snapshots.
"""

import time

import pytest

from freqprob.performance.profiling import (
    DistributionMemoryAnalyzer,
    MemoryMonitor,
    MemoryProfiler,
    MemorySnapshot,
    PerformanceMetrics,
    force_garbage_collection,
    get_object_memory_usage,
    profile_memory_usage,
)

# get_object_memory_usage returns dict values typed as ``int | float | str``,
# so numeric comparisons need the operator code relaxed; the frequency-dist
# arguments are concrete dicts where the API declares an invariant Mapping.
# mypy: disable-error-code="operator, arg-type"


class TestMemoryProfiler:
    """Behaviour of the MemoryProfiler class."""

    def test_take_snapshot_returns_snapshot(self) -> None:
        """take_snapshot returns a MemorySnapshot and records it."""
        profiler = MemoryProfiler()
        snapshot = profiler.take_snapshot()

        assert isinstance(snapshot, MemorySnapshot)
        assert snapshot.timestamp > 0
        # Recorded in the profiler's snapshot history.
        assert profiler.get_snapshots() == [snapshot]

    def test_profile_operation_records_metrics(self) -> None:
        """profile_operation records a PerformanceMetrics entry."""
        profiler = MemoryProfiler()

        assert profiler.get_latest_metrics() is None

        with profiler.profile_operation("noop"):
            _ = list(range(1000))

        metrics = profiler.get_latest_metrics()
        assert isinstance(metrics, PerformanceMetrics)
        assert metrics.operation_name == "noop"
        assert metrics.execution_time >= 0.0
        # The delta properties should be computable numbers.
        assert isinstance(metrics.memory_delta_mb, float)
        assert isinstance(metrics.python_objects_delta_mb, float)

        assert len(profiler.get_all_metrics()) == 1

    def test_get_all_metrics_returns_copy(self) -> None:
        """get_all_metrics returns a copy, not the internal list."""
        profiler = MemoryProfiler()
        with profiler.profile_operation("op"):
            pass

        metrics = profiler.get_all_metrics()
        metrics.clear()
        # Mutating the returned list must not affect the profiler.
        assert len(profiler.get_all_metrics()) == 1

    def test_clear_history(self) -> None:
        """clear_history empties both metrics and snapshots."""
        profiler = MemoryProfiler()
        profiler.take_snapshot()
        with profiler.profile_operation("op"):
            pass

        assert profiler.get_snapshots()
        assert profiler.get_all_metrics()

        profiler.clear_history()

        assert profiler.get_snapshots() == []
        assert profiler.get_all_metrics() == []

    def test_memory_summary_empty(self) -> None:
        """get_memory_summary reports an error when there are no snapshots."""
        profiler = MemoryProfiler()
        summary = profiler.get_memory_summary()
        assert summary == {"error": "No snapshots available"}

    def test_memory_summary_with_snapshots(self) -> None:
        """get_memory_summary aggregates recorded snapshots."""
        profiler = MemoryProfiler()
        profiler.take_snapshot()
        profiler.take_snapshot()

        summary = profiler.get_memory_summary()
        assert summary["total_snapshots"] == 2
        assert "time_range" in summary
        assert set(summary["rss_memory"]) == {"current_mb", "min_mb", "max_mb", "avg_mb"}
        assert set(summary["python_objects"]) == {"current_mb", "min_mb", "max_mb", "avg_mb"}

    def test_tracemalloc_disabled(self) -> None:
        """With tracemalloc disabled, peak fields stay None but snapshots work."""
        profiler = MemoryProfiler(enable_tracemalloc=False)
        snapshot = profiler.take_snapshot()
        assert snapshot.peak_mb is None
        with profiler.profile_operation("op"):
            pass
        assert profiler.get_latest_metrics() is not None


class TestProfileMemoryUsageDecorator:
    """The profile_memory_usage decorator."""

    def test_decorator_profiles_calls(self) -> None:
        """The decorator profiles each call under the given operation name."""

        @profile_memory_usage("my_op")
        def build_list() -> list[int]:
            return [i * i for i in range(1000)]

        result = build_list()
        assert len(result) == 1000

        profiler = build_list.get_profiler()
        assert profiler is not None
        metrics = profiler.get_all_metrics()
        assert len(metrics) == 1
        assert metrics[0].operation_name == "my_op"

        # A second call reuses the same profiler instance.
        build_list()
        assert build_list.get_profiler() is profiler
        assert len(build_list.get_profiler().get_all_metrics()) == 2

    def test_decorator_defaults_to_function_name(self) -> None:
        """Without an explicit name, the function name is used."""

        @profile_memory_usage()
        def compute() -> int:
            return 42

        assert compute() == 42
        metrics = compute.get_profiler().get_all_metrics()
        assert metrics[0].operation_name == "compute"


class TestDistributionMemoryAnalyzer:
    """The DistributionMemoryAnalyzer helpers."""

    def test_measure_distribution_memory(self) -> None:
        """measure_distribution_memory reports element count and sizes."""
        analyzer = DistributionMemoryAnalyzer()
        freqdist = {"a": 100, "b": 50, "c": 1}
        measurement = analyzer.measure_distribution_memory(freqdist)

        assert measurement["num_elements"] == 3
        assert measurement["total_mb"] > 0
        assert measurement["bytes_per_element"] > 0

    def test_measure_empty_distribution(self) -> None:
        """An empty distribution avoids division by zero."""
        analyzer = DistributionMemoryAnalyzer()
        measurement = analyzer.measure_distribution_memory({})
        assert measurement["num_elements"] == 0
        assert measurement["bytes_per_element"] == 0

    def test_compare_representations(self) -> None:
        """compare_representations returns savings for each representation."""
        analyzer = DistributionMemoryAnalyzer()
        freqdist = {f"word_{i}": max(1, 100 - i) for i in range(50)}

        comparison = analyzer.compare_representations(freqdist)

        assert set(comparison) >= {
            "original",
            "compressed",
            "quantized",
            "sparse",
            "memory_savings",
            "profiling_metrics",
        }
        for name in ("compressed", "quantized", "sparse"):
            savings = comparison["memory_savings"][name]
            assert set(savings) == {
                "absolute_savings_mb",
                "percentage_savings",
                "compression_ratio",
            }
        assert comparison["profiling_metrics"]  # non-empty list of metric dicts

    def test_benchmark_scoring_methods_default(self) -> None:
        """benchmark_scoring_methods benchmarks the default methods."""
        analyzer = DistributionMemoryAnalyzer()
        freqdist = {"a": 10, "b": 5, "c": 2, "d": 1}
        results = analyzer.benchmark_scoring_methods(freqdist, ["a", "b", "z"])

        assert set(results) == {"MLE", "Laplace", "StreamingMLE"}
        for method in ("MLE", "Laplace"):
            assert "creation" in results[method]
            assert "scoring" in results[method]
            assert results[method]["num_scores"] == 3

    def test_benchmark_scoring_methods_unknown_is_skipped(self) -> None:
        """Unknown method names are silently skipped."""
        analyzer = DistributionMemoryAnalyzer()
        freqdist = {"a": 3, "b": 2}
        results = analyzer.benchmark_scoring_methods(
            freqdist, ["a"], methods_to_test=["MLE", "NotARealMethod"]
        )
        assert "MLE" in results
        assert "NotARealMethod" not in results


class TestMemoryMonitor:
    """The MemoryMonitor alerting and reporting."""

    def test_start_and_stop_monitoring(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Start/stop toggle the flag and print status messages."""
        monitor = MemoryMonitor(memory_threshold_mb=500.0)
        monitor.start_monitoring()
        started = monitor._monitoring
        monitor.stop_monitoring()
        stopped = monitor._monitoring
        assert started is True
        assert stopped is False

        out = capsys.readouterr().out
        assert "Started memory monitoring" in out
        assert "Stopped memory monitoring" in out

    def test_check_memory_no_alert(self) -> None:
        """A very high threshold never triggers an alert."""
        monitor = MemoryMonitor(memory_threshold_mb=1e9)
        assert monitor.check_memory() is None
        assert monitor._alerts == []

    def test_check_memory_triggers_alert(self, capsys: pytest.CaptureFixture[str]) -> None:
        """A negative threshold forces the alert path (RSS is always >= 0)."""
        monitor = MemoryMonitor(memory_threshold_mb=-1.0)
        alert = monitor.check_memory()

        assert alert is not None
        assert set(alert) == {"timestamp", "memory_mb", "threshold_mb", "excess_mb"}
        assert alert["threshold_mb"] == -1.0
        assert monitor._alerts == [alert]
        assert "MEMORY ALERT" in capsys.readouterr().out

    def test_monitoring_report_empty(self) -> None:
        """An empty monitor reports no data available."""
        monitor = MemoryMonitor()
        assert monitor.get_monitoring_report() == {"error": "No monitoring data available"}

    def test_monitoring_report_with_data(self) -> None:
        """The report aggregates collected snapshots."""
        monitor = MemoryMonitor(memory_threshold_mb=1e9)
        monitor.check_memory()
        monitor.check_memory()

        report = monitor.get_monitoring_report()
        assert report["total_snapshots"] == 2
        assert report["threshold_violations"] == 0
        assert set(report["memory_statistics"]) == {"min_mb", "max_mb", "avg_mb", "current_mb"}
        assert report["memory_trend"] in {"increasing", "decreasing", "stable", "insufficient_data"}

    def _snapshot(self, rss: float) -> MemorySnapshot:
        """Build a snapshot with a given RSS value for trend tests."""
        return MemorySnapshot(
            timestamp=time.time(),
            rss_mb=rss,
            vms_mb=0.0,
            python_objects_mb=0.0,
        )

    def test_memory_trend_insufficient_data(self) -> None:
        """A single snapshot yields 'insufficient_data'."""
        monitor = MemoryMonitor()
        monitor._snapshots = [self._snapshot(10.0)]
        assert monitor._calculate_memory_trend() == "insufficient_data"

    def test_memory_trend_increasing(self) -> None:
        """A rising second half yields 'increasing'."""
        monitor = MemoryMonitor()
        monitor._snapshots = [self._snapshot(10.0), self._snapshot(10.0), self._snapshot(100.0)]
        assert monitor._calculate_memory_trend() == "increasing"

    def test_memory_trend_decreasing(self) -> None:
        """A falling second half yields 'decreasing'."""
        monitor = MemoryMonitor()
        monitor._snapshots = [self._snapshot(100.0), self._snapshot(100.0), self._snapshot(10.0)]
        assert monitor._calculate_memory_trend() == "decreasing"

    def test_memory_trend_stable(self) -> None:
        """A flat series yields 'stable'."""
        monitor = MemoryMonitor()
        monitor._snapshots = [self._snapshot(50.0), self._snapshot(50.0), self._snapshot(50.0)]
        assert monitor._calculate_memory_trend() == "stable"


class TestMemoryUtilities:
    """The standalone memory-inspection utilities."""

    def test_get_object_memory_usage_dict(self) -> None:
        """Dict inspection reports item count and per-item size."""
        info = get_object_memory_usage({"a": 1, "b": 2})
        assert info["num_items"] == 2
        assert info["total_size"] >= info["basic_size"]
        assert info["avg_item_size"] > 0

    def test_get_object_memory_usage_empty_dict(self) -> None:
        """An empty dict yields zero average item size."""
        info = get_object_memory_usage({})
        assert info["num_items"] == 0
        assert info["avg_item_size"] == 0

    def test_get_object_memory_usage_list(self) -> None:
        """List inspection reports item count and per-item size."""
        info = get_object_memory_usage([1, 2, 3, 4])
        assert info["num_items"] == 4
        assert info["avg_item_size"] > 0

    def test_get_object_memory_usage_tuple(self) -> None:
        """Tuples are handled like lists."""
        info = get_object_memory_usage((1, 2, 3))
        assert info["num_items"] == 3

    def test_get_object_memory_usage_empty_list(self) -> None:
        """An empty list yields zero average item size."""
        info = get_object_memory_usage([])
        assert info["num_items"] == 0
        assert info["avg_item_size"] == 0

    def test_get_object_memory_usage_scalar(self) -> None:
        """Scalars report their type and a single size."""
        info = get_object_memory_usage(42)
        assert info["type"] == "int"
        assert info["basic_size"] == info["total_size"]

    def test_force_garbage_collection(self) -> None:
        """force_garbage_collection returns collection statistics."""
        stats = force_garbage_collection()
        assert set(stats) == {
            "objects_collected",
            "objects_before",
            "objects_after",
            "objects_freed",
            "gc_generations",
        }
        assert stats["gc_generations"] >= 1
