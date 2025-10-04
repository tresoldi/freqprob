#!/usr/bin/env python3
"""FreqProb Performance Benchmarks.

This script provides comprehensive performance benchmarks for FreqProb
smoothing methods across different scenarios and datasets.

Benchmarks v0.4.0 features:
- SimpleGoodTuring with bins parameter (per-word probability)
- InterpolatedSmoothing with n-gram mode (trigram + bigram)
- KneserNey and ModifiedKneserNey methods
- Cross-entropy evaluation
- N-gram datasets (bigrams, trigrams)

Usage:
    python benchmarks.py [--output OUTPUT_DIR] [--format FORMAT] [--quick]

Formats: json, csv, html, all
"""

import argparse
import csv
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import freqprob

HAS_PLOTTING = False

try:
    import psutil

    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("Warning: psutil not available. Memory measurements will be limited.")


class BenchmarkResult:
    """Container for benchmark results."""

    def __init__(
        self,
        method: str,
        dataset: str,
        metric: str,
        value: float,
        unit: str = "",
        metadata: dict[str, Any] | None = None,
    ):
        """Initialize benchmark result.

        Args:
            method: Name of the benchmarked method
            dataset: Dataset used for benchmarking
            metric: Performance metric measured
            value: Measured value
            unit: Unit of measurement
            metadata: Additional metadata
        """
        self.method = method
        self.dataset = dataset
        self.metric = metric
        self.value = value
        self.unit = unit
        self.metadata = metadata or {}
        self.timestamp = time.time()

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary format.

        Returns:
            Dictionary representation of the benchmark result
        """

        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy(obj: Any) -> Any:
            if hasattr(obj, "item"):  # numpy scalar
                return obj.item()
            if isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj

        return {
            "method": self.method,
            "dataset": self.dataset,
            "metric": self.metric,
            "value": convert_numpy(self.value),
            "unit": self.unit,
            "metadata": convert_numpy(self.metadata),
            "timestamp": self.timestamp,
        }


class DatasetGenerator:
    """Generate synthetic datasets for benchmarking."""

    @staticmethod
    def create_zipf_distribution(
        vocab_size: int, total_count: int, alpha: float = 1.2
    ) -> dict[str, int]:
        """Create a frequency distribution following Zipf's law."""
        np.random.seed(42)  # For reproducibility

        # Generate Zipfian frequencies
        frequencies = np.random.zipf(alpha, vocab_size)

        # Normalize to desired total count
        frequencies = (frequencies / frequencies.sum()) * total_count
        frequencies = frequencies.astype(int)
        frequencies[frequencies == 0] = 1  # Ensure no zero counts

        # Create vocabulary
        words = [f"word_{i:06d}" for i in range(vocab_size)]

        return dict(zip(words, frequencies, strict=False))

    @staticmethod
    def create_uniform_distribution(vocab_size: int, total_count: int) -> dict[str, int]:
        """Create a uniform frequency distribution."""
        count_per_word = total_count // vocab_size
        remainder = total_count % vocab_size

        words = [f"word_{i:06d}" for i in range(vocab_size)]
        frequencies = [count_per_word] * vocab_size

        # Distribute remainder
        for i in range(remainder):
            frequencies[i] += 1

        return dict(zip(words, frequencies, strict=False))

    @staticmethod
    def create_power_law_distribution(
        vocab_size: int, total_count: int, exponent: float = -1.5
    ) -> dict[str, int]:
        """Create a power-law frequency distribution."""
        np.random.seed(42)

        # Generate ranks
        ranks = np.arange(1, vocab_size + 1)

        # Power law: frequency ∝ rank^exponent
        frequencies = ranks**exponent
        frequencies = frequencies / frequencies.sum() * total_count
        frequencies = frequencies.astype(int)
        frequencies[frequencies == 0] = 1

        words = [f"word_{i:06d}" for i in range(vocab_size)]

        return dict(zip(words, frequencies, strict=False))

    @staticmethod
    def create_bigram_distribution(
        vocab_size: int, total_count: int, alpha: float = 1.2
    ) -> dict[tuple[str, str], int]:
        """Create a bigram frequency distribution following Zipf's law."""
        np.random.seed(42)

        # Generate vocabulary
        words = [f"word_{i:06d}" for i in range(vocab_size)]

        # Generate bigrams with Zipfian frequencies
        num_bigrams = min(vocab_size * vocab_size, total_count // 10)
        frequencies = np.random.zipf(alpha, num_bigrams)
        frequencies = (frequencies / frequencies.sum()) * total_count
        frequencies = frequencies.astype(int)
        frequencies[frequencies == 0] = 1

        # Create bigrams
        bigrams = []
        for _ in range(num_bigrams):
            w1 = words[np.random.randint(0, vocab_size)]
            w2 = words[np.random.randint(0, vocab_size)]
            bigrams.append((w1, w2))

        return dict(zip(bigrams, frequencies[: len(bigrams)], strict=False))

    @staticmethod
    def create_trigram_distribution(
        vocab_size: int, total_count: int, alpha: float = 1.2
    ) -> dict[tuple[str, str, str], int]:
        """Create a trigram frequency distribution following Zipf's law."""
        np.random.seed(42)

        # Generate vocabulary
        words = [f"word_{i:06d}" for i in range(vocab_size)]

        # Generate trigrams with Zipfian frequencies
        num_trigrams = min(vocab_size * vocab_size, total_count // 20)
        frequencies = np.random.zipf(alpha, num_trigrams)
        frequencies = (frequencies / frequencies.sum()) * total_count
        frequencies = frequencies.astype(int)
        frequencies[frequencies == 0] = 1

        # Create trigrams
        trigrams = []
        for _ in range(num_trigrams):
            w1 = words[np.random.randint(0, vocab_size)]
            w2 = words[np.random.randint(0, vocab_size)]
            w3 = words[np.random.randint(0, vocab_size)]
            trigrams.append((w1, w2, w3))

        return dict(zip(trigrams, frequencies[: len(trigrams)], strict=False))


class PerformanceBenchmark:
    """Main benchmarking class."""

    def __init__(self) -> None:
        """Initialize performance benchmark."""
        self.results: list[BenchmarkResult] = []
        self.datasets: dict[str, Any] = {}
        self.test_datasets: dict[str, Any] = {}

        # Smoothing methods to benchmark (unigram methods)
        self.smoothing_methods = {
            "MLE": lambda freq: freqprob.MLE(freq, logprob=True),
            "Laplace": lambda freq: freqprob.Laplace(freq, bins=len(freq) * 2, logprob=True),
            "ELE": lambda freq: freqprob.ELE(freq, bins=len(freq) * 2, logprob=True),
            "Lidstone_0.1": lambda freq: freqprob.Lidstone(
                freq, gamma=0.1, bins=len(freq) * 2, logprob=True
            ),
            "Lidstone_0.5": lambda freq: freqprob.Lidstone(
                freq, gamma=0.5, bins=len(freq) * 2, logprob=True
            ),
            "Bayesian_0.5": lambda freq: freqprob.BayesianSmoothing(freq, alpha=0.5, logprob=True),
            "Bayesian_1.0": lambda freq: freqprob.BayesianSmoothing(freq, alpha=1.0, logprob=True),
        }

        # Add SimpleGoodTuring with different bins configurations (v0.4.0)
        def try_sgt(freq: Any) -> Any:
            try:
                return freqprob.SimpleGoodTuring(freq, logprob=True)
            except Exception as e:
                print(f"  Warning: SimpleGoodTuring failed: {e}")
                return None

        def try_sgt_custom_bins(freq: Any) -> Any:
            try:
                # Test with custom bins parameter (v0.4.0 feature)
                return freqprob.SimpleGoodTuring(freq, bins=len(freq) * 3, logprob=True)
            except Exception as e:
                print(f"  Warning: SimpleGoodTuring (custom bins) failed: {e}")
                return None

        self.smoothing_methods["SimpleGoodTuring"] = try_sgt
        self.smoothing_methods["SimpleGoodTuring_3x"] = try_sgt_custom_bins

        # N-gram methods (for bigram/trigram datasets)
        self.ngram_methods: dict[str, Any] = {
            "KneserNey_0.75": lambda freq: self._try_method(
                lambda: freqprob.KneserNey(freq, discount=0.75, logprob=True), "KneserNey_0.75"
            ),
            "KneserNey_0.5": lambda freq: self._try_method(
                lambda: freqprob.KneserNey(freq, discount=0.5, logprob=True), "KneserNey_0.5"
            ),
            "ModifiedKneserNey": lambda freq: self._try_method(
                lambda: freqprob.ModifiedKneserNey(freq, logprob=True), "ModifiedKneserNey"
            ),
        }

    def _try_method(self, method_func: Any, method_name: str) -> Any:
        """Try to create a method, return None if it fails."""
        try:
            return method_func()
        except Exception as e:
            print(f"  Warning: {method_name} failed: {e}")
            return None

    def generate_datasets(self) -> None:
        """Generate benchmark datasets."""
        print("Generating benchmark datasets...")

        # Small datasets
        self.datasets["small_zipf"] = DatasetGenerator.create_zipf_distribution(100, 1000)
        self.datasets["small_uniform"] = DatasetGenerator.create_uniform_distribution(100, 1000)

        # Medium datasets
        self.datasets["medium_zipf"] = DatasetGenerator.create_zipf_distribution(1000, 10000)
        self.datasets["medium_power"] = DatasetGenerator.create_power_law_distribution(1000, 10000)

        # Large datasets
        self.datasets["large_zipf"] = DatasetGenerator.create_zipf_distribution(5000, 50000)
        self.datasets["large_power"] = DatasetGenerator.create_power_law_distribution(5000, 50000)

        # Very sparse dataset
        sparse_data = {}
        for i in range(1000):
            if np.random.random() < 0.1:  # Only 10% of words have counts
                sparse_data[f"word_{i:06d}"] = max(1, int(np.random.exponential(2)))
        self.datasets["sparse"] = sparse_data

        # N-gram datasets (for KneserNey, ModifiedKneserNey, InterpolatedSmoothing)
        self.datasets["bigram_small"] = DatasetGenerator.create_bigram_distribution(50, 500)
        self.datasets["bigram_medium"] = DatasetGenerator.create_bigram_distribution(100, 2000)
        self.datasets["trigram_small"] = DatasetGenerator.create_trigram_distribution(30, 300)
        self.datasets["trigram_medium"] = DatasetGenerator.create_trigram_distribution(50, 1000)

        # Generate test datasets (held-out data)
        for dataset_name in ["small_zipf", "medium_zipf", "large_zipf"]:
            base_dataset = self.datasets[dataset_name]
            vocab = list(base_dataset.keys())

            # Create test data with some unseen words
            test_data = {}
            for word in vocab[: len(vocab) // 2]:  # Use half the vocabulary
                if np.random.random() < 0.8:  # 80% chance to include
                    test_data[word] = max(1, int(np.random.poisson(2)))

            # Add some unseen words
            for i in range(10):
                test_data[f"unseen_word_{i}"] = 1

            self.test_datasets[dataset_name] = test_data

        print(
            f"Generated {len(self.datasets)} training datasets and {len(self.test_datasets)} test datasets"
        )

    def benchmark_creation_time(self) -> None:
        """Benchmark model creation time."""
        print("Benchmarking model creation time...")

        for dataset_name, dataset in self.datasets.items():
            print(
                f"  Dataset: {dataset_name} ({len(dataset)} words, {sum(dataset.values())} total)"
            )

            for method_name, method_func in self.smoothing_methods.items():
                times = []

                # Run multiple times for statistical significance
                for _ in range(5):
                    start_time = time.perf_counter()
                    try:
                        model = method_func(dataset)
                        end_time = time.perf_counter()
                        if model is not None:
                            times.append(end_time - start_time)
                    except Exception:
                        times.append(float("inf"))  # Failed

                # Calculate statistics
                valid_times = [t for t in times if t != float("inf")]
                if valid_times:
                    mean_time = statistics.mean(valid_times)
                    std_time = statistics.stdev(valid_times) if len(valid_times) > 1 else 0

                    result = BenchmarkResult(
                        method=method_name,
                        dataset=dataset_name,
                        metric="creation_time",
                        value=mean_time,
                        unit="seconds",
                        metadata={
                            "std_dev": std_time,
                            "num_runs": len(valid_times),
                            "vocab_size": len(dataset),
                            "total_count": sum(dataset.values()),
                        },
                    )
                    self.results.append(result)
                else:
                    # Method failed
                    result = BenchmarkResult(
                        method=method_name,
                        dataset=dataset_name,
                        metric="creation_time",
                        value=float("inf"),
                        unit="seconds",
                        metadata={"failed": True},
                    )
                    self.results.append(result)

    def benchmark_query_performance(self) -> None:
        """Benchmark query performance."""
        print("Benchmarking query performance...")

        for dataset_name, dataset in self.datasets.items():
            if len(dataset) > 2000:  # Skip very large datasets for query benchmarks
                continue

            print(f"  Dataset: {dataset_name}")

            # Create test queries
            vocab = list(dataset.keys())
            test_queries = vocab[: min(100, len(vocab))]  # Sample 100 words

            for method_name, method_func in self.smoothing_methods.items():
                try:
                    # Create model
                    model = method_func(dataset)
                    times = []
                    for _ in range(5):  # Multiple runs
                        start_time = time.perf_counter()
                        for word in test_queries:
                            _ = model(word)
                        end_time = time.perf_counter()
                        times.append((end_time - start_time) / len(test_queries))

                    mean_time = statistics.mean(times)
                    queries_per_sec = 1.0 / mean_time if mean_time > 0 else float("inf")

                    result = BenchmarkResult(
                        method=method_name,
                        dataset=dataset_name,
                        metric="query_time",
                        value=mean_time,
                        unit="seconds",
                        metadata={
                            "queries_per_second": queries_per_sec,
                            "num_queries": len(test_queries),
                        },
                    )
                    self.results.append(result)

                except Exception:
                    pass  # Skip failed methods

    def benchmark_memory_usage(self) -> None:
        """Benchmark memory usage."""
        print("Benchmarking memory usage...")

        if not HAS_PSUTIL:
            print("  Skipping memory benchmarks (psutil not available)")
            return

        for dataset_name, dataset in self.datasets.items():
            print(f"  Dataset: {dataset_name}")

            for method_name, method_func in self.smoothing_methods.items():
                try:
                    # Measure memory before
                    process = psutil.Process()
                    memory_before = process.memory_info().rss

                    # Create model
                    model = method_func(dataset)
                    memory_after = process.memory_info().rss
                    memory_delta = (memory_after - memory_before) / 1024 / 1024  # MB

                    result = BenchmarkResult(
                        method=method_name,
                        dataset=dataset_name,
                        metric="memory_usage",
                        value=memory_delta,
                        unit="MB",
                        metadata={
                            "vocab_size": len(dataset),
                            "total_count": sum(dataset.values()),
                        },
                    )
                    self.results.append(result)

                    # Clean up
                    del model

                except Exception:
                    pass  # Skip failed methods

    def benchmark_perplexity(self) -> None:
        """Benchmark perplexity on test data."""
        print("Benchmarking perplexity...")

        for dataset_name, dataset in self.datasets.items():
            if dataset_name not in self.test_datasets:
                continue

            test_data = self.test_datasets[dataset_name]
            test_words = []
            for word, count in test_data.items():
                test_words.extend([word] * count)

            print(f"  Dataset: {dataset_name} (test set: {len(test_words)} words)")

            for method_name, method_func in self.smoothing_methods.items():
                try:
                    model = method_func(dataset)
                    perplexity = freqprob.perplexity(model, test_words)

                    result = BenchmarkResult(
                        method=method_name,
                        dataset=dataset_name,
                        metric="perplexity",
                        value=perplexity,
                        unit="",
                        metadata={
                            "test_words": len(test_words),
                            "unique_test_words": len(set(test_words)),
                        },
                    )
                    self.results.append(result)

                except Exception:
                    # Some methods might fail on certain datasets
                    pass

    def benchmark_ngram_methods(self) -> None:
        """Benchmark n-gram specific methods (KneserNey, ModifiedKneserNey, InterpolatedSmoothing)."""
        print("Benchmarking n-gram methods...")

        # Benchmark KneserNey and ModifiedKneserNey on bigram datasets
        ngram_dataset_names = [
            name for name in self.datasets if "bigram" in name or "trigram" in name
        ]

        for dataset_name in ngram_dataset_names:
            dataset = self.datasets[dataset_name]
            print(f"  Dataset: {dataset_name} ({len(dataset)} n-grams)")

            for method_name, method_func in self.ngram_methods.items():
                times = []

                # Run multiple times
                for _ in range(3):
                    start_time = time.perf_counter()
                    try:
                        model = method_func(dataset)
                        end_time = time.perf_counter()
                        if model is not None:
                            times.append(end_time - start_time)
                        else:
                            times.append(float("inf"))
                    except Exception as e:
                        print(f"    Warning: {method_name} failed on {dataset_name}: {e}")
                        times.append(float("inf"))

                # Calculate statistics
                valid_times = [t for t in times if t != float("inf")]
                if valid_times:
                    mean_time = statistics.mean(valid_times)

                    result = BenchmarkResult(
                        method=method_name,
                        dataset=dataset_name,
                        metric="creation_time",
                        value=mean_time,
                        unit="seconds",
                        metadata={
                            "ngram_count": len(dataset),
                            "total_count": sum(dataset.values()),
                        },
                    )
                    self.results.append(result)

        # Benchmark InterpolatedSmoothing (trigram + bigram)
        print("  Benchmarking InterpolatedSmoothing...")
        for lambda_weight in [0.5, 0.7, 0.9]:
            # Find matching trigram and bigram datasets
            if "trigram_small" in self.datasets and "bigram_small" in self.datasets:
                trigram_data = self.datasets["trigram_small"]
                bigram_data = self.datasets["bigram_small"]

                times = []
                for _ in range(3):
                    start_time = time.perf_counter()
                    try:
                        model = freqprob.InterpolatedSmoothing(
                            trigram_data, bigram_data, lambda_weight=lambda_weight, logprob=True
                        )
                        end_time = time.perf_counter()
                        times.append(end_time - start_time)
                    except Exception as e:
                        print(f"    Warning: InterpolatedSmoothing λ={lambda_weight} failed: {e}")
                        times.append(float("inf"))

                valid_times = [t for t in times if t != float("inf")]
                if valid_times:
                    mean_time = statistics.mean(valid_times)

                    result = BenchmarkResult(
                        method=f"InterpolatedSmoothing_λ{lambda_weight}",
                        dataset="trigram_small+bigram_small",
                        metric="creation_time",
                        value=mean_time,
                        unit="seconds",
                        metadata={
                            "lambda_weight": lambda_weight,
                            "high_order_count": len(trigram_data),
                            "low_order_count": len(bigram_data),
                        },
                    )
                    self.results.append(result)

    def benchmark_cross_entropy(self) -> None:
        """Benchmark cross-entropy on test data."""
        print("Benchmarking cross-entropy...")

        for dataset_name, dataset in self.datasets.items():
            # Skip n-gram datasets for unigram methods
            if "bigram" in dataset_name or "trigram" in dataset_name:
                continue

            if dataset_name not in self.test_datasets:
                continue

            test_data = self.test_datasets[dataset_name]
            test_words = []
            for word, count in test_data.items():
                test_words.extend([word] * count)

            print(f"  Dataset: {dataset_name} (test set: {len(test_words)} words)")

            for method_name, method_func in self.smoothing_methods.items():
                try:
                    model = method_func(dataset)
                    ce = freqprob.cross_entropy(model, test_words)

                    result = BenchmarkResult(
                        method=method_name,
                        dataset=dataset_name,
                        metric="cross_entropy",
                        value=ce,
                        unit="bits",
                        metadata={
                            "test_words": len(test_words),
                            "unique_test_words": len(set(test_words)),
                        },
                    )
                    self.results.append(result)

                except Exception as e:
                    print(f"    Warning: {method_name} cross-entropy failed: {e}")

    def benchmark_scaling(self) -> None:
        """Benchmark scaling behavior."""
        print("Benchmarking scaling behavior...")

        # Create datasets of increasing size
        sizes = [100, 500, 1000, 2000, 5000]

        for size in sizes:
            if size > 2000:  # Skip very large for time reasons
                continue

            dataset = DatasetGenerator.create_zipf_distribution(size, size * 10)
            dataset_name = f"scaling_{size}"

            print(f"  Size: {size} words")

            # Test a subset of methods for scaling
            scaling_methods = ["MLE", "Laplace", "ELE", "Bayesian_0.5"]

            for method_name in scaling_methods:
                if method_name not in self.smoothing_methods:
                    continue

                method_func = self.smoothing_methods[method_name]

                try:
                    start_time = time.perf_counter()
                    model = method_func(dataset)
                    end_time = time.perf_counter()

                    if model is not None:
                        creation_time = end_time - start_time

                        result = BenchmarkResult(
                            method=method_name,
                            dataset=dataset_name,
                            metric="scaling_time",
                            value=creation_time,
                            unit="seconds",
                            metadata={"vocab_size": size, "total_count": size * 10},
                        )
                        self.results.append(result)

                except Exception:
                    pass

    def run_all_benchmarks(self) -> None:
        """Run all benchmarks."""
        print("Starting FreqProb Performance Benchmarks")
        print("=" * 50)

        self.generate_datasets()
        self.benchmark_creation_time()
        self.benchmark_query_performance()
        self.benchmark_memory_usage()
        self.benchmark_perplexity()
        self.benchmark_cross_entropy()
        self.benchmark_ngram_methods()
        self.benchmark_scaling()

        print(f"\nCompleted benchmarks: {len(self.results)} results")

    def analyze_results(self) -> dict[str, Any]:
        """Analyze benchmark results."""
        analysis: dict[str, Any] = {
            "summary": {},
            "best_performers": {},
            "scaling_analysis": {},
            "failure_analysis": {},
        }

        # Group results by metric
        by_metric = defaultdict(list)
        for result in self.results:
            by_metric[result.metric].append(result)

        # Summary statistics
        analysis["summary"] = {
            "total_results": len(self.results),
            "metrics_tested": list(by_metric.keys()),
            "methods_tested": list({r.method for r in self.results}),
            "datasets_tested": list({r.dataset for r in self.results}),
        }

        # Best performers for each metric
        for metric, results in by_metric.items():
            if metric == "perplexity":
                # Lower is better for perplexity
                valid_results = [r for r in results if r.value != float("inf")]
                if valid_results:
                    best = min(valid_results, key=lambda x: x.value)
                    analysis["best_performers"][metric] = {
                        "method": best.method,
                        "dataset": best.dataset,
                        "value": best.value,
                    }
            elif metric in ["creation_time", "query_time"]:
                # Lower is better for time metrics
                valid_results = [r for r in results if r.value != float("inf")]
                if valid_results:
                    best = min(valid_results, key=lambda x: x.value)
                    analysis["best_performers"][metric] = {
                        "method": best.method,
                        "dataset": best.dataset,
                        "value": best.value,
                    }

        # Scaling analysis
        scaling_results = [r for r in self.results if r.metric == "scaling_time"]
        if scaling_results:
            by_method = defaultdict(list)
            for result in scaling_results:
                by_method[result.method].append(result)

            scaling_trends = {}
            for method, method_results in by_method.items():
                # Sort by vocabulary size
                sorted_results = sorted(method_results, key=lambda x: x.metadata["vocab_size"])

                sizes = [r.metadata["vocab_size"] for r in sorted_results]
                times = [r.value for r in sorted_results]

                scaling_trends[method] = {
                    "sizes": sizes,
                    "times": times,
                    "complexity": "Unknown",  # Could fit polynomial here
                }

            analysis["scaling_analysis"] = scaling_trends

        # Failure analysis
        failed_results = [
            r for r in self.results if r.value == float("inf") or r.metadata.get("failed", False)
        ]

        failure_by_method: dict[str, int] = defaultdict(int)
        for result in failed_results:
            failure_by_method[result.method] += 1

        analysis["failure_analysis"] = {
            "total_failures": len(failed_results),
            "failures_by_method": dict(failure_by_method),
        }

        return analysis

    def save_results(self, output_dir: str, format_type: str = "json") -> None:
        """Save benchmark results."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # Convert results to dictionaries
        results_data = [result.to_dict() for result in self.results]

        if format_type in ["json", "all"]:
            json_file = output_path / "benchmark_results.json"
            with open(json_file, "w") as f:
                json.dump(results_data, f, indent=2)
            print(f"Saved JSON results to {json_file}")

        if format_type in ["csv", "all"]:
            csv_file = output_path / "benchmark_results.csv"
            with open(csv_file, "w", newline="") as f:
                if results_data:
                    fieldnames = [
                        "method",
                        "dataset",
                        "metric",
                        "value",
                        "unit",
                        "timestamp",
                    ]
                    writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                    writer.writeheader()
                    for result in results_data:
                        writer.writerow(result)
            print(f"Saved CSV results to {csv_file}")

        # Save analysis
        analysis = self.analyze_results()

        if format_type in ["json", "all"]:
            analysis_file = output_path / "benchmark_analysis.json"
            with open(analysis_file, "w") as f:
                json.dump(analysis, f, indent=2)
            print(f"Saved analysis to {analysis_file}")

        if format_type in ["html", "all"]:
            self.generate_html_report(output_path, analysis)

    def generate_html_report(self, output_path: Path, analysis: dict[str, Any]) -> None:
        """Generate HTML report."""
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>FreqProb Performance Benchmarks</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1, h2, h3 {{ color: #333; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .metric {{ font-weight: bold; color: #666; }}
        .best {{ background-color: #e8f5e8; }}
        .failed {{ background-color: #ffe8e8; }}
    </style>
</head>
<body>
    <h1>FreqProb Performance Benchmarks</h1>
    <p>Generated on {time.strftime("%Y-%m-%d %H:%M:%S")}</p>

    <h2>Summary</h2>
    <ul>
        <li>Total Results: {analysis["summary"]["total_results"]}</li>
        <li>Methods Tested: {", ".join(analysis["summary"]["methods_tested"])}</li>
        <li>Datasets Tested: {", ".join(analysis["summary"]["datasets_tested"])}</li>
        <li>Metrics: {", ".join(analysis["summary"]["metrics_tested"])}</li>
    </ul>

    <h2>Best Performers</h2>
    <table>
        <tr><th>Metric</th><th>Best Method</th><th>Dataset</th><th>Value</th></tr>
"""

        for metric, best in analysis["best_performers"].items():
            html_content += f"""
        <tr>
            <td class="metric">{metric}</td>
            <td class="best">{best["method"]}</td>
            <td>{best["dataset"]}</td>
            <td>{best["value"]:.4f}</td>
        </tr>"""

        html_content += """
    </table>

    <h2>All Results</h2>
    <table>
        <tr><th>Method</th><th>Dataset</th><th>Metric</th><th>Value</th><th>Unit</th></tr>
"""

        for result in self.results:
            css_class = "failed" if result.value == float("inf") else ""
            value_str = "FAILED" if result.value == float("inf") else f"{result.value:.4f}"

            html_content += f"""
        <tr class="{css_class}">
            <td>{result.method}</td>
            <td>{result.dataset}</td>
            <td>{result.metric}</td>
            <td>{value_str}</td>
            <td>{result.unit}</td>
        </tr>"""

        html_content += (
            """
    </table>

    <h2>Failure Analysis</h2>
    <p>Total Failures: """
            + str(analysis["failure_analysis"]["total_failures"])
            + """</p>
    <table>
        <tr><th>Method</th><th>Failure Count</th></tr>
"""
        )

        for method, count in analysis["failure_analysis"]["failures_by_method"].items():
            html_content += f"""
        <tr>
            <td>{method}</td>
            <td>{count}</td>
        </tr>"""

        html_content += """
    </table>
</body>
</html>
"""

        html_file = output_path / "benchmark_report.html"
        with open(html_file, "w") as f:
            f.write(html_content)
        print(f"Saved HTML report to {html_file}")

    def print_summary(self) -> None:
        """Print a summary of results."""
        analysis = self.analyze_results()

        print("\nBENCHMARK SUMMARY")
        print("=" * 50)
        print(f"Total results: {analysis['summary']['total_results']}")
        print(f"Methods tested: {len(analysis['summary']['methods_tested'])}")
        print(f"Datasets tested: {len(analysis['summary']['datasets_tested'])}")
        print()

        print("BEST PERFORMERS:")
        print("-" * 20)
        for metric, best in analysis["best_performers"].items():
            print(f"{metric:15}: {best['method']} on {best['dataset']} ({best['value']:.4f})")
        print()

        if analysis["failure_analysis"]["total_failures"] > 0:
            print("FAILURES:")
            print("-" * 10)
            for method, count in analysis["failure_analysis"]["failures_by_method"].items():
                print(f"{method}: {count} failures")
            print()

        # Performance insights
        print("KEY INSIGHTS:")
        print("-" * 15)

        # Find fastest method overall
        time_results = [
            r for r in self.results if r.metric == "creation_time" and r.value != float("inf")
        ]
        if time_results:
            fastest = min(time_results, key=lambda x: x.value)
            print(f"• Fastest creation: {fastest.method} ({fastest.value:.4f}s)")

        # Find most accurate method
        perplexity_results = [
            r for r in self.results if r.metric == "perplexity" and r.value != float("inf")
        ]
        if perplexity_results:
            best_perplexity = min(perplexity_results, key=lambda x: x.value)
            print(f"• Best perplexity: {best_perplexity.method} ({best_perplexity.value:.2f})")

        # Memory efficiency
        memory_results = [
            r for r in self.results if r.metric == "memory_usage" and r.value != float("inf")
        ]
        if memory_results:
            most_efficient = min(memory_results, key=lambda x: x.value)
            print(
                f"• Most memory efficient: {most_efficient.method} ({most_efficient.value:.2f} MB)"
            )


def main() -> None:
    """Run the benchmarking script."""
    parser = argparse.ArgumentParser(description="Run FreqProb performance benchmarks")
    parser.add_argument(
        "--output", default="benchmark_results", help="Output directory for results"
    )
    parser.add_argument(
        "--format",
        choices=["json", "csv", "html", "all"],
        default="all",
        help="Output format",
    )
    parser.add_argument(
        "--quick", action="store_true", help="Run quick benchmarks (fewer datasets)"
    )

    args = parser.parse_args()

    # Set numpy seed for reproducibility
    np.random.seed(42)

    benchmark = PerformanceBenchmark()

    if args.quick:
        # Reduce datasets for quick testing
        print("Running quick benchmarks...")
        benchmark.datasets = {
            "small_zipf": DatasetGenerator.create_zipf_distribution(50, 500),
            "medium_zipf": DatasetGenerator.create_zipf_distribution(200, 2000),
        }
        benchmark.test_datasets = {
            "small_zipf": DatasetGenerator.create_zipf_distribution(25, 100),
            "medium_zipf": DatasetGenerator.create_zipf_distribution(100, 500),
        }
        # Run limited benchmarks
        benchmark.benchmark_creation_time()
        benchmark.benchmark_query_performance()
        benchmark.benchmark_perplexity()
        benchmark.benchmark_cross_entropy()
    else:
        # Run comprehensive benchmarks
        benchmark.run_all_benchmarks()

    benchmark.print_summary()
    benchmark.save_results(args.output, args.format)

    print(f"\nBenchmark complete! Results saved to {args.output}/")


if __name__ == "__main__":
    main()
