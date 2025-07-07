#!/usr/bin/env python3
"""
FreqProb Validation Report Generator

This script runs comprehensive validation tests and generates detailed reports
on numerical stability, statistical correctness, performance, and regression testing.
"""

import argparse
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import freqprob
from freqprob.validation import BenchmarkSuite, PerformanceProfiler, ValidationSuite

try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import matplotlib.pyplot as plt
    import seaborn as sns

    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False


class ValidationReportGenerator:
    """Comprehensive validation report generator for FreqProb."""

    def __init__(self, output_dir: Path, verbose: bool = True):
        """Initialize the validation report generator.


        Args:
            output_dir: Directory to save reports and artifacts
            verbose: Whether to print progress information
        """
        self.output_dir = Path(output_dir)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose

        self.profiler = PerformanceProfiler()
        self.validator = ValidationSuite(self.profiler)
        self.benchmarker = BenchmarkSuite(self.profiler)

        self.report_data = {
            "metadata": {
                "freqprob_version": getattr(freqprob, "__version__", "unknown"),
                "generation_time": time.time(),
                "python_version": sys.version,
                "platform": sys.platform,
            },
            "test_results": {},
            "performance_results": {},
            "benchmark_results": {},
            "summary": {},
        }

    def log(self, message: str) -> None:
        """Log a message if verbose mode is enabled."""
        if self.verbose:

            print(f"[{time.strftime('%H:%M:%S')}] {message}")

    def generate_test_distributions(self) -> list[dict[str, int]]:
        """Generate diverse test distributions for validation."""
        distributions = []

        # Small uniform distribution
        distributions.append({f"word_{i}": 10 for i in range(5)})

        # Medium Zipfian distribution
        zipf_dist = {}
        for i in range(100):
            freq = max(1, int(1000 / (i + 1)))
            zipf_dist[f"term_{i}"] = freq
        distributions.append(zipf_dist)

        # Large sparse distribution
        sparse_dist = {}
        for i in range(1000):
            if i % 10 == 0:  # Only every 10th word has a count
                sparse_dist[f"sparse_{i}"] = max(1, int(100 / (i // 10 + 1)))
        distributions.append(sparse_dist)

        # Extreme frequency differences
        extreme_dist = {
            "very_common": 10000,
            "common": 1000,
            "medium": 100,
            "rare": 10,
            "very_rare": 1,
        }
        distributions.append(extreme_dist)

        # Single element distribution
        distributions.append({"singleton": 42})

        # Power law distribution
        power_dist = {}
        for i in range(200):
            freq = max(1, int(500 * ((i + 1) ** -1.5)))
            power_dist[f"power_{i}"] = freq
        distributions.append(power_dist)

        return distributions

    def run_numerical_stability_tests(self) -> dict[str, Any]:
        """Run numerical stability tests."""
        self.log("Running numerical stability tests...")

        test_distributions = self.generate_test_distributions()
        stability_results = {}

        # Test basic smoothing methods
        basic_methods = [
            (freqprob.MLE, {}),
            (freqprob.Laplace, {"bins": 1000}),
            (freqprob.ELE, {"bins": 1000}),
            (freqprob.Lidstone, {"gamma": 0.5, "bins": 1000}),
            (freqprob.BayesianSmoothing, {"alpha": 0.5}),
            (freqprob.Uniform, {"unobs_prob": 0.1}),
        ]

        for method_class, kwargs in basic_methods:
            method_name = method_class.__name__
            self.log(f"  Testing {method_name}...")

            method_results = self.validator.validate_numerical_stability(
                method_class, test_distributions, **kwargs
            )

            stability_results[method_name] = {
                "total_tests": len(method_results),
                "passed": sum(1 for r in method_results if r.passed),
                "failed": sum(1 for r in method_results if not r.passed),
                "details": [r.to_dict() for r in method_results],
            }

        return stability_results

    def run_statistical_correctness_tests(self) -> dict[str, Any]:
        """Run statistical correctness validation."""
        self.log("Running statistical correctness tests...")

        test_dist = {"apple": 60, "banana": 30, "cherry": 10}
        correctness_results = {}

        methods_to_test = [
            (freqprob.MLE, {}),
            (freqprob.Laplace, {"bins": 100}),
            (freqprob.ELE, {"bins": 100}),
            (freqprob.Lidstone, {"gamma": 0.5, "bins": 100}),
            (freqprob.BayesianSmoothing, {"alpha": 0.5}),
        ]

        for method_class, kwargs in methods_to_test:
            method_name = method_class.__name__
            self.log(f"  Testing {method_name}...")

            result = self.validator.validate_statistical_correctness(
                method_class, test_dist, **kwargs
            )

            correctness_results[method_name] = result.to_dict()

        return correctness_results

    def run_regression_tests(self) -> dict[str, Any]:
        """Run regression tests against reference implementations."""
        self.log("Running regression tests...")

        # Run the regression test module
        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "tests/test_regression_reference.py",
                    "-v",
                    "--tb=short",
                ],
                capture_output=True,
                text=True,
                cwd=self.output_dir.parent,
            )

            regression_results = {
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "summary": "PASSED" if result.returncode == 0 else "FAILED",
            }
        except Exception as e:
            regression_results = {"returncode": -1, "error": str(e), "summary": "ERROR"}

        return regression_results

    def run_performance_benchmarks(self) -> dict[str, Any]:
        """Run performance benchmarks."""
        self.log("Running performance benchmarks...")

        test_dist = {f"word_{i}": max(1, int(1000 / (i + 1))) for i in range(100)}
        performance_results = {}

        # Benchmark creation scaling
        self.log("  Benchmarking creation scaling...")
        vocab_sizes = [10, 50, 100, 500] if HAS_NUMPY else [10, 50]

        methods_to_benchmark = [freqprob.MLE, freqprob.Laplace, freqprob.ELE]

        scaling_results = self.benchmarker.benchmark_creation_scaling(
            methods_to_benchmark, vocab_sizes, bins=1000
        )

        performance_results["creation_scaling"] = {
            method: [m.to_dict() for m in metrics] for method, metrics in scaling_results.items()
        }

        # Benchmark query performance
        self.log("  Benchmarking query performance...")
        test_methods = {
            "MLE": freqprob.MLE(test_dist, logprob=False),
            "Laplace": freqprob.Laplace(test_dist, bins=200, logprob=False),
        }

        query_counts = [10, 50, 100] if HAS_NUMPY else [10, 25]
        query_results = self.benchmarker.benchmark_query_scaling(test_methods, query_counts)

        performance_results["query_scaling"] = {
            method: [m.to_dict() for m in metrics] for method, metrics in query_results.items()
        }

        # Benchmark method comparison
        self.log("  Benchmarking method comparison...")
        method_configs = [
            (freqprob.MLE, {}),
            (freqprob.Laplace, {"bins": 100}),
            (freqprob.ELE, {"bins": 100}),
            (freqprob.Lidstone, {"gamma": 0.5, "bins": 100}),
        ]

        comparison_results = self.benchmarker.compare_methods(method_configs, test_dist)
        performance_results["method_comparison"] = {
            name: metrics.to_dict() for name, metrics in comparison_results.items()
        }

        return performance_results

    def run_property_based_tests(self) -> dict[str, Any]:
        """Run property-based tests with Hypothesis."""
        self.log("Running property-based tests...")

        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "tests/test_property_based.py",
                    "-v",
                    "--tb=short",
                    "--hypothesis-show-statistics",
                ],
                capture_output=True,
                text=True,
                cwd=self.output_dir.parent,
            )

            property_results = {
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "summary": "PASSED" if result.returncode == 0 else "FAILED",
            }
        except Exception as e:
            property_results = {"returncode": -1, "error": str(e), "summary": "ERROR"}

        return property_results

    def generate_summary_statistics(self, results: dict[str, Any]) -> dict[str, Any]:
        """Generate overall summary statistics."""
        summary = {
            "validation_overview": {},
            "performance_overview": {},
            "overall_status": "UNKNOWN",
        }

        # Validation summary
        if "numerical_stability" in results:
            stability_total = 0
            stability_passed = 0

            for method_results in results["numerical_stability"].values():
                stability_total += method_results["total_tests"]
                stability_passed += method_results["passed"]

            summary["validation_overview"]["numerical_stability"] = {
                "total_tests": stability_total,
                "passed_tests": stability_passed,
                "success_rate": stability_passed / stability_total if stability_total > 0 else 0,
            }

        if "statistical_correctness" in results:
            correctness_passed = sum(
                1 for result in results["statistical_correctness"].values() if result["passed"]
            )
            correctness_total = len(results["statistical_correctness"])

            summary["validation_overview"]["statistical_correctness"] = {
                "total_tests": correctness_total,
                "passed_tests": correctness_passed,
                "success_rate": (
                    correctness_passed / correctness_total if correctness_total > 0 else 0
                ),
            }

        # Performance summary
        if "performance_benchmarks" in results:
            perf_data = results["performance_benchmarks"]

            if "method_comparison" in perf_data:
                fastest_method = min(
                    perf_data["method_comparison"].items(), key=lambda x: x[1]["duration_seconds"]
                )

                summary["performance_overview"]["fastest_method"] = {
                    "name": fastest_method[0],
                    "duration": fastest_method[1]["duration_seconds"],
                }

        # Overall status
        all_passed = []

        if "regression_tests" in results:
            all_passed.append(results["regression_tests"]["summary"] == "PASSED")

        if "property_based_tests" in results:
            all_passed.append(results["property_based_tests"]["summary"] == "PASSED")

        if all_passed:
            summary["overall_status"] = "PASSED" if all(all_passed) else "FAILED"

        return summary

    def generate_html_report(self, results: dict[str, Any]) -> None:
        """Generate HTML validation report."""
        html_content = f"""
        
<!DOCTYPE html>
<html>
<head>
    <title>FreqProb Validation Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
        h1, h2, h3 {{ color: #333; }}
        .summary {{ background: #f0f0f0; padding: 20px; border-radius: 5px; margin: 20px 0; }}
        .passed {{ color: #28a745; font-weight: bold; }}
        .failed {{ color: #dc3545; font-weight: bold; }}
        .warning {{ color: #ffc107; font-weight: bold; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .metric {{ font-family: monospace; }}
        pre {{ background: #f8f9fa; padding: 10px; border-radius: 3px; overflow-x: auto; }}
        .status-passed {{ background-color: #d4edda; }}
        .status-failed {{ background-color: #f8d7da; }}
    </style>
</head>
<body>
    <h1>FreqProb Validation Report</h1>

    <div class="summary">
        <h2>Summary</h2>
        <p><strong>Generated:</strong> {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p><strong>FreqProb Version:</strong> {results.get('metadata', {}).get('freqprob_version', 'unknown')}</p>
        <p><strong>Python Version:</strong> {results.get('metadata', {}).get('python_version', 'unknown')}</p>
        <p><strong>Overall Status:</strong>
           <span class="{'passed' if results.get('summary', {}).get('overall_status') == 'PASSED' else 'failed'}">
           {results.get('summary', {}).get('overall_status', 'UNKNOWN')}
           </span>
        </p>
    </div>
"""

        # Numerical Stability Section
        if "numerical_stability" in results:
            html_content += """
    <h2>Numerical Stability Tests</h2>
    <table>
        <tr><th>Method</th><th>Total Tests</th><th>Passed</th><th>Failed</th><th>Success Rate</th></tr>
"""
            for method, data in results["numerical_stability"].items():

                success_rate = (
                    data["passed"] / data["total_tests"] if data["total_tests"] > 0 else 0
                )
                status_class = "status-passed" if success_rate > 0.95 else "status-failed"

                html_content += f"""
        <tr class="{status_class}">
            <td>{method}</td>
            <td>{data['total_tests']}</td>
            <td>{data['passed']}</td>
            <td>{data['failed']}</td>
            <td>{success_rate:.1%}</td>
        </tr>"""
        html_content += "</table>"

        # Statistical Correctness Section
        if "statistical_correctness" in results:
            html_content += """
    <h2>Statistical Correctness Tests</h2>
    <table>
        <tr><th>Method</th><th>Status</th><th>Error Message</th></tr>
"""
            for method, data in results["statistical_correctness"].items():

                status = "PASSED" if data["passed"] else "FAILED"
                status_class = "status-passed" if data["passed"] else "status-failed"
                error_msg = data.get("error_message", "") or "N/A"

                html_content += f"""
        <tr class="{status_class}">
            <td>{method}</td>
            <td><span class="{'passed' if data['passed'] else 'failed'}">{status}</span></td>
            <td>{error_msg}</td>
        </tr>"""
        html_content += "</table>"

        # Performance Benchmarks Section
        if "performance_benchmarks" in results:
            html_content += """
    <h2>Performance Benchmarks</h2>
    <h3>Method Comparison</h3>
    <table>
        <tr><th>Method</th><th>Duration (s)</th><th>Memory Peak (MB)</th><th>Memory Delta (MB)</th></tr>
"""
            if "method_comparison" in results["performance_benchmarks"]:

                for method, data in results["performance_benchmarks"]["method_comparison"].items():
                    html_content += f"""
        <tr>
            <td>{method}</td>
            <td class="metric">{data['duration_seconds']:.4f}</td>
            <td class="metric">{data['memory_peak_mb']:.2f}</td>
            <td class="metric">{data['memory_delta_mb']:.2f}</td>
        </tr>"""
        html_content += "</table>"

        # Test Results Summary
        regression_status = results.get("regression_tests", {}).get("summary", "NOT_RUN")
        property_status = results.get("property_based_tests", {}).get("summary", "NOT_RUN")

        html_content += f"""
    <h2>Additional Test Results</h2>
    <table>
        <tr><th>Test Suite</th><th>Status</th></tr>
        <tr class="{'status-passed' if regression_status == 'PASSED' else 'status-failed'}">
            <td>Regression Tests</td>
            <td><span class="{'passed' if regression_status == 'PASSED' else 'failed'}">{regression_status}</span></td>
        </tr>
        <tr class="{'status-passed' if property_status == 'PASSED' else 'status-failed'}">
            <td>Property-Based Tests</td>
            <td><span class="{'passed' if property_status == 'PASSED' else 'failed'}">{property_status}</span></td>
        </tr>
    </table>

    <h2>Raw Data</h2>
    <p>Complete validation data is available in the JSON report files.</p>

</body>
</html>"""

        # Save HTML report
        html_path = self.output_dir / "validation_report.html"
        with open(html_path, "w") as f:
            f.write(html_content)

        self.log(f"HTML report saved to {html_path}")

    def run_full_validation(self) -> dict[str, Any]:
        """Run complete validation suite."""
        self.log("Starting comprehensive FreqProb validation...")

        try:
            # Run all validation components
            self.report_data["test_results"][
                "numerical_stability"
            ] = self.run_numerical_stability_tests()
            self.report_data["test_results"][
                "statistical_correctness"
            ] = self.run_statistical_correctness_tests()
            self.report_data["test_results"]["regression_tests"] = self.run_regression_tests()
            self.report_data["test_results"][
                "property_based_tests"
            ] = self.run_property_based_tests()

            self.report_data["performance_results"][
                "performance_benchmarks"
            ] = self.run_performance_benchmarks()

            # Generate summary
            self.report_data["summary"] = self.generate_summary_statistics(
                self.report_data["test_results"]
            )

            # Save detailed JSON report
            json_path = self.output_dir / "validation_report.json"
            with open(json_path, "w") as f:
                json.dump(self.report_data, f, indent=2)

            self.log(f"JSON report saved to {json_path}")

            # Generate HTML report
            combined_results = {
                **self.report_data["test_results"],
                **self.report_data["performance_results"],
            }
            combined_results["summary"] = self.report_data["summary"]
            combined_results["metadata"] = self.report_data["metadata"]
            self.generate_html_report(combined_results)

            # Export profiler results
            profiler_path = self.output_dir / "profiler_results.json"
            self.profiler.export_results(profiler_path, format="json")

            # Export benchmark results
            benchmark_path = self.output_dir / "benchmark_results.json"
            self.benchmarker.export_benchmark_report(benchmark_path)

            self.log("Validation complete!")
            return self.report_data

        except Exception as e:
            self.log(f"Error during validation: {e}")
            self.report_data["error"] = {"message": str(e), "traceback": traceback.format_exc()}
            return self.report_data


def main():
    """Main entry point for validation report generation."""
    parser = argparse.ArgumentParser(description="Generate FreqProb validation report")

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("validation_results"),
        help="Directory to save validation results",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output")
    parser.add_argument("--quick", action="store_true", help="Run quick validation (fewer tests)")

    args = parser.parse_args()

    # Create report generator
    generator = ValidationReportGenerator(output_dir=args.output_dir, verbose=not args.quiet)

    # Run validation
    results = generator.run_full_validation()

    # Print summary
    if not args.quiet:
        print("\n" + "=" * 60)
        print("VALIDATION SUMMARY")
        print("=" * 60)

        overall_status = results.get("summary", {}).get("overall_status", "UNKNOWN")
        print(f"Overall Status: {overall_status}")

        if "test_results" in results:
            if "numerical_stability" in results["test_results"]:
                stability_data = results["test_results"]["numerical_stability"]
                total_stability = sum(data["total_tests"] for data in stability_data.values())
                passed_stability = sum(data["passed"] for data in stability_data.values())
                print(f"Numerical Stability: {passed_stability}/{total_stability} tests passed")

            if "statistical_correctness" in results["test_results"]:
                correctness_data = results["test_results"]["statistical_correctness"]
                passed_correctness = sum(1 for data in correctness_data.values() if data["passed"])
                total_correctness = len(correctness_data)
                print(
                    f"Statistical Correctness: {passed_correctness}/{total_correctness} tests passed"
                )

        print(f"\nDetailed reports saved to: {args.output_dir}")
        print("- validation_report.html (human-readable)")
        print("- validation_report.json (detailed data)")
        print("- profiler_results.json (performance data)")
        print("- benchmark_results.json (benchmark data)")

    # Exit with appropriate code
    overall_status = results.get("summary", {}).get("overall_status", "UNKNOWN")
    if overall_status == "PASSED":
        sys.exit(0)
    elif overall_status == "FAILED":
        sys.exit(1)
    else:
        sys.exit(2)


if __name__ == "__main__":
    main()
