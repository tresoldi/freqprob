#!/usr/bin/env python3
"""FreqProb Validation Suite.

Comprehensive validation and testing for FreqProb smoothing methods.
Tests numerical stability, statistical correctness, and v0.4.0 features.

Usage:
    python scripts/validation.py [--output-dir DIR] [--quick] [--quiet]
"""

import argparse
import json
import math
import subprocess
import sys
import time
import traceback
import tracemalloc
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import freqprob

try:
    import psutil

    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


@dataclass
class PerformanceMetrics:
    """Performance measurement results."""

    operation_name: str
    duration_seconds: float
    memory_peak_mb: float
    memory_delta_mb: float
    cpu_percent: float = 0.0
    iterations: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "operation_name": self.operation_name,
            "duration_seconds": self.duration_seconds,
            "memory_peak_mb": self.memory_peak_mb,
            "memory_delta_mb": self.memory_delta_mb,
            "cpu_percent": self.cpu_percent,
            "iterations": self.iterations,
            "metadata": self.metadata,
        }


@dataclass
class ValidationResult:
    """Validation test result."""

    test_name: str
    passed: bool
    error_message: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "test_name": self.test_name,
            "passed": self.passed,
            "error_message": self.error_message,
            "details": self.details,
        }


class ValidationSuite:
    """Comprehensive validation suite for FreqProb."""

    def __init__(self, output_dir: Path, verbose: bool = True):
        """Initialize validation suite.

        Args:
            output_dir: Directory for output files
            verbose: Print progress messages
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose

        self.report_data: dict[str, Any] = {
            "metadata": {
                "freqprob_version": getattr(freqprob, "__version__", "unknown"),
                "generation_time": time.time(),
                "python_version": sys.version,
            },
            "results": {},
            "summary": {},
        }

    def log(self, message: str) -> None:
        """Log message if verbose."""
        if self.verbose:
            print(f"[{time.strftime('%H:%M:%S')}] {message}")

    def generate_test_distributions(self) -> dict[str, dict]:
        """Generate diverse test distributions."""
        distributions: dict[str, dict] = {}

        # Unigram distributions
        distributions["small_uniform"] = {f"word_{i}": 10 for i in range(5)}

        distributions["medium_zipf"] = {}
        for i in range(100):
            freq = max(1, int(1000 / (i + 1)))
            distributions["medium_zipf"][f"term_{i}"] = freq

        distributions["extreme_freq"] = {
            "very_common": 10000,
            "common": 1000,
            "medium": 100,
            "rare": 10,
            "very_rare": 1,
        }

        distributions["singleton"] = {"single": 42}

        # N-gram distributions (v0.4.0)
        distributions["bigram_small"] = {
            ("the", "cat"): 5,
            ("the", "dog"): 3,
            ("a", "cat"): 2,
            ("a", "dog"): 1,
            ("big", "cat"): 1,
        }

        distributions["trigram_small"] = {
            ("the", "big", "cat"): 3,
            ("a", "big", "dog"): 2,
            ("the", "small", "cat"): 1,
        }

        return distributions

    def profile_method(self, method_class: type, freq_dist: dict, **kwargs: Any) -> PerformanceMetrics:
        """Profile a single method creation.

        Args:
            method_class: Method class to test
            freq_dist: Frequency distribution
            **kwargs: Method parameters

        Returns:
            Performance metrics
        """
        tracemalloc.start()
        mem_before = tracemalloc.get_traced_memory()[0] / 1024 / 1024

        start_time = time.perf_counter()
        try:
            _ = method_class(freq_dist, **kwargs)  # type: ignore[call-arg]
            duration = time.perf_counter() - start_time

            mem_current, mem_peak = tracemalloc.get_traced_memory()
            mem_peak_mb = mem_peak / 1024 / 1024
            mem_delta_mb = (mem_current / 1024 / 1024) - mem_before

            tracemalloc.stop()

            return PerformanceMetrics(
                operation_name=method_class.__name__,
                duration_seconds=duration,
                memory_peak_mb=mem_peak_mb,
                memory_delta_mb=mem_delta_mb,
                metadata={"vocab_size": len(freq_dist), **kwargs},
            )
        except Exception as e:
            tracemalloc.stop()
            raise e

    def validate_numerical_stability(
        self, method_class: type, test_dists: dict[str, dict], **kwargs: Any
    ) -> list[ValidationResult]:
        """Test numerical stability across distributions.

        Args:
            method_class: Method to test
            test_dists: Test distributions
            **kwargs: Method parameters

        Returns:
            List of validation results
        """
        results = []

        for dist_name, dist in test_dists.items():
            # Skip n-gram dists for unigram methods
            if isinstance(next(iter(dist.keys())), tuple):
                continue

            try:
                scorer = method_class(dist, logprob=False, **kwargs)  # type: ignore[call-arg]

                # Test all elements
                for elem in dist.keys():
                    prob = scorer(elem)

                    # Check for invalid values
                    if math.isnan(prob) or math.isinf(prob) or prob < 0:
                        results.append(
                            ValidationResult(
                                test_name=f"{method_class.__name__}_{dist_name}_{elem}",
                                passed=False,
                                error_message=f"Invalid probability: {prob}",
                                details={"element": elem, "probability": float(prob)},
                            )
                        )
                    else:
                        results.append(
                            ValidationResult(
                                test_name=f"{method_class.__name__}_{dist_name}_{elem}",
                                passed=True,
                                details={"element": elem, "probability": float(prob)},
                            )
                        )

            except Exception as e:
                results.append(
                    ValidationResult(
                        test_name=f"{method_class.__name__}_{dist_name}",
                        passed=False,
                        error_message=str(e),
                    )
                )

        return results

    def validate_statistical_correctness(
        self, method_class: type, test_dist: dict[str, int], **kwargs: Any
    ) -> ValidationResult:
        """Test statistical correctness (probabilities sum to ~1).

        Args:
            method_class: Method to test
            test_dist: Test distribution
            **kwargs: Method parameters

        Returns:
            Validation result
        """
        try:
            scorer = method_class(test_dist, logprob=False, **kwargs)  # type: ignore[call-arg]

            # Sum probabilities for all elements
            total_prob = sum(scorer(elem) for elem in test_dist.keys())

            # Check if sums to ~1.0
            tolerance = 0.01
            if abs(total_prob - 1.0) > tolerance:
                return ValidationResult(
                    test_name=f"{method_class.__name__}_sum_to_one",
                    passed=False,
                    error_message=f"Probabilities sum to {total_prob}, expected ~1.0",
                    details={"total_probability": float(total_prob), "tolerance": tolerance},
                )

            return ValidationResult(
                test_name=f"{method_class.__name__}_sum_to_one",
                passed=True,
                details={"total_probability": float(total_prob)},
            )

        except Exception as e:
            return ValidationResult(
                test_name=f"{method_class.__name__}_sum_to_one",
                passed=False,
                error_message=str(e),
            )

    def run_numerical_stability_tests(self) -> dict[str, Any]:
        """Run numerical stability tests on all methods."""
        self.log("Running numerical stability tests...")

        test_dists = self.generate_test_distributions()
        results = {}

        # Unigram methods
        unigram_methods = [
            (freqprob.MLE, {}),
            (freqprob.Laplace, {"bins": 1000}),
            (freqprob.ELE, {"bins": 1000}),
            (freqprob.Lidstone, {"gamma": 0.5, "bins": 1000}),
            (freqprob.BayesianSmoothing, {"alpha": 0.5}),
            (freqprob.Uniform, {"unobs_prob": 0.1}),
            (freqprob.WittenBell, {"bins": 1000}),  # v0.4.0
        ]

        # v0.4.0: Add SimpleGoodTuring with bins parameter
        try:
            unigram_methods.append((freqprob.SimpleGoodTuring, {}))
            unigram_methods.append((freqprob.SimpleGoodTuring, {"bins": 2000}))  # Test bins
        except AttributeError:
            self.log("  SimpleGoodTuring not available")

        for method_class, kwargs in unigram_methods:
            method_name = method_class.__name__
            if kwargs:
                method_name = f"{method_name}_{list(kwargs.values())[0]}"

            self.log(f"  Testing {method_name}...")
            method_results = self.validate_numerical_stability(method_class, test_dists, **kwargs)

            results[method_name] = {
                "total_tests": len(method_results),
                "passed": sum(1 for r in method_results if r.passed),
                "failed": sum(1 for r in method_results if not r.passed),
                "details": [r.to_dict() for r in method_results],
            }

        # v0.4.0: Test n-gram methods
        self.log("  Testing n-gram methods...")
        ngram_methods = [
            (freqprob.KneserNey, {"discount": 0.75}, test_dists["bigram_small"]),
            (freqprob.ModifiedKneserNey, {}, test_dists["bigram_small"]),
        ]

        for method_class, kwargs, dist in ngram_methods:
            method_name = f"{method_class.__name__}"
            try:
                scorer = method_class(dist, logprob=False, **kwargs)  # type: ignore[call-arg]
                test_elem = list(dist.keys())[0]
                prob = scorer(test_elem)

                passed = not (math.isnan(prob) or math.isinf(prob) or prob < 0)
                results[method_name] = {
                    "total_tests": 1,
                    "passed": 1 if passed else 0,
                    "failed": 0 if passed else 1,
                    "details": [
                        {
                            "test_name": method_name,
                            "passed": passed,
                            "details": {"probability": float(prob)},
                        }
                    ],
                }
            except Exception as e:
                results[method_name] = {
                    "total_tests": 1,
                    "passed": 0,
                    "failed": 1,
                    "details": [{"test_name": method_name, "passed": False, "error": str(e)}],
                }

        # v0.4.0: Test InterpolatedSmoothing
        self.log("  Testing InterpolatedSmoothing...")
        try:
            trigram_dist = test_dists["trigram_small"]
            bigram_dist = test_dists["bigram_small"]
            interp = freqprob.InterpolatedSmoothing(
                trigram_dist, bigram_dist, lambda_weight=0.7, logprob=False
            )
            test_trigram = list(trigram_dist.keys())[0]
            prob = interp(test_trigram)

            passed = not (math.isnan(prob) or math.isinf(prob) or prob < 0)
            # Also check 1e-10 floor
            if prob > 0 and prob < 1e-10:
                passed = False

            results["InterpolatedSmoothing"] = {
                "total_tests": 1,
                "passed": 1 if passed else 0,
                "failed": 0 if passed else 1,
                "details": [
                    {
                        "test_name": "InterpolatedSmoothing",
                        "passed": passed,
                        "details": {"probability": float(prob), "floor_check": prob >= 1e-10},
                    }
                ],
            }
        except Exception as e:
            results["InterpolatedSmoothing"] = {
                "total_tests": 1,
                "passed": 0,
                "failed": 1,
                "details": [{"test_name": "InterpolatedSmoothing", "passed": False, "error": str(e)}],
            }

        return results

    def run_statistical_correctness_tests(self) -> dict[str, Any]:
        """Run statistical correctness tests."""
        self.log("Running statistical correctness tests...")

        test_dist = {"apple": 60, "banana": 30, "cherry": 10}
        results = {}

        methods = [
            (freqprob.MLE, {}),
            (freqprob.Laplace, {"bins": 100}),
            (freqprob.ELE, {"bins": 100}),
            (freqprob.Lidstone, {"gamma": 0.5, "bins": 100}),
            (freqprob.BayesianSmoothing, {"alpha": 0.5}),
        ]

        for method_class, kwargs in methods:
            self.log(f"  Testing {method_class.__name__}...")
            result = self.validate_statistical_correctness(method_class, test_dist, **kwargs)
            results[method_class.__name__] = result.to_dict()

        return results

    def run_pytest_tests(self, test_file: str, test_name: str) -> dict[str, Any]:
        """Run pytest test file via subprocess.

        Args:
            test_file: Path to test file
            test_name: Name for results

        Returns:
            Test results
        """
        self.log(f"Running {test_name}...")

        try:
            result = subprocess.run(
                [sys.executable, "-m", "pytest", test_file, "-v", "--tb=short", "--no-cov"],
                capture_output=True,
                text=True,
                cwd=self.output_dir.parent,
                timeout=120,
            )

            return {
                "returncode": result.returncode,
                "stdout": result.stdout[:5000],  # Limit output
                "stderr": result.stderr[:5000],
                "summary": "PASSED" if result.returncode == 0 else "FAILED",
            }
        except Exception as e:
            return {"returncode": -1, "error": str(e), "summary": "ERROR"}

    def generate_summary(self) -> dict[str, Any]:
        """Generate summary statistics."""
        summary: dict[str, Any] = {
            "validation_overview": {},
            "overall_status": "UNKNOWN",
        }

        results = self.report_data["results"]

        # Numerical stability summary
        if "numerical_stability" in results:
            total = sum(r["total_tests"] for r in results["numerical_stability"].values())
            passed = sum(r["passed"] for r in results["numerical_stability"].values())

            summary["validation_overview"]["numerical_stability"] = {
                "total_tests": total,
                "passed_tests": passed,
                "success_rate": passed / total if total > 0 else 0,
            }

        # Statistical correctness summary
        if "statistical_correctness" in results:
            total = len(results["statistical_correctness"])
            passed = sum(1 for r in results["statistical_correctness"].values() if r["passed"])

            summary["validation_overview"]["statistical_correctness"] = {
                "total_tests": total,
                "passed_tests": passed,
                "success_rate": passed / total if total > 0 else 0,
            }

        # Overall status
        all_passed = []
        for test_type in ["regression_tests", "property_tests"]:
            if test_type in results:
                all_passed.append(results[test_type]["summary"] == "PASSED")

        if all_passed:
            summary["overall_status"] = "PASSED" if all(all_passed) else "FAILED"

        return summary

    def generate_html_report(self) -> None:
        """Generate HTML report."""
        results = self.report_data["results"]
        summary = self.report_data["summary"]

        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>FreqProb Validation Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1, h2 {{ color: #333; }}
        .summary {{ background: #f0f0f0; padding: 20px; margin: 20px 0; }}
        .passed {{ color: #28a745; font-weight: bold; }}
        .failed {{ color: #dc3545; font-weight: bold; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; }}
        th {{ background: #f2f2f2; }}
        .status-passed {{ background: #d4edda; }}
        .status-failed {{ background: #f8d7da; }}
    </style>
</head>
<body>
    <h1>FreqProb Validation Report</h1>
    <div class="summary">
        <h2>Summary</h2>
        <p><strong>Generated:</strong> {time.strftime("%Y-%m-%d %H:%M:%S")}</p>
        <p><strong>Version:</strong> {self.report_data["metadata"]["freqprob_version"]}</p>
        <p><strong>Status:</strong> <span class="{"passed" if summary.get("overall_status") == "PASSED" else "failed"}">
           {summary.get("overall_status", "UNKNOWN")}
        </span></p>
    </div>

    <h2>Numerical Stability Tests</h2>
    <table>
        <tr><th>Method</th><th>Total</th><th>Passed</th><th>Failed</th><th>Rate</th></tr>
"""

        if "numerical_stability" in results:
            for method, data in results["numerical_stability"].items():
                rate = data["passed"] / data["total_tests"] if data["total_tests"] > 0 else 0
                status_class = "status-passed" if rate > 0.95 else "status-failed"
                html_content += f"""
        <tr class="{status_class}">
            <td>{method}</td>
            <td>{data["total_tests"]}</td>
            <td>{data["passed"]}</td>
            <td>{data["failed"]}</td>
            <td>{rate:.1%}</td>
        </tr>"""

        html_content += """
    </table>

    <h2>Statistical Correctness Tests</h2>
    <table>
        <tr><th>Method</th><th>Status</th><th>Error</th></tr>
"""

        if "statistical_correctness" in results:
            for method, data in results["statistical_correctness"].items():
                status = "PASSED" if data["passed"] else "FAILED"
                status_class = "status-passed" if data["passed"] else "status-failed"
                error = data.get("error_message", "") or "N/A"
                html_content += f"""
        <tr class="{status_class}">
            <td>{method}</td>
            <td><span class="{"passed" if data["passed"] else "failed"}">{status}</span></td>
            <td>{error}</td>
        </tr>"""

        html_content += """
    </table>
</body>
</html>"""

        html_path = self.output_dir / "validation_report.html"
        html_path.write_text(html_content)
        self.log(f"HTML report saved to {html_path}")

    def run_full_validation(self) -> dict[str, Any]:
        """Run complete validation suite."""
        self.log("Starting FreqProb validation...")

        try:
            # Core validation tests
            self.report_data["results"]["numerical_stability"] = self.run_numerical_stability_tests()
            self.report_data["results"]["statistical_correctness"] = (
                self.run_statistical_correctness_tests()
            )

            # Pytest-based tests
            self.report_data["results"]["regression_tests"] = self.run_pytest_tests(
                "tests/test_regression_reference.py", "regression tests"
            )
            self.report_data["results"]["property_tests"] = self.run_pytest_tests(
                "tests/test_property_based.py", "property-based tests"
            )

            # Generate summary
            self.report_data["summary"] = self.generate_summary()

            # Save JSON report
            json_path = self.output_dir / "validation_report.json"
            with open(json_path, "w") as f:
                json.dump(self.report_data, f, indent=2)
            self.log(f"JSON report saved to {json_path}")

            # Generate HTML report
            self.generate_html_report()

            self.log("Validation complete!")
            return self.report_data

        except Exception as e:
            self.log(f"Error during validation: {e}")
            self.report_data["error"] = {
                "message": str(e),
                "traceback": traceback.format_exc(),
            }
            return self.report_data


def main() -> None:
    """Run validation from command line."""
    parser = argparse.ArgumentParser(description="FreqProb validation suite")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("validation_results"),
        help="Output directory",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress output")
    parser.add_argument("--quick", action="store_true", help="Quick validation (skip slow tests)")

    args = parser.parse_args()

    # Run validation
    validator = ValidationSuite(output_dir=args.output_dir, verbose=not args.quiet)
    results = validator.run_full_validation()

    # Print summary
    if not args.quiet:
        print("\n" + "=" * 60)
        print("VALIDATION SUMMARY")
        print("=" * 60)

        overall_status = results.get("summary", {}).get("overall_status", "UNKNOWN")
        print(f"Overall Status: {overall_status}")

        if "results" in results and "numerical_stability" in results["results"]:
            stability = results["results"]["numerical_stability"]
            total = sum(d["total_tests"] for d in stability.values())
            passed = sum(d["passed"] for d in stability.values())
            print(f"Numerical Stability: {passed}/{total} tests passed")

        if "results" in results and "statistical_correctness" in results["results"]:
            correctness = results["results"]["statistical_correctness"]
            total = len(correctness)
            passed = sum(1 for d in correctness.values() if d["passed"])
            print(f"Statistical Correctness: {passed}/{total} tests passed")

        print(f"\nReports saved to: {args.output_dir}")

    # Exit with appropriate code
    overall_status = results.get("summary", {}).get("overall_status", "UNKNOWN")
    sys.exit(0 if overall_status == "PASSED" else 1 if overall_status == "FAILED" else 2)


if __name__ == "__main__":
    main()
