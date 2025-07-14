#!/usr/bin/env python3
"""FreqProb Benchmark Runner (Python replacement for run_benchmarks.sh)."""

import argparse
import datetime
import json
import os
import shutil
import subprocess
import sys


def check_python():
    """Check if Python 3 is available."""
    if not shutil.which("python3"):
        print("Error: Python 3 is required but not found", file=sys.stderr)
        sys.exit(1)


def check_freqprob():
    """Check if FreqProb is available."""
    try:
        import freqprob  # noqa: F401
    except ImportError:
        print("Error: FreqProb not found. Please install it first:")
        print("  pip install -e .")
        sys.exit(1)


def run_benchmark(name, args, output_dir):
    """Run a specific benchmark configuration."""
    output_subdir = os.path.join(output_dir, name)
    os.makedirs(output_subdir, exist_ok=True)
    cmd = [
        sys.executable,
        os.path.join(os.path.dirname(__file__), "..", "..", "docs", "benchmarks.py"),
    ]
    cmd = [*cmd, *args, "--output", output_subdir]
    print(f"Running {name} benchmark...")
    print(f"Command: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
        print(f"✓ {name} benchmark completed successfully")
        print(f"  Results saved to: {output_subdir}/")
    except subprocess.CalledProcessError:
        print(f"✗ {name} benchmark failed", file=sys.stderr)
        sys.exit(1)
    print()


def main():
    """Main benchmark runner function."""
    parser = argparse.ArgumentParser(description="FreqProb Benchmark Runner (Python)")
    parser.add_argument(
        "--quick", action="store_true", help="Run quick benchmarks (fewer datasets)"
    )
    parser.add_argument("--plots", action="store_true", help="Generate plots (requires matplotlib)")
    args = parser.parse_args()

    check_python()
    check_freqprob()

    output_dir = f"benchmark_results_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}\n")

    # Run benchmarks
    if args.quick:
        print("Running quick benchmarks (suitable for CI/testing)...")
        run_benchmark("quick", ["--quick", "--format", "all"], output_dir)
    else:
        print(
            "Running comprehensive benchmarks...\nThis may take several minutes depending on your system.\n"
        )
        run_benchmark("comprehensive", ["--format", "all"], output_dir)
        # Memory-focused benchmark (if psutil available)
        try:
            import psutil  # noqa: F401

            print("✓ psutil available - including memory benchmarks")
        except ImportError:
            print("⚠ psutil not available - memory benchmarks will be limited")
            print("  Install with: pip install psutil")

    # Generate summary report
    summary_file = os.path.join(output_dir, "benchmark_summary.txt")
    with open(summary_file, "w") as f:
        f.write("FreqProb Benchmark Summary\n=========================\n\n")
        f.write(f"Benchmark run completed: {datetime.datetime.now()}\n")
        f.write(f"System information:\n- OS: {os.uname().sysname} {os.uname().release}\n")
        f.write(f"- Python: {sys.version.split()[0]}\n")
        try:
            import freqprob

            version = getattr(freqprob, "__version__", "dev")
        except Exception:
            version = "dev"
        f.write(f"- FreqProb: {version}\n\n")
        f.write(
            f"Benchmark configuration:\n- Quick mode: {args.quick}\n- Output directory: {output_dir}\n\n"
        )
        f.write("Files generated:\n")
        for root, _, files in os.walk(output_dir):
            for file in files:
                if file.endswith((".json", ".csv", ".html")):
                    f.write(f"- {os.path.relpath(os.path.join(root, file), output_dir)}\n")
    print(f"✓ Summary report saved to: {summary_file}\n")

    # Optional: Generate plots (not implemented)
    if args.plots:
        try:
            import matplotlib.pyplot as plt  # noqa: F401

            print("Generating performance plots...")
            print(
                "⚠ Plot generation not yet implemented. Results can be visualized by loading the JSON data."
            )
        except ImportError:
            print("⚠ matplotlib not available - skipping plots")
            print("  Install with: pip install matplotlib seaborn")

    # Display final summary
    print("Benchmark Results Summary\n========================\n")
    for result_dir in os.listdir(output_dir):
        result_path = os.path.join(output_dir, result_dir)
        if os.path.isdir(result_path):
            analysis_file = os.path.join(result_path, "benchmark_analysis.json")
            if os.path.isfile(analysis_file):
                print(f"Results from: {result_dir}")
                try:
                    with open(analysis_file) as f:
                        analysis = json.load(f)
                    summary = analysis.get("summary", {})
                    best = analysis.get("best_performers", {})
                    failures = analysis.get("failure_analysis", {})
                    print(f"  Total results: {summary.get('total_results', 'N/A')}")
                    print(f"  Methods tested: {len(summary.get('methods_tested', []))}")
                    print(f"  Datasets tested: {len(summary.get('datasets_tested', []))}")
                    if best:
                        print("  Best performers:")
                        for metric, info in best.items():
                            method = info.get("method", "N/A")
                            value = info.get("value", "N/A")
                            print(f"    {metric}: {method} ({value})")
                    total_failures = failures.get("total_failures", 0)
                    if total_failures > 0:
                        print(f"  ⚠ {total_failures} method failures detected")
                except Exception as e:
                    print(f"  Error reading analysis: {e}")
                print()
    print("📊 View detailed results:")
    print(f"  - Open {output_dir}/*/benchmark_report.html in a web browser")
    print(f"  - Load {output_dir}/*/benchmark_results.json for analysis")
    print(f"  - Import {output_dir}/*/benchmark_results.csv into spreadsheet\n")
    print("🔍 To run specific benchmarks:")
    print(f"  {sys.executable} docs/benchmarks.py --help\n")
    print("Benchmark suite completed successfully!")


if __name__ == "__main__":
    main()
