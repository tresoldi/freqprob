#!/bin/bash

# FreqProb Benchmark Runner
# This script runs comprehensive performance benchmarks for FreqProb

set -e  # Exit on error

echo "FreqProb Performance Benchmark Runner"
echo "====================================="
echo

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required but not found"
    exit 1
fi

# Check if FreqProb is importable
python3 -c "import freqprob" 2>/dev/null || {
    echo "Error: FreqProb not found. Please install it first:"
    echo "  pip install -e ."
    exit 1
}

# Create output directory
OUTPUT_DIR="benchmark_results_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

echo "Output directory: $OUTPUT_DIR"
echo

# Function to run benchmark with error handling
run_benchmark() {
    local name="$1"
    local args="$2"
    local output_subdir="$OUTPUT_DIR/$name"

    echo "Running $name benchmark..."
    echo "Command: python3 benchmarks.py $args --output $output_subdir"

    if python3 benchmarks.py $args --output "$output_subdir"; then
        echo "✓ $name benchmark completed successfully"
        echo "  Results saved to: $output_subdir/"
    else
        echo "✗ $name benchmark failed"
        return 1
    fi
    echo
}

# Check for command line arguments
QUICK_MODE=false
INCLUDE_PLOTS=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)
            QUICK_MODE=true
            shift
            ;;
        --plots)
            INCLUDE_PLOTS=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo
            echo "Options:"
            echo "  --quick    Run quick benchmarks (fewer datasets)"
            echo "  --plots    Generate plots (requires matplotlib)"
            echo "  --help     Show this help message"
            echo
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Run benchmarks based on mode
if [ "$QUICK_MODE" = true ]; then
    echo "Running quick benchmarks (suitable for CI/testing)..."
    run_benchmark "quick" "--quick --format all"
else
    echo "Running comprehensive benchmarks..."
    echo "This may take several minutes depending on your system."
    echo

    # Full benchmark suite
    run_benchmark "comprehensive" "--format all"

    # Additional specialized benchmarks if time permits
    echo "Running additional specialized benchmarks..."

    # Memory-focused benchmark (if psutil available)
    if python3 -c "import psutil" 2>/dev/null; then
        echo "✓ psutil available - including memory benchmarks"
    else
        echo "⚠ psutil not available - memory benchmarks will be limited"
        echo "  Install with: pip install psutil"
    fi
fi

# Generate summary report
echo "Generating summary report..."
SUMMARY_FILE="$OUTPUT_DIR/benchmark_summary.txt"

cat > "$SUMMARY_FILE" << EOF
FreqProb Benchmark Summary
=========================

Benchmark run completed: $(date)
System information:
- OS: $(uname -s -r)
- Python: $(python3 --version)
- FreqProb: $(python3 -c "import freqprob; print(getattr(freqprob, '__version__', 'dev'))" 2>/dev/null || echo "dev")

Benchmark configuration:
- Quick mode: $QUICK_MODE
- Output directory: $OUTPUT_DIR

Files generated:
EOF

# List all generated files
find "$OUTPUT_DIR" -type f -name "*.json" -o -name "*.csv" -o -name "*.html" | while read file; do
    echo "- $(basename "$file")" >> "$SUMMARY_FILE"
done

echo "✓ Summary report saved to: $SUMMARY_FILE"
echo

# Check for matplotlib and offer to generate plots
if [ "$INCLUDE_PLOTS" = true ]; then
    if python3 -c "import matplotlib.pyplot" 2>/dev/null; then
        echo "Generating performance plots..."
        # Note: This would require a separate plotting script
        echo "⚠ Plot generation not yet implemented"
        echo "  Results can be visualized by loading the JSON data"
    else
        echo "⚠ matplotlib not available - skipping plots"
        echo "  Install with: pip install matplotlib seaborn"
    fi
fi

# Display final summary
echo "Benchmark Results Summary"
echo "========================"
echo

# Try to extract key metrics from results
for result_dir in "$OUTPUT_DIR"/*; do
    if [ -d "$result_dir" ] && [ -f "$result_dir/benchmark_analysis.json" ]; then
        echo "Results from: $(basename "$result_dir")"

        # Extract summary using Python
        python3 << EOF
import json
import sys

try:
    with open("$result_dir/benchmark_analysis.json", "r") as f:
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
            method = info.get('method', 'N/A')
            value = info.get('value', 'N/A')
            print(f"    {metric}: {method} ({value})")

    total_failures = failures.get('total_failures', 0)
    if total_failures > 0:
        print(f"  ⚠ {total_failures} method failures detected")

except Exception as e:
    print(f"  Error reading analysis: {e}")

EOF
        echo
    fi
done

echo "📊 View detailed results:"
echo "  - Open $OUTPUT_DIR/*/benchmark_report.html in a web browser"
echo "  - Load $OUTPUT_DIR/*/benchmark_results.json for analysis"
echo "  - Import $OUTPUT_DIR/*/benchmark_results.csv into spreadsheet"
echo

echo "🔍 To run specific benchmarks:"
echo "  python3 benchmarks.py --help"
echo

echo "Benchmark suite completed successfully!"

# Make the script executable
chmod +x "$0" 2>/dev/null || true
