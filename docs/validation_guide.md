# FreqProb Validation and Testing Guide

This comprehensive guide covers validation procedures, testing methodologies, and quality assurance practices for FreqProb.

## Overview

FreqProb employs a multi-layered validation approach to ensure correctness, stability, and performance:

1. **Numerical Stability Testing** - Edge cases and extreme value handling
2. **Statistical Correctness Validation** - Mathematical property verification  
3. **Regression Testing** - Compatibility with reference implementations
4. **Property-Based Testing** - Invariant verification across input spaces
5. **Performance Profiling** - Benchmarking and optimization validation

## Validation Framework

### Core Components

#### ValidationSuite
The main validation orchestrator that coordinates different test types:

```python
from freqprob.validation import ValidationSuite, PerformanceProfiler

# Create validation suite
profiler = PerformanceProfiler()
validator = ValidationSuite(profiler)

# Run comprehensive validation
results = validator.run_comprehensive_validation(
    method_classes=[freqprob.MLE, freqprob.Laplace, freqprob.ELE],
    test_distributions=[your_test_distributions]
)
```

#### PerformanceProfiler
Advanced profiling for performance analysis:

```python
from freqprob.validation import PerformanceProfiler

profiler = PerformanceProfiler()

# Profile method creation
with profiler.profile_operation("method_creation"):
    method = freqprob.Laplace(freq_dist, bins=1000)

# Get detailed metrics
metrics = profiler.results[-1]
print(f"Duration: {metrics.duration_seconds}s")
print(f"Memory: {metrics.memory_delta_mb}MB")
```

## Testing Categories

### 1. Numerical Stability Testing

Tests robustness under extreme conditions and edge cases.

#### Test Categories

**Empty Distributions**
```python
# Should handle gracefully or raise appropriate errors
empty_dist = {}
try:
    method = freqprob.MLE(empty_dist)
    assert False, "Should have raised an error"
except (ValueError, ZeroDivisionError):
    pass  # Expected behavior
```

**Single Element Distributions**
```python
single_dist = {'word': 1}
mle = freqprob.MLE(single_dist, logprob=False)
assert abs(mle('word') - 1.0) < 1e-15
```

**Extreme Count Values**
```python
large_dist = {
    'common': 10**9,
    'rare': 1
}
mle = freqprob.MLE(large_dist, logprob=False)
# Should not overflow or underflow
assert not math.isnan(mle('rare'))
assert not math.isinf(mle('rare'))
```

**Precision Loss Scenarios**
```python
# Test with many equal-frequency words
uniform_dist = {f'word_{i}': 1 for i in range(1000)}
mle = freqprob.MLE(uniform_dist, logprob=False)

expected_prob = 1.0 / 1000.0
for word in list(uniform_dist.keys())[:10]:
    actual_prob = mle(word)
    relative_error = abs(actual_prob - expected_prob) / expected_prob
    assert relative_error < 1e-14
```

#### Running Stability Tests

```bash
# Run numerical stability test suite
pytest tests/test_numerical_stability.py -v

# Run with coverage
pytest tests/test_numerical_stability.py --cov=freqprob --cov-report=html
```

### 2. Statistical Correctness Validation

Verifies mathematical properties and theoretical correctness.

#### Property Categories

**Probability Axioms**
- Non-negativity: P(x) ≥ 0 for all x
- Normalization: Σ P(x) ≤ 1 (with equality for complete distributions)
- Monotonicity: Higher counts → higher probabilities (generally)

**Method-Specific Properties**
```python
# MLE should match exact relative frequencies
mle = freqprob.MLE({'a': 3, 'b': 2}, logprob=False)
assert abs(mle('a') - 0.6) < 1e-15  # 3/5
assert abs(mle('b') - 0.4) < 1e-15  # 2/5

# Laplace formula verification
laplace = freqprob.Laplace({'a': 3, 'b': 2}, bins=10, logprob=False)
expected_a = (3 + 1) / (5 + 10)  # (count + 1) / (total + bins)
assert abs(laplace('a') - expected_a) < 1e-15
```

**Consistency Properties**
```python
# Log/linear probability consistency
linear_method = freqprob.MLE(freq_dist, logprob=False)
log_method = freqprob.MLE(freq_dist, logprob=True)

for word in freq_dist:
    linear_prob = linear_method(word)
    log_prob = log_method(word)
    converted_prob = math.exp(log_prob)
    assert abs(linear_prob - converted_prob) < 1e-12
```

#### Running Correctness Tests

```bash
# Run statistical correctness tests
pytest tests/test_statistical_correctness.py -v

# Run with property-based testing
pytest tests/test_property_based.py -v --hypothesis-show-statistics
```

### 3. Regression Testing

Ensures compatibility with established reference implementations.

#### Reference Implementation Comparisons

**NLTK Compatibility**
```python
import nltk
from nltk.probability import MLEProbDist, LaplaceeProbDist

# Compare MLE implementations
words = ['cat', 'dog', 'cat', 'bird', 'cat']
freq_counts = freqprob.word_frequency(words)

# FreqProb
freqprob_mle = freqprob.MLE(freq_counts, logprob=False)

# NLTK
nltk_freqdist = nltk.FreqDist(words)
nltk_mle = MLEProbDist(nltk_freqdist)

# Should be identical
for word in freq_counts:
    freqprob_prob = freqprob_mle(word)
    nltk_prob = nltk_mle.prob(word)
    assert abs(freqprob_prob - nltk_prob) < 1e-14
```

**SciPy Compatibility**
```python
from scipy import stats

# Test entropy calculations
counts = {'a': 40, 'b': 30, 'c': 20, 'd': 10}
total = sum(counts.values())
probs = [count/total for count in counts.values()]

# FreqProb entropy
mle = freqprob.MLE(counts, logprob=True)
freqprob_entropy = -sum(
    math.exp(mle(word)) * mle(word) for word in counts
)

# SciPy entropy
scipy_entropy = stats.entropy(probs, base=math.e)

assert abs(freqprob_entropy - scipy_entropy) < 1e-12
```

#### Running Regression Tests

```bash
# Run regression tests (requires optional dependencies)
pytest tests/test_regression_reference.py -v

# Run specific regression test categories
pytest tests/test_regression_reference.py::TestNLTKRegression -v
pytest tests/test_regression_reference.py::TestScipyRegression -v
```

### 4. Property-Based Testing

Uses Hypothesis to verify properties across wide input ranges.

#### Property Examples

**Probability Axiom Verification**
```python
from hypothesis import given, strategies as st

@given(freq_dist=frequency_distribution())
def test_probability_axioms(freq_dist):
    mle = freqprob.MLE(freq_dist, logprob=False)

    # All probabilities non-negative
    for word in freq_dist:
        assert mle(word) >= 0

    # Probabilities sum to 1
    total_prob = sum(mle(word) for word in freq_dist)
    assert abs(total_prob - 1.0) < 1e-14
```

**Scaling Invariance**
```python
@given(freq_dist=frequency_distribution())
def test_scaling_invariance(freq_dist):
    scale_factor = 5
    scaled_dist = {word: count * scale_factor
                   for word, count in freq_dist.items()}

    original_mle = freqprob.MLE(freq_dist, logprob=False)
    scaled_mle = freqprob.MLE(scaled_dist, logprob=False)

    # Probabilities should be identical
    for word in freq_dist:
        assert abs(original_mle(word) - scaled_mle(word)) < 1e-14
```

#### Running Property-Based Tests

```bash
# Run with default settings
pytest tests/test_property_based.py -v

# Run with more examples for thorough testing
pytest tests/test_property_based.py -v --hypothesis-max-examples=100

# Run stateful testing
pytest tests/test_property_based.py::TestFreqProbStateMachine -v
```

### 5. Performance Profiling

Validates performance characteristics and identifies regressions.

#### Profiling Categories

**Creation Time Scaling**
```python
from freqprob.validation import PerformanceProfiler

profiler = PerformanceProfiler()

# Test scaling with vocabulary size
vocab_sizes = [100, 500, 1000, 2000]
for size in vocab_sizes:
    test_dist = {f'word_{i}': max(1, int(1000/(i+1)))
                 for i in range(size)}

    metrics = profiler.profile_method_creation(
        freqprob.Laplace, test_dist, bins=size*2
    )

    print(f"Size {size}: {metrics.duration_seconds:.4f}s")
```

**Query Performance**
```python
# Profile query performance
method = freqprob.Laplace(freq_dist, bins=1000)
test_words = list(freq_dist.keys())[:100]

metrics = profiler.profile_query_performance(
    method, test_words, iterations=1000
)

queries_per_sec = metrics.iterations / metrics.duration_seconds
print(f"Query rate: {queries_per_sec:.0f} queries/second")
```

**Memory Usage Analysis**
```python
# Profile memory usage
with profiler.profile_operation("memory_test"):
    # Create large method
    large_dist = {f'word_{i}': i+1 for i in range(10000)}
    method = freqprob.SimpleGoodTuring(large_dist)

metrics = profiler.results[-1]
print(f"Memory usage: {metrics.memory_delta_mb:.2f} MB")
```

#### Running Performance Tests

```bash
# Run performance benchmarks
python docs/benchmarks.py --quick --output perf_results

# Run validation report (includes performance)
python scripts/validation_report.py --output-dir validation_results
```

## Comprehensive Validation

### Full Validation Suite

Run the complete validation suite:

```bash
# Generate comprehensive validation report
python scripts/validation_report.py --output-dir validation_results

# Quick validation for CI/CD
python scripts/validation_report.py --quick --output-dir quick_validation
```

This generates:
- `validation_report.html` - Human-readable summary
- `validation_report.json` - Detailed test results
- `profiler_results.json` - Performance profiling data
- `benchmark_results.json` - Benchmark comparisons

### Custom Validation Workflows

Create custom validation workflows:

```python
from freqprob.validation import ValidationSuite, PerformanceProfiler

# Create custom validator
profiler = PerformanceProfiler(enable_detailed_tracking=True)
validator = ValidationSuite(profiler)

# Define test distributions
test_distributions = [
    {'word1': 100, 'word2': 50, 'word3': 25},
    {f'term_{i}': max(1, int(1000/(i+1))) for i in range(500)}
]

# Test specific methods
methods_to_test = [
    freqprob.MLE,
    freqprob.Laplace,
    freqprob.ELE,
    freqprob.BayesianSmoothing
]

# Run validation
results = validator.run_comprehensive_validation(
    methods_to_test,
    test_distributions,
    bins=1000,  # Common parameter
    alpha=0.5   # For Bayesian smoothing
)

# Generate report
validator.generate_validation_report(Path('custom_validation.json'))
```

## Continuous Integration

### GitHub Actions Integration

The validation suite integrates with CI/CD pipelines:

```yaml
# .github/workflows/validation.yml
name: Validation

on: [push, pull_request]

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install -e .[dev]
          pip install hypothesis nltk scipy

      - name: Run validation suite
        run: |
          python scripts/validation_report.py --quick --output-dir validation_results

      - name: Upload validation results
        uses: actions/upload-artifact@v3
        with:
          name: validation-results
          path: validation_results/
```

### Local Development Workflow

For local development:

```bash
# Quick validation during development
pytest tests/test_numerical_stability.py -x

# Full validation before commits
python scripts/validation_report.py --output-dir pre_commit_validation

# Performance regression check
python docs/benchmarks.py --quick --output benchmark_check
```

## Validation Best Practices

### 1. Test Design Principles

**Comprehensive Coverage**
- Test edge cases and boundary conditions
- Include realistic and pathological distributions
- Verify mathematical properties and invariants

**Reproducibility**
- Use fixed random seeds for deterministic tests
- Document test assumptions and expected behaviors
- Version test data and reference implementations

**Performance Awareness**
- Profile performance-critical operations
- Set reasonable performance regression thresholds
- Monitor memory usage and scaling behavior

### 2. Error Handling

**Expected Failures**
```python
# Some methods may legitimately fail on certain inputs
try:
    sgt = freqprob.SimpleGoodTuring(problematic_dist)
except (ValueError, RuntimeError):
    # SGT can fail on irregular distributions
    pytest.skip("SGT failed on this distribution (expected)")
```

**Graceful Degradation**
```python
# Test fallback mechanisms
try:
    advanced_method = freqprob.ModifiedKneserNey(bigrams)
except RuntimeError:
    # Fall back to simpler method
    fallback_method = freqprob.KneserNey(bigrams, discount=0.75)
```

### 3. Reporting and Analysis

**Automated Reports**
- Generate validation reports in CI/CD pipelines
- Track validation metrics over time
- Alert on significant performance regressions

**Manual Analysis**
- Review failed tests to understand root causes
- Analyze performance trends and bottlenecks
- Validate theoretical properties against implementations

## Troubleshooting

### Common Issues

**Import Errors**
```bash
# Install optional dependencies for comprehensive testing
pip install hypothesis nltk scipy matplotlib seaborn
```

**Performance Test Failures**
```python
# Adjust performance thresholds for different hardware
validator.validate_performance_regression(
    method_class, test_dist,
    max_duration_seconds=20.0,  # Increased threshold
    max_memory_mb=2000.0        # Increased memory limit
)
```

**Hypothesis Test Failures**
```bash
# Increase example count for more thorough testing
pytest tests/test_property_based.py --hypothesis-max-examples=200

# Debug hypothesis failures
pytest tests/test_property_based.py --hypothesis-verbosity=verbose
```

### Debug Utilities

**Profiler Analysis**
```python
# Analyze profiler results
profiler = PerformanceProfiler()
# ... run tests ...

summary = profiler.get_summary_statistics()
print(json.dumps(summary, indent=2))

# Export for external analysis
profiler.export_results('debug_profile.json', format='json')
```

**Validation Details**
```python
# Get detailed validation results
validator = ValidationSuite()
# ... run validation ...

for result in validator.results:
    if not result.passed:
        print(f"Failed: {result.test_name}")
        print(f"Error: {result.error_message}")
        print(f"Details: {result.details}")
```

This validation framework ensures FreqProb maintains high quality, correctness, and performance standards across all development and deployment scenarios.
