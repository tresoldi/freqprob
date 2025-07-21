# FreqProb Library: Complete LLM Coding Agent Documentation

## Table of Contents
1. [Library Overview](#library-overview)
2. [Core Architecture](#core-architecture)
3. [Mathematical Foundations](#mathematical-foundations)
4. [API Reference](#api-reference)
5. [Implementation Patterns](#implementation-patterns)
6. [Testing and Validation](#testing-and-validation)
7. [Performance Optimization](#performance-optimization)
8. [Extension Guidelines](#extension-guidelines)
9. [Common Workflows](#common-workflows)

## Library Overview

FreqProb is a comprehensive Python library for probability smoothing and frequency-based language modeling. It provides 10+ smoothing methods ranging from basic MLE to advanced techniques like Kneser-Ney and Simple Good-Turing. The library emphasizes mathematical correctness, performance optimization, and extensive validation.

### Key Features
- **Comprehensive Smoothing Methods**: MLE, Laplace, Lidstone, ELE, Kneser-Ney, Good-Turing, Bayesian, and more
- **High Performance**: Vectorized operations, streaming algorithms, intelligent caching
- **Memory Efficiency**: Compressed representations, sparse storage, lazy evaluation
- **Strict Type Safety**: Full type hints with custom domain types
- **Mathematical Validation**: Property-based testing, numerical stability checks
- **Production Ready**: 400+ tests, performance profiling, regression testing

### Dependencies
- **Core**: numpy>=1.20.0, scipy>=1.7.0
- **Optional**: psutil (memory profiling), hypothesis (validation), nltk (reference testing)
- **Development**: pytest, ruff, mypy, bandit, pre-commit

## Core Architecture

### Type System
```python
# Core type aliases for domain clarity
Element = str | int | float | tuple[Any, ...] | frozenset[Any]
Count = int
Probability = float
LogProbability = float
FrequencyDistribution = Mapping[Element, Count]

# Generic type for method chaining
T = TypeVar("T", bound="ScoringMethod")
```

### Abstract Base Class
```python
class ScoringMethod(ABC):
    """Abstract base class for all probability estimation methods.

    Core contract that all methods must implement:
    - __call__(element: Element) -> Probability | LogProbability
    - _compute_probabilities(freqdist: FrequencyDistribution) -> None
    """

    def __init__(self, freqdist: FrequencyDistribution,
                 unobs_prob: float | None = None,
                 logprob: bool = False, **kwargs):
        self.freqdist = freqdist
        self.config = ScoringMethodConfig(unobs_prob=unobs_prob, logprob=logprob, **kwargs)
        self._probabilities: dict[Element, Probability] = {}
        self._compute_probabilities(freqdist)

    @abstractmethod
    def _compute_probabilities(self, freqdist: FrequencyDistribution) -> None:
        """Compute probabilities for all elements in the distribution."""
        pass

    def __call__(self, element: Element) -> Probability | LogProbability:
        """Return probability for given element."""
        if element in self._probabilities:
            prob = self._probabilities[element]
        else:
            prob = self._get_unobserved_probability()

        return math.log(prob) if self.config.logprob else prob
```

### Configuration System
```python
@dataclass
class ScoringMethodConfig:
    """Type-safe configuration for smoothing methods."""
    unobs_prob: float | None = None  # Reserved probability mass (0.0 ≤ p ≤ 1.0)
    gamma: float | None = None       # Additive smoothing parameter (γ ≥ 0)
    bins: int | None = None          # Total vocabulary size (B ≥ 1)
    logprob: bool = False           # Return log-probabilities

    def __post_init__(self) -> None:
        """Validate parameters after initialization."""
        if self.unobs_prob is not None and not 0.0 <= self.unobs_prob <= 1.0:
            raise ValueError("Reserved probability must be between 0.0 and 1.0")
        if self.gamma is not None and self.gamma < 0:
            raise ValueError("Gamma must be non-negative")
        if self.bins is not None and self.bins < 1:
            raise ValueError("Bins must be positive")
```

## Mathematical Foundations

### Basic Smoothing Methods

#### Maximum Likelihood Estimation (MLE)
```python
class MLE(ScoringMethod):
    """P(w) = c(w) / N where N = Σc(w)"""

    def _compute_probabilities(self, freqdist: FrequencyDistribution) -> None:
        total_count = sum(freqdist.values())
        self._probabilities = {
            element: count / total_count
            for element, count in freqdist.items()
        }
```

#### Laplace Smoothing (Add-1)
```python
class Laplace(ScoringMethod):
    """P(w) = (c(w) + 1) / (N + B)

    Args:
        bins: Total vocabulary size B
    """

    def _compute_probabilities(self, freqdist: FrequencyDistribution) -> None:
        total_count = sum(freqdist.values())
        bins = self.config.bins or len(freqdist)

        self._probabilities = {
            element: (count + 1) / (total_count + bins)
            for element, count in freqdist.items()
        }

    def _get_unobserved_probability(self) -> float:
        """Unobserved elements get probability 1 / (N + B)"""
        total_count = sum(self.freqdist.values())
        bins = self.config.bins or len(self.freqdist)
        return 1 / (total_count + bins)
```

#### Lidstone Smoothing (Add-k)
```python
class Lidstone(ScoringMethod):
    """P(w) = (c(w) + γ) / (N + B×γ)

    Generalization of Laplace with configurable gamma parameter.
    """

    def _compute_probabilities(self, freqdist: FrequencyDistribution) -> None:
        total_count = sum(freqdist.values())
        gamma = self.config.gamma or 1.0
        bins = self.config.bins or len(freqdist)

        self._probabilities = {
            element: (count + gamma) / (total_count + bins * gamma)
            for element, count in freqdist.items()
        }
```

#### Expected Likelihood Estimation (ELE)
```python
class ELE(Lidstone):
    """Lidstone smoothing with γ = 0.5"""

    def __init__(self, freqdist: FrequencyDistribution, **kwargs):
        kwargs['gamma'] = 0.5
        super().__init__(freqdist, **kwargs)
```

### Advanced Smoothing Methods

#### Kneser-Ney Smoothing
```python
class KneserNey(ScoringMethod):
    """Absolute discounting with continuation probability.

    P_KN(w_i|w_{i-1}) = max(c(w_{i-1},w_i) - d, 0) / c(w_{i-1}) + λ(w_{i-1}) × P_cont(w_i)

    Where:
    - d: discount parameter (0 < d < 1)
    - P_cont: continuation probability based on context diversity
    - λ: normalization constant
    """

    def __init__(self, freqdist: FrequencyDistribution, discount: float = 0.75, **kwargs):
        if not 0 < discount < 1:
            raise ValueError("Discount must be between 0 and 1")
        self.discount = discount
        super().__init__(freqdist, **kwargs)

    def _compute_probabilities(self, freqdist: FrequencyDistribution) -> None:
        # Calculate continuation probabilities
        context_counts = {}  # c(w_{i-1})
        continuation_counts = {}  # N_1+(•w_i) - number of contexts where w_i appears

        for ngram, count in freqdist.items():
            if isinstance(ngram, tuple) and len(ngram) >= 2:
                context = ngram[:-1]
                word = ngram[-1]

                context_counts[context] = context_counts.get(context, 0) + count
                if count > 0:
                    continuation_counts[word] = continuation_counts.get(word, 0) + 1

        total_continuation = sum(continuation_counts.values())

        # Compute probabilities using Kneser-Ney formula
        for ngram, count in freqdist.items():
            if isinstance(ngram, tuple) and len(ngram) >= 2:
                context = ngram[:-1]
                word = ngram[-1]

                # Discounted probability
                discounted = max(count - self.discount, 0) / context_counts[context]

                # Continuation probability
                continuation = continuation_counts.get(word, 0) / total_continuation

                # Interpolation weight (lambda)
                lambda_weight = (self.discount / context_counts[context]) * len([
                    w for (ctx, w) in freqdist.keys()
                    if isinstance((ctx, w), tuple) and ctx == context
                ])

                self._probabilities[ngram] = discounted + lambda_weight * continuation
```

#### Simple Good-Turing Smoothing
```python
class SimpleGoodTuring(ScoringMethod):
    """Good-Turing with log-linear smoothing.

    Uses frequency-of-frequencies to estimate probabilities:
    r* = (r+1) × N_{r+1} / N_r

    Switches between empirical and smoothed estimates based on confidence.
    """

    def _compute_probabilities(self, freqdist: FrequencyDistribution) -> None:
        from collections import Counter
        import scipy.stats

        # Calculate frequency of frequencies
        counts = list(freqdist.values())
        freq_of_freq = Counter(counts)  # N_r values

        max_count = max(counts)
        smoothed_estimates = {}

        # Use log-linear smoothing for sparse r values
        r_values = sorted(freq_of_freq.keys())
        nr_values = [freq_of_freq[r] for r in r_values]

        # Linear regression in log space: log(N_r) ~ a + b×log(r)
        log_r = [math.log(r) for r in r_values if r > 0]
        log_nr = [math.log(nr) for r, nr in zip(r_values, nr_values) if r > 0 and nr > 0]

        if len(log_r) >= 2:
            slope, intercept, _, _, _ = scipy.stats.linregress(log_r, log_nr)

            # Compute Good-Turing estimates
            for count in range(1, max_count + 1):
                if count + 1 in freq_of_freq:
                    # Use empirical estimate
                    r_star = (count + 1) * freq_of_freq[count + 1] / freq_of_freq[count]
                else:
                    # Use smoothed estimate
                    smoothed_nr_plus1 = math.exp(intercept + slope * math.log(count + 1))
                    r_star = (count + 1) * smoothed_nr_plus1 / freq_of_freq.get(count, 1)

                smoothed_estimates[count] = r_star

        # Compute probabilities
        total_adjusted = sum(smoothed_estimates.get(count, count)
                           for count in freqdist.values())

        self._probabilities = {
            element: smoothed_estimates.get(count, count) / total_adjusted
            for element, count in freqdist.items()
        }
```

#### Bayesian Smoothing
```python
class BayesianSmoothing(ScoringMethod):
    """Bayesian smoothing with Dirichlet prior.

    P(w) = (c(w) + α) / (N + α×V)

    Where α is the Dirichlet concentration parameter.
    """

    def __init__(self, freqdist: FrequencyDistribution, alpha: float = 1.0, **kwargs):
        if alpha <= 0:
            raise ValueError("Alpha must be positive")
        self.alpha = alpha
        super().__init__(freqdist, **kwargs)

    def _compute_probabilities(self, freqdist: FrequencyDistribution) -> None:
        total_count = sum(freqdist.values())
        vocab_size = len(freqdist)

        self._probabilities = {
            element: (count + self.alpha) / (total_count + self.alpha * vocab_size)
            for element, count in freqdist.items()
        }
```

## API Reference

### Core Classes

#### ScoringMethod Hierarchy
```
ScoringMethod (ABC)
├── MLE
├── Uniform
├── Random
├── Lidstone
│   ├── Laplace (γ=1.0)
│   └── ELE (γ=0.5)
├── BayesianSmoothing
├── WittenBell
├── CertaintyDegree
├── SimpleGoodTuring
├── KneserNey
├── ModifiedKneserNey
└── InterpolatedSmoothing
```

#### Memory-Efficient Classes
```python
# Compressed frequency distributions
compressed_dist = freqprob.create_compressed_distribution(
    freqdist, quantization_levels=256
)

# Sparse representations (only non-zero counts)
sparse_dist = freqprob.create_sparse_distribution(freqdist)

# Quantized probability tables
quantized = freqprob.QuantizedProbabilityTable(
    probabilities, quantization_bits=8
)
```

#### Streaming Classes
```python
# Streaming MLE with vocabulary limits
streaming_mle = freqprob.StreamingMLE(
    initial_data=freqdist,
    max_vocabulary_size=1000,
    compression_threshold=0.8,
    logprob=False
)

# Update with new data
streaming_mle.update_single("new_word", 5)
streaming_mle.update_batch(["word1", "word2"], [3, 7])

# Streaming with decay (for time-series data)
streaming_laplace = freqprob.StreamingLaplace(
    initial_data=freqdist,
    decay_factor=0.99,
    max_vocabulary_size=500
)
```

#### Vectorized Classes
```python
# High-performance batch scoring
vectorized = freqprob.VectorizedScorer(scoring_method)
words = ["word1", "word2", "word3"] * 1000
probabilities = vectorized.score_batch(words)  # NumPy array

# Multi-method batch scoring
methods = {
    "mle": freqprob.MLE(freqdist),
    "laplace": freqprob.Laplace(freqdist, bins=1000),
    "kneser_ney": freqprob.KneserNey(bigram_freqdist)
}
batch_scorer = freqprob.create_vectorized_batch_scorer(methods)
results = batch_scorer.score_batch(words)  # Dict of NumPy arrays
```

#### Lazy Evaluation Classes
```python
# Lazy computation for expensive methods
lazy_method = freqprob.create_lazy_mle(freqdist, unobs_prob=0.01)
lazy_scorer = freqprob.LazyBatchScorer(lazy_method)

# Computations only happen when needed
prob = lazy_method("word")  # Triggers computation if not cached
batch_probs = lazy_scorer.score_batch(words)  # Vectorized lazy evaluation
```

### Utility Functions

#### Evaluation Metrics
```python
# Perplexity calculation
test_data = ["word1", "word2", "word3"] * 100
perplexity = freqprob.perplexity(scoring_method, test_data)

# Cross-entropy between models
ce = freqprob.cross_entropy(model1, model2, test_data)

# KL divergence
kl_div = freqprob.kl_divergence(model1, model2, vocabulary)

# Multi-model comparison
models = {"mle": mle, "laplace": laplace, "kneser_ney": kn}
comparison = freqprob.model_comparison(models, test_data)
# Returns: {"model_name": {"perplexity": float, "cross_entropy": float, ...}}
```

#### N-gram Processing
```python
# Generate n-grams from text
text = "the quick brown fox jumps"
bigrams = freqprob.generate_ngrams(text.split(), n=2)
# Returns: [("the", "quick"), ("quick", "brown"), ...]

# N-gram frequency counting
tokens = text.split()
trigram_counts = freqprob.ngram_frequency(tokens, n=3)

# Word frequency counting
word_counts = freqprob.word_frequency(tokens)
```

### Caching System

#### Automatic Caching
```python
# Methods automatically cache expensive computations
method = freqprob.SimpleGoodTuring(large_freqdist)
prob1 = method("word")  # Computed and cached
prob2 = method("word")  # Retrieved from cache

# Cache statistics
stats = freqprob.get_cache_stats()
# Returns: {"hits": int, "misses": int, "hit_rate": float}

# Clear all caches
freqprob.clear_all_caches()
```

#### Custom Caching
```python
from freqprob.cache import cached_computation

@cached_computation(max_cache_size=1000)
def expensive_calculation(data, parameters):
    # Expensive computation here
    return result
```

### Profiling and Validation

#### Memory Profiling
```python
# Memory usage analysis
analyzer = freqprob.DistributionMemoryAnalyzer()
comparison = analyzer.compare_representations(large_freqdist)

# Memory monitoring during operations
monitor = freqprob.MemoryMonitor()
with monitor.profile_operation("model_creation"):
    model = freqprob.KneserNey(huge_ngram_data)

metrics = monitor.get_latest_metrics()
# Returns: MemoryMetrics(peak_memory_mb, execution_time, etc.)

# Memory profiler for detailed analysis
profiler = freqprob.MemoryProfiler()
metrics = profiler.profile_method_creation(
    freqprob.SimpleGoodTuring,
    large_freqdist,
    iterations=10
)
```

#### Performance Validation
```python
# Validate method correctness and performance
validator = freqprob.ValidationSuite()
result = validator.validate_method(
    freqprob.Laplace,
    test_distributions=[small_dist, medium_dist, large_dist]
)

# Quick validation
is_valid = freqprob.quick_validate_method(freqprob.MLE, test_dist)

# Performance comparison
comparison = freqprob.compare_method_performance(
    methods=[freqprob.MLE, freqprob.Laplace, freqprob.KneserNey],
    distributions=[test_dist],
    test_data=validation_data
)
```

## Implementation Patterns

### Method Creation Pattern
```python
def create_smoothing_method(method_name: str, freqdist: FrequencyDistribution,
                          **kwargs) -> ScoringMethod:
    """Factory function for creating smoothing methods."""

    method_map = {
        "mle": freqprob.MLE,
        "laplace": freqprob.Laplace,
        "lidstone": freqprob.Lidstone,
        "ele": freqprob.ELE,
        "kneser_ney": freqprob.KneserNey,
        "simple_gt": freqprob.SimpleGoodTuring,
        "bayesian": freqprob.BayesianSmoothing,
    }

    if method_name not in method_map:
        raise ValueError(f"Unknown method: {method_name}")

    return method_map[method_name](freqdist, **kwargs)
```

### Configuration Validation Pattern
```python
def validate_smoothing_config(method_class, **kwargs):
    """Validate configuration parameters before method creation."""

    if method_class == freqprob.KneserNey:
        discount = kwargs.get("discount", 0.75)
        if not 0 < discount < 1:
            raise ValueError("KneserNey discount must be in (0, 1)")

    elif method_class == freqprob.BayesianSmoothing:
        alpha = kwargs.get("alpha", 1.0)
        if alpha <= 0:
            raise ValueError("Bayesian alpha must be positive")

    elif issubclass(method_class, freqprob.Lidstone):
        gamma = kwargs.get("gamma", 1.0)
        if gamma < 0:
            raise ValueError("Lidstone gamma must be non-negative")

    return kwargs
```

### Error Handling Pattern
```python
def safe_probability_computation(method: ScoringMethod, element: Element) -> float:
    """Safe probability computation with error handling."""
    try:
        prob = method(element)

        # Validate result
        if math.isnan(prob) or math.isinf(prob):
            raise ValueError(f"Invalid probability computed: {prob}")

        if not method.config.logprob and (prob < 0 or prob > 1):
            raise ValueError(f"Probability out of range [0,1]: {prob}")

        return prob

    except Exception as e:
        logging.warning(f"Probability computation failed for {element}: {e}")
        # Return safe fallback
        return math.log(1e-10) if method.config.logprob else 1e-10
```

### Batch Processing Pattern
```python
def process_large_vocabulary(freqdist: FrequencyDistribution,
                           method_class: type[ScoringMethod],
                           batch_size: int = 10000,
                           **kwargs) -> ScoringMethod:
    """Process large vocabularies in batches to manage memory."""

    if len(freqdist) <= batch_size:
        return method_class(freqdist, **kwargs)

    # Use streaming approach for large vocabularies
    if hasattr(method_class, 'Streaming'):
        streaming_class = getattr(freqprob, f"Streaming{method_class.__name__}")
        return streaming_class(freqdist, max_vocabulary_size=batch_size, **kwargs)

    # Use compressed representation
    compressed_dist = freqprob.create_compressed_distribution(freqdist)
    return method_class(compressed_dist.to_dict(), **kwargs)
```

## Testing and Validation

### Property-Based Testing with Hypothesis
```python
from hypothesis import given, strategies as st, settings

@st.composite
def frequency_distribution(draw, min_vocab=1, max_vocab=100):
    """Generate valid frequency distributions."""
    vocab_size = draw(st.integers(min_value=min_vocab, max_value=max_vocab))
    words = [f"word_{i}" for i in range(vocab_size)]
    counts = draw(st.lists(
        st.integers(min_value=1, max_value=1000),
        min_size=vocab_size, max_size=vocab_size
    ))
    return dict(zip(words, counts))

@given(freq_dist=frequency_distribution())
@settings(max_examples=50, deadline=5000)
def test_probability_axioms(freq_dist):
    """Test fundamental probability axioms."""
    method = freqprob.MLE(freq_dist, logprob=False)

    # All probabilities non-negative
    for word in freq_dist:
        assert method(word) >= 0

    # Probabilities sum to 1 (within numerical precision)
    total_prob = sum(method(word) for word in freq_dist)
    assert abs(total_prob - 1.0) < 1e-14

    # Higher counts → higher probabilities (monotonicity)
    sorted_items = sorted(freq_dist.items(), key=lambda x: x[1])
    for i in range(len(sorted_items) - 1):
        word1, count1 = sorted_items[i]
        word2, count2 = sorted_items[i + 1]
        if count1 < count2:
            assert method(word1) < method(word2)
```

### Mathematical Validation Tests
```python
def test_smoothing_formulas():
    """Test mathematical formulas against theoretical expectations."""

    # Test Laplace formula: P(w) = (c(w) + 1) / (N + V)
    counts = {"a": 10, "b": 5, "c": 2}
    laplace = freqprob.Laplace(counts, bins=100, logprob=False)

    total_count = sum(counts.values())
    bins = 100

    for word, count in counts.items():
        expected = (count + 1) / (total_count + bins)
        actual = laplace(word)
        assert abs(actual - expected) < 1e-15

    # Test ELE formula: P(w) = (c(w) + 0.5) / (N + 0.5×V)
    ele = freqprob.ELE(counts, bins=100, logprob=False)
    for word, count in counts.items():
        expected = (count + 0.5) / (total_count + 0.5 * bins)
        actual = ele(word)
        assert abs(actual - expected) < 1e-15
```

### Performance Regression Tests
```python
def test_performance_regression():
    """Test that performance doesn't degrade over time."""
    import time

    # Large test distribution
    large_dist = {f"word_{i}": max(1, 1000 - i) for i in range(10000)}

    # Baseline performance expectations
    start_time = time.perf_counter()
    mle = freqprob.MLE(large_dist, logprob=False)
    creation_time = time.perf_counter() - start_time

    # Should create large MLE in reasonable time
    assert creation_time < 1.0  # Max 1 second

    # Batch scoring performance
    test_words = [f"word_{i}" for i in range(0, 1000, 10)]
    start_time = time.perf_counter()
    probs = [mle(word) for word in test_words]
    scoring_time = time.perf_counter() - start_time

    # Should score 100 words quickly
    assert scoring_time < 0.1  # Max 0.1 seconds
```

### Numerical Stability Tests
```python
def test_numerical_stability():
    """Test behavior with extreme numerical values."""

    # Very large counts
    extreme_dist = {"word": 2**50}
    mle = freqprob.MLE(extreme_dist, logprob=False)
    assert mle("word") == 1.0
    assert not math.isnan(mle("word"))

    # Very small counts (test underflow handling)
    tiny_dist = {"word": 1e-100}
    mle_tiny = freqprob.MLE(tiny_dist, logprob=False)
    assert mle_tiny("word") == 1.0

    # Log probabilities should handle extreme values
    mle_log = freqprob.MLE(tiny_dist, logprob=True)
    log_prob = mle_log("word")
    assert not math.isnan(log_prob) and not math.isinf(log_prob)
```

## Performance Optimization

### Caching Strategies
```python
# Automatic method-level caching
@cached_computation(max_cache_size=1000, key_prefix="sgt")
def simple_good_turing_computation(freqdist_hash, p_value, allow_fail):
    """Cache expensive Good-Turing computations."""
    # Expensive computation here
    return smoothed_probabilities

# Cache configuration
cache_config = {
    "max_cache_size": 10000,
    "eviction_policy": "lru",  # least-recently-used
    "key_hash_algo": "sha256"
}
```

### Memory Optimization Techniques
```python
# String interning for repeated elements
import sys

def intern_elements(freqdist: FrequencyDistribution) -> FrequencyDistribution:
    """Intern string elements to reduce memory usage."""
    return {
        (sys.intern(elem) if isinstance(elem, str) else elem): count
        for elem, count in freqdist.items()
    }

# Compressed integer storage
def compress_counts(counts: list[int]) -> bytes:
    """Compress integer counts using variable-length encoding."""
    import array
    return array.array('i', counts).tobytes()

# Quantized probability storage
def quantize_probabilities(probs: list[float], bits: int = 8) -> list[int]:
    """Quantize probabilities to reduce storage."""
    max_val = (1 << bits) - 1
    return [int(p * max_val) for p in probs]
```

### Vectorization Patterns
```python
import numpy as np

def vectorized_mle_computation(counts: np.ndarray) -> np.ndarray:
    """Vectorized MLE computation using NumPy."""
    total_count = np.sum(counts)
    return counts / total_count

def vectorized_laplace_computation(counts: np.ndarray, vocab_size: int) -> np.ndarray:
    """Vectorized Laplace smoothing."""
    total_count = np.sum(counts)
    return (counts + 1) / (total_count + vocab_size)

# Batch probability lookup
class FastProbabilityLookup:
    """O(1) probability lookup using pre-computed arrays."""

    def __init__(self, method: ScoringMethod, vocabulary: list[Element]):
        self.vocabulary = vocabulary
        self.prob_array = np.array([method(word) for word in vocabulary])
        self.word_to_index = {word: i for i, word in enumerate(vocabulary)}

    def __call__(self, word: Element) -> float:
        """O(1) probability lookup."""
        if word in self.word_to_index:
            return self.prob_array[self.word_to_index[word]]
        return 1e-10  # Unobserved word fallback
```

### Streaming Optimization
```python
class MemoryEfficientStreaming:
    """Memory-efficient streaming with automatic compression."""

    def __init__(self, max_memory_mb: float = 100.0):
        self.max_memory_mb = max_memory_mb
        self.freqdist = {}
        self.total_count = 0

    def update(self, element: Element, count: int):
        """Update with automatic memory management."""
        self.freqdist[element] = self.freqdist.get(element, 0) + count
        self.total_count += count

        # Check memory usage
        if self._get_memory_usage() > self.max_memory_mb:
            self._compress_distribution()

    def _compress_distribution(self):
        """Compress distribution when memory limit reached."""
        # Remove low-frequency items
        min_count = max(1, self.total_count // (len(self.freqdist) * 10))
        self.freqdist = {
            elem: count for elem, count in self.freqdist.items()
            if count >= min_count
        }
```

## Extension Guidelines

### Creating Custom Smoothing Methods
```python
class CustomSmoothingMethod(ScoringMethod):
    """Template for custom smoothing methods."""

    def __init__(self, freqdist: FrequencyDistribution,
                 custom_param: float = 1.0, **kwargs):
        # Validate custom parameters
        if custom_param <= 0:
            raise ValueError("Custom parameter must be positive")

        self.custom_param = custom_param
        super().__init__(freqdist, **kwargs)

    def _compute_probabilities(self, freqdist: FrequencyDistribution) -> None:
        """Implement your smoothing formula here."""
        # Example: Custom additive smoothing
        total_count = sum(freqdist.values())

        self._probabilities = {
            element: (count + self.custom_param) / (total_count + len(freqdist) * self.custom_param)
            for element, count in freqdist.items()
        }

    def _get_unobserved_probability(self) -> float:
        """Define probability for unseen elements."""
        total_count = sum(self.freqdist.values())
        vocab_size = len(self.freqdist)
        return self.custom_param / (total_count + vocab_size * self.custom_param)
```

### Adding Caching to Custom Methods
```python
from freqprob.cache import cached_computation

class CachedCustomMethod(ScoringMethod):
    """Custom method with intelligent caching."""

    @cached_computation(max_cache_size=1000)
    def _expensive_computation(self, param1: float, param2: int) -> dict:
        """Cache expensive computations."""
        # Expensive computation here
        return result

    def _compute_probabilities(self, freqdist: FrequencyDistribution) -> None:
        # Use cached computation
        cached_result = self._expensive_computation(self.param1, self.param2)
        self._probabilities = cached_result
```

### Creating Streaming Variants
```python
from freqprob.streaming import IncrementalScoringMethod

class StreamingCustomMethod(ScoringMethod, IncrementalScoringMethod):
    """Streaming version of custom method."""

    def __init__(self, initial_data: FrequencyDistribution = None,
                 max_vocabulary_size: int = 10000, **kwargs):
        self.max_vocabulary_size = max_vocabulary_size
        initial_data = initial_data or {}
        super().__init__(initial_data, **kwargs)

    def update_single(self, element: Element, count: int) -> None:
        """Update with single element."""
        self.freqdist[element] = self.freqdist.get(element, 0) + count

        # Manage vocabulary size
        if len(self.freqdist) > self.max_vocabulary_size:
            self._compress_vocabulary()

        # Recompute probabilities
        self._compute_probabilities(self.freqdist)

    def _compress_vocabulary(self):
        """Remove low-frequency elements."""
        sorted_items = sorted(self.freqdist.items(), key=lambda x: x[1])
        keep_count = self.max_vocabulary_size // 2
        self.freqdist = dict(sorted_items[-keep_count:])
```

## Common Workflows

### Language Model Training
```python
def train_language_model(text_corpus: list[str], n: int = 2) -> ScoringMethod:
    """Train n-gram language model from text corpus."""

    # Tokenize and generate n-grams
    all_ngrams = []
    for text in text_corpus:
        tokens = text.lower().split()
        ngrams = freqprob.generate_ngrams(tokens, n=n)
        all_ngrams.extend(ngrams)

    # Count n-gram frequencies
    ngram_counts = freqprob.ngram_frequency(all_ngrams, n=1)

    # Choose appropriate smoothing method
    if n == 1:
        # Unigram model with Laplace smoothing
        vocab_size = len(set(token for text in text_corpus for token in text.split()))
        return freqprob.Laplace(ngram_counts, bins=vocab_size, logprob=True)
    else:
        # Higher-order model with Kneser-Ney
        return freqprob.KneserNey(ngram_counts, discount=0.75, logprob=True)

# Usage
corpus = ["the quick brown fox", "the lazy dog sleeps", "brown fox jumps high"]
bigram_model = train_language_model(corpus, n=2)
prob = bigram_model(("the", "quick"))  # P(quick|the)
```

### Model Comparison and Selection
```python
def compare_smoothing_methods(train_data: FrequencyDistribution,
                            test_data: list[Element]) -> dict:
    """Compare different smoothing methods on test data."""

    # Define methods to compare
    methods = {
        "mle": freqprob.MLE(train_data, logprob=True),
        "laplace": freqprob.Laplace(train_data, bins=1000, logprob=True),
        "ele": freqprob.ELE(train_data, bins=1000, logprob=True),
        "bayesian": freqprob.BayesianSmoothing(train_data, alpha=0.5, logprob=True),
    }

    # Add Good-Turing if data is suitable
    try:
        methods["good_turing"] = freqprob.SimpleGoodTuring(train_data, logprob=True)
    except ValueError:
        pass  # Skip if Good-Turing fails

    # Comprehensive comparison
    comparison = freqprob.model_comparison(methods, test_data)

    # Find best model by perplexity
    best_method = min(comparison.items(), key=lambda x: x[1]["perplexity"])

    return {
        "best_method": best_method[0],
        "best_perplexity": best_method[1]["perplexity"],
        "all_results": comparison
    }
```

### Large-Scale Processing
```python
def process_large_dataset(data_stream, method_class: type[ScoringMethod],
                         batch_size: int = 10000) -> ScoringMethod:
    """Process large datasets efficiently."""

    # Use streaming approach
    if hasattr(freqprob, f"Streaming{method_class.__name__}"):
        streaming_class = getattr(freqprob, f"Streaming{method_class.__name__}")
        method = streaming_class(max_vocabulary_size=batch_size)

        # Process data in batches
        batch = []
        for item in data_stream:
            batch.append(item)
            if len(batch) >= batch_size:
                # Update streaming method
                counts = freqprob.word_frequency(batch)
                for word, count in counts.items():
                    method.update_single(word, count)
                batch = []

        # Process remaining items
        if batch:
            counts = freqprob.word_frequency(batch)
            for word, count in counts.items():
                method.update_single(word, count)

        return method

    else:
        # Collect all data first, then process
        all_data = list(data_stream)
        counts = freqprob.word_frequency(all_data)
        return method_class(counts)

# Memory monitoring during processing
def process_with_monitoring(data_stream, method_class):
    """Process with memory monitoring."""
    monitor = freqprob.MemoryMonitor()

    with monitor.profile_operation("large_dataset_processing"):
        method = process_large_dataset(data_stream, method_class)

    metrics = monitor.get_latest_metrics()
    print(f"Peak memory: {metrics.peak_memory_mb:.1f} MB")
    print(f"Processing time: {metrics.execution_time:.2f} seconds")

    return method
```

### Text Classification with Smoothed Features
```python
def create_text_classifier(labeled_documents: list[tuple[str, str]]) -> dict:
    """Create Naive Bayes classifier with smoothed probabilities."""

    # Separate documents by class
    class_documents = {}
    for text, label in labeled_documents:
        if label not in class_documents:
            class_documents[label] = []
        class_documents[label].append(text)

    # Train class-specific language models
    class_models = {}
    for label, documents in class_documents.items():
        # Combine all documents for this class
        all_words = []
        for doc in documents:
            all_words.extend(doc.lower().split())

        # Count word frequencies
        word_counts = freqprob.word_frequency(all_words)

        # Create smoothed model
        vocab_size = len(set(word for docs in class_documents.values()
                           for doc in docs for word in doc.lower().split()))

        class_models[label] = freqprob.Laplace(
            word_counts, bins=vocab_size, logprob=True
        )

    return class_models

def classify_document(text: str, class_models: dict) -> tuple[str, float]:
    """Classify document using trained models."""
    words = text.lower().split()

    class_scores = {}
    for label, model in class_models.items():
        # Sum log probabilities (Naive Bayes assumption)
        score = sum(model(word) for word in words)
        class_scores[label] = score

    # Return most likely class
    best_class = max(class_scores.items(), key=lambda x: x[1])
    return best_class[0], best_class[1]
```

This comprehensive documentation provides LLM coding agents with detailed understanding of the FreqProb library's architecture, mathematical foundations, implementation patterns, and practical usage examples. It covers everything from basic usage to advanced optimization techniques and extension guidelines.
