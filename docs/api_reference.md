# FreqProb API Reference

Complete API documentation for all FreqProb classes and functions.

## Table of Contents

1. [Core Classes](#core-classes)
2. [Basic Smoothing Methods](#basic-smoothing-methods)
3. [Advanced Smoothing Methods](#advanced-smoothing-methods)
4. [Utility Functions](#utility-functions)
5. [Efficiency Features](#efficiency-features)
6. [Memory Management](#memory-management)
7. [Streaming Classes](#streaming-classes)
8. [Type Definitions](#type-definitions)

---

## Core Classes

### ScoringMethod

Base class for all probability scoring methods.

```python
class ScoringMethod(ABC)
```

**Abstract Methods:**
- `__call__(element: Element) -> Union[Probability, LogProbability]`

**Properties:**
- `logprob: bool` - Whether the method returns log probabilities
- `config: ScoringMethodConfig` - Configuration object

**Methods:**
- `get_config() -> ScoringMethodConfig` - Get method configuration

---

### ScoringMethodConfig

Configuration class for scoring methods.

```python
class ScoringMethodConfig
```

**Parameters:**
- `unobs_prob: Optional[float] = None` - Probability for unobserved elements
- `logprob: bool = True` - Whether to return log probabilities

---

## Basic Smoothing Methods

### MLE

Maximum Likelihood Estimation (no smoothing).

```python
class MLE(ScoringMethod)
```

**Constructor:**
```python
MLE(freqdist: FrequencyDistribution,
    unobs_prob: Optional[float] = None,
    logprob: bool = True)
```

**Parameters:**
- `freqdist` - Dictionary mapping elements to counts
- `unobs_prob` - Probability mass for unobserved elements (default: 0.0)
- `logprob` - Return log probabilities if True

**Example:**
```python
freqdist = {'cat': 3, 'dog': 2, 'bird': 1}
mle = freqprob.MLE(freqdist, logprob=False)
print(mle('cat'))  # 0.5 (3/6)
```

**Mathematical Formula:**
$$P_{MLE}(w) = \frac{c(w)}{N}$$

---

### Laplace

Laplace smoothing (add-one smoothing).

```python
class Laplace(ScoringMethod)
```

**Constructor:**
```python
Laplace(freqdist: FrequencyDistribution,
        bins: Optional[int] = None,
        logprob: bool = True)
```

**Parameters:**
- `freqdist` - Dictionary mapping elements to counts
- `bins` - Total number of possible elements (vocabulary size)
- `logprob` - Return log probabilities if True

**Example:**
```python
freqdist = {'cat': 3, 'dog': 2, 'bird': 1}
laplace = freqprob.Laplace(freqdist, bins=1000, logprob=False)
print(laplace('cat'))    # (3+1)/(6+1000) ≈ 0.004
print(laplace('mouse'))  # (0+1)/(6+1000) ≈ 0.001
```

**Mathematical Formula:**
$$P_{Laplace}(w) = \frac{c(w) + 1}{N + B}$$

---

### Lidstone

Lidstone smoothing (add-k smoothing).

```python
class Lidstone(ScoringMethod)
```

**Constructor:**
```python
Lidstone(freqdist: FrequencyDistribution,
         gamma: float,
         bins: Optional[int] = None,
         logprob: bool = True)
```

**Parameters:**
- `freqdist` - Dictionary mapping elements to counts
- `gamma` - Smoothing parameter (k in add-k)
- `bins` - Total number of possible elements
- `logprob` - Return log probabilities if True

**Example:**
```python
freqdist = {'cat': 3, 'dog': 2, 'bird': 1}
lidstone = freqprob.Lidstone(freqdist, gamma=0.5, bins=1000, logprob=False)
print(lidstone('cat'))  # (3+0.5)/(6+500) ≈ 0.007
```

**Mathematical Formula:**
$$P_{Lidstone}(w) = \frac{c(w) + \gamma}{N + \gamma \cdot B}$$

---

### ELE

Expected Likelihood Estimation (Lidstone with γ=0.5).

```python
class ELE(ScoringMethod)
```

**Constructor:**
```python
ELE(freqdist: FrequencyDistribution,
    bins: Optional[int] = None,
    logprob: bool = True)
```

**Parameters:**
- `freqdist` - Dictionary mapping elements to counts
- `bins` - Total number of possible elements
- `logprob` - Return log probabilities if True

**Example:**
```python
freqdist = {'cat': 3, 'dog': 2, 'bird': 1}
ele = freqprob.ELE(freqdist, bins=1000, logprob=False)
print(ele('cat'))  # (3+0.5)/(6+500) ≈ 0.007
```

**Mathematical Formula:**
$$P_{ELE}(w) = \frac{c(w) + 0.5}{N + 0.5 \cdot B}$$

---

### Uniform

Uniform distribution over observed elements.

```python
class Uniform(ScoringMethod)
```

**Constructor:**
```python
Uniform(freqdist: FrequencyDistribution,
        unobs_prob: float = 0.1,
        logprob: bool = True)
```

**Parameters:**
- `freqdist` - Dictionary mapping elements to counts
- `unobs_prob` - Probability for unobserved elements
- `logprob` - Return log probabilities if True

**Mathematical Formula:**
$$P_{Uniform}(w) = \begin{cases}
\frac{1-p_{unobs}}{V} & \text{if } c(w) > 0 \\
p_{unobs} & \text{if } c(w) = 0
\end{cases}$$

---

### Random

Random probability assignment.

```python
class Random(ScoringMethod)
```

**Constructor:**
```python
Random(freqdist: FrequencyDistribution,
       unobs_prob: float = 0.1,
       seed: Optional[int] = None,
       logprob: bool = True)
```

**Parameters:**
- `freqdist` - Dictionary mapping elements to counts
- `unobs_prob` - Probability for unobserved elements
- `seed` - Random seed for reproducibility
- `logprob` - Return log probabilities if True

---

## Advanced Smoothing Methods

### SimpleGoodTuring

Simple Good-Turing smoothing using frequency-of-frequencies.

```python
class SimpleGoodTuring(ScoringMethod)
```

**Constructor:**
```python
SimpleGoodTuring(freqdist: FrequencyDistribution,
                 p_value: float = 0.05,
                 allow_fail: bool = False,
                 logprob: bool = True)
```

**Parameters:**
- `freqdist` - Dictionary mapping elements to counts
- `p_value` - Confidence level for smoothing decisions
- `allow_fail` - Whether to raise exception on failure
- `logprob` - Return log probabilities if True

**Example:**
```python
freqdist = {'cat': 3, 'dog': 2, 'bird': 1, 'fish': 1}
try:
    sgt = freqprob.SimpleGoodTuring(freqdist, logprob=False)
    print(sgt('cat'))
    print(sgt('unknown'))  # Non-zero probability
except ValueError:
    print("SGT failed - frequency distribution unsuitable")
```

**Mathematical Foundation:**
Uses Good-Turing frequency estimation with log-linear smoothing of frequency-of-frequencies.

---

### KneserNey

Kneser-Ney smoothing for n-gram language models.

```python
class KneserNey(ScoringMethod)
```

**Constructor:**
```python
KneserNey(freqdist: FrequencyDistribution,
          discount: float = 0.75,
          logprob: bool = True)
```

**Parameters:**
- `freqdist` - Dictionary mapping n-grams to counts
- `discount` - Absolute discount parameter (0 < d < 1)
- `logprob` - Return log probabilities if True

**Example:**
```python
# Bigram counts: context -> word
bigrams = {
    ('the', 'cat'): 5, ('the', 'dog'): 3,
    ('a', 'cat'): 2, ('a', 'dog'): 1
}
kn = freqprob.KneserNey(bigrams, discount=0.75, logprob=False)
print(kn(('the', 'cat')))
```

**Mathematical Formula:**
$$P_{KN}(w_i|w_{i-1}) = \frac{\max(c(w_{i-1}, w_i) - d, 0)}{c(w_{i-1})} + \lambda(w_{i-1}) \cdot P_{cont}(w_i)$$

---

### ModifiedKneserNey

Modified Kneser-Ney with count-dependent discounting.

```python
class ModifiedKneserNey(ScoringMethod)
```

**Constructor:**
```python
ModifiedKneserNey(freqdist: FrequencyDistribution,
                  logprob: bool = True)
```

**Parameters:**
- `freqdist` - Dictionary mapping n-grams to counts
- `logprob` - Return log probabilities if True

**Example:**
```python
bigrams = {('the', 'cat'): 5, ('the', 'dog'): 3, ('a', 'cat'): 2}
mkn = freqprob.ModifiedKneserNey(bigrams, logprob=False)
print(mkn(('the', 'cat')))
```

**Features:**
- Automatically estimates optimal discount parameters
- Uses different discounts for counts 1, 2, and 3+
- Generally superior to standard Kneser-Ney

---

### InterpolatedSmoothing

Linear interpolation between two models.

```python
class InterpolatedSmoothing(ScoringMethod)
```

**Constructor:**
```python
InterpolatedSmoothing(high_order_freqdist: FrequencyDistribution,
                      low_order_freqdist: FrequencyDistribution,
                      lambda_weight: float = 0.7,
                      logprob: bool = True)
```

**Parameters:**
- `high_order_freqdist` - Higher-order model frequency distribution
- `low_order_freqdist` - Lower-order model frequency distribution
- `lambda_weight` - Interpolation weight (0 < λ < 1)
- `logprob` - Return log probabilities if True

**Example:**
```python
trigrams = {('the', 'big', 'cat'): 3}
bigrams = {('big', 'cat'): 5}
interpolated = freqprob.InterpolatedSmoothing(
    trigrams, bigrams, lambda_weight=0.7, logprob=False
)
```

**Mathematical Formula:**
$$P_{interp}(w|context) = \lambda P_{high}(w|context) + (1-\lambda) P_{low}(w|context)$$

---

### BayesianSmoothing

Bayesian smoothing with Dirichlet prior.

```python
class BayesianSmoothing(ScoringMethod)
```

**Constructor:**
```python
BayesianSmoothing(freqdist: FrequencyDistribution,
                  alpha: float = 1.0,
                  logprob: bool = True)
```

**Parameters:**
- `freqdist` - Dictionary mapping elements to counts
- `alpha` - Dirichlet concentration parameter
- `logprob` - Return log probabilities if True

**Example:**
```python
freqdist = {'cat': 3, 'dog': 2, 'bird': 1}
bayesian = freqprob.BayesianSmoothing(freqdist, alpha=0.5, logprob=False)
print(bayesian('cat'))
```

**Mathematical Formula:**
$$P_{Bayes}(w) = \frac{c(w) + \alpha}{N + V \cdot \alpha}$$

**Parameter Guidelines:**
- `α = 0.5`: Jeffreys prior (often optimal)
- `α = 1.0`: Uniform prior (equivalent to Laplace)
- `α > 1.0`: Stronger uniformity bias

---

### WittenBell

Witten-Bell smoothing.

```python
class WittenBell(ScoringMethod)
```

**Constructor:**
```python
WittenBell(freqdist: FrequencyDistribution,
           bins: Optional[int] = None,
           logprob: bool = True)
```

**Parameters:**
- `freqdist` - Dictionary mapping elements to counts
- `bins` - Total number of possible elements
- `logprob` - Return log probabilities if True

**Mathematical Formula:**
$$P_{WB}(w) = \begin{cases}
\frac{c(w)}{N + T} & \text{if } c(w) > 0 \\
\frac{T}{Z(N + T)} & \text{if } c(w) = 0
\end{cases}$$

---

### CertaintyDegree

Experimental certainty degree estimation.

```python
class CertaintyDegree(ScoringMethod)
```

**Constructor:**
```python
CertaintyDegree(freqdist: FrequencyDistribution,
                bins: Optional[int] = None,
                logprob: bool = True)
```

**Note:** This is an experimental method with variable performance across domains.

---

## Utility Functions

### Model Evaluation

#### perplexity

Calculate perplexity of a model on test data.

```python
def perplexity(model: ScoringMethod,
               test_data: List[Element]) -> float
```

**Parameters:**
- `model` - Trained scoring method (must use `logprob=True`)
- `test_data` - List of test elements

**Returns:**
- Perplexity value (lower is better)

**Example:**
```python
model = freqprob.KneserNey(bigrams, logprob=True)
test_words = ['the', 'cat', 'sat', 'on', 'mat']
pp = freqprob.perplexity(model, test_words)
print(f"Perplexity: {pp:.2f}")
```

**Formula:**
$$PP = \exp\left(-\frac{1}{N}\sum_{i=1}^{N} \log P(w_i)\right)$$

---

#### cross_entropy

Calculate cross-entropy between model and test data.

```python
def cross_entropy(model: ScoringMethod,
                  test_data: List[Element]) -> float
```

**Parameters:**
- `model` - Trained scoring method (must use `logprob=True`)
- `test_data` - List of test elements

**Returns:**
- Cross-entropy in bits (lower is better)

**Example:**
```python
ce = freqprob.cross_entropy(model, test_words)
print(f"Cross-entropy: {ce:.2f} bits")
```

---

#### kl_divergence

Calculate KL divergence between two models.

```python
def kl_divergence(p_model: ScoringMethod,
                  q_model: ScoringMethod,
                  test_data: List[Element]) -> float
```

**Parameters:**
- `p_model` - Reference model (must use `logprob=True`)
- `q_model` - Comparison model (must use `logprob=True`)
- `test_data` - List of test elements

**Returns:**
- KL divergence (lower means more similar)

---

#### model_comparison

Compare multiple models on test data.

```python
def model_comparison(models: Dict[str, ScoringMethod],
                     test_data: List[Element]) -> Dict[str, Dict[str, float]]
```

**Parameters:**
- `models` - Dictionary of model name -> model
- `test_data` - List of test elements

**Returns:**
- Dictionary with metrics for each model

**Example:**
```python
models = {
    'mle': freqprob.MLE(freqdist, logprob=True),
    'laplace': freqprob.Laplace(freqdist, logprob=True)
}
results = freqprob.model_comparison(models, test_data)
for name, metrics in results.items():
    print(f"{name}: PP={metrics['perplexity']:.2f}")
```

---

### Text Processing

#### generate_ngrams

Generate n-grams from text.

```python
def generate_ngrams(tokens: List[str], n: int) -> List[Tuple[str, ...]]
```

**Parameters:**
- `tokens` - List of tokens
- `n` - N-gram order

**Returns:**
- List of n-gram tuples

**Example:**
```python
tokens = ['the', 'cat', 'sat']
bigrams = freqprob.generate_ngrams(tokens, 2)
# [('<s>', 'the'), ('the', 'cat'), ('cat', 'sat'), ('sat', '</s>')]
```

---

#### word_frequency

Calculate word frequencies from text.

```python
def word_frequency(tokens: List[str]) -> Dict[str, int]
```

**Parameters:**
- `tokens` - List of word tokens

**Returns:**
- Dictionary mapping words to frequencies

---

#### ngram_frequency

Calculate n-gram frequencies from text.

```python
def ngram_frequency(tokens: List[str], n: int) -> Dict[Tuple[str, ...], int]
```

**Parameters:**
- `tokens` - List of word tokens
- `n` - N-gram order

**Returns:**
- Dictionary mapping n-grams to frequencies

---

## Efficiency Features

### VectorizedScorer

Efficient batch scoring operations.

```python
class VectorizedScorer
```

**Constructor:**
```python
VectorizedScorer(scorer: ScoringMethod)
```

**Methods:**

#### score_batch

Score multiple elements efficiently.

```python
def score_batch(self, elements: List[Element]) -> np.ndarray
```

**Example:**
```python
mle = freqprob.MLE(freqdist, logprob=False)
vectorized = freqprob.VectorizedScorer(mle)
elements = ['cat', 'dog', 'bird']
scores = vectorized.score_batch(elements)
# Returns numpy array of scores
```

#### score_matrix

Score 2D array of elements.

```python
def score_matrix(self, elements_2d: List[List[Element]]) -> np.ndarray
```

#### top_k_elements

Get top-k highest scoring elements.

```python
def top_k_elements(self, k: int) -> Tuple[List[Element], np.ndarray]
```

#### normalize_scores

Normalize scores to probabilities.

```python
def normalize_scores(self, scores: np.ndarray) -> np.ndarray
```

---

### BatchScorer

Score with multiple methods simultaneously.

```python
class BatchScorer
```

**Constructor:**
```python
BatchScorer(scorers: Dict[str, ScoringMethod])
```

**Methods:**

#### score_batch

Score elements with all methods.

```python
def score_batch(self, elements: List[Element]) -> Dict[str, np.ndarray]
```

**Example:**
```python
scorers = {
    'mle': freqprob.MLE(freqdist),
    'laplace': freqprob.Laplace(freqdist)
}
batch_scorer = freqprob.BatchScorer(scorers)
results = batch_scorer.score_batch(['cat', 'dog'])
# Returns: {'mle': array([...]), 'laplace': array([...])}
```

---

### Lazy Evaluation

#### create_lazy_mle

Create lazy MLE scorer.

```python
def create_lazy_mle(freqdist: FrequencyDistribution,
                    logprob: bool = True) -> LazyScoringMethod
```

#### create_lazy_laplace

Create lazy Laplace scorer.

```python
def create_lazy_laplace(freqdist: FrequencyDistribution,
                        bins: Optional[int] = None,
                        logprob: bool = True) -> LazyScoringMethod
```

**Example:**
```python
lazy_mle = freqprob.create_lazy_mle(huge_freqdist)
# Only computes probabilities when accessed
prob = lazy_mle('word')  # Computed on first access
prob = lazy_mle('word')  # Cached on subsequent access
```

### LazyScoringMethod

Lazy evaluation scoring method.

**Methods:**

#### get_computed_elements

Get list of computed elements.

```python
def get_computed_elements(self) -> List[Element]
```

#### clear_cache

Clear computed probability cache.

```python
def clear_cache(self) -> None
```

---

### Caching

#### get_cache_stats

Get caching statistics.

```python
def get_cache_stats() -> Dict[str, int]
```

#### clear_all_caches

Clear all internal caches.

```python
def clear_all_caches() -> None
```

---

## Memory Management

### Compressed Representations

#### create_compressed_distribution

Create memory-efficient compressed distribution.

```python
def create_compressed_distribution(
    freqdist: FrequencyDistribution,
    quantization_levels: Optional[int] = None,
    use_compression: bool = True
) -> CompressedFrequencyDistribution
```

**Parameters:**
- `freqdist` - Original frequency distribution
- `quantization_levels` - Number of quantization levels for compression
- `use_compression` - Whether to use data compression

**Example:**
```python
large_freqdist = {f'word_{i}': max(1, 1000-i) for i in range(10000)}
compressed = freqprob.create_compressed_distribution(
    large_freqdist,
    quantization_levels=1024
)
print(compressed.get_memory_usage())
```

---

#### create_sparse_distribution

Create sparse representation for sparse data.

```python
def create_sparse_distribution(
    freqdist: FrequencyDistribution,
    default_count: int = 0
) -> SparseFrequencyDistribution
```

---

### CompressedFrequencyDistribution

Memory-efficient frequency distribution.

**Methods:**

#### get_count

```python
def get_count(self, element: Element) -> int
```

#### get_memory_usage

```python
def get_memory_usage(self) -> Dict[str, int]
```

#### compress_to_bytes

```python
def compress_to_bytes(self) -> bytes
```

#### decompress_from_bytes

```python
@classmethod
def decompress_from_bytes(cls, data: bytes) -> 'CompressedFrequencyDistribution'
```

---

### SparseFrequencyDistribution

Sparse representation optimized for distributions with many zeros.

**Methods:**

#### get_top_k

```python
def get_top_k(self, k: int) -> List[Tuple[Element, int]]
```

#### get_elements_with_count_range

```python
def get_elements_with_count_range(self, min_count: int, max_count: int) -> List[Element]
```

#### get_count_histogram

```python
def get_count_histogram(self) -> Dict[int, int]
```

---

### Memory Profiling

#### MemoryProfiler

Monitor memory usage during operations.

```python
class MemoryProfiler
```

**Methods:**

#### profile_operation

Context manager for profiling operations.

```python
def profile_operation(self, operation_name: str)
```

#### get_latest_metrics

```python
def get_latest_metrics() -> MemoryMetrics
```

**Example:**
```python
profiler = freqprob.MemoryProfiler()
with profiler.profile_operation("model_creation"):
    model = freqprob.SimpleGoodTuring(large_freqdist)

metrics = profiler.get_latest_metrics()
print(f"Memory used: {metrics.memory_delta_mb:.2f} MB")
```

---

## Streaming Classes

### StreamingMLE

Streaming Maximum Likelihood Estimation with incremental updates.

```python
class StreamingMLE(ScoringMethod, IncrementalScoringMethod)
```

**Constructor:**
```python
StreamingMLE(initial_freqdist: Optional[Dict[Element, int]] = None,
             max_vocabulary_size: Optional[int] = None,
             unobs_prob: Optional[float] = None,
             logprob: bool = True)
```

**Methods:**

#### update_single

Update with a single observation.

```python
def update_single(self, element: Element, count: int = 1) -> None
```

#### update_batch

Update with multiple observations.

```python
def update_batch(self, elements: List[Element],
                 counts: Optional[List[int]] = None) -> None
```

#### get_streaming_statistics

```python
def get_streaming_statistics(self) -> Dict[str, Any]
```

#### save_state / load_state

```python
def save_state(self, filepath: str) -> None

@classmethod
def load_state(cls, filepath: str) -> 'StreamingMLE'
```

**Example:**
```python
streaming_mle = freqprob.StreamingMLE(max_vocabulary_size=10000, logprob=False)
streaming_mle.update_single('word1', 5)
streaming_mle.update_batch(['word2', 'word3'])
prob = streaming_mle('word1')
```

---

### StreamingLaplace

Streaming Laplace smoothing.

```python
class StreamingLaplace(StreamingMLE)
```

**Constructor:**
```python
StreamingLaplace(initial_freqdist: Optional[Dict[Element, int]] = None,
                 max_vocabulary_size: Optional[int] = None,
                 bins: Optional[int] = None,
                 logprob: bool = True)
```

---

### StreamingDataProcessor

High-level processor for streaming text data.

```python
class StreamingDataProcessor
```

**Constructor:**
```python
StreamingDataProcessor(scoring_methods: Dict[str, IncrementalScoringMethod],
                       batch_size: int = 1000,
                       auto_save_interval: Optional[int] = None)
```

**Methods:**

#### process_text_stream

```python
def process_text_stream(self, text_stream: Iterator[str]) -> None
```

#### get_score

```python
def get_score(self, method_name: str, element: Element) -> float
```

#### get_statistics

```python
def get_statistics(self) -> Dict[str, Any]
```

---

## Type Definitions

### Basic Types

```python
# Core element type - can be string, int, float, tuple, or frozenset
Element = Union[str, int, float, tuple, frozenset]

# Count type for frequency distributions
Count = int

# Probability types
Probability = float
LogProbability = float

# Frequency distribution type
FrequencyDistribution = Dict[Element, Count]
```

### Configuration Types

```python
class ScoringMethodConfig:
    unobs_prob: Optional[float]
    logprob: bool
```

### Streaming Types

```python
class IncrementalScoringMethod(ABC):
    @abstractmethod
    def update_single(self, element: Element, count: int = 1) -> None: ...

    @abstractmethod
    def update_batch(self, elements: List[Element],
                     counts: Optional[List[int]] = None) -> None: ...

    @abstractmethod
    def get_update_count(self) -> int: ...
```

---

## Error Handling

### Common Exceptions

- `ValueError`: Invalid parameters or incompatible data
- `ZeroDivisionError`: Empty frequency distributions
- `KeyError`: Missing required elements
- `RuntimeError`: Method-specific failures (e.g., SGT convergence)

### Error Prevention

```python
# Always check for empty distributions
if not freqdist:
    raise ValueError("Empty frequency distribution")

# Validate smoothing parameters
if gamma <= 0:
    raise ValueError("Gamma must be positive")

# Handle method failures gracefully
try:
    sgt = freqprob.SimpleGoodTuring(freqdist)
except RuntimeError:
    # Fallback to simpler method
    sgt = freqprob.ELE(freqdist)
```

---

## Usage Examples

### Basic Usage

```python
import freqprob

# Create frequency distribution
freqdist = {'the': 100, 'cat': 50, 'dog': 30, 'bird': 10}

# Basic smoothing
mle = freqprob.MLE(freqdist, logprob=False)
laplace = freqprob.Laplace(freqdist, bins=1000, logprob=False)

print(f"MLE P(cat) = {mle('cat'):.4f}")
print(f"Laplace P(cat) = {laplace('cat'):.4f}")
print(f"Laplace P(unseen) = {laplace('elephant'):.6f}")
```

### Advanced Usage

```python
# N-gram language modeling
bigrams = {('the', 'cat'): 5, ('the', 'dog'): 3, ('a', 'cat'): 2}
kn = freqprob.KneserNey(bigrams, discount=0.75, logprob=True)

# Model evaluation
test_bigrams = [('the', 'cat'), ('a', 'dog')]
pp = freqprob.perplexity(kn, test_bigrams)
print(f"Perplexity: {pp:.2f}")

# Efficiency features
vectorized = freqprob.VectorizedScorer(laplace)
batch_scores = vectorized.score_batch(['cat', 'dog', 'bird'])
```

### Streaming Usage

```python
# Real-time learning
streaming = freqprob.StreamingMLE(max_vocabulary_size=10000, logprob=False)

# Process incoming data
for word in data_stream:
    streaming.update_single(word)
    if streaming.get_update_count() % 1000 == 0:
        print(f"Processed {streaming.get_update_count()} updates")

# Get current probability
current_prob = streaming('word')
```

This API reference provides comprehensive documentation for all FreqProb functionality. For additional examples and tutorials, see the user guide and Jupyter notebooks.
