# FreqProb User Guide

A comprehensive guide to frequency-based probability estimation and smoothing methods.

## Table of Contents

1. [Introduction](#introduction)
2. [Mathematical Background](#mathematical-background)
3. [Basic Smoothing Methods](#basic-smoothing-methods)
4. [Advanced Smoothing Methods](#advanced-smoothing-methods)
5. [Modern Smoothing Techniques](#modern-smoothing-techniques)
6. [Computational Efficiency](#computational-efficiency)
7. [Memory Management](#memory-management)
8. [Model Evaluation](#model-evaluation)
9. [Best Practices](#best-practices)
10. [Common Use Cases](#common-use-cases)

## Introduction

FreqProb is a comprehensive library for frequency-based probability estimation and smoothing methods, primarily designed for natural language processing applications. The library provides implementations of classical and modern smoothing techniques, along with advanced features for computational efficiency and memory management.

### Key Features

- **Comprehensive Smoothing Methods**: From basic MLE to advanced Kneser-Ney smoothing
- **Computational Efficiency**: Vectorized operations, caching, and lazy evaluation
- **Memory Management**: Streaming updates and compressed representations
- **Flexible Data Types**: Support for strings, integers, tuples, and custom types
- **Performance Optimization**: Built for large-scale NLP applications
- **Model Evaluation**: Built-in metrics for comparing different smoothing approaches

### Installation

```bash
pip install freqprob
```

### Quick Start

```python
import freqprob

# Create a frequency distribution
freqdist = {'the': 1000, 'cat': 50, 'dog': 30, 'bird': 10}

# Basic Maximum Likelihood Estimation
mle = freqprob.MLE(freqdist, logprob=False)
print(f"P(cat) = {mle('cat')}")  # 0.045

# Laplace Smoothing
laplace = freqprob.Laplace(freqdist, logprob=False)
print(f"P(unknown) = {laplace('unknown')}")  # Non-zero probability

# Model Comparison
models = {'mle': mle, 'laplace': laplace}
test_data = ['the', 'cat', 'unknown']
comparison = freqprob.model_comparison(models, test_data)
print(comparison)
```

## Mathematical Background

### The Probability Estimation Problem

Given a frequency distribution $D = \{(w_1, c_1), (w_2, c_2), \ldots, (w_n, c_n)\}$ where $w_i$ are elements (words, n-grams, etc.) and $c_i$ are their observed counts, we want to estimate the probability $P(w)$ for any element $w$.

The fundamental challenge is handling **unseen events** - elements that don't appear in our training data but might occur in test data. This is known as the **zero probability problem**.

### Mathematical Notation

Throughout this guide, we use the following notation:

- $N = \sum_{i=1}^{n} c_i$ : Total count of observations
- $V$ : Vocabulary size (number of unique elements)
- $c(w)$ : Count of element $w$ in the training data
- $P(w)$ : Estimated probability of element $w$
- $B$ : Total number of possible elements (bins)

### The Zero Probability Problem

Maximum Likelihood Estimation (MLE) assigns probability:

$$P_{MLE}(w) = \frac{c(w)}{N}$$

This gives $P(w) = 0$ for unseen elements, which is problematic because:

1. **Product Rule**: $P(w_1, w_2, \ldots, w_k) = \prod_{i=1}^{k} P(w_i) = 0$ if any $P(w_i) = 0$
2. **Generalization**: Real applications often encounter unseen elements
3. **Evaluation Metrics**: Many metrics (perplexity, cross-entropy) are undefined with zero probabilities

## Basic Smoothing Methods

### Maximum Likelihood Estimation (MLE)

The simplest approach that assigns probability proportional to observed frequency.

**Formula:**
$$P_{MLE}(w) = \frac{c(w)}{N}$$

**Properties:**
- No smoothing for unseen elements
- Optimal for training data
- Poor generalization to unseen data

**Implementation:**
```python
mle = freqprob.MLE(freqdist, logprob=False)
prob = mle('word')
```

### Uniform Distribution

Assigns equal probability to all observed elements.

**Formula:**
$$P_{Uniform}(w) = \begin{cases}
\frac{1}{V} & \text{if } c(w) > 0 \\
\alpha & \text{if } c(w) = 0
\end{cases}$$

where $\alpha$ is the reserved probability mass for unseen elements.

**Implementation:**
```python
uniform = freqprob.Uniform(freqdist, unobs_prob=0.1, logprob=False)
```

### Random Distribution

Assigns random probabilities from a uniform distribution, normalized to sum to 1.

**Properties:**
- Useful for baseline comparisons
- Non-deterministic (can set seed for reproducibility)
- Provides non-zero probabilities for all elements

**Implementation:**
```python
random_dist = freqprob.Random(freqdist, unobs_prob=0.1, seed=42, logprob=False)
```

## Advanced Smoothing Methods

### Lidstone Smoothing (Add-k Smoothing)

Adds a constant $\gamma$ to all counts before normalization.

**Formula:**
$$P_{Lidstone}(w) = \frac{c(w) + \gamma}{N + \gamma \cdot B}$$

**Parameters:**
- $\gamma > 0$ : Smoothing parameter
- $B$ : Total number of possible elements (bins)

**Special Cases:**
- $\gamma = 1$ : Laplace smoothing (add-one)
- $\gamma \to 0$ : Approaches MLE
- $\gamma \to \infty$ : Approaches uniform distribution

**Implementation:**
```python
# General Lidstone smoothing
lidstone = freqprob.Lidstone(freqdist, gamma=0.5, bins=10000, logprob=False)

# Laplace smoothing (special case)
laplace = freqprob.Laplace(freqdist, bins=10000, logprob=False)
```

### Expected Likelihood Estimation (ELE)

A special case of Lidstone smoothing with $\gamma = 0.5$.

**Formula:**
$$P_{ELE}(w) = \frac{c(w) + 0.5}{N + 0.5 \cdot B}$$

**Properties:**
- Theoretically motivated by Bayesian inference
- Good balance between smoothing and preserving frequency patterns
- Less aggressive smoothing than Laplace

**Implementation:**
```python
ele = freqprob.ELE(freqdist, bins=10000, logprob=False)
```

### Witten-Bell Smoothing

Reserves probability mass for unseen elements based on the number of singletons.

**Formula:**
$$P_{WB}(w) = \begin{cases}
\frac{c(w)}{N + T} & \text{if } c(w) > 0 \\
\frac{T}{Z(N + T)} & \text{if } c(w) = 0
\end{cases}$$

where:
- $T$ = number of observed types (vocabulary size)
- $Z$ = number of unseen types = $B - T$

**Key Insight:** Uses the number of word types as an estimate of the probability of seeing a new type.

**Implementation:**
```python
witten_bell = freqprob.WittenBell(freqdist, bins=10000, logprob=False)
```

### Simple Good-Turing Smoothing

Based on the Good-Turing frequency estimation method, using frequency-of-frequencies statistics.

**Mathematical Foundation:**

For count $r$, the adjusted count is:
$$r^* = (r + 1) \frac{n_{r+1}}{n_r}$$

where $n_r$ is the number of elements that occur exactly $r$ times.

**Algorithm:**
1. Compute frequency-of-frequencies: $n_r = |\{w : c(w) = r\}|$
2. Apply log-linear smoothing for reliable estimates
3. Use confidence intervals to decide between empirical and smoothed estimates

**Implementation:**
```python
sgt = freqprob.SimpleGoodTuring(freqdist, p_value=0.05, logprob=False)
```

**Parameters:**
- `p_value`: Confidence level for smoothing threshold
- `allow_fail`: Whether to raise errors on invalid assumptions

### Certainty Degree Estimation

An experimental method that estimates probability mass for unseen elements based on the "certainty" of having observed all important elements.

**Formula:**
$$P_{CD}(w) = \frac{c(w)}{N} \cdot \left(1 - \left(\frac{V}{Z+1}\right)^N\right)$$

**Properties:**
- Experimental method under development
- Attempts to model the probability that no important elements are missing
- Performance varies significantly across domains

**Implementation:**
```python
certainty = freqprob.CertaintyDegree(freqdist, bins=10000, logprob=False)
```

## Modern Smoothing Techniques

### Kneser-Ney Smoothing

One of the most effective smoothing methods for language modeling, based on absolute discounting and continuation probabilities.

**Mathematical Foundation:**

For bigram model $P(w_i|w_{i-1})$:

$$P_{KN}(w_i|w_{i-1}) = \frac{\max(c(w_{i-1}, w_i) - d, 0)}{c(w_{i-1})} + \lambda(w_{i-1}) \cdot P_{cont}(w_i)$$

where:
- $d$ : Discount parameter $(0 < d < 1)$
- $\lambda(w_{i-1}) = \frac{d \cdot |\{w : c(w_{i-1}, w) > 0\}|}{c(w_{i-1})}$ : Backoff weight
- $P_{cont}(w_i) = \frac{|\{w : c(w, w_i) > 0\}|}{|\{(w, w') : c(w, w') > 0\}|}$ : Continuation probability

**Key Insights:**
- Uses absolute discounting instead of relative discounting
- Continuation probability captures how likely a word is to appear in novel contexts
- Particularly effective for n-gram language models

**Implementation:**
```python
# Input should be bigrams: {(context, word): count}
bigram_counts = {
    ('the', 'cat'): 5, ('the', 'dog'): 3, ('a', 'cat'): 2,
    ('a', 'dog'): 1, ('big', 'cat'): 1, ('small', 'dog'): 1
}
kn = freqprob.KneserNey(bigram_counts, discount=0.75, logprob=False)
prob = kn(('the', 'cat'))
```

### Modified Kneser-Ney Smoothing

An enhanced version that uses different discount values for different frequency counts.

**Mathematical Foundation:**

$$P_{MKN}(w_i|w_{i-1}) = \frac{\max(c(w_{i-1}, w_i) - D(c(w_{i-1}, w_i)), 0)}{c(w_{i-1})} + \lambda(w_{i-1}) \cdot P_{cont}(w_i)$$

where $D(c)$ is a count-dependent discount:
- $D(1) = d_1$ for singleton counts
- $D(2) = d_2$ for doubleton counts  
- $D(c) = d_3$ for $c \geq 3$

The discounts are estimated from the data:
- $d_1 = 1 - 2 \frac{n_2}{n_1} \frac{n_1}{n_1 + 2n_2}$
- $d_2 = 2 - 3 \frac{n_3}{n_2} \frac{n_2}{n_2 + 3n_3}$
- $d_3 = 3 - 4 \frac{n_4}{n_3} \frac{n_3}{n_3 + 4n_4}$

**Implementation:**
```python
mkn = freqprob.ModifiedKneserNey(bigram_counts, logprob=False)
```

### Interpolated Smoothing

Combines estimates from multiple models using weighted linear interpolation.

**Formula:**
$$P_{interp}(w|context) = \lambda P_{high}(w|context) + (1-\lambda) P_{low}(w|context)$$

**Use Cases:**
- Combining different n-gram orders
- Blending domain-specific and general models
- Ensemble methods

**Implementation:**
```python
# Combine trigram and bigram models
trigrams = {('the', 'big', 'cat'): 3, ('a', 'big', 'dog'): 2}
bigrams = {('big', 'cat'): 5, ('big', 'dog'): 3}

interpolated = freqprob.InterpolatedSmoothing(
    trigrams, bigrams, lambda_weight=0.7, logprob=False
)
```

### Bayesian Smoothing

Uses a Dirichlet prior distribution for theoretically principled probability estimates.

**Formula:**
$$P_{Bayes}(w) = \frac{c(w) + \alpha}{N + V \cdot \alpha}$$

where $\alpha$ is the Dirichlet concentration parameter.

**Parameter Interpretation:**
- $\alpha \to 0$ : Approaches MLE (minimal smoothing)
- $\alpha = 1$ : Uniform prior (equivalent to Laplace smoothing)
- $\alpha > 1$ : Stronger preference for uniformity

**Implementation:**
```python
bayesian = freqprob.BayesianSmoothing(freqdist, alpha=1.0, logprob=False)
```

## Computational Efficiency

### Vectorized Operations

For processing large datasets efficiently, FreqProb provides vectorized operations using numpy.

**VectorizedScorer:**
```python
from freqprob import VectorizedScorer

scorer = freqprob.MLE(large_freqdist, logprob=False)
vectorized = VectorizedScorer(scorer)

# Score batch of elements efficiently
elements = ['word1', 'word2', 'word3', ...]
scores = vectorized.score_batch(elements)  # Returns numpy array

# Matrix operations
elements_2d = [['word1', 'word2'], ['word3', 'word4']]
score_matrix = vectorized.score_matrix(elements_2d)

# Top-k most probable elements
top_elements, top_scores = vectorized.top_k_elements(10)
```

**BatchScorer for Multiple Methods:**
```python
from freqprob import BatchScorer

scorers = {
    'mle': freqprob.MLE(freqdist),
    'laplace': freqprob.Laplace(freqdist),
    'kneser_ney': freqprob.KneserNey(bigram_freqdist)
}

batch_scorer = BatchScorer(scorers)
results = batch_scorer.score_batch(test_elements)
# Returns: {'mle': array([...]), 'laplace': array([...]), ...}
```

### Caching and Memoization

FreqProb automatically caches expensive computations to improve performance.

**Automatic Caching:**
```python
# First computation is cached
sgt1 = freqprob.SimpleGoodTuring(large_freqdist)  # Slow
sgt2 = freqprob.SimpleGoodTuring(large_freqdist)  # Fast (cached)

# Cache management
freqprob.clear_all_caches()
stats = freqprob.get_cache_stats()
print(f"Cache size: {stats['sgt_cache_size']} entries")
```

### Lazy Evaluation

Compute probabilities only when needed, reducing memory usage and computation time.

**Lazy Scoring:**
```python
from freqprob import create_lazy_mle

# Only computes probabilities for accessed elements
lazy_scorer = create_lazy_mle(huge_freqdist, logprob=False)

# First access triggers computation
prob1 = lazy_scorer('frequent_word')  # Computed

# Subsequent accesses use cached value
prob2 = lazy_scorer('frequent_word')  # Cached

# Check what's been computed
computed = lazy_scorer.get_computed_elements()
print(f"Computed {len(computed)} out of {len(huge_freqdist)} elements")
```

## Memory Management

### Streaming Updates

For real-time applications or very large datasets, use streaming frequency distributions.

**StreamingFrequencyDistribution:**
```python
from freqprob import StreamingFrequencyDistribution

# Bounded memory usage regardless of stream size
stream_dist = StreamingFrequencyDistribution(
    max_vocabulary_size=10000,
    min_count_threshold=2,
    decay_factor=0.99  # Exponential forgetting
)

# Process streaming data
for token in data_stream:
    stream_dist.update(token)

# Automatically maintains vocabulary size limit
print(f"Vocabulary size: {stream_dist.get_vocabulary_size()}")
```

**StreamingMLE:**
```python
from freqprob import StreamingMLE

# Incremental probability updates
streaming_mle = StreamingMLE(max_vocabulary_size=10000, logprob=False)

# Update with new observations
streaming_mle.update_single('word', 5)
streaming_mle.update_batch(['word1', 'word2', 'word1'])

# Get current probabilities
prob = streaming_mle('word')

# Save/load state
streaming_mle.save_state('model.pkl')
loaded_model = StreamingMLE.load_state('model.pkl')
```

### Memory-Efficient Representations

For large vocabularies, use compressed representations to reduce memory usage.

**Compressed Distributions:**
```python
from freqprob import create_compressed_distribution

# 50-90% memory savings
large_freqdist = {f'word_{i}': max(1, 10000-i) for i in range(100000)}

# With quantization for additional compression
compressed = create_compressed_distribution(
    large_freqdist,
    quantization_levels=1024,  # Trade-off: memory vs precision
    use_compression=True
)

# Check memory usage
memory_info = compressed.get_memory_usage()
print(f"Total memory: {memory_info['total'] / 1024 / 1024:.2f} MB")
```

**Sparse Distributions:**
```python
from freqprob import create_sparse_distribution

# Optimized for distributions with many zeros
sparse_freqdist = {'rare_word': 1, 'common_word': 10000}
sparse = create_sparse_distribution(sparse_freqdist)

# Efficient queries
top_10 = sparse.get_top_k(10)
mid_freq_words = sparse.get_elements_with_count_range(10, 100)
```

### Memory Profiling

Monitor and analyze memory usage patterns.

**MemoryProfiler:**
```python
from freqprob import MemoryProfiler

profiler = MemoryProfiler()

with profiler.profile_operation("model_creation"):
    model = freqprob.SimpleGoodTuring(large_freqdist)

metrics = profiler.get_latest_metrics()
print(f"Memory used: {metrics.memory_delta_mb:.2f} MB")
print(f"Execution time: {metrics.execution_time:.2f} seconds")
```

## Model Evaluation

### Built-in Metrics

FreqProb provides standard evaluation metrics for comparing different smoothing methods.

**Perplexity:**
```python
from freqprob import perplexity

model = freqprob.KneserNey(train_bigrams, logprob=True)
test_data = [('the', 'cat'), ('a', 'dog'), ('big', 'house')]

pp = perplexity(model, test_data)
print(f"Perplexity: {pp:.2f}")  # Lower is better
```

**Cross-Entropy:**
```python
from freqprob import cross_entropy

ce = cross_entropy(model, test_data)
print(f"Cross-entropy: {ce:.2f} bits")  # Lower is better
```

**Model Comparison:**
```python
from freqprob import model_comparison

models = {
    'mle': freqprob.MLE(train_data, logprob=True),
    'laplace': freqprob.Laplace(train_data, logprob=True),
    'kneser_ney': freqprob.KneserNey(train_bigrams, logprob=True)
}

comparison = model_comparison(models, test_data)
for model_name, metrics in comparison.items():
    print(f"{model_name}: PP={metrics['perplexity']:.2f}, "
          f"CE={metrics['cross_entropy']:.2f}")
```

**KL Divergence:**
```python
from freqprob import kl_divergence

# Compare two models
kl_div = kl_divergence(reference_model, test_model, test_data)
print(f"KL divergence: {kl_div:.4f}")  # Lower means more similar
```

### Custom Evaluation

**Hold-out Validation:**
```python
def evaluate_smoothing_method(method_class, train_data, test_data, **kwargs):
    """Evaluate a smoothing method on held-out data."""
    model = method_class(train_data, logprob=True, **kwargs)

    metrics = {
        'perplexity': perplexity(model, test_data),
        'cross_entropy': cross_entropy(model, test_data),
        'coverage': sum(1 for item in test_data if model(item) > -20) / len(test_data)
    }

    return metrics

# Compare different gamma values for Lidstone
gammas = [0.01, 0.1, 0.5, 1.0, 2.0]
results = {}

for gamma in gammas:
    results[gamma] = evaluate_smoothing_method(
        freqprob.Lidstone, train_data, test_data, gamma=gamma
    )

# Find best gamma
best_gamma = min(results.keys(), key=lambda g: results[g]['perplexity'])
print(f"Best gamma: {best_gamma} (PP: {results[best_gamma]['perplexity']:.2f})")
```

## Best Practices

### Choosing the Right Smoothing Method

**For Language Modeling:**
1. **Modified Kneser-Ney** - Best overall performance for n-gram models
2. **Kneser-Ney** - Good alternative when MKN is too complex
3. **Interpolated Smoothing** - For combining multiple model orders

**For General Frequency Estimation:**
1. **Laplace Smoothing** - Simple, robust baseline
2. **Lidstone with tuned γ** - When you can optimize parameters
3. **Simple Good-Turing** - When frequency-of-frequencies are reliable

**For Large-Scale Applications:**
1. **StreamingMLE/StreamingLaplace** - For real-time updates
2. **Compressed representations** - For memory-constrained environments
3. **Vectorized operations** - For batch processing

### Parameter Tuning

**Cross-Validation for Smoothing Parameters:**
```python
from sklearn.model_selection import ParameterGrid
import numpy as np

def tune_lidstone_gamma(train_data, val_data):
    """Find optimal gamma for Lidstone smoothing."""
    param_grid = {'gamma': np.logspace(-3, 1, 20)}  # 0.001 to 10

    best_gamma = None
    best_perplexity = float('inf')

    for params in ParameterGrid(param_grid):
        model = freqprob.Lidstone(train_data, logprob=True, **params)
        pp = perplexity(model, val_data)

        if pp < best_perplexity:
            best_perplexity = pp
            best_gamma = params['gamma']

    return best_gamma, best_perplexity

gamma, pp = tune_lidstone_gamma(train_freqdist, val_data)
print(f"Optimal gamma: {gamma:.4f} (Perplexity: {pp:.2f})")
```

### Memory Optimization

**For Large Vocabularies:**
```python
# Use compressed representations
compressed_dist = create_compressed_distribution(
    large_freqdist,
    quantization_levels=2048,  # Balance memory vs accuracy
    use_compression=True
)

# Monitor memory usage
from freqprob import MemoryProfiler
profiler = MemoryProfiler()

with profiler.profile_operation("model_training"):
    model = freqprob.MLE(compressed_dist.to_dict())

print(f"Memory usage: {profiler.get_latest_metrics().memory_delta_mb:.2f} MB")
```

**For Streaming Data:**
```python
# Configure streaming with appropriate limits
streaming_model = freqprob.StreamingMLE(
    max_vocabulary_size=50000,  # Limit vocabulary
    logprob=True
)

# Process data in batches for efficiency
batch_size = 1000
for i in range(0, len(data_stream), batch_size):
    batch = data_stream[i:i+batch_size]
    streaming_model.update_batch(batch)
```

### Performance Optimization

**Batch Processing:**
```python
# Instead of individual scoring
scores = [model(element) for element in large_list]  # Slow

# Use vectorized operations
vectorized = VectorizedScorer(model)
scores = vectorized.score_batch(large_list)  # Fast
```

**Lazy Evaluation for Sparse Access:**
```python
# When you only need a subset of probabilities
lazy_model = create_lazy_mle(huge_freqdist)

# Only compute what you need
important_words = ['the', 'of', 'and', 'to', 'a']
important_probs = [lazy_model(word) for word in important_words]

# Check computation efficiency
computed_ratio = len(lazy_model.get_computed_elements()) / len(huge_freqdist)
print(f"Computed only {computed_ratio:.1%} of probabilities")
```

## Common Use Cases

### Language Modeling

**N-gram Language Model with Kneser-Ney:**
```python
def build_ngram_model(text_corpus, n=3):
    """Build an n-gram language model with Kneser-Ney smoothing."""
    from freqprob import generate_ngrams, ngram_frequency, KneserNey

    # Generate n-grams
    ngrams = []
    for sentence in text_corpus:
        tokens = sentence.split()
        # Add sentence boundaries
        padded = ['<s>'] * (n-1) + tokens + ['</s>']
        ngrams.extend(generate_ngrams(padded, n))

    # Create frequency distribution
    ngram_freqdist = ngram_frequency(ngrams, n=1)  # Already n-grams

    # Train Kneser-Ney model
    model = KneserNey(ngram_freqdist, discount=0.75, logprob=True)

    return model

# Usage
corpus = ["the cat sat on the mat", "a dog ran in the park"]
model = build_ngram_model(corpus, n=3)

# Calculate sentence probability
def sentence_probability(model, sentence, n=3):
    tokens = sentence.split()
    padded = ['<s>'] * (n-1) + tokens + ['</s>']
    ngrams = generate_ngrams(padded, n)

    log_prob = sum(model(ngram) for ngram in ngrams)
    return math.exp(log_prob)

prob = sentence_probability(model, "the cat ran")
print(f"P(sentence) = {prob:.2e}")
```

### Text Classification Feature Extraction

**TF-IDF with Smoothed Probabilities:**
```python
def extract_smoothed_features(documents, smoothing_method=freqprob.Laplace):
    """Extract features with smoothed probability estimates."""
    from collections import Counter

    # Build vocabulary and document frequencies
    vocab = set()
    doc_freqs = []

    for doc in documents:
        words = doc.split()
        vocab.update(words)
        doc_freqs.append(Counter(words))

    # Create smoothed models for each document
    models = []
    for doc_freq in doc_freqs:
        model = smoothing_method(doc_freq, logprob=False)
        models.append(model)

    # Extract feature vectors
    features = []
    for model in models:
        feature_vector = [model(word) for word in sorted(vocab)]
        features.append(feature_vector)

    return features, sorted(vocab)

# Usage
docs = [
    "the cat is happy",
    "the dog is sad",
    "cats and dogs are pets"
]

features, vocabulary = extract_smoothed_features(docs)
print(f"Vocabulary size: {len(vocabulary)}")
print(f"Feature vectors: {len(features)} x {len(features[0])}")
```

### Information Retrieval

**Document Scoring with Language Models:**
```python
def build_document_language_model(document, background_model, lambda_mix=0.8):
    """Build a smoothed document language model."""
    from freqprob import word_frequency, InterpolatedSmoothing

    # Document term frequencies
    doc_freqdist = word_frequency(document.split())

    # Create document-specific and background models
    doc_model = freqprob.MLE(doc_freqdist, logprob=True)

    # Interpolate with background model
    interpolated = InterpolatedSmoothing(
        doc_freqdist, background_model._freqdist,
        lambda_weight=lambda_mix, logprob=True
    )

    return interpolated

def score_query(query, doc_models):
    """Score a query against document models."""
    query_terms = query.split()

    scores = []
    for doc_model in doc_models:
        # Query likelihood
        log_likelihood = sum(doc_model(term) for term in query_terms)
        scores.append(log_likelihood)

    return scores

# Usage
documents = [
    "machine learning algorithms for classification",
    "natural language processing with neural networks",
    "computer vision and image recognition"
]

# Build background model from all documents
all_text = " ".join(documents)
background_freqdist = word_frequency(all_text.split())
background_model = freqprob.MLE(background_freqdist, logprob=True)

# Build document models
doc_models = []
for doc in documents:
    model = build_document_language_model(doc, background_model)
    doc_models.append(model)

# Score query
query = "machine learning classification"
scores = score_query(query, doc_models)

# Rank documents
ranked_docs = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
for rank, (doc_idx, score) in enumerate(ranked_docs, 1):
    print(f"Rank {rank}: Document {doc_idx+1} (score: {score:.2f})")
```

### Real-time Text Processing

**Streaming Topic Detection:**
```python
class StreamingTopicDetector:
    """Real-time topic detection using streaming language models."""

    def __init__(self, topics, max_vocab_size=10000):
        self.topics = topics
        self.topic_models = {}

        # Initialize streaming models for each topic
        for topic in topics:
            self.topic_models[topic] = freqprob.StreamingMLE(
                max_vocabulary_size=max_vocab_size,
                logprob=True
            )

    def update_topic(self, topic, text):
        """Update topic model with new text."""
        words = text.split()
        self.topic_models[topic].update_batch(words)

    def classify_text(self, text):
        """Classify text into most likely topic."""
        words = text.split()

        topic_scores = {}
        for topic, model in self.topic_models.items():
            # Calculate log-likelihood
            score = sum(model(word) for word in words)
            topic_scores[topic] = score

        # Return most likely topic
        best_topic = max(topic_scores, key=topic_scores.get)
        return best_topic, topic_scores[best_topic]

    def get_topic_statistics(self):
        """Get statistics for all topic models."""
        stats = {}
        for topic, model in self.topic_models.items():
            stats[topic] = model.get_streaming_statistics()
        return stats

# Usage
detector = StreamingTopicDetector(['sports', 'technology', 'politics'])

# Train with streaming data
training_data = [
    ('sports', 'football soccer basketball game score'),
    ('technology', 'computer software algorithm programming'),
    ('politics', 'government election policy vote democracy')
]

for topic, text in training_data:
    detector.update_topic(topic, text)

# Classify new text
new_text = "the basketball game had a high score"
predicted_topic, confidence = detector.classify_text(new_text)
print(f"Predicted topic: {predicted_topic} (confidence: {confidence:.2f})")
```

This comprehensive user guide provides both theoretical understanding and practical implementation guidance for using FreqProb effectively across a wide range of applications.
