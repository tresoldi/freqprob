# FreqProb: LLM Agent Documentation

## Overview

FreqProb is a Python library for probability smoothing and frequency-based language modeling. It converts raw frequency counts into probability estimates using mathematically rigorous smoothing techniques. This documentation is designed for LLM coding agents to integrate FreqProb into their projects.

**Key Use Cases:**
- Natural language processing and text classification
- N-gram language modeling
- Information retrieval and ranking
- Statistical text analysis
- Zero-shot learning with frequency-based priors

**Installation:**
```bash
pip install freqprob
```

**Optional dependencies:**
```bash
pip install freqprob[memory]      # Memory profiling with psutil
pip install freqprob[validation]  # Testing with hypothesis, nltk, scikit-learn
pip install freqprob[all]         # All optional features
```

---

## Quick Start

### Basic Usage

```python
import freqprob
from collections import Counter

# Count word frequencies in your data
text = "the cat sat on the mat the cat slept"
word_counts = Counter(text.split())
# Result: {'the': 3, 'cat': 2, 'sat': 1, 'on': 1, 'mat': 1, 'slept': 1}

# Create a smoothed probability model
model = freqprob.Laplace(word_counts, bins=1000, logprob=False)

# Get probabilities for words
prob_the = model("the")      # High probability (frequent word)
prob_dog = model("dog")      # Low probability (unseen word)

# Use log-probabilities for numerical stability
log_model = freqprob.Laplace(word_counts, bins=1000, logprob=True)
log_prob = log_model("the")  # Returns log(P(the))
```

### Model Evaluation

```python
# Evaluate model quality with perplexity
test_data = ["the", "cat", "sat", "the", "dog"]
perplexity = freqprob.perplexity(model, test_data)
# Lower perplexity = better model

# Compare multiple models
models = {
    "laplace": freqprob.Laplace(word_counts, bins=1000, logprob=True),
    "ele": freqprob.ELE(word_counts, bins=1000, logprob=True),
}
comparison = freqprob.model_comparison(models, test_data)
# Returns: {"laplace": {"perplexity": 15.2, ...}, "ele": {...}}
```

---

## Core Concepts

### Type System

FreqProb uses strict typing for clarity and safety:

```python
from freqprob.base import Element, FrequencyDistribution, Probability, LogProbability

# Element: Any hashable type (str, int, tuple, frozenset)
word: Element = "cat"
bigram: Element = ("the", "cat")
ngram: Element = ("the", "quick", "brown", "fox")

# FrequencyDistribution: Mapping from elements to counts
freq_dist: FrequencyDistribution = {"cat": 5, "dog": 3, "bird": 1}

# Probability: Regular probability (0.0 to 1.0)
prob: Probability = 0.25

# LogProbability: Natural logarithm of probability
log_prob: LogProbability = -1.386  # log(0.25)
```

### Common Parameters

All smoothing methods share these parameters:

```python
model = freqprob.SomeMethod(
    freqdist,           # Required: frequency distribution (dict-like)
    unobs_prob=0.01,    # Optional: probability mass for unseen elements
    logprob=True,       # Optional: return log-probabilities (default: True)
    bins=10000,         # Optional: total vocabulary size (method-specific)
)
```

**Parameter Guidelines:**
- `freqdist`: Dictionary mapping elements to counts (must be positive integers)
- `unobs_prob`: Reserve probability mass for unseen words (0.0 ≤ p ≤ 1.0)
- `logprob`: Use `True` for numerical stability in language modeling
- `bins`: Total vocabulary size including unseen words

---

## Smoothing Methods

### 1. Maximum Likelihood Estimation (MLE)

**Formula:** `P(w) = c(w) / N`

**When to use:**
- Baseline model for comparison
- When you have dense, complete data
- Not recommended for sparse data (assigns zero to unseen words)

```python
import freqprob

counts = {"apple": 10, "banana": 5, "orange": 3}

# Basic MLE
mle = freqprob.MLE(counts, logprob=False)
print(mle("apple"))    # 10/18 = 0.556
print(mle("grape"))    # 0.0 (unseen word)

# MLE with reserved mass for unseen words
mle_smooth = freqprob.MLE(counts, unobs_prob=0.1, logprob=False)
print(mle_smooth("grape"))  # 0.1
```

### 2. Laplace Smoothing (Add-1)

**Formula:** `P(w) = (c(w) + 1) / (N + B)`

**When to use:**
- Simple baseline for sparse data
- Small vocabularies (< 10,000 words)
- Quick prototyping

**Limitations:** Over-smooths for large vocabularies

```python
# Laplace smoothing
counts = {"the": 100, "cat": 50, "sat": 20}
vocab_size = 10000  # Estimated total vocabulary

laplace = freqprob.Laplace(counts, bins=vocab_size, logprob=False)

# All words get non-zero probability
print(laplace("the"))        # (100+1) / (170+10000)
print(laplace("unseen"))     # 1 / (170+10000)
```

### 3. Lidstone Smoothing (Add-k)

**Formula:** `P(w) = (c(w) + γ) / (N + B×γ)`

**When to use:**
- Tune smoothing strength with gamma parameter
- More flexible than Laplace
- Good for medium-sized vocabularies

```python
counts = {"word1": 100, "word2": 50, "word3": 20}

# Conservative smoothing (γ = 0.1)
lidstone_light = freqprob.Lidstone(counts, gamma=0.1, bins=5000, logprob=False)

# Aggressive smoothing (γ = 2.0)
lidstone_heavy = freqprob.Lidstone(counts, gamma=2.0, bins=5000, logprob=False)

# Compare smoothing effects
print(f"Light: {lidstone_light('unseen')}")  # Smaller probability
print(f"Heavy: {lidstone_heavy('unseen')}")  # Larger probability
```

### 4. Expected Likelihood Estimation (ELE)

**Formula:** `P(w) = (c(w) + 0.5) / (N + 0.5×B)`

**When to use:**
- Theoretically-justified alternative to Laplace (γ = 0.5)
- Good default for sparse data
- Better than Laplace for large vocabularies

```python
counts = {"token1": 80, "token2": 40, "token3": 15}

# ELE is Lidstone with γ=0.5
ele = freqprob.ELE(counts, bins=5000, logprob=True)

# Equivalent to:
lidstone = freqprob.Lidstone(counts, gamma=0.5, bins=5000, logprob=True)
```

### 5. Bayesian Smoothing

**Formula:** `P(w) = (c(w) + α) / (N + α×V)` (Dirichlet prior)

**When to use:**
- Incorporate prior knowledge via alpha parameter
- Principled probabilistic framework
- Good for small datasets

```python
counts = {"positive": 20, "negative": 5, "neutral": 2}

# Weak prior (α = 0.5)
weak_prior = freqprob.BayesianSmoothing(counts, alpha=0.5, logprob=False)

# Strong prior (α = 10.0) - assumes uniform distribution
strong_prior = freqprob.BayesianSmoothing(counts, alpha=10.0, logprob=False)

# Strong prior pulls probabilities toward uniform
print(weak_prior("positive"))    # Closer to MLE
print(strong_prior("positive"))  # Closer to 1/3
```

### 6. Simple Good-Turing

**When to use:**
- Large, sparse datasets
- You need accurate estimates for low-frequency items
- Gold standard for unseen word probability

**Requirements:** Needs diverse frequency-of-frequency distribution

```python
import freqprob

# Good-Turing works best with large, diverse datasets
large_counts = {f"word_{i}": max(1, 1000 - i*2) for i in range(500)}

try:
    sgt = freqprob.SimpleGoodTuring(large_counts, logprob=True)

    # Good-Turing excels at low-frequency and unseen words
    prob_rare = sgt("word_400")      # Rare word
    prob_unseen = sgt("new_word")    # Unseen word

except ValueError as e:
    # May fail if frequency distribution is too uniform
    print(f"Good-Turing failed: {e}")
    # Fallback to ELE or Bayesian smoothing
```

### 7. Kneser-Ney Smoothing

**When to use:**
- N-gram language models (bigrams, trigrams)
- State-of-the-art for text modeling
- You need continuation probability (not just frequency)

**Key insight:** Estimates how likely a word appears in novel contexts

```python
# Kneser-Ney for bigram language model
bigram_counts = {
    ("the", "cat"): 10,
    ("the", "dog"): 8,
    ("a", "cat"): 5,
    ("a", "dog"): 4,
    ("big", "cat"): 2,
}

kn = freqprob.KneserNey(bigram_counts, discount=0.75, logprob=True)

# P(dog | the)
prob = kn(("the", "dog"))
```

**Modified Kneser-Ney:** More sophisticated variant

```python
# Automatically selects optimal discount parameters
mkn = freqprob.ModifiedKneserNey(bigram_counts, logprob=True)
```

### 8. Interpolated Smoothing

**When to use:**
- Combine predictions from multiple models
- Balance different smoothing strengths

```python
# Create base models
mle_model = freqprob.MLE(counts, logprob=True)
laplace_model = freqprob.Laplace(counts, bins=5000, logprob=True)

# Interpolate with custom weights
interpolated = freqprob.InterpolatedSmoothing(
    [mle_model, laplace_model],
    weights=[0.7, 0.3],  # 70% MLE, 30% Laplace
    logprob=True
)

prob = interpolated("word")
```

### 9. Witten-Bell Smoothing

**When to use:**
- Text compression applications
- Variable-length vocabularies

```python
wb = freqprob.WittenBell(counts, logprob=True)
```

### 10. Certainty Degree Smoothing

**When to use:**
- Adjust smoothing based on data confidence

```python
cd = freqprob.CertaintyDegree(counts, certainty=0.8, logprob=True)
```

---

## Practical Workflows

### Building a Text Classifier

```python
import freqprob
from collections import Counter

# Training data: list of (text, label) tuples
training_data = [
    ("this movie is great amazing wonderful", "positive"),
    ("loved it best film ever", "positive"),
    ("terrible worst movie ever", "negative"),
    ("hated it boring bad", "negative"),
]

# Build class-specific models
class_models = {}
all_words = set()

for label in ["positive", "negative"]:
    # Collect words for this class
    class_words = []
    for text, doc_label in training_data:
        if doc_label == label:
            class_words.extend(text.split())
            all_words.update(text.split())

    # Count and smooth
    word_counts = Counter(class_words)
    vocab_size = len(all_words)

    class_models[label] = freqprob.Laplace(
        word_counts, bins=vocab_size, logprob=True
    )

# Classify new text
def classify(text):
    words = text.split()
    scores = {}

    for label, model in class_models.items():
        # Sum log probabilities (Naive Bayes)
        score = sum(model(word) for word in words)
        scores[label] = score

    # Return highest scoring class
    return max(scores.items(), key=lambda x: x[1])[0]

# Test
result = classify("great wonderful movie")
print(f"Predicted class: {result}")  # "positive"
```

### N-gram Language Model

```python
import freqprob

# Generate bigrams from text
def generate_bigrams(text):
    words = text.split()
    return list(zip(words[:-1], words[1:]))

# Training
corpus = [
    "the cat sat on the mat",
    "the dog sat on the floor",
    "the cat slept on the bed",
]

all_bigrams = []
for sentence in corpus:
    all_bigrams.extend(generate_bigrams(sentence))

bigram_counts = Counter(all_bigrams)

# Build Kneser-Ney model
lm = freqprob.KneserNey(bigram_counts, discount=0.75, logprob=True)

# Compute sentence probability
def sentence_probability(sentence, model):
    bigrams = generate_bigrams(sentence)
    log_prob = sum(model(bg) for bg in bigrams)
    return log_prob

# Evaluate
test_sentence = "the cat sat"
prob = sentence_probability(test_sentence, lm)
print(f"Log probability: {prob}")

# Compare sentences
sentence1 = "the cat sat"
sentence2 = "sat cat the"  # Unnatural order
print(f"Natural: {sentence_probability(sentence1, lm)}")
print(f"Unnatural: {sentence_probability(sentence2, lm)}")
```

### Word Recommendation System

```python
import freqprob
from collections import Counter

# User interaction history
user_clicks = [
    "electronics", "phone", "laptop", "phone", "tablet",
    "electronics", "laptop", "laptop", "headphones", "phone"
]

# Build frequency distribution
click_counts = Counter(user_clicks)

# Create smoothed model (handles unseen categories)
model = freqprob.ELE(click_counts, bins=100, logprob=False)

# Rank categories by probability
categories = ["phone", "laptop", "tablet", "camera", "speaker"]
ranked = sorted(
    categories,
    key=lambda cat: model(cat),
    reverse=True
)

print("Recommended categories:", ranked)
# Output: ['phone', 'laptop', 'tablet', 'speaker', 'camera']
```

---

## Performance Optimization

### Batch Processing

For scoring many elements efficiently:

```python
import freqprob

counts = {f"word_{i}": i for i in range(1, 1000)}
model = freqprob.Laplace(counts, bins=5000, logprob=True)

# Create vectorized scorer for batch operations
vectorized = freqprob.VectorizedScorer(model)

# Score many words at once (returns NumPy array)
words = [f"word_{i}" for i in range(1, 100)]
probabilities = vectorized.score_batch(words)

# Much faster than:
# probabilities = [model(word) for word in words]
```

### Multi-Model Batch Scoring

```python
# Compare multiple models on same data
models = {
    "mle": freqprob.MLE(counts, logprob=True),
    "laplace": freqprob.Laplace(counts, bins=5000, logprob=True),
    "ele": freqprob.ELE(counts, bins=5000, logprob=True),
}

batch_scorer = freqprob.create_vectorized_batch_scorer(models)
results = batch_scorer.score_batch(words)

# results = {"mle": array([...]), "laplace": array([...]), "ele": array([...])}
```

### Streaming for Large Datasets

When data doesn't fit in memory:

```python
# Start with small initial data
initial_counts = {"word1": 10, "word2": 5}

streaming_model = freqprob.StreamingMLE(
    initial_data=initial_counts,
    max_vocabulary_size=10000,  # Limit memory usage
    logprob=True
)

# Update incrementally as new data arrives
streaming_model.update_single("word3", 8)
streaming_model.update_batch(["word1", "word4"], [3, 12])

# Use model immediately
prob = streaming_model("word1")
```

### Memory-Efficient Representations

For very large vocabularies:

```python
import freqprob

# Original: large frequency distribution
huge_counts = {f"word_{i}": i for i in range(100000)}

# Compress to reduce memory (quantizes counts)
compressed = freqprob.create_compressed_distribution(
    huge_counts,
    quantization_levels=256  # 8-bit quantization
)

# Use compressed distribution
model = freqprob.Laplace(compressed.to_dict(), bins=500000, logprob=True)

# Sparse representation (only stores non-zero counts)
sparse = freqprob.create_sparse_distribution(huge_counts)
```

### Lazy Evaluation

Defer computation until needed:

```python
# Create lazy model (doesn't compute all probabilities upfront)
lazy_model = freqprob.create_lazy_mle(large_counts)

# Probabilities computed on-demand
prob1 = lazy_model("word1")  # Computed now
prob2 = lazy_model("word1")  # Retrieved from cache

# Lazy batch scoring
lazy_scorer = freqprob.LazyBatchScorer(lazy_model)
batch_probs = lazy_scorer.score_batch(words)
```

---

## Evaluation and Validation

### Perplexity

Measure how well model predicts test data (lower is better):

```python
import freqprob

# Train on training set
train_counts = Counter(["the", "cat", "sat"] * 10)
model = freqprob.Laplace(train_counts, bins=1000, logprob=True)

# Evaluate on test set
test_data = ["the", "cat", "sat", "on", "mat"]
perplexity = freqprob.perplexity(model, test_data)

print(f"Perplexity: {perplexity:.2f}")
# Lower perplexity = better predictions
```

### Cross-Entropy

```python
model1 = freqprob.MLE(counts, logprob=True)
model2 = freqprob.Laplace(counts, bins=1000, logprob=True)

ce = freqprob.cross_entropy(model1, model2, test_data)
```

### KL Divergence

```python
vocabulary = list(counts.keys())
kl_div = freqprob.kl_divergence(model1, model2, vocabulary)
```

### Model Comparison

```python
models = {
    "mle": freqprob.MLE(train_counts, logprob=True),
    "laplace": freqprob.Laplace(train_counts, bins=1000, logprob=True),
    "ele": freqprob.ELE(train_counts, bins=1000, logprob=True),
    "bayesian": freqprob.BayesianSmoothing(train_counts, alpha=1.0, logprob=True),
}

comparison = freqprob.model_comparison(models, test_data)

# Find best model
best_model_name = min(comparison.items(), key=lambda x: x[1]["perplexity"])[0]
print(f"Best model: {best_model_name}")
print(f"Perplexity: {comparison[best_model_name]['perplexity']:.2f}")
```

---

## Advanced Features

### Caching

FreqProb automatically caches expensive computations:

```python
# SimpleGoodTuring automatically caches results
sgt = freqprob.SimpleGoodTuring(large_counts, logprob=True)

# First call: computed and cached
prob1 = sgt("word")

# Second call: retrieved from cache (fast)
prob2 = sgt("word")

# Check cache statistics
stats = freqprob.get_cache_stats()
print(f"Cache hits: {stats['hits']}")
print(f"Cache misses: {stats['misses']}")
print(f"Hit rate: {stats['hit_rate']:.2%}")

# Clear all caches if needed
freqprob.clear_all_caches()
```

### Memory Profiling

Analyze memory usage:

```python
import freqprob

# Profile memory usage during model creation
profiler = freqprob.MemoryProfiler()

metrics = profiler.profile_method_creation(
    freqprob.SimpleGoodTuring,
    large_counts,
    iterations=5
)

print(f"Peak memory: {metrics.peak_memory_mb:.1f} MB")
print(f"Average time: {metrics.avg_time:.3f} seconds")

# Monitor memory during operations
monitor = freqprob.MemoryMonitor()

with monitor.profile_operation("batch_scoring"):
    vectorized = freqprob.VectorizedScorer(model)
    results = vectorized.score_batch(large_word_list)

latest = monitor.get_latest_metrics()
print(f"Operation memory: {latest.peak_memory_mb:.1f} MB")
```

### Validation (Optional)

Requires `pip install freqprob[validation]`:

```python
import freqprob

# Quick validation check
is_valid = freqprob.quick_validate_method(
    freqprob.Laplace,
    test_distribution
)

# Comprehensive validation
validator = freqprob.ValidationSuite()
result = validator.validate_method(
    freqprob.ELE,
    test_distributions=[small_dist, medium_dist, large_dist]
)

# Performance comparison
comparison = freqprob.compare_method_performance(
    methods=[freqprob.MLE, freqprob.Laplace, freqprob.KneserNey],
    distributions=[test_dist],
    test_data=validation_data
)
```

---

## Utility Functions

### N-gram Generation

```python
import freqprob

text = "the quick brown fox jumps"
tokens = text.split()

# Generate bigrams
bigrams = freqprob.generate_ngrams(tokens, n=2)
# [("the", "quick"), ("quick", "brown"), ("brown", "fox"), ("fox", "jumps")]

# Generate trigrams
trigrams = freqprob.generate_ngrams(tokens, n=3)
# [("the", "quick", "brown"), ("quick", "brown", "fox"), ...]

# Count n-gram frequencies
from collections import Counter
bigram_counts = Counter(bigrams)
```

### Frequency Counting

```python
import freqprob

tokens = ["the", "cat", "sat", "on", "the", "mat"]

# Word frequency
word_freq = freqprob.word_frequency(tokens)
# {"the": 2, "cat": 1, "sat": 1, "on": 1, "mat": 1}

# N-gram frequency
ngrams = freqprob.generate_ngrams(tokens, n=2)
ngram_freq = freqprob.ngram_frequency(ngrams, n=1)
```

---

## Common Patterns

### Pattern 1: Model Selection with Cross-Validation

```python
import freqprob
from collections import Counter

def select_best_model(train_data, validation_data, vocab_size):
    """Select best smoothing method via validation perplexity."""

    train_counts = Counter(train_data)

    # Try different methods
    models = {
        "laplace": freqprob.Laplace(train_counts, bins=vocab_size, logprob=True),
        "ele": freqprob.ELE(train_counts, bins=vocab_size, logprob=True),
        "bayesian_0.5": freqprob.BayesianSmoothing(train_counts, alpha=0.5, logprob=True),
        "bayesian_1.0": freqprob.BayesianSmoothing(train_counts, alpha=1.0, logprob=True),
    }

    # Evaluate on validation set
    results = {}
    for name, model in models.items():
        ppl = freqprob.perplexity(model, validation_data)
        results[name] = ppl

    # Return best model
    best_name = min(results.items(), key=lambda x: x[1])[0]
    return models[best_name], results

# Usage
train = ["word1", "word2", "word1"] * 100
validation = ["word1", "word3", "word2"] * 20

best_model, scores = select_best_model(train, validation, vocab_size=1000)
print(f"Best model perplexity: {min(scores.values()):.2f}")
```

### Pattern 2: Hierarchical Smoothing

```python
import freqprob

# Use different smoothing for different frequency ranges
def hierarchical_smoothing(counts, vocab_size):
    """Use Good-Turing for rare words, MLE for frequent words."""

    # Separate frequent and rare words
    threshold = 10
    frequent = {w: c for w, c in counts.items() if c >= threshold}
    rare = {w: c for w, c in counts.items() if c < threshold}

    # Different models for different frequencies
    frequent_model = freqprob.MLE(frequent, logprob=True)
    rare_model = freqprob.SimpleGoodTuring(rare, logprob=True)

    # Wrapper to route to appropriate model
    class HierarchicalModel:
        def __call__(self, word):
            if counts.get(word, 0) >= threshold:
                return frequent_model(word)
            else:
                return rare_model(word)

    return HierarchicalModel()
```

### Pattern 3: Domain Adaptation

```python
import freqprob

def adapt_model(general_counts, domain_counts, domain_weight=0.7):
    """Interpolate general and domain-specific models."""

    # Train on both corpora
    general_model = freqprob.ELE(general_counts, bins=10000, logprob=True)
    domain_model = freqprob.ELE(domain_counts, bins=5000, logprob=True)

    # Interpolate with domain preference
    adapted = freqprob.InterpolatedSmoothing(
        [domain_model, general_model],
        weights=[domain_weight, 1 - domain_weight],
        logprob=True
    )

    return adapted

# Usage
general_text = ["common", "word", "the"] * 1000
domain_text = ["technical", "jargon", "specific"] * 100

general_counts = Counter(general_text)
domain_counts = Counter(domain_text)

model = adapt_model(general_counts, domain_counts, domain_weight=0.8)
```

---

## Error Handling

### Common Issues and Solutions

```python
import freqprob

# Issue 1: Empty frequency distribution
try:
    model = freqprob.MLE({}, logprob=True)
except ValueError as e:
    print(f"Error: {e}")
    # Solution: Ensure non-empty distribution

# Issue 2: Invalid counts (must be positive integers)
try:
    model = freqprob.Laplace({"word": -5}, bins=100, logprob=True)
except ValueError as e:
    print(f"Error: {e}")
    # Solution: Ensure all counts are positive

# Issue 3: SimpleGoodTuring fails on uniform distributions
try:
    uniform_counts = {f"word_{i}": 10 for i in range(100)}
    sgt = freqprob.SimpleGoodTuring(uniform_counts, logprob=True)
except ValueError as e:
    print(f"Good-Turing failed: {e}")
    # Solution: Use ELE or Bayesian smoothing instead
    fallback = freqprob.ELE(uniform_counts, bins=1000, logprob=True)

# Issue 4: Numerical underflow with regular probabilities
counts = {f"word_{i}": i for i in range(1, 1000)}
model_regular = freqprob.Laplace(counts, bins=10000, logprob=False)

# Computing product of many small probabilities
test_words = [f"word_{i}" for i in range(1, 100)]
product = 1.0
for word in test_words:
    product *= model_regular(word)  # May underflow to 0.0!

# Solution: Use log-probabilities
model_log = freqprob.Laplace(counts, bins=10000, logprob=True)
log_sum = sum(model_log(word) for word in test_words)
# Then: actual_probability = exp(log_sum)
```

---

## Best Practices

### 1. Choose Appropriate Smoothing

| Data Characteristics | Recommended Method |
|---------------------|-------------------|
| Dense, complete data | MLE or Bayesian (small α) |
| Sparse, small vocabulary | Laplace or ELE |
| Sparse, large vocabulary | ELE or Bayesian |
| Very large, sparse | SimpleGoodTuring |
| N-gram models | KneserNey or ModifiedKneserNey |
| Need theoretical justification | Bayesian with domain-appropriate α |

### 2. Use Log-Probabilities

```python
# Always use logprob=True for language modeling
model = freqprob.Laplace(counts, bins=5000, logprob=True)

# Combine log-probabilities with addition
log_p1 = model("word1")
log_p2 = model("word2")
log_product = log_p1 + log_p2  # log(P1 × P2)

# Convert back to probability if needed
import math
probability = math.exp(log_product)
```

### 3. Set Vocabulary Size Appropriately

```python
# Count actual vocabulary
actual_vocab = set(all_training_words)

# Estimate unseen words (rule of thumb: 20-50% more)
estimated_total = len(actual_vocab) * 1.3

model = freqprob.Laplace(counts, bins=int(estimated_total), logprob=True)
```

### 4. Validate on Held-Out Data

```python
# Split data: 80% train, 20% validation
from random import shuffle

all_data = list(word_sequence)
shuffle(all_data)

split_point = int(len(all_data) * 0.8)
train_data = all_data[:split_point]
val_data = all_data[split_point:]

# Train and validate
train_counts = Counter(train_data)
model = freqprob.ELE(train_counts, bins=5000, logprob=True)

perplexity = freqprob.perplexity(model, val_data)
print(f"Validation perplexity: {perplexity:.2f}")
```

### 5. Profile Before Optimizing

```python
# Start simple
simple_model = freqprob.Laplace(counts, bins=5000, logprob=True)

# Profile if performance is an issue
profiler = freqprob.MemoryProfiler()
metrics = profiler.profile_method_creation(
    freqprob.Laplace, counts, iterations=10
)

# Only optimize if needed
if metrics.avg_time > 1.0:  # > 1 second
    # Try vectorized or streaming approach
    vectorized = freqprob.VectorizedScorer(simple_model)
```

---

## Mathematical Reference

### Probability Axioms Satisfied

All FreqProb methods satisfy:

1. **Non-negativity:** `P(w) ≥ 0` for all w
2. **Normalization:** `Σ P(w) = 1` (within numerical precision)
3. **Monotonicity:** Higher counts → higher probabilities (for simple methods)

### Log-Probability Arithmetic

```python
import math

# Multiplication → Addition
log_p1 = model("word1")
log_p2 = model("word2")
log_product = log_p1 + log_p2

# Division → Subtraction
log_ratio = log_p1 - log_p2

# Power → Multiplication
log_power = log_p1 * n

# Sum of probabilities (use logsumexp for stability)
from scipy.special import logsumexp
log_probs = [model(w) for w in words]
log_sum = logsumexp(log_probs)
```

---

## Package Information

**Version:** 0.3.1
**Python:** Requires 3.10+
**Core dependencies:** numpy, scipy
**License:** MIT
**Repository:** https://github.com/tresoldi/freqprob

### Import Structure

```python
# Main classes
from freqprob import (
    MLE, Laplace, Lidstone, ELE,
    BayesianSmoothing, SimpleGoodTuring,
    KneserNey, ModifiedKneserNey,
    WittenBell, CertaintyDegree,
    InterpolatedSmoothing,
)

# Utilities
from freqprob import (
    perplexity, cross_entropy, kl_divergence,
    model_comparison,
    generate_ngrams, ngram_frequency, word_frequency,
)

# Performance
from freqprob import (
    VectorizedScorer, BatchScorer,
    create_vectorized_batch_scorer,
    StreamingMLE, StreamingLaplace,
    LazyBatchScorer,
)

# Memory optimization
from freqprob import (
    create_compressed_distribution,
    create_sparse_distribution,
)

# Caching
from freqprob import (
    get_cache_stats, clear_all_caches,
)

# Profiling (optional)
from freqprob import (
    MemoryMonitor, MemoryProfiler,
    DistributionMemoryAnalyzer,
)

# Validation (requires freqprob[validation])
from freqprob import (
    quick_validate_method,
    compare_method_performance,
    ValidationSuite,
)
```

---

## Complete Example: Text Categorization Pipeline

```python
import freqprob
from collections import Counter
from typing import List, Tuple

class TextCategorizer:
    """Multi-class text categorization using smoothed language models."""

    def __init__(self, smoothing="ele", vocab_scale=1.3):
        self.smoothing = smoothing
        self.vocab_scale = vocab_scale
        self.models = {}
        self.vocab_size = 0

    def train(self, documents: List[Tuple[str, str]]):
        """Train on labeled documents.

        Args:
            documents: List of (text, category) tuples
        """
        # Organize by category
        category_data = {}
        all_words = set()

        for text, category in documents:
            words = text.lower().split()
            all_words.update(words)

            if category not in category_data:
                category_data[category] = []
            category_data[category].extend(words)

        # Estimate vocabulary size
        self.vocab_size = int(len(all_words) * self.vocab_scale)

        # Build model for each category
        for category, words in category_data.items():
            counts = Counter(words)

            if self.smoothing == "laplace":
                model = freqprob.Laplace(
                    counts, bins=self.vocab_size, logprob=True
                )
            elif self.smoothing == "ele":
                model = freqprob.ELE(
                    counts, bins=self.vocab_size, logprob=True
                )
            elif self.smoothing == "bayesian":
                model = freqprob.BayesianSmoothing(
                    counts, alpha=1.0, logprob=True
                )
            else:
                raise ValueError(f"Unknown smoothing: {self.smoothing}")

            self.models[category] = model

    def predict(self, text: str) -> Tuple[str, float]:
        """Predict category for text.

        Returns:
            (category, log_probability)
        """
        words = text.lower().split()

        scores = {}
        for category, model in self.models.items():
            # Sum log-probabilities
            log_prob = sum(model(word) for word in words)
            scores[category] = log_prob

        # Return highest scoring category
        best_category = max(scores.items(), key=lambda x: x[1])
        return best_category

    def evaluate(self, test_documents: List[Tuple[str, str]]) -> float:
        """Compute accuracy on test set."""
        correct = 0

        for text, true_category in test_documents:
            pred_category, _ = self.predict(text)
            if pred_category == true_category:
                correct += 1

        return correct / len(test_documents)


# Usage example
if __name__ == "__main__":
    # Training data
    train_data = [
        ("python programming language code", "tech"),
        ("machine learning algorithms data", "tech"),
        ("software development debugging", "tech"),
        ("football soccer match goal", "sports"),
        ("basketball game player score", "sports"),
        ("tennis tournament championship", "sports"),
        ("stock market trading investment", "finance"),
        ("banking credit loan mortgage", "finance"),
        ("economy inflation interest rates", "finance"),
    ]

    # Test data
    test_data = [
        ("java programming variables", "tech"),
        ("baseball pitcher homerun", "sports"),
        ("cryptocurrency blockchain", "finance"),
    ]

    # Train categorizer
    categorizer = TextCategorizer(smoothing="ele")
    categorizer.train(train_data)

    # Evaluate
    accuracy = categorizer.evaluate(test_data)
    print(f"Accuracy: {accuracy:.1%}")

    # Predict new text
    new_text = "deep learning neural networks"
    category, score = categorizer.predict(new_text)
    print(f"Text: '{new_text}'")
    print(f"Category: {category}")
    print(f"Log-probability: {score:.2f}")
```

---

This documentation provides comprehensive guidance for LLM agents to effectively use FreqProb in their projects. For the latest updates and examples, visit the [GitHub repository](https://github.com/tresoldi/freqprob).
