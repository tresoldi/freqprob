#' # FreqProb Tutorial 3: Computational Efficiency and Memory Management
#' 
#' This tutorial demonstrates the efficiency and memory management features of FreqProb for large-scale applications:
#' 
#' 1. **Vectorized Operations** - Batch processing for performance
#' 2. **Caching and Memoization** - Avoiding redundant computations
#' 3. **Lazy Evaluation** - Computing only what's needed
#' 4. **Streaming Updates** - Real-time incremental learning
#' 5. **Memory-Efficient Representations** - Handling large vocabularies
#' 6. **Memory Profiling** - Monitoring and optimization
#' 
#' ## Setup

#| hide
import gc
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import psutil
import seaborn as sns

import freqprob

# Import efficiency features
from freqprob import (
    BatchScorer,
    DistributionMemoryAnalyzer,
    MemoryProfiler,
    StreamingLaplace,
    StreamingMLE,
    VectorizedScorer,
    create_compressed_distribution,
    create_lazy_mle,
    create_sparse_distribution,
)

# Set up plotting
plt.style.use("default")
sns.set_palette("husl")
random.seed(42)
np.random.seed(42)

print("FreqProb Efficiency and Memory Management Tutorial")
print("=" * 50)
print(f"NumPy version: {np.__version__}")
print(f"Current memory usage: {psutil.Process().memory_info().rss / 1024 / 1024:.1f} MB")
#|

#' ## Creating Large-Scale Data
#' 
#' First, let's create realistic large-scale datasets to demonstrate efficiency features.

def create_realistic_frequency_distribution(vocab_size, total_count, zipf_exponent=1.2):
    """Create a realistic frequency distribution following Zipf's law."""
    # Generate Zipfian frequencies
    frequencies = np.random.zipf(zipf_exponent, vocab_size)

    # Normalize to desired total count
    frequencies = (frequencies / frequencies.sum()) * total_count
    frequencies = frequencies.astype(int)
    frequencies[frequencies == 0] = 1  # Ensure no zero counts

    # Create word-like keys
    words = [f"word_{i:06d}" for i in range(vocab_size)]

    return dict(zip(words, frequencies, strict=False))


# Create datasets of different sizes
print("Creating realistic datasets...")
datasets = {
    "small": create_realistic_frequency_distribution(1000, 10000),
    "medium": create_realistic_frequency_distribution(10000, 100000),
    "large": create_realistic_frequency_distribution(50000, 500000),
}

for name, dataset in datasets.items():
    print(
        f"{name.capitalize()} dataset: {len(dataset):,} words, {sum(dataset.values()):,} total count"
    )

    # Show frequency distribution characteristics
    counts = list(dataset.values())
    print(f"  Frequency range: {min(counts)} - {max(counts):,}")
    print(
        f"  Top 10 account for {sum(sorted(counts, reverse=True)[:10]) / sum(counts) * 100:.1f}% of data"
    )
    print()

# Visualize the Zipfian distribution
plt.figure(figsize=(12, 4))

for i, (name, dataset) in enumerate(datasets.items()):
    plt.subplot(1, 3, i + 1)
    counts = sorted(dataset.values(), reverse=True)
    ranks = np.arange(1, len(counts) + 1)

    plt.loglog(ranks, counts, "o-", alpha=0.7, markersize=2)
    plt.title(f"{name.capitalize()} Dataset\n(Zipf Distribution)")
    plt.xlabel("Rank")
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

#' ## Vectorized Operations for Batch Processing
#' 
#' Vectorized operations allow efficient batch processing of multiple elements.

# Create models for comparison
medium_dataset = datasets["medium"]
mle_model = freqprob.MLE(medium_dataset, logprob=False)
laplace_model = freqprob.Laplace(medium_dataset, bins=20000, logprob=False)

print("VECTORIZED OPERATIONS DEMONSTRATION")
print("=" * 40)
print()

# Create vectorized scorer
vectorized_mle = VectorizedScorer(mle_model)

# Prepare test data
test_elements = list(medium_dataset.keys())[:1000]  # First 1000 words
print(f"Testing with {len(test_elements)} elements...")

# Compare individual vs batch scoring
print("\nPerformance Comparison:")
print("-" * 25)

# Individual scoring (traditional approach)
start_time = time.time()
individual_scores = [mle_model(element) for element in test_elements]
individual_time = time.time() - start_time

# Batch scoring (vectorized approach)
start_time = time.time()
batch_scores = vectorized_mle.score_batch(test_elements)
batch_time = time.time() - start_time

print(f"Individual scoring: {individual_time:.4f} seconds")
print(f"Batch scoring:      {batch_time:.4f} seconds")
print(f"Speedup:           {individual_time / batch_time:.1f}x")

# Verify results are equivalent
max_diff = np.max(np.abs(np.array(individual_scores) - batch_scores))
print(f"Maximum difference: {max_diff:.2e} (should be ~0)")

# Demonstrate matrix operations
print("\nMatrix Operations:")
print("-" * 18)

# Create 2D array of elements
elements_2d = [test_elements[:10], test_elements[10:20]]  # 2 rows, 10 columns
score_matrix = vectorized_mle.score_matrix(elements_2d)

print(f"Score matrix shape: {score_matrix.shape}")
print("Sample scores:")
print(score_matrix[:2, :5])  # Show first 2 rows, 5 columns

# Top-k most probable elements
print("\nTop-K Operations:")
print("-" * 17)

top_elements, top_scores = vectorized_mle.top_k_elements(10)
print("Top 10 most probable elements:")
for i, (element, score) in enumerate(zip(top_elements, top_scores, strict=False)):
    count = medium_dataset[element]
    print(f"{i + 1:2d}. {element} (count={count:,}, prob={score:.6f})")

# Score normalization - import the function from vectorized module
print("\nScore Normalization:")
print("-" * 19)

sample_scores = batch_scores[:20]
# Import the normalize_scores function from vectorized module
from freqprob.vectorized import normalize_scores

normalized_scores = normalize_scores(sample_scores, method="softmax")

print(f"Original scores range:    [{np.min(sample_scores):.6f}, {np.max(sample_scores):.6f}]")
print(
    f"Normalized scores range:  [{np.min(normalized_scores):.6f}, {np.max(normalized_scores):.6f}]"
)
print(f"Normalized scores sum:    {np.sum(normalized_scores):.6f} (should be 1.0)")

#' ## Batch Scoring with Multiple Methods
#' 
#' BatchScorer allows comparing multiple models efficiently.

print("BATCH SCORING WITH MULTIPLE METHODS")
print("=" * 40)
print()

# Create multiple scorers
scorers = {
    "mle": freqprob.MLE(medium_dataset, logprob=False),
    "laplace": freqprob.Laplace(medium_dataset, bins=20000, logprob=False),
    "ele": freqprob.ELE(medium_dataset, bins=20000, logprob=False),
}

# Create batch scorer
batch_scorer = BatchScorer(scorers)

# Test elements (mix of frequent, rare, and unseen)
test_mix = (
    list(medium_dataset.keys())[:5]
    + list(medium_dataset.keys())[-5:]  # Frequent words
    + ["unseen_word_1", "unseen_word_2"]  # Rare words  # Unseen words
)

print(f"Testing {len(test_mix)} elements with {len(scorers)} methods...")

# Batch score all methods
start_time = time.time()
batch_results = batch_scorer.score_batch(test_mix)
batch_time = time.time() - start_time

print(f"Batch scoring completed in {batch_time:.4f} seconds")
print()

# Display results
print("Results by method:")
print("-" * 18)
for method_name, scores in batch_results.items():
    print(f"{method_name.upper()}:")
    for i, (element, score) in enumerate(zip(test_mix, scores, strict=False)):
        count = medium_dataset.get(element, 0)
        element_type = "frequent" if i < 5 else "rare" if i < 10 else "unseen"
        print(f"  {element:<15} ({element_type:<8}, c={count:>3}): {score:.6f}")
    print()

# Visualize method comparison
plt.figure(figsize=(14, 8))

# Prepare data for plotting
methods = list(batch_results.keys())
element_types = ["Frequent"] * 5 + ["Rare"] * 5 + ["Unseen"] * 2
colors = {"Frequent": "green", "Rare": "orange", "Unseen": "red"}

for i, method in enumerate(methods):
    plt.subplot(2, 2, i + 1)
    scores = batch_results[method]

    # Color bars by element type
    bar_colors = [colors[t] for t in element_types]
    bars = plt.bar(range(len(test_mix)), scores, color=bar_colors, alpha=0.7)

    plt.title(f"{method.upper()} Scores")
    plt.xlabel("Test Elements")
    plt.ylabel("Probability")
    plt.xticks(
        range(len(test_mix)), [e[:8] + "..." if len(e) > 8 else e for e in test_mix], rotation=45
    )
    plt.yscale("log")
    plt.grid(True, alpha=0.3)

# Add legend
plt.subplot(2, 2, 4)
legend_elements = [plt.Rectangle((0, 0), 1, 1, facecolor=colors[t], alpha=0.7) for t in colors]
plt.legend(legend_elements, colors.keys(), loc="center")
plt.axis("off")

plt.tight_layout()
plt.show()

# Performance analysis
print("Method Performance Analysis:")
print("-" * 30)

for method_name in methods:
    scores = batch_results[method_name]
    frequent_scores = scores[:5]
    rare_scores = scores[5:10]
    unseen_scores = scores[10:]

    print(f"{method_name.upper()}:")
    print(
        f"  Frequent words: avg = {np.mean(frequent_scores):.6f}, std = {np.std(frequent_scores):.6f}"
    )
    print(f"  Rare words:     avg = {np.mean(rare_scores):.6f}, std = {np.std(rare_scores):.6f}")
    print(
        f"  Unseen words:   avg = {np.mean(unseen_scores):.6f}, std = {np.std(unseen_scores):.6f}"
    )
    print()

#' ## Caching and Memoization
#' 
#' FreqProb automatically caches expensive computations like Simple Good-Turing.

import sys
from functools import lru_cache

print("CACHING AND MEMOIZATION")
print("=" * 25)
print()

# Demonstrate SGT caching (if available)
small_dataset = datasets["small"]

print("Simple Good-Turing Caching Demo:")
print("-" * 35)

try:
    # First creation (cold cache)
    print("Creating first SGT model (cold cache)...")
    start_time = time.time()
    sgt1 = freqprob.SimpleGoodTuring(small_dataset, logprob=False)
    first_time = time.time() - start_time
    print(f"Time: {first_time:.4f} seconds")

    # Second creation (hot cache)
    print("\nCreating second SGT model (hot cache)...")
    start_time = time.time()
    sgt2 = freqprob.SimpleGoodTuring(small_dataset, logprob=False)
    second_time = time.time() - start_time
    print(f"Time: {second_time:.4f} seconds")

    print(f"\nSpeedup from caching: {first_time / second_time:.1f}x")

    # Verify cache is working by checking results are identical
    test_words = list(small_dataset.keys())[:10]
    scores1 = [sgt1(word) for word in test_words]
    scores2 = [sgt2(word) for word in test_words]

    max_diff = max(abs(s1 - s2) for s1, s2 in zip(scores1, scores2, strict=False))
    print(f"Maximum score difference: {max_diff:.2e} (should be 0)")

    # Cache statistics
    cache_stats = freqprob.get_cache_stats()
    print("\nCache Statistics:")
    for key, value in cache_stats.items():
        print(f"  {key}: {value}")

    # Clear cache and show effect
    print("\nClearing all caches...")
    freqprob.clear_all_caches()

    print("Creating third SGT model (cache cleared)...")
    start_time = time.time()
    sgt3 = freqprob.SimpleGoodTuring(small_dataset, logprob=False)
    third_time = time.time() - start_time
    print(f"Time: {third_time:.4f} seconds (should be similar to first time)")

except Exception as e:
    print(f"SGT caching demo failed: {e}")
    print("This can happen with certain frequency distributions")

# Demonstrate custom caching for expensive operations
print("\n\nCustom Caching Example:")
print("-" * 25)

# Simulate expensive computation


@lru_cache(maxsize=1000)
def expensive_computation(x):
    """Simulate expensive computation with caching."""
    time.sleep(0.001)  # Simulate work
    return x**2 + np.sin(x) * 100


# Test caching effect
test_values = np.random.uniform(0, 10, 500)
repeated_values = np.tile(test_values[:100], 5)  # Repeat first 100 values 5 times

print(f"Testing with {len(repeated_values)} values (many repeats)...")

# Without cache (clear cache first)
expensive_computation.cache_clear()
start_time = time.time()
results_no_cache = [expensive_computation(x) for x in test_values[:100]]  # Fresh computation
no_cache_time = time.time() - start_time

# With cache (subsequent calls)
start_time = time.time()
results_with_cache = [expensive_computation(x) for x in repeated_values]  # Many cached hits
with_cache_time = time.time() - start_time

print(f"Time without cache benefit: {no_cache_time:.4f} seconds")
print(f"Time with cache benefit:    {with_cache_time:.4f} seconds")
print(f"Speedup from caching:      {no_cache_time * 5 / with_cache_time:.1f}x")

# Cache info
cache_info = expensive_computation.cache_info()
print("\nCache statistics:")
print(f"  Hits: {cache_info.hits}")
print(f"  Misses: {cache_info.misses}")
print(f"  Hit rate: {cache_info.hits / (cache_info.hits + cache_info.misses) * 100:.1f}%")

#' ## Lazy Evaluation
#' 
#' Lazy evaluation computes probabilities only when needed, saving time and memory.

from freqprob import LazyBatchScorer

print("LAZY EVALUATION")
print("=" * 16)
print()

# Use large dataset for dramatic effect
large_dataset = datasets["large"]
print(
    f"Working with dataset: {len(large_dataset):,} words, {sum(large_dataset.values()):,} total count"
)

# Create lazy and eager models
print("\nCreating models...")

# Eager evaluation (traditional)
start_time = time.time()
eager_mle = freqprob.MLE(large_dataset, logprob=False)
eager_creation_time = time.time() - start_time
print(f"Eager MLE creation: {eager_creation_time:.4f} seconds")

# Lazy evaluation
start_time = time.time()
lazy_mle = create_lazy_mle(large_dataset, logprob=False)
lazy_creation_time = time.time() - start_time
print(f"Lazy MLE creation:  {lazy_creation_time:.4f} seconds")

print(f"Creation speedup: {eager_creation_time / lazy_creation_time:.1f}x")

# Test queries on subset of elements
test_subset = list(large_dataset.keys())[::1000]  # Every 1000th element
print(f"\nQuerying {len(test_subset)} elements...")

# Eager model queries
start_time = time.time()
eager_scores = [eager_mle(word) for word in test_subset]
eager_query_time = time.time() - start_time

# Lazy model queries (first time)
start_time = time.time()
lazy_scores = [lazy_mle(word) for word in test_subset]
lazy_query_time = time.time() - start_time

print(f"Eager query time: {eager_query_time:.4f} seconds")
print(f"Lazy query time:  {lazy_query_time:.4f} seconds")

# Verify results are identical
max_diff = max(
    abs(eager_score - lazy_score)
    for eager_score, lazy_score in zip(eager_scores, lazy_scores, strict=False)
)
print(f"Maximum difference: {max_diff:.2e}")

# Show lazy evaluation statistics
computed_elements = lazy_mle.get_computed_elements()
total_elements = len(large_dataset)
computation_ratio = len(computed_elements) / total_elements

print("\nLazy Evaluation Statistics:")
print(f"Total elements:    {total_elements:,}")
print(f"Computed elements: {len(computed_elements):,}")
print(f"Computation ratio: {computation_ratio:.1%}")
print(f"Memory savings:    ~{(1 - computation_ratio) * 100:.1f}%")

# Demonstrate repeated queries (cached)
print("\nRepeated queries (should be cached):")
start_time = time.time()
lazy_scores_repeat = [lazy_mle(word) for word in test_subset]
lazy_repeat_time = time.time() - start_time

print(f"Lazy repeat time: {lazy_repeat_time:.4f} seconds")
print(f"Speedup vs first: {lazy_query_time / lazy_repeat_time:.1f}x")

# Batch lazy evaluation
print("\nBatch Lazy Evaluation:")
print("-" * 25)

# Create lazy batch scorer - use the lazy_mle we already created
lazy_batch = LazyBatchScorer(lazy_mle)

# Test batch processing
batch_test_elements = list(large_dataset.keys())[::2000]  # Every 2000th element
print(f"Batch processing {len(batch_test_elements)} elements...")

start_time = time.time()
batch_scores = lazy_batch.score_batch(batch_test_elements)
batch_time = time.time() - start_time

print(f"Batch processing time: {batch_time:.4f} seconds")
print(f"Average per element:   {batch_time / len(batch_test_elements) * 1000:.2f} ms")

# Demonstrate streaming evaluation
print("\nStreaming Evaluation:")
print("-" * 20)

# Create a stream of elements (some repeated)
stream_elements = batch_test_elements[:5] * 3  # Repeat first 5 elements 3 times
print(f"Processing stream of {len(stream_elements)} elements (with repetitions)...")

start_time = time.time()
streaming_scores = list(lazy_batch.score_streaming(stream_elements))
streaming_time = time.time() - start_time

print(f"Streaming processing time: {streaming_time:.4f} seconds")
print(f"Average per element:       {streaming_time / len(stream_elements) * 1000:.2f} ms")

# Memory usage comparison (simplified approach)
print("\nMemory Usage Comparison:")
print("-" * 27)

# Get current process memory for comparison
import psutil

current_process = psutil.Process()
current_memory = current_process.memory_info().rss / 1024 / 1024

# Estimate memory usage based on computed elements
eager_total_elements = len(large_dataset)
lazy_computed_elements = len(computed_elements)

# Estimated memory usage (simplified calculation)
estimated_eager_memory = eager_total_elements * 0.1  # Assume ~0.1KB per element
estimated_lazy_memory = lazy_computed_elements * 0.1  # Only computed elements

print(f"Current process memory:     {current_memory:.1f} MB")
print(
    f"Estimated eager model:      {estimated_eager_memory / 1024:.1f} MB (all {eager_total_elements:,} elements)"
)
print(
    f"Estimated lazy model:       {estimated_lazy_memory / 1024:.1f} MB (only {lazy_computed_elements:,} computed)"
)
print(f"Computational efficiency:   {computation_ratio:.1%} of full computation")
print(f"Memory efficiency:          ~{(1 - computation_ratio) * 100:.1f}% savings")

#' ## Streaming Updates and Real-time Learning
#' 
#' Streaming models allow incremental updates for real-time applications.

print("STREAMING UPDATES AND REAL-TIME LEARNING")
print("=" * 42)
print()

# Initialize streaming models
initial_data = {f"word_{i}": max(1, 100 - i) for i in range(50)}  # Small initial vocabulary

streaming_mle = StreamingMLE(initial_data, max_vocabulary_size=100, logprob=False)
streaming_laplace = StreamingLaplace(initial_data, max_vocabulary_size=100, logprob=False)

print(f"Initial vocabulary size: {len(initial_data)}")
print("Maximum vocabulary size: 100")
print()

# Simulate streaming data


def generate_streaming_data(n_updates=500):
    """Generate streaming data with evolving vocabulary."""
    stream = []

    # Mix of existing and new words
    existing_words = list(initial_data.keys())

    for i in range(n_updates):
        word = random.choice(existing_words) if random.random() < 0.7 else f"new_word_{i}"
        count = random.randint(1, 5)
        stream.append((word, count))

    return stream


streaming_data = generate_streaming_data(500)
print(f"Generated {len(streaming_data)} streaming updates")

# Process streaming data and track statistics
print("\nProcessing streaming data...")

vocab_sizes = []
update_times = []
probabilities_tracked = {"word_0": [], "word_10": [], "new_word_100": []}

for i, (word, count) in enumerate(streaming_data):
    start_time = time.time()

    # Update both models
    streaming_mle.update_single(word, count)
    streaming_laplace.update_single(word, count)

    update_time = time.time() - start_time
    update_times.append(update_time * 1000)  # Convert to milliseconds

    # Track statistics every 50 updates
    if i % 50 == 0:
        stats = streaming_mle.get_streaming_statistics()
        vocab_sizes.append(stats["vocabulary_size"])

        # Track specific word probabilities
        for tracked_word in probabilities_tracked:
            prob = streaming_mle(tracked_word)
            probabilities_tracked[tracked_word].append(prob)

print(f"Average update time: {np.mean(update_times):.3f} ms")
print(f"Maximum update time: {np.max(update_times):.3f} ms")

# Final statistics
final_stats = streaming_mle.get_streaming_statistics()
print("\nFinal Statistics:")
print(f"  Vocabulary size: {final_stats['vocabulary_size']}")
print(f"  Total count: {final_stats['total_count']:,}")
print(f"  Update count: {final_stats['update_count']}")
print(f"  Most frequent: {final_stats['most_frequent']}")

# Visualize streaming behavior
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# 1. Vocabulary size over time
axes[0, 0].plot(range(0, len(streaming_data), 50), vocab_sizes, "b-", linewidth=2)
axes[0, 0].axhline(y=100, color="r", linestyle="--", alpha=0.7, label="Max vocabulary")
axes[0, 0].set_title("Vocabulary Size Over Time")
axes[0, 0].set_xlabel("Updates")
axes[0, 0].set_ylabel("Vocabulary Size")
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. Update time distribution
axes[0, 1].hist(update_times, bins=30, alpha=0.7, color="green")
axes[0, 1].axvline(
    x=np.mean(update_times),
    color="red",
    linestyle="--",
    label=f"Mean: {np.mean(update_times):.2f} ms",
)
axes[0, 1].set_title("Update Time Distribution")
axes[0, 1].set_xlabel("Update Time (ms)")
axes[0, 1].set_ylabel("Frequency")
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 3. Probability evolution
colors = ["blue", "orange", "green"]
for i, (word, probs) in enumerate(probabilities_tracked.items()):
    if probs:  # Only plot if we have data
        axes[1, 0].plot(
            range(0, len(probs) * 50, 50),
            probs,
            color=colors[i],
            linewidth=2,
            label=word,
            marker="o",
            markersize=4,
        )

axes[1, 0].set_title("Probability Evolution")
axes[1, 0].set_xlabel("Updates")
axes[1, 0].set_ylabel("Probability")
axes[1, 0].set_yscale("log")
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 4. Model comparison on test words
test_words = ["word_0", "word_25", "new_word_100", "unseen_word"]
mle_probs = [streaming_mle(word) for word in test_words]
laplace_probs = [streaming_laplace(word) for word in test_words]

x = np.arange(len(test_words))
width = 0.35

axes[1, 1].bar(x - width / 2, mle_probs, width, label="Streaming MLE", alpha=0.8)
axes[1, 1].bar(x + width / 2, laplace_probs, width, label="Streaming Laplace", alpha=0.8)

axes[1, 1].set_title("Model Comparison After Streaming")
axes[1, 1].set_xlabel("Test Words")
axes[1, 1].set_ylabel("Probability")
axes[1, 1].set_xticks(x)
axes[1, 1].set_xticklabels(test_words, rotation=45)
axes[1, 1].set_yscale("log")
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\nKey Observations:")
print("- Vocabulary size stabilizes at the maximum limit")
print("- Update times remain consistently low")
print("- Probabilities adapt to new data in real-time")
print("- Streaming Laplace provides smoother probability estimates")

#' ## Memory-Efficient Representations
#' 
#' For large vocabularies, compressed and sparse representations can significantly reduce memory usage.

print("MEMORY-EFFICIENT REPRESENTATIONS")
print("=" * 33)
print()

# Create large frequency distribution for demonstration
large_freq_dist = create_realistic_frequency_distribution(25000, 250000)
print(
    f"Original dataset: {len(large_freq_dist):,} words, {sum(large_freq_dist.values()):,} total count"
)

# Memory usage of original dictionary
original_size = sys.getsizeof(large_freq_dist)
for k, v in large_freq_dist.items():
    original_size += sys.getsizeof(k) + sys.getsizeof(v)

print(f"Original memory usage: {original_size / 1024 / 1024:.1f} MB")
print()

# Create different representations
print("Creating memory-efficient representations...")

# 1. Compressed representation
start_time = time.time()
compressed_dist = create_compressed_distribution(large_freq_dist, quantization_levels=1024)
compressed_time = time.time() - start_time
compressed_memory = compressed_dist.get_memory_usage()

# 2. Sparse representation
start_time = time.time()
sparse_dist = create_sparse_distribution(large_freq_dist)
sparse_time = time.time() - start_time
sparse_memory = sparse_dist.get_memory_usage()

# 3. Highly compressed representation
start_time = time.time()
highly_compressed = create_compressed_distribution(
    large_freq_dist, quantization_levels=256, use_compression=True
)
highly_compressed_time = time.time() - start_time
highly_compressed_memory = highly_compressed.get_memory_usage()

print(f"Compressed creation time:        {compressed_time:.3f} seconds")
print(f"Sparse creation time:            {sparse_time:.3f} seconds")
print(f"Highly compressed creation time: {highly_compressed_time:.3f} seconds")
print()

# Memory comparison
print("Memory Usage Comparison:")
print("-" * 28)

representations = [
    ("Original Dictionary", original_size),
    ("Compressed", compressed_memory["total"]),
    ("Sparse", sparse_memory["total"]),
    ("Highly Compressed", highly_compressed_memory["total"]),
]

for name, size in representations:
    size_mb = size / 1024 / 1024
    savings = (1 - size / original_size) * 100
    print(f"{name:<20}: {size_mb:6.1f} MB ({savings:+5.1f}% vs original)")

# Accuracy comparison (for compressed representations)
print("\nAccuracy Comparison:")
print("-" * 20)

test_sample = list(large_freq_dist.items())[:100]  # Test on first 100 words

# Calculate errors for compressed representations
compressed_errors = []
highly_compressed_errors = []

for word, original_count in test_sample:
    compressed_count = compressed_dist.get_count(word)
    highly_compressed_count = highly_compressed.get_count(word)

    compressed_error = abs(original_count - compressed_count) / original_count
    highly_compressed_error = abs(original_count - highly_compressed_count) / original_count

    compressed_errors.append(compressed_error)
    highly_compressed_errors.append(highly_compressed_error)

print("Compressed (1024 levels):")
print(f"  Mean relative error: {np.mean(compressed_errors) * 100:.2f}%")
print(f"  Max relative error:  {np.max(compressed_errors) * 100:.2f}%")

print("Highly compressed (256 levels):")
print(f"  Mean relative error: {np.mean(highly_compressed_errors) * 100:.2f}%")
print(f"  Max relative error:  {np.max(highly_compressed_errors) * 100:.2f}%")

# Sparse distribution benefits
print("\nSparse distribution benefits:")
print(f"  Stored elements: {sparse_dist.get_vocabulary_size():,}")
print(
    f"  Storage efficiency: {sparse_dist.get_vocabulary_size() / len(large_freq_dist) * 100:.1f}% of original"
)

# Demonstrate sparse operations
top_10 = sparse_dist.get_top_k(10)
print(f"  Top 10 elements: {top_10[:3]}...")  # Show first 3

mid_freq_elements = sparse_dist.get_elements_with_count_range(100, 1000)
print(f"  Elements with count 100-1000: {len(mid_freq_elements)}")

# Visualize memory usage breakdown
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Memory usage comparison
names = [name for name, _ in representations]
sizes = [size / 1024 / 1024 for _, size in representations]
colors = ["red", "blue", "green", "orange"]

bars = ax1.bar(names, sizes, color=colors, alpha=0.7)
ax1.set_title("Memory Usage Comparison")
ax1.set_ylabel("Memory Usage (MB)")
ax1.tick_params(axis="x", rotation=45)

# Add value labels on bars
for bar, size in zip(bars, sizes, strict=False):
    ax1.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.5,
        f"{size:.1f}",
        ha="center",
        va="bottom",
    )

# Accuracy vs compression trade-off
compression_levels = [1024, 256]
mean_errors = [np.mean(compressed_errors) * 100, np.mean(highly_compressed_errors) * 100]
memory_savings = [
    (1 - compressed_memory["total"] / original_size) * 100,
    (1 - highly_compressed_memory["total"] / original_size) * 100,
]

ax2.scatter(memory_savings, mean_errors, s=100, c=["blue", "orange"], alpha=0.7)
for i, level in enumerate(compression_levels):
    ax2.annotate(
        f"{level} levels",
        (memory_savings[i], mean_errors[i]),
        xytext=(5, 5),
        textcoords="offset points",
    )

ax2.set_title("Accuracy vs Memory Savings Trade-off")
ax2.set_xlabel("Memory Savings (%)")
ax2.set_ylabel("Mean Relative Error (%)")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Performance comparison
print("\nQuery Performance Comparison:")
print("-" * 31)

query_words = list(large_freq_dist.keys())[:1000]

# Time each representation
query_times = {}

# Original dictionary
start_time = time.time()
_ = [large_freq_dist.get(word, 0) for word in query_words]
query_times["Original"] = time.time() - start_time

# Compressed
start_time = time.time()
_ = [compressed_dist.get_count(word) for word in query_words]
query_times["Compressed"] = time.time() - start_time

# Sparse
start_time = time.time()
_ = [sparse_dist.get_count(word) for word in query_words]
query_times["Sparse"] = time.time() - start_time

for name, time_taken in query_times.items():
    queries_per_sec = len(query_words) / time_taken
    print(f"{name:<12}: {time_taken:.4f}s ({queries_per_sec:,.0f} queries/sec)")

print("\nKey Insights:")
print("- Compressed representations can save 50-80% memory")
print("- Sparse representations excel for distributions with many zeros")
print("- Trade-off between compression ratio and accuracy")
print("- Query performance generally comparable")

#' ## Memory Profiling and Monitoring
#' 
#' FreqProb provides tools for monitoring memory usage and optimizing performance.

print("MEMORY PROFILING AND MONITORING")
print("=" * 33)
print()

# Initialize memory profiler
profiler = MemoryProfiler()

print("Memory Profiling Demo:")
print("-" * 22)

# Profile different operations
operations = [
    ("Create MLE Model", lambda: freqprob.MLE(datasets["medium"], logprob=False)),
    (
        "Create Laplace Model",
        lambda: freqprob.Laplace(datasets["medium"], bins=20000, logprob=False),
    ),
    ("Create Compressed Dist", lambda: create_compressed_distribution(datasets["medium"])),
    (
        "Vectorized Scoring",
        lambda: VectorizedScorer(freqprob.MLE(datasets["small"])).score_batch(
            list(datasets["small"].keys())[:100]
        ),
    ),
]

profiling_results = []

for op_name, operation in operations:
    print(f"Profiling: {op_name}...")

    with profiler.profile_operation(op_name):
        result = operation()
        # Force garbage collection to clean measurement
        gc.collect()

    metrics = profiler.get_latest_metrics()
    profiling_results.append((op_name, metrics))

    print(f"  Memory delta: {metrics.memory_delta_mb:+6.2f} MB")
    print(f"  Execution time: {metrics.execution_time:.4f} seconds")
    print()

# Display profiling summary
print("Profiling Summary:")
print("-" * 18)

summary = profiler.get_memory_summary()
print(f"Total snapshots taken: {summary['total_snapshots']}")

# Fix: Use the correct keys from the actual memory summary structure
if "rss_memory" in summary:
    rss_memory = summary["rss_memory"]
    print(f"Peak memory usage: {rss_memory['max_mb']:.1f} MB")
    print(f"Memory range: {rss_memory['min_mb']:.1f} - {rss_memory['max_mb']:.1f} MB")
else:
    print("Memory summary not available (no snapshots taken)")
print()

# Analyze memory efficiency of different representations
print("Distribution Memory Analysis:")
print("-" * 30)

analyzer = DistributionMemoryAnalyzer()

# Analyze medium dataset
analysis_dataset = datasets["medium"]
print(f"Analyzing dataset with {len(analysis_dataset):,} elements...")

# Measure original distribution
original_memory = analyzer.measure_distribution_memory(analysis_dataset)
print("\nOriginal distribution:")
print(f"  Total memory: {original_memory['total_mb']:.1f} MB")
print(f"  Elements: {original_memory['num_elements']:,}")
print(
    f"  Memory per element: {original_memory['total_mb'] * 1024 / original_memory['num_elements']:.1f} KB"
)

# Compare all representations
comparison = analyzer.compare_representations(analysis_dataset)

print("\nRepresentation Comparison:")
for repr_name, memory_info in comparison.items():
    if repr_name != "memory_savings" and repr_name != "profiling_metrics":
        print(f"  {repr_name.capitalize()}:")

        # Fix: Handle different key structures
        if "total_mb" in memory_info:
            # Original distribution uses 'total_mb'
            memory_mb = memory_info["total_mb"]
        elif "total" in memory_info:
            # Other representations use 'total' in bytes
            memory_mb = memory_info["total"] / 1024 / 1024
        else:
            memory_mb = 0.0

        print(f"    Memory: {memory_mb:.1f} MB")
        if "creation_time" in memory_info:
            print(f"    Creation time: {memory_info['creation_time']:.3f} seconds")

print("\nMemory Savings:")
if "memory_savings" in comparison:
    savings = comparison["memory_savings"]
    for method, saving_info in savings.items():
        print(f"  {method.capitalize()}: {saving_info['percentage_savings']:.1f}% savings")
        print(f"    Absolute: {saving_info['absolute_savings_mb']:.1f} MB")

# Memory trend monitoring
print("\nMemory Trend Analysis:")
print("-" * 23)

# Simulate memory usage over time
memory_monitor = freqprob.MemoryMonitor(memory_threshold_mb=100.0, monitoring_interval=0.1)
memory_monitor.start_monitoring()

# Simulate workload
workload_memory = []
for _i in range(10):
    # Create and destroy objects to simulate memory usage
    temp_data = {f"temp_{j}": j for j in range(1000)}
    temp_model = freqprob.MLE(temp_data, logprob=False)

    # Check memory
    alert = memory_monitor.check_memory()
    current_memory = psutil.Process().memory_info().rss / 1024 / 1024
    workload_memory.append(current_memory)

    time.sleep(0.05)  # Brief pause
    del temp_data, temp_model
    gc.collect()

memory_monitor.stop_monitoring()

# Get monitoring report
monitoring_report = memory_monitor.get_monitoring_report()
print(f"Monitoring duration: {monitoring_report['monitoring_duration']:.2f} seconds")
print("Memory statistics:")
for stat_name, value in monitoring_report["memory_statistics"].items():
    print(f"  {stat_name}: {value:.1f} MB")

# Visualize profiling results
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# 1. Memory usage by operation
op_names = [name for name, _ in profiling_results]
memory_deltas = [metrics.memory_delta_mb for _, metrics in profiling_results]
execution_times = [metrics.execution_time for _, metrics in profiling_results]

bars = axes[0, 0].bar(range(len(op_names)), memory_deltas, alpha=0.7)
axes[0, 0].set_title("Memory Usage by Operation")
axes[0, 0].set_xlabel("Operations")
axes[0, 0].set_ylabel("Memory Delta (MB)")
axes[0, 0].set_xticks(range(len(op_names)))
axes[0, 0].set_xticklabels([name.replace(" ", "\n") for name in op_names], rotation=0)

# Add value labels
for bar, value in zip(bars, memory_deltas, strict=False):
    axes[0, 0].text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.1,
        f"{value:.1f}",
        ha="center",
        va="bottom",
    )

# 2. Execution time by operation
axes[0, 1].bar(range(len(op_names)), execution_times, alpha=0.7, color="orange")
axes[0, 1].set_title("Execution Time by Operation")
axes[0, 1].set_xlabel("Operations")
axes[0, 1].set_ylabel("Execution Time (seconds)")
axes[0, 1].set_xticks(range(len(op_names)))
axes[0, 1].set_xticklabels([name.replace(" ", "\n") for name in op_names], rotation=0)

# 3. Memory efficiency comparison - fix to handle different key structures
repr_names = ["Original", "Compressed", "Sparse"]
memory_usage = []

# Extract memory usage with proper key handling
for repr_name in ["original", "compressed", "sparse"]:
    if repr_name in comparison:
        memory_info = comparison[repr_name]
        if "total_mb" in memory_info:
            memory_usage.append(memory_info["total_mb"])
        elif "total" in memory_info:
            memory_usage.append(memory_info["total"] / 1024 / 1024)
        else:
            memory_usage.append(0.0)
    else:
        memory_usage.append(0.0)

if memory_usage and memory_usage[0] > 0:  # Only plot if we have valid data
    bars = axes[1, 0].bar(repr_names, memory_usage, alpha=0.7, color=["red", "blue", "green"])
    axes[1, 0].set_title("Memory Usage by Representation")
    axes[1, 0].set_ylabel("Memory Usage (MB)")

    # Add percentage savings
    for i, (bar, usage) in enumerate(zip(bars, memory_usage, strict=False)):
        if i > 0 and memory_usage[0] > 0:  # Skip original
            savings_pct = (1 - usage / memory_usage[0]) * 100
            axes[1, 0].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"-{savings_pct:.0f}%",
                ha="center",
                va="bottom",
                color="green",
                fontweight="bold",
            )
else:
    axes[1, 0].text(
        0.5,
        0.5,
        "Memory comparison data not available",
        ha="center",
        va="center",
        transform=axes[1, 0].transAxes,
    )
    axes[1, 0].set_title("Memory Usage by Representation")

# 4. Memory trend during workload
axes[1, 1].plot(range(len(workload_memory)), workload_memory, "b-", linewidth=2, marker="o")
axes[1, 1].set_title("Memory Usage During Workload")
axes[1, 1].set_xlabel("Time Steps")
axes[1, 1].set_ylabel("Memory Usage (MB)")
axes[1, 1].grid(True, alpha=0.3)

# Add threshold line if available
axes[1, 1].axhline(y=100, color="red", linestyle="--", alpha=0.7, label="Threshold")
axes[1, 1].legend()

plt.tight_layout()
plt.show()

# Recommendations
print("\nMemory Optimization Recommendations:")
print("=" * 38)
print("1. Use compressed representations for large vocabularies (50%+ savings)")
print("2. Consider sparse representations for distributions with many zeros")
print("3. Monitor memory usage during long-running processes")
print("4. Use streaming models for real-time applications with bounded memory")
print("5. Profile operations to identify memory bottlenecks")
print("6. Implement garbage collection strategies for batch processing")

#' ## Performance Benchmarks and Best Practices
#' 
#' Let's conclude with comprehensive benchmarks and practical recommendations.

print("PERFORMANCE BENCHMARKS AND BEST PRACTICES")
print("=" * 44)
print()

# Comprehensive benchmark suite


def run_comprehensive_benchmark():
    """Run comprehensive performance benchmark."""

    benchmark_results = {}

    # Test datasets of different sizes
    test_datasets = {
        "Small (1K)": datasets["small"],
        "Medium (10K)": datasets["medium"],
        "Large (50K)": datasets["large"],
    }

    for dataset_name, dataset in test_datasets.items():
        print(f"Benchmarking {dataset_name}...")

        dataset_results = {}
        test_elements = list(dataset.keys())[: min(1000, len(dataset))]

        # 1. Model creation time
        def create_mle(ds):
            return freqprob.MLE(ds, logprob=False)

        def create_laplace(ds):
            return freqprob.Laplace(ds, bins=len(ds) * 2, logprob=False)

        def create_lazy_mle_func(ds):
            return create_lazy_mle(ds, logprob=False)

        models_to_test = [
            ("MLE", create_mle),
            ("Laplace", create_laplace),
            ("Lazy MLE", create_lazy_mle_func),
        ]

        for model_name, model_creator in models_to_test:
            # Creation time
            start_time = time.time()
            model = model_creator(dataset)
            creation_time = time.time() - start_time

            # Query time
            start_time = time.time()
            _ = [model(element) for element in test_elements[:100]]
            query_time = time.time() - start_time

            dataset_results[model_name] = {
                "creation_time": creation_time,
                "query_time": query_time,
                "queries_per_sec": 100 / query_time,
            }

        # 2. Vectorized vs individual scoring
        mle_model = freqprob.MLE(dataset, logprob=False)
        vectorized_model = VectorizedScorer(mle_model)

        # Individual scoring
        start_time = time.time()
        _ = [mle_model(element) for element in test_elements]
        individual_time = time.time() - start_time

        # Batch scoring
        start_time = time.time()
        _ = vectorized_model.score_batch(test_elements)
        batch_time = time.time() - start_time

        dataset_results["Vectorization"] = {
            "individual_time": individual_time,
            "batch_time": batch_time,
            "speedup": individual_time / batch_time,
        }

        # 3. Memory representations
        if len(dataset) >= 5000:  # Only for larger datasets
            # Compressed
            start_time = time.time()
            compressed = create_compressed_distribution(dataset)
            compressed_creation_time = time.time() - start_time

            # Sparse
            start_time = time.time()
            sparse = create_sparse_distribution(dataset)
            sparse_creation_time = time.time() - start_time

            dataset_results["Memory Representations"] = {
                "compressed_creation": compressed_creation_time,
                "sparse_creation": sparse_creation_time,
                "compressed_memory": compressed.get_memory_usage()["total"] / 1024 / 1024,
                "sparse_memory": sparse.get_memory_usage()["total"] / 1024 / 1024,
            }

        benchmark_results[dataset_name] = dataset_results

    return benchmark_results


# Run benchmarks
print("Running comprehensive benchmarks...")
benchmark_results = run_comprehensive_benchmark()

# Display results
print("\nBenchmark Results:")
print("=" * 18)

for dataset_name, results in benchmark_results.items():
    print(f"\n{dataset_name}:")
    print("-" * len(dataset_name))

    # Model performance
    print("Model Performance:")
    for model_name in ["MLE", "Laplace", "Lazy MLE"]:
        if model_name in results:
            r = results[model_name]
            print(
                f"  {model_name:<10}: {r['creation_time']:.4f}s creation, {r['queries_per_sec']:,.0f} queries/sec"
            )

    # Vectorization performance
    if "Vectorization" in results:
        v = results["Vectorization"]
        print(f"  Vectorization: {v['speedup']:.1f}x speedup")

    # Memory representations
    if "Memory Representations" in results:
        m = results["Memory Representations"]
        print(f"  Memory (Compressed): {m['compressed_memory']:.1f} MB")
        print(f"  Memory (Sparse): {m['sparse_memory']:.1f} MB")

# Performance scaling analysis
print("\n\nPerformance Scaling Analysis:")
print("=" * 31)

dataset_sizes = [len(datasets["small"]), len(datasets["medium"]), len(datasets["large"])]
mle_creation_times = []
lazy_creation_times = []
vectorization_speedups = []

for dataset_name in ["Small (1K)", "Medium (10K)", "Large (50K)"]:
    results = benchmark_results[dataset_name]
    mle_creation_times.append(results["MLE"]["creation_time"])
    lazy_creation_times.append(results["Lazy MLE"]["creation_time"])
    vectorization_speedups.append(results["Vectorization"]["speedup"])

# Visualize scaling
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# 1. Creation time scaling
axes[0, 0].loglog(dataset_sizes, mle_creation_times, "o-", label="MLE", linewidth=2, markersize=8)
axes[0, 0].loglog(
    dataset_sizes, lazy_creation_times, "s-", label="Lazy MLE", linewidth=2, markersize=8
)
axes[0, 0].set_title("Model Creation Time Scaling")
axes[0, 0].set_xlabel("Dataset Size")
axes[0, 0].set_ylabel("Creation Time (seconds)")
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. Vectorization speedup
axes[0, 1].semilogx(
    dataset_sizes, vectorization_speedups, "o-", color="green", linewidth=2, markersize=8
)
axes[0, 1].set_title("Vectorization Speedup by Dataset Size")
axes[0, 1].set_xlabel("Dataset Size")
axes[0, 1].set_ylabel("Speedup Factor")
axes[0, 1].grid(True, alpha=0.3)

# 3. Memory usage comparison (for larger datasets)
memory_datasets = ["Medium (10K)", "Large (50K)"]
compressed_memory = []
sparse_memory = []

for dataset_name in memory_datasets:
    if "Memory Representations" in benchmark_results[dataset_name]:
        m = benchmark_results[dataset_name]["Memory Representations"]
        compressed_memory.append(m["compressed_memory"])
        sparse_memory.append(m["sparse_memory"])

if compressed_memory and sparse_memory:
    x = np.arange(len(memory_datasets))
    width = 0.35

    axes[1, 0].bar(x - width / 2, compressed_memory, width, label="Compressed", alpha=0.8)
    axes[1, 0].bar(x + width / 2, sparse_memory, width, label="Sparse", alpha=0.8)

    axes[1, 0].set_title("Memory Usage by Representation")
    axes[1, 0].set_xlabel("Dataset")
    axes[1, 0].set_ylabel("Memory Usage (MB)")
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(memory_datasets)
    axes[1, 0].legend()

# 4. Queries per second comparison
model_names = ["MLE", "Laplace", "Lazy MLE"]
queries_per_sec = {model: [] for model in model_names}

for dataset_name in ["Small (1K)", "Medium (10K)", "Large (50K)"]:
    results = benchmark_results[dataset_name]
    for model_name in model_names:
        if model_name in results:
            queries_per_sec[model_name].append(results[model_name]["queries_per_sec"])

x = np.arange(len(["Small", "Medium", "Large"]))
width = 0.25

for i, (model_name, qps_values) in enumerate(queries_per_sec.items()):
    if qps_values:
        offset = (i - 1) * width
        axes[1, 1].bar(x + offset, qps_values, width, label=model_name, alpha=0.8)

axes[1, 1].set_title("Query Performance by Dataset Size")
axes[1, 1].set_xlabel("Dataset Size")
axes[1, 1].set_ylabel("Queries per Second")
axes[1, 1].set_xticks(x)
axes[1, 1].set_xticklabels(["Small", "Medium", "Large"])
axes[1, 1].legend()
axes[1, 1].set_yscale("log")

plt.tight_layout()
plt.show()

# Best practices summary
print("\nBEST PRACTICES SUMMARY")
print("=" * 23)
print()

print("🚀 PERFORMANCE OPTIMIZATION:")
print("  • Use vectorized operations for batch processing (2-10x speedup)")
print("  • Choose lazy evaluation for sparse access patterns")
print("  • Implement caching for expensive computations (SGT, etc.)")
print("  • Profile operations to identify bottlenecks")
print()

print("💾 MEMORY MANAGEMENT:")
print("  • Use compressed representations for large vocabularies (50%+ savings)")
print("  • Consider sparse representations for distributions with many zeros")
print("  • Implement streaming for real-time applications")
print("  • Monitor memory usage and set appropriate limits")
print()

print("⚖️ TRADE-OFFS TO CONSIDER:")
print("  • Accuracy vs. Memory: Higher compression = lower accuracy")
print("  • Speed vs. Memory: Eager computation = faster queries, more memory")
print("  • Complexity vs. Performance: Advanced methods may not always be better")
print("  • Real-time vs. Batch: Streaming good for real-time, batch good for throughput")
print()

print("🎯 CHOOSING THE RIGHT APPROACH:")
print("  • Small datasets (<10K): Standard implementations sufficient")
print("  • Medium datasets (10K-100K): Consider vectorization and compression")
print("  • Large datasets (>100K): Use streaming, compression, and lazy evaluation")
print("  • Real-time applications: Streaming models with bounded memory")
print("  • Batch processing: Vectorized operations with memory monitoring")
print()

print("🔧 IMPLEMENTATION CHECKLIST:")
print("  ☐ Profile your specific use case")
print("  ☐ Choose appropriate data representations")
print("  ☐ Implement memory monitoring")
print("  ☐ Test with realistic data sizes")
print("  ☐ Validate accuracy vs. efficiency trade-offs")
print("  ☐ Plan for scaling and growth")

#' ## Summary and Next Steps
#' 
#' In this tutorial, you learned how to optimize FreqProb for large-scale, production applications:
#' 
#' ### Key Techniques Covered:
#' 
#' 1. **Vectorized Operations** - 2-10x speedup for batch processing
#' 2. **Caching and Memoization** - Automatic optimization for expensive computations
#' 3. **Lazy Evaluation** - Compute only what's needed, save memory
#' 4. **Streaming Updates** - Real-time learning with bounded memory
#' 5. **Memory-Efficient Representations** - 50-80% memory savings for large vocabularies
#' 6. **Memory Profiling** - Monitor and optimize memory usage
#' 
#' ### Performance Insights:
#' 
#' - **Vectorization** provides consistent speedups across dataset sizes
#' - **Lazy evaluation** is most beneficial for sparse access patterns
#' - **Compressed representations** offer significant memory savings with minimal accuracy loss
#' - **Streaming models** enable real-time applications with predictable memory usage
#' 
#' ### When to Use What:
#' 
#' - **Small datasets** (<10K): Standard implementations are sufficient
#' - **Medium datasets** (10K-100K): Add vectorization and consider compression
#' - **Large datasets** (>100K): Use all efficiency features
#' - **Real-time applications**: Streaming models with memory monitoring
#' - **Batch processing**: Vectorized operations with profiling
#' 
#' ### Next Steps:
#' 
#' 1. **Apply these techniques** to your specific use case
#' 2. **Profile your applications** to identify bottlenecks
#' 3. **Try Tutorial 4**: Real-world NLP Applications
#' 4. **Experiment** with different combinations of techniques
#' 5. **Scale gradually** and monitor performance metrics
#' 
#' Remember: **optimization is application-specific**. Always profile your specific use case and validate that optimizations actually improve performance for your data and usage patterns!
