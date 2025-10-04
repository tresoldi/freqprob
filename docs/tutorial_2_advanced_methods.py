#' # FreqProb Tutorial 2: Advanced Smoothing Methods
#'
#' This tutorial covers advanced smoothing techniques that are essential for modern NLP applications:
#'
#' 1. **Simple Good-Turing smoothing** - Using frequency-of-frequencies
#' 2. **Kneser-Ney smoothing** - The gold standard for n-gram models
#' 3. **Modified Kneser-Ney** - Enhanced version with count-dependent discounting
#' 4. **Interpolated smoothing** - Combining multiple models
#' 5. **Bayesian smoothing** - Principled probabilistic approach
#'
#' ## Setup

from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import freqprob

# Set up plotting
plt.style.use("default")
sns.set_palette("husl")
np.random.seed(1305)

#' ## Dataset Preparation
#'
#' We'll use a larger, more realistic dataset to demonstrate advanced smoothing methods.
#' The corpus is built from Python's built-in module docstrings, which provide natural
#' language text with realistic frequency distributions.

# Create corpus from built-in Python module docstrings
import re
import sys

# Collect docstrings from sys and re modules
corpus_text = ""

# Add sys module docstrings
corpus_text += sys.__doc__ or ""
for name in dir(sys):
    obj = getattr(sys, name)
    if hasattr(obj, "__doc__") and obj.__doc__ and isinstance(obj.__doc__, str):
        corpus_text += "\n" + obj.__doc__

# Add re module docstrings
corpus_text += "\n" + (re.__doc__ or "")
for name in dir(re):
    obj = getattr(re, name)
    if hasattr(obj, "__doc__") and obj.__doc__ and isinstance(obj.__doc__, str):
        corpus_text += "\n" + obj.__doc__

# Tokenize: simple whitespace splitting and lowercase
all_words = corpus_text.lower().split()

freqdist = Counter(all_words)
print("Corpus statistics:")
print(f"Total tokens: {len(all_words)}")
print(f"Unique words: {len(freqdist)}")
print(f"Average frequency: {len(all_words) / len(freqdist):.2f}")

# Show frequency distribution
print("\nMost common words:")
for word, count in freqdist.most_common(10):
    print(f"{word}: {count}")

# Analyze frequency-of-frequencies (crucial for Good-Turing)
freq_of_freqs = Counter(freqdist.values())
print("\nFrequency-of-frequencies (r -> Nr):")
for r in sorted(freq_of_freqs.keys()):
    print(f"r={r}: {freq_of_freqs[r]} words appear {r} time(s)")

#' ## Simple Good-Turing Smoothing
#'
#' Simple Good-Turing (SGT) is a frequency-based smoothing method that uses frequency-of-frequencies
#' statistics to estimate probabilities. The key insight: if we observed N₁ words that appear once,
#' we should expect about N₁ words we haven't seen yet.
#'
#' ### Understanding Total Mass vs Per-Word Probability
#'
#' Good-Turing estimates p₀, the **total probability mass** for ALL unseen words combined.
#' To get the probability for a single unseen word, we need to divide by the estimated number
#' of unseen types. The `bins` parameter controls this calculation:
#'
#' - **bins**: Total vocabulary size (observed + unobserved types)
#' - **Default**: bins = V + N₁ (observed types + singleton count)
#' - **Per-word unseen probability**: p₀ / (bins - V)
#'
#' This ensures probabilities are meaningful: P(word₁) + P(word₂) makes sense for different words.

# Create Simple Good-Turing model
try:
    # Default behavior: bins = V + N₁ (observed types + singleton count)
    sgt = freqprob.SimpleGoodTuring(freqdist, logprob=False)
    print("Simple Good-Turing model created successfully")
    print(
        f"Default bins: {len(freqdist)} (observed) + {freq_of_freqs.get(1, 0)} (singletons) = {len(freqdist) + freq_of_freqs.get(1, 0)}"
    )

    # Test on sample words from the corpus (sys/re module docstrings)
    # Use common programming terms and one unseen word
    test_words = ["the", "string", "pattern", "xyzabc"]  # Last one is unseen

    print("\nSimple Good-Turing probabilities:")
    for word in test_words:
        original_count = freqdist.get(word, 0)
        sgt_prob = sgt(word)
        status = "unseen" if original_count == 0 else f"count: {original_count}"
        print(f"P({word:<10}) = {sgt_prob:.6f} ({status})")

    # Compare with MLE
    mle_comparison = freqprob.MLE(freqdist, logprob=False)

    print("\nMLE vs Simple Good-Turing comparison:")
    print(f"{'Word':<10} {'MLE':<10} {'SGT':<10} {'Difference':<10}")
    print("-" * 40)
    for word in test_words:
        mle_prob = mle_comparison(word)
        sgt_prob = sgt(word)
        diff = sgt_prob - mle_prob
        print(f"{word:<10} {mle_prob:<10.6f} {sgt_prob:<10.6f} {diff:<10.6f}")

    # Demonstrate total mass vs per-word probability
    print("\n" + "=" * 60)
    print("TOTAL MASS vs PER-WORD PROBABILITY")
    print("=" * 60)

    per_word_prob = sgt("unseen_word")
    total_unseen_mass = sgt.total_unseen_mass
    total_observed_mass = sum(sgt(word) for word in freqdist)

    print(f"\nTotal probability mass (p₀): {total_unseen_mass:.6f}")
    print(f"Per-word unseen probability: {per_word_prob:.6f}")
    print(f"Ratio (total/per-word): {total_unseen_mass / per_word_prob:.0f} estimated unseen types")
    print(f"\nTotal probability for observed words: {total_observed_mass:.6f}")
    print(f"Total probability for all unseen words (p₀): {total_unseen_mass:.6f}")
    print(f"Sum (should be ≈ 1.0): {total_observed_mass + total_unseen_mass:.6f}")

    # Verify probability semantics
    print("\n" + "=" * 60)
    print("VERIFYING PROBABILITY SEMANTICS")
    print("=" * 60)

    # Two different unseen words should have same probability
    unseen1 = sgt("xyzabc")
    unseen2 = sgt("qwerty")
    print(f"\nP(xyzabc) = {unseen1:.6f}")
    print(f"P(qwerty) = {unseen2:.6f}")
    print(f"P(xyzabc) + P(qwerty) = {unseen1 + unseen2:.6f}")
    print("This sum is meaningful because each returns PER-WORD probability")

    # Demonstrate effect of bins parameter
    print("\n" + "=" * 60)
    print("EFFECT OF BINS PARAMETER")
    print("=" * 60)

    bins_values = [
        len(freqdist) + freq_of_freqs.get(1, 0),  # Default: V + N₁
        len(freqdist) * 2,  # 2x observed vocabulary
        len(freqdist) * 5,  # 5x observed vocabulary
        10000,  # Fixed large vocabulary
    ]

    print(f"\nObserved vocabulary size (V): {len(freqdist)}")
    print(f"Singleton count (N₁): {freq_of_freqs.get(1, 0)}")
    print(f"\n{'bins':<10} {'P(unseen)':<12} {'Est. Unseen Types':<20} {'Description':<30}")
    print("-" * 80)

    for bins in bins_values:
        sgt_test = freqprob.SimpleGoodTuring(freqdist, bins=bins, logprob=False)
        unseen_prob = sgt_test("xyzabc")
        est_unseen = int(sgt_test.total_unseen_mass / unseen_prob)

        if bins == bins_values[0]:
            desc = "Default (V + N₁)"
        elif bins == bins_values[1]:
            desc = "2x observed vocabulary"
        elif bins == bins_values[2]:
            desc = "5x observed vocabulary"
        else:
            desc = "Fixed large vocabulary"

        print(f"{bins:<10} {unseen_prob:<12.6f} {est_unseen:<20} {desc:<30}")

    print("\nKey insight: Larger bins → smaller per-word unseen probability")
    print("Choose bins based on your domain knowledge of total vocabulary size")

    # Show compatibility with perplexity calculation
    print("\n" + "=" * 60)
    print("COMPATIBILITY WITH PERPLEXITY")
    print("=" * 60)

    # Create log-probability version for perplexity
    sgt_log = freqprob.SimpleGoodTuring(freqdist, logprob=True)

    # Small test set with words relevant to sys/re documentation
    test_sample = ["the", "string", "module", "function", "xyzabc"]

    print(f"\nTest words: {test_sample}")
    print("\nLog-probabilities:")
    for word in test_sample:
        logprob = sgt_log(word)
        prob = np.exp(logprob)
        status = "(unseen)" if word == "xyzabc" else "(observed)"
        print(f"  {word:<10}: log P = {logprob:8.4f}, P = {prob:.6f} {status}")

    # Calculate perplexity
    perp = freqprob.perplexity(sgt_log, test_sample)
    print(f"\nPerplexity: {perp:.2f}")
    print("Lower perplexity = better model fit to this test data")

except Exception as e:
    print(f"Error creating Simple Good-Turing model: {e}")
    print("This can happen with small datasets or irregular frequency patterns")
    sgt = None

#' ## N-gram Language Models Setup
#'
#' For Kneser-Ney smoothing, we need to work with n-grams. Let's create bigram data.

# Generate bigrams from our corpus
# We'll use sentence boundaries based on punctuation


def generate_bigrams(words):
    """Generate bigrams from a list of words, treating periods as sentence boundaries."""
    bigrams = []
    # Add sentence start marker
    prev_word = "<s>"
    for word in words:
        # Check if this word ends with sentence-ending punctuation
        if word.endswith((".", "!", "?", ":", ";")):
            # Add bigram with current word
            bigrams.append((prev_word, word))
            # Next bigram starts a new sentence
            prev_word = "<s>"
        else:
            bigrams.append((prev_word, word))
            prev_word = word
    # Add final sentence end marker
    bigrams.append((prev_word, "</s>"))
    return bigrams


bigrams = generate_bigrams(all_words)
bigram_freqdist = Counter(bigrams)

print("Bigram statistics:")
print(f"Total bigrams: {len(bigrams)}")
print(f"Unique bigrams: {len(bigram_freqdist)}")

print("\nMost common bigrams:")
for bigram, count in bigram_freqdist.most_common(10):
    print(f"{bigram}: {count}")

# Also create context counts for Kneser-Ney
context_counts = Counter()
word_contexts = defaultdict(set)

for (context, word), count in bigram_freqdist.items():
    context_counts[context] += count
    word_contexts[word].add(context)

print(f"\nNumber of unique contexts: {len(context_counts)}")
print("Most frequent contexts:")
for context, count in context_counts.most_common(5):
    print(f"'{context}': {count}")

#' ## Kneser-Ney Smoothing
#'
#' Kneser-Ney is the gold standard for n-gram language models. It uses absolute discounting and continuation probabilities.

# Create Kneser-Ney model
try:
    # Default discount of 0.75 is commonly used
    kn = freqprob.KneserNey(bigram_freqdist, discount=0.75, logprob=False)
    print("Kneser-Ney model created successfully")

    # Test on various bigrams from programming documentation
    test_bigrams = [
        ("the", "string"),
        ("the", "function"),
        ("in", "the"),
        ("of", "module"),
        ("xyzabc", "qwerty"),  # Unseen bigram
    ]

    print("\nKneser-Ney probabilities:")
    for bigram in test_bigrams:
        original_count = bigram_freqdist.get(bigram, 0)
        kn_prob = kn(bigram)
        print(f"{bigram!s:<20} (count={original_count}): P = {kn_prob:.6f}")

    # Compare with bigram MLE
    bigram_mle = freqprob.MLE(bigram_freqdist, logprob=False)

    observed_bigrams = [bg for bg in test_bigrams if bg in bigram_freqdist]
    kn_probs = [kn(bg) for bg in observed_bigrams]
    mle_probs = [bigram_mle(bg) for bg in observed_bigrams]

    plt.figure(figsize=(12, 6))
    x = np.arange(len(observed_bigrams))
    width = 0.35

    plt.bar(x - width / 2, mle_probs, width, label="Bigram MLE", alpha=0.8)
    plt.bar(x + width / 2, kn_probs, width, label="Kneser-Ney", alpha=0.8)

    plt.title("Bigram MLE vs Kneser-Ney")
    plt.xlabel("Bigrams")
    plt.ylabel("Probability")
    plt.xticks(x, [str(bg) for bg in observed_bigrams], rotation=45)
    plt.legend()
    plt.tight_layout()

    # Demonstrate continuation probability concept
    print("\nContinuation probability insight:")
    words_to_analyze = ["the", "string", "module"]

    for word in words_to_analyze:
        # Count how many different contexts this word appears in
        contexts = len(word_contexts[word])
        total_count = sum(count for (ctx, w), count in bigram_freqdist.items() if w == word)
        print(f"'{word}': appears {total_count} times in {contexts} different contexts")

    print("\nKneser-Ney favors words that appear in many different contexts!")

except Exception as e:
    print(f"Error creating Kneser-Ney model: {e}")
    kn = None

#' ## Modified Kneser-Ney Smoothing
#'
#' Modified Kneser-Ney uses different discount values for different frequency counts.

# Create Modified Kneser-Ney model
try:
    mkn = freqprob.ModifiedKneserNey(bigram_freqdist, logprob=False)
    print("Modified Kneser-Ney model created successfully")

    # Compare KN vs MKN
    if kn is not None:
        print("\nComparison: Kneser-Ney vs Modified Kneser-Ney")
        print("-" * 55)

        for bigram in test_bigrams[:4]:  # Skip unseen for cleaner comparison
            count = bigram_freqdist.get(bigram, 0)
            kn_prob = kn(bigram)
            mkn_prob = mkn(bigram)
            print(f"{bigram!s:<20} (c={count}): KN={kn_prob:.6f}, MKN={mkn_prob:.6f}")

        # Visualize differences
        bigrams_to_plot = [bg for bg in test_bigrams[:4] if bg in bigram_freqdist]
        kn_probs_plot = [kn(bg) for bg in bigrams_to_plot]
        mkn_probs_plot = [mkn(bg) for bg in bigrams_to_plot]

        plt.figure(figsize=(12, 6))
        x = np.arange(len(bigrams_to_plot))
        width = 0.35

        plt.bar(x - width / 2, kn_probs_plot, width, label="Kneser-Ney", alpha=0.8)
        plt.bar(x + width / 2, mkn_probs_plot, width, label="Modified Kneser-Ney", alpha=0.8)

        plt.title("Kneser-Ney vs Modified Kneser-Ney")
        plt.xlabel("Bigrams")
        plt.ylabel("Probability")
        plt.xticks(x, [str(bg) for bg in bigrams_to_plot], rotation=45)
        plt.legend()
        plt.tight_layout()

        # Show discount values used by MKN
        print("\nModified Kneser-Ney uses count-dependent discounts:")
        print("- Different discount values for counts 1, 2, and 3+")
        print("- Automatically estimated from frequency-of-frequencies")

    else:
        print("Regular Kneser-Ney not available for comparison")

except Exception as e:
    print(f"Error creating Modified Kneser-Ney model: {e}")
    mkn = None

#' ## Interpolated Smoothing
#'
#' Interpolated smoothing combines multiple models (e.g., trigram with bigram fallback).
#' The method automatically detects n-gram interpolation mode when distributions have
#' different tuple lengths, extracting lower-order context from higher-order n-grams.

# Generate trigrams for interpolation example


def generate_trigrams(words):
    """Generate trigrams from a list of words, treating periods as sentence boundaries."""
    trigrams = []
    # Start with two sentence boundary markers
    context = ["<s>", "<s>"]

    for word in words:
        # Add the trigram
        trigrams.append((context[0], context[1], word))

        # Check if this word ends with sentence-ending punctuation
        # Reset context for new sentence, or shift context window
        context = ["<s>", "<s>"] if word.endswith((".", "!", "?", ":", ";")) else [context[1], word]

    # Add final trigrams with sentence end markers
    trigrams.append((context[0], context[1], "</s>"))
    trigrams.append((context[1], "</s>", "</s>"))

    return trigrams


trigrams = generate_trigrams(all_words)
trigram_freqdist = Counter(trigrams)

print("Trigram statistics:")
print(f"Total trigrams: {len(trigrams)}")
print(f"Unique trigrams: {len(trigram_freqdist)}")

print("\nMost common trigrams:")
for trigram, count in trigram_freqdist.most_common(5):
    print(f"{trigram}: {count}")

# Create interpolated model (trigram + bigram)
try:
    # Lambda weight controls interpolation: λ * P_high + (1-λ) * P_low
    lambda_weight = 0.7  # Favor trigrams

    interpolated = freqprob.InterpolatedSmoothing(
        trigram_freqdist, bigram_freqdist, lambda_weight=lambda_weight, logprob=False
    )

    print(f"\nInterpolated model created (λ = {lambda_weight})")

    # Test on trigrams from programming documentation
    test_trigrams = [
        ("the", "string", "object"),
        ("in", "the", "module"),
        ("<s>", "the", "function"),
        ("pattern", "in", "the"),
    ]

    print("\nInterpolated smoothing probabilities:")
    print("(Using n-gram mode: extracts bigram context from trigrams)")
    for trigram in test_trigrams:
        count = trigram_freqdist.get(trigram, 0)
        prob = interpolated(trigram)
        # Extract bigram context for display
        bigram_context = (trigram[1], trigram[2])
        bigram_count = bigram_freqdist.get(bigram_context, 0)
        print(
            f"{trigram!s:<25} (count={count}, context {bigram_context} count={bigram_count}): P = {prob:.6f}"
        )

    # Compare different lambda values
    lambda_values = [0.1, 0.3, 0.5, 0.7, 0.9]
    test_trigram = ("in", "the", "module")

    probs_by_lambda = []
    for lam in lambda_values:
        interp_model = freqprob.InterpolatedSmoothing(
            trigram_freqdist, bigram_freqdist, lambda_weight=lam, logprob=False
        )
        prob = interp_model(test_trigram)
        probs_by_lambda.append(prob)

    plt.figure(figsize=(10, 6))
    plt.plot(lambda_values, probs_by_lambda, "o-", linewidth=2, markersize=8)
    plt.title(f"Effect of lambda on P{test_trigram}")
    plt.xlabel("Lambda (lambda) - Weight for Higher-order Model")
    plt.ylabel("Probability")
    plt.grid(True, alpha=0.3)
    plt.axvline(
        x=lambda_weight,
        color="red",
        linestyle="--",
        alpha=0.7,
        label=f"Current lambda = {lambda_weight}",
    )
    plt.legend()
    plt.tight_layout()

    print("\nAs lambda increases, we rely more on trigram model (more specific context)")
    print("As lambda decreases, we rely more on bigram model (more general, better coverage)")

except Exception as e:
    print(f"Error creating interpolated model: {e}")
    interpolated = None

#' ## Bayesian Smoothing
#'
#' Bayesian smoothing uses a Dirichlet prior for theoretically principled probability estimates.

# Create Bayesian smoothing models with different priors
alpha_values = [0.1, 0.5, 1.0, 2.0, 5.0]
bayesian_models = {}

for alpha in alpha_values:
    bayesian_models[alpha] = freqprob.BayesianSmoothing(freqdist, alpha=alpha, logprob=False)

print("Bayesian Smoothing with different alpha (concentration parameters):")
print("=" * 65)

test_words = ["the", "string", "module", "xyzabc"]  # Last one unseen

for word in test_words:
    count = freqdist.get(word, 0)
    print(f"\nWord: '{word}' (count = {count})")
    print(f"{'alpha':<6} {'Probability':<12} {'Effect':<20}")
    print("-" * 40)

    for alpha in alpha_values:
        prob = bayesian_models[alpha](word)
        if alpha == 0.1:
            effect = "Minimal smoothing"
        elif alpha == 1.0:
            effect = "Uniform prior (Laplace)"
        elif alpha > 1.0:
            effect = "Strong uniform bias"
        else:
            effect = "Light smoothing"

        print(f"{alpha:<6.1f} {prob:<12.6f} {effect}")

# Visualize the effect of alpha
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
axes = axes.ravel()

for i, word in enumerate(test_words):
    if i >= 4:
        break

    probs = [bayesian_models[alpha](word) for alpha in alpha_values]

    axes[i].semilogx(alpha_values, probs, "o-", linewidth=2, markersize=8)
    axes[i].set_title(f'P("{word}") vs alpha (count = {freqdist.get(word, 0)})')
    axes[i].set_xlabel("Alpha (alpha)")
    axes[i].set_ylabel("Probability")
    axes[i].grid(True, alpha=0.3)

plt.tight_layout()

# Compare Bayesian with other methods
print("\nComparison with other smoothing methods:")
print("=" * 45)

# Create comparison models
laplace_model = freqprob.Laplace(freqdist, bins=1000, logprob=False)
bayesian_model = bayesian_models[1.0]  # alpha = 1 is equivalent to Laplace
optimal_bayesian = bayesian_models[0.5]  # Often a good choice

comparison_models = {
    "Laplace": laplace_model,
    "Bayesian (alpha=1.0)": bayesian_model,
    "Bayesian (alpha=0.5)": optimal_bayesian,
}

for word in ["the", "string", "xyzabc"]:
    print(f"\nP('{word}'):")
    for name, model in comparison_models.items():
        prob = model(word)
        print(f"  {name:<18}: {prob:.6f}")

print("\nKey insight: Bayesian smoothing with alpha=1.0 is equivalent to Laplace!")
print("The alpha parameter controls the strength of the uniform prior.")

#' # Find best models
#' if unigram_perplexities:
#'     best_unigram = min(unigram_perplexities.items(), key=lambda x: x[1])
#'     print(f"\nBest unigram model: {best_unigram[0]} (PP = {best_unigram[1]:.2f})")
#'
#' if bigram_perplexities:
#'     best_bigram = min(bigram_perplexities.items(), key=lambda x: x[1])
#'     print(f"Best bigram model: {best_bigram[0]} (PP = {best_bigram[1]:.2f})")

# Create test set
test_corpus = [
    "the function returns a string object from the module",
    "pattern matching uses regular expressions for text processing",
    "the module provides methods for searching and replacing strings",
]

test_words = []
for sentence in test_corpus:
    test_words.extend(sentence.split())

print(f"Test set: {len(test_words)} words")
print(f"Words: {test_words}")

# Evaluate unigram models
print("\nUnigram Model Evaluation (Perplexity):")
print("=" * 40)

unigram_models = {
    "MLE": freqprob.MLE(freqdist, logprob=True),
    "Laplace": freqprob.Laplace(freqdist, bins=1000, logprob=True),
    "Bayesian (alpha=0.5)": freqprob.BayesianSmoothing(freqdist, alpha=0.5, logprob=True),
}

if sgt is not None:
    unigram_models["Simple Good-Turing"] = freqprob.SimpleGoodTuring(freqdist, logprob=True)

unigram_perplexities = {}
for name, model in unigram_models.items():
    try:
        pp = freqprob.perplexity(model, test_words)
        unigram_perplexities[name] = pp
        print(f"{name:<20}: {pp:.2f}")
    except Exception as e:
        print(f"{name:<20}: Error - {str(e)[:30]}...")

# Evaluate bigram models (on bigram test data)
test_bigrams = []
for sentence in test_corpus:
    words = ["<s>", *sentence.split(), "</s>"]
    for i in range(len(words) - 1):
        test_bigrams.append((words[i], words[i + 1]))

print("\nBigram Model Evaluation (Perplexity):")
print("=" * 40)

bigram_models = {"Bigram MLE": freqprob.MLE(bigram_freqdist, logprob=True)}

if kn is not None:
    bigram_models["Kneser-Ney"] = freqprob.KneserNey(bigram_freqdist, discount=0.75, logprob=True)

if mkn is not None:
    bigram_models["Modified Kneser-Ney"] = freqprob.ModifiedKneserNey(bigram_freqdist, logprob=True)

bigram_perplexities = {}
for name, model in bigram_models.items():
    try:
        pp = freqprob.perplexity(model, test_bigrams)
        bigram_perplexities[name] = pp
        print(f"{name:<20}: {pp:.2f}")
    except Exception as e:
        print(f"{name:<20}: Error - {str(e)[:30]}...")

# Visualize results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Unigram models
if unigram_perplexities:
    methods = list(unigram_perplexities.keys())
    values = list(unigram_perplexities.values())

    bars1 = ax1.bar(methods, values, alpha=0.8, color=plt.cm.Set3(np.linspace(0, 1, len(methods))))
    ax1.set_title("Unigram Model Perplexity Comparison")
    ax1.set_ylabel("Perplexity (lower is better)")
    ax1.tick_params(axis="x", rotation=45)

    # Add value labels
    for bar, value in zip(bars1, values, strict=False):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{value:.1f}",
            ha="center",
            va="bottom",
        )

# Bigram models
if bigram_perplexities:
    methods = list(bigram_perplexities.keys())
    values = list(bigram_perplexities.values())

    bars2 = ax2.bar(methods, values, alpha=0.8, color=plt.cm.Set2(np.linspace(0, 1, len(methods))))
    ax2.set_title("Bigram Model Perplexity Comparison")
    ax2.set_ylabel("Perplexity (lower is better)")
    ax2.tick_params(axis="x", rotation=45)

    # Add value labels
    for bar, value in zip(bars2, values, strict=False):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{value:.1f}",
            ha="center",
            va="bottom",
        )

plt.tight_layout()

# Find best models
if unigram_perplexities:
    best_unigram = min(unigram_perplexities.items(), key=lambda x: x[1])
    print(f"\nBest unigram model: {best_unigram[0]} (PP = {best_unigram[1]:.2f})")

if bigram_perplexities:
    best_bigram = min(bigram_perplexities.items(), key=lambda x: x[1])
    print(f"Best bigram model: {best_bigram[0]} (PP = {best_bigram[1]:.2f})")

# Show computational complexity
print("⚡ COMPUTATIONAL COMPLEXITY:")
print("=" * 28)
methods_complexity = {
    "Laplace/Bayesian": "O(1) per query",
    "Simple Good-Turing": "O(V) preprocessing, O(1) query",
    "Kneser-Ney": "O(N) preprocessing, O(1) query",
    "Modified Kneser-Ney": "O(N) preprocessing, O(1) query",
    "Interpolated": "O(k) per query (k models)",
}

for method, complexity in methods_complexity.items():
    print(f"  {method:<18}: {complexity}")

print("\n  V = vocabulary size, N = total n-grams, k = number of models")

#' ADVANCED SMOOTHING METHODS: KEY INSIGHTS
#'
#' 🔵 SIMPLE GOOD-TURING:
#'   • Uses frequency-of-frequencies statistics
#'   • Returns per-word probability (divides p₀ by estimated unseen types)
#'   • bins parameter controls vocabulary size estimate (default: V + N₁)
#'   • total_unseen_mass property provides access to p₀
#'   • Works well when frequency patterns are reliable
#'   • Can fail with sparse data or irregular patterns
#'
#' 🟢 KNESER-NEY:
#'   • Gold standard for n-gram language models
#'   • Uses absolute discounting (subtract fixed amount)
#'   • Continuation probability: how likely is word in new contexts?
#'   • Particularly effective for bigrams and trigrams
#'
#' 🟡 MODIFIED KNESER-NEY:
#'   • Enhanced version of Kneser-Ney
#'   • Different discount values for different frequency counts
#'   • Automatically estimates discounts from data
#'   • Generally performs better than standard Kneser-Ney
#'
#' 🔴 INTERPOLATED SMOOTHING:
#'   • Combines multiple models (e.g., trigram + bigram)
#'   • Automatic n-gram mode: extracts lower-order context from higher-order n-grams
#'   • Linear interpolation: lambda*P_high(ngram) + (1-lambda)*P_low(context)
#'   • Balances specificity with coverage
#'   • Essential for practical n-gram systems
#'
#' 🟣 BAYESIAN SMOOTHING:
#'   • Theoretically principled using Dirichlet prior
#'   • alpha parameter controls prior strength
#'   • alpha=1.0 equivalent to Laplace smoothing
#'   • Good theoretical foundation
#'
#' 🎯 PRACTICAL RECOMMENDATIONS:
#'
#' For Language Modeling:
#'  1. Start with Modified Kneser-Ney for n-grams
#'  2. Use interpolation for robustness
#'  3. Consider neural models for large datasets
#'
#' For General Frequency Estimation:
#'   1. Try Simple Good-Turing first
#'   2. Fall back to Bayesian smoothing if SGT fails
#'   3. Tune alpha parameter using validation data
#'
#' For Production Systems:
#'   1. Use interpolated smoothing for robustness
#'   2. Consider computational costs
#'   3. Validate on domain-specific data
