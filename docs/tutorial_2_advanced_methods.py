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
np.random.seed(42)

print("Advanced Smoothing Methods Tutorial")
print("===================================")

#' ## Dataset Preparation
#'
#' We'll use a larger, more realistic dataset to demonstrate advanced smoothing methods.

# Create a more substantial corpus for realistic frequency patterns
corpus = [
    "the quick brown fox jumps over the lazy dog",
    "a dog runs fast in the park every morning",
    "the cat sits quietly on the warm windowsill",
    "brown bears roam freely in the dense forest",
    "quick movements help animals escape from predators",
    "lazy cats sleep most of the day in sunny spots",
    "the forest contains many different species of animals",
    "fast cars drive on the highway during rush hour",
    "morning light filters through the trees in the forest",
    "animals in the wild must find food and shelter",
    "the dog barks loudly when strangers approach the house",
    "cats and dogs are popular pets in many households",
    "sunny weather brings people outdoors to enjoy nature",
    "dense fog covers the mountains in the early morning",
    "species diversity is important for ecosystem health",
]

# Create word frequency distribution
all_words = []
for sentence in corpus:
    all_words.extend(sentence.split())

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
#' Good-Turing uses the frequency-of-frequencies to estimate probabilities for unseen events.

# Create Simple Good-Turing model
try:
    sgt = freqprob.SimpleGoodTuring(freqdist, logprob=False)
    print("Simple Good-Turing model created successfully")

    # Test on sample words
    test_words = ["the", "cat", "forest", "elephant"]  # Last one is unseen

    print("Simple Good-Turing probabilities:")
    for word in test_words:
        original_count = freqdist.get(word, 0)
        sgt_prob = sgt(word)
        print(f"P({word}) = {sgt_prob:.6f} (original count: {original_count})")

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

    # Show probability mass for unseen events
    unseen_prob = sgt("unseen_word")
    total_observed_mass = sum(sgt(word) for word in freqdist)
    print(f"\nProbability for unseen words: {unseen_prob:.6f}")
    print(f"Total probability mass for observed words: {total_observed_mass:.4f}")
    print(f"Reserved mass for unseen events: {1 - total_observed_mass:.4f}")

except Exception as e:
    print(f"Error creating Simple Good-Turing model: {e}")
    print("This can happen with small datasets or irregular frequency patterns")
    sgt = None

#' ## N-gram Language Models Setup
#'
#' For Kneser-Ney smoothing, we need to work with n-grams. Let's create bigram data.

# Generate bigrams from our corpus


def generate_bigrams(text_corpus):
    """Generate bigrams with sentence boundaries."""
    bigrams = []
    for sentence in text_corpus:
        words = ["<s>", *sentence.split(), "</s>"]  # Add sentence boundaries
        for i in range(len(words) - 1):
            bigrams.append((words[i], words[i + 1]))
    return bigrams


bigrams = generate_bigrams(corpus)
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

    # Test on various bigrams
    test_bigrams = [
        ("the", "cat"),
        ("the", "dog"),
        ("in", "the"),
        ("of", "animals"),
        ("elephant", "runs"),  # Unseen bigram
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
    words_to_analyze = ["the", "cat", "forest"]

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

# Generate trigrams for interpolation example


def generate_trigrams(text_corpus):
    """Generate trigrams with sentence boundaries."""
    trigrams = []
    for sentence in text_corpus:
        words = ["<s>", "<s>", *sentence.split(), "</s>", "</s>"]
        for i in range(len(words) - 2):
            trigrams.append((words[i], words[i + 1], words[i + 2]))
    return trigrams


trigrams = generate_trigrams(corpus)
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

    # Test on trigrams
    test_trigrams = [
        ("the", "cat", "sits"),
        ("in", "the", "forest"),
        ("<s>", "the", "quick"),
        ("animals", "in", "the"),
    ]

    print("\nInterpolated smoothing probabilities:")
    for trigram in test_trigrams:
        count = trigram_freqdist.get(trigram, 0)
        prob = interpolated(trigram)
        print(f"{trigram!s:<25} (count={count}): P = {prob:.6f}")

    # Compare different lambda values
    lambda_values = [0.1, 0.3, 0.5, 0.7, 0.9]
    test_trigram = ("in", "the", "forest")

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

test_words = ["the", "cat", "forest", "elephant"]  # Last one unseen

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

for word in ["the", "cat", "elephant"]:
    print(f"\nP('{word}'):")
    for name, model in comparison_models.items():
        prob = model(word)
        print(f"  {name:<18}: {prob:.6f}")

print("\nKey insight: Bayesian smoothing with alpha=1.0 is equivalent to Laplace!")
print("The alpha parameter controls the strength of the uniform prior.")

#' # Create test set
#' test_corpus = [
#'     "the elephant walks slowly through the dense jungle",
#'     "wild animals search for food in the morning",
#'     "cats climb trees to escape from dangerous predators",
#' ]
#'
#' test_words = []
#' for sentence in test_corpus:
#'     test_words.extend(sentence.split())
#'
#' print(f"Test set: {len(test_words)} words")
#' print(f"Words: {test_words}")
#'
#' # Evaluate unigram models
#' print("\nUnigram Model Evaluation (Perplexity):")
#' print("=" * 40)
#'
#' unigram_models = {
#'     "MLE": freqprob.MLE(freqdist, logprob=True),
#'     "Laplace": freqprob.Laplace(freqdist, bins=1000, logprob=True),
#'     "Bayesian (alpha=0.5)": freqprob.BayesianSmoothing(freqdist, alpha=0.5, logprob=True),
#' }
#'
#' if sgt is not None:
#'     unigram_models["Simple Good-Turing"] = freqprob.SimpleGoodTuring(freqdist, logprob=True)
#'
#' unigram_perplexities = {}
#' for name, model in unigram_models.items():
#'     try:
#'         pp = freqprob.perplexity(model, test_words)
#'         unigram_perplexities[name] = pp
#'         print(f"{name:<20}: {pp:.2f}")
#'     except Exception as e:
#'         print(f"{name:<20}: Error - {str(e)[:30]}...")
#'
#' # Evaluate bigram models (on bigram test data)
#' test_bigrams = []
#' for sentence in test_corpus:
#'     words = ["<s>", *sentence.split(), "</s>"]
#'     for i in range(len(words) - 1):
#'         test_bigrams.append((words[i], words[i + 1]))
#'
#' print("\nBigram Model Evaluation (Perplexity):")
#' print("=" * 40)
#'
#' bigram_models = {"Bigram MLE": freqprob.MLE(bigram_freqdist, logprob=True)}
#'
#' if kn is not None:
#'     bigram_models["Kneser-Ney"] = freqprob.KneserNey(bigram_freqdist, discount=0.75, logprob=True)
#'
#' if mkn is not None:
#'     bigram_models["Modified Kneser-Ney"] = freqprob.ModifiedKneserNey(bigram_freqdist, logprob=True)
#'
#' bigram_perplexities = {}
#' for name, model in bigram_models.items():
#'     try:
#'         pp = freqprob.perplexity(model, test_bigrams)
#'         bigram_perplexities[name] = pp
#'         print(f"{name:<20}: {pp:.2f}")
#'     except Exception as e:
#'         print(f"{name:<20}: Error - {str(e)[:30]}...")
#'
#' # Visualize results
#' fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
#'
#' # Unigram models
#' if unigram_perplexities:
#'     methods = list(unigram_perplexities.keys())
#'     values = list(unigram_perplexities.values())
#'
#'     bars1 = ax1.bar(methods, values, alpha=0.8, color=plt.cm.Set3(np.linspace(0, 1, len(methods))))
#'     ax1.set_title("Unigram Model Perplexity Comparison")
#'     ax1.set_ylabel("Perplexity (lower is better)")
#'     ax1.tick_params(axis="x", rotation=45)
#'
#'     # Add value labels
#'     for bar, value in zip(bars1, values, strict=False):
#'         ax1.text(
#'             bar.get_x() + bar.get_width() / 2,
#'             bar.get_height() + 0.5,
#'             f"{value:.1f}",
#'             ha="center",
#'             va="bottom",
#'         )
#'
#' # Bigram models
#' if bigram_perplexities:
#'     methods = list(bigram_perplexities.keys())
#'     values = list(bigram_perplexities.values())
#'
#'     bars2 = ax2.bar(methods, values, alpha=0.8, color=plt.cm.Set2(np.linspace(0, 1, len(methods))))
#'     ax2.set_title("Bigram Model Perplexity Comparison")
#'     ax2.set_ylabel("Perplexity (lower is better)")
#'     ax2.tick_params(axis="x", rotation=45)
#'
#'     # Add value labels
#'     for bar, value in zip(bars2, values, strict=False):
#'         ax2.text(
#'             bar.get_x() + bar.get_width() / 2,
#'             bar.get_height() + 0.5,
#'             f"{value:.1f}",
#'             ha="center",
#'             va="bottom",
#'         )
#'
#' plt.tight_layout()
#'
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
    "the elephant walks slowly through the dense jungle",
    "wild animals search for food in the morning",
    "cats climb trees to escape from dangerous predators",
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

print("ADVANCED SMOOTHING METHODS: KEY INSIGHTS")
print("=" * 50)
print()

print("🔵 SIMPLE GOOD-TURING:")
print("   • Uses frequency-of-frequencies statistics")
print("   • Estimates probability of unseen events from singletons")
print("   • Works well when frequency patterns are reliable")
print("   • Can fail with sparse data or irregular patterns")
print()

print("🟢 KNESER-NEY:")
print("   • Gold standard for n-gram language models")
print("   • Uses absolute discounting (subtract fixed amount)")
print("   • Continuation probability: how likely is word in new contexts?")
print("   • Particularly effective for bigrams and trigrams")
print()

print("🟡 MODIFIED KNESER-NEY:")
print("   • Enhanced version of Kneser-Ney")
print("   • Different discount values for different frequency counts")
print("   • Automatically estimates discounts from data")
print("   • Generally performs better than standard Kneser-Ney")
print()

print("🔴 INTERPOLATED SMOOTHING:")
print("   • Combines multiple models (e.g., trigram + bigram)")
print("   • Linear interpolation: lambda*P_high + (1-lambda)*P_low")
print("   • Balances specificity with coverage")
print("   • Essential for practical n-gram systems")
print()

print("🟣 BAYESIAN SMOOTHING:")
print("   • Theoretically principled using Dirichlet prior")
print("   • alpha parameter controls prior strength")
print("   • alpha=1.0 equivalent to Laplace smoothing")
print("   • Good theoretical foundation")
print()

# Practical recommendations
print("🎯 PRACTICAL RECOMMENDATIONS:")
print("=" * 30)
print()

print("For Language Modeling:")
print("  1. Start with Modified Kneser-Ney for n-grams")
print("  2. Use interpolation for robustness")
print("  3. Consider neural models for large datasets")
print()

print("For General Frequency Estimation:")
print("  1. Try Simple Good-Turing first")
print("  2. Fall back to Bayesian smoothing if SGT fails")
print("  3. Tune alpha parameter using validation data")
print()

print("For Production Systems:")
print("  1. Use interpolated smoothing for robustness")
print("  2. Consider computational costs")
print("  3. Validate on domain-specific data")
print()

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

#' ## Exercise: Advanced Method Selection
#'
#' Practice choosing the right advanced method for different scenarios.

#' ## Summary
#'
#' In this tutorial, you learned about advanced smoothing methods that are essential for modern NLP:
#'
#' ### Methods Covered:
#' 1. **Simple Good-Turing** - Uses frequency-of-frequencies for principled probability estimation
#' 2. **Kneser-Ney** - The gold standard for n-gram language models with continuation probabilities
#' 3. **Modified Kneser-Ney** - Enhanced version with count-dependent discounting
#' 4. **Interpolated Smoothing** - Combines multiple models for robustness
#' 5. **Bayesian Smoothing** - Theoretically principled approach with Dirichlet priors
#'
#' ### Key Insights:
#' - **Kneser-Ney** dominates for n-gram language modeling
#' - **Good-Turing** provides theoretical foundation for unseen event estimation
#' - **Interpolation** is crucial for practical systems
#' - **Bayesian methods** offer principled parameter control
#' - **Context matters** - different methods excel in different scenarios
#'
#' ### Next Steps:
#' - **Tutorial 3**: Computational Efficiency and Memory Management
#' - **Tutorial 4**: Real-world Applications and Case Studies
#' - Practice implementing these methods on your own datasets
#' - Experiment with hyperparameter tuning
#'
#' **Remember**: The best method depends on your specific use case, data characteristics, and computational constraints!
