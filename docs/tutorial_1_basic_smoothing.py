#' # FreqProb Tutorial 1: Basic Smoothing Methods
#'
#' This tutorial introduces the fundamental smoothing methods in FreqProb. You'll learn how to:
#'
#' 1. Create frequency distributions from text data
#' 2. Apply different smoothing techniques
#' 3. Compare model performance
#' 4. Handle unseen elements
#'
#' ## Setup
#'
#' First, let's import the necessary libraries and set up our environment.

# | hide
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import freqprob

# Set style for better plots
plt.style.use("default")
sns.set_palette("husl")

print(f"FreqProb version: {freqprob.__version__ if hasattr(freqprob, '__version__') else 'dev'}")
# |

#' ## Creating a Frequency Distribution
#'
#' Let's start with a simple text corpus and create a frequency distribution.

# Sample text corpus
corpus = [
    "the cat sat on the mat",
    "the dog ran in the park",
    "a cat and a dog played in the garden",
    "the cat likes the warm sun",
    "dogs and cats are pets",
]

# Create word frequency distribution
all_words = []
for sentence in corpus:
    all_words.extend(sentence.split())

freqdist = Counter(all_words)
print("Frequency Distribution:")
for word, count in freqdist.most_common():
    print(f"{word}: {count}")

print(f"\nTotal words: {sum(freqdist.values())}")
print(f"Unique words: {len(freqdist)}")

#' ## Maximum Likelihood Estimation (MLE)
#'
#' Let's start with the simplest approach - Maximum Likelihood Estimation.

# Create MLE model
mle = freqprob.MLE(freqdist, logprob=False)

print("MLE Probabilities:")
test_words = ["the", "cat", "dog", "garden", "unknown_word"]
for word in test_words:
    prob = mle(word)
    print(f"P({word}) = {prob:.4f}")

# Visualize the distribution
words = list(freqdist.keys())
probs = [mle(word) for word in words]

plt.figure(figsize=(12, 6))
plt.bar(words, probs)
plt.title("MLE Probability Distribution")
plt.xlabel("Words")
plt.ylabel("Probability")
plt.xticks(rotation=45)
plt.tight_layout()

print(f"\nProbability sum: {sum(probs):.4f}")
print(f"Problem: P(unknown_word) = {mle('unknown_word'):.4f} (zero probability!)")

#' ## Laplace Smoothing (Add-One)
#'
#' Laplace smoothing solves the zero probability problem by adding 1 to all counts.

# Create Laplace smoothing model
# We need to specify the number of possible words (bins)
vocabulary_size = 1000  # Assume a vocabulary of 1000 possible words
laplace = freqprob.Laplace(freqdist, bins=vocabulary_size, logprob=False)

print("Laplace Smoothing Probabilities:")
for word in test_words:
    prob = laplace(word)
    print(f"P({word}) = {prob:.6f}")

# Compare MLE vs Laplace
observed_words = list(freqdist.keys())
mle_probs = [mle(word) for word in observed_words]
laplace_probs = [laplace(word) for word in observed_words]

plt.figure(figsize=(12, 6))
x = np.arange(len(observed_words))
width = 0.35

plt.bar(x - width / 2, mle_probs, width, label="MLE", alpha=0.8)
plt.bar(x + width / 2, laplace_probs, width, label="Laplace", alpha=0.8)

plt.title("MLE vs Laplace Smoothing")
plt.xlabel("Words")
plt.ylabel("Probability")
plt.xticks(x, observed_words, rotation=45)
plt.legend()
plt.tight_layout()

print(f"\nUnseen word probability (Laplace): {laplace('unknown_word'):.6f}")
print(f"Unseen word probability (MLE): {mle('unknown_word'):.6f}")

#' ## Lidstone Smoothing (Add-k)
#'
#' Lidstone smoothing is a generalization of Laplace smoothing where we can adjust the smoothing parameter.

# Test different gamma values for Lidstone smoothing
gamma_list = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0]
common_word = "the"
unseen_word = "unknown_word"

# Create Lidstone models with different gamma values
lidstone_models = {}
common_probs = []
unseen_probs = []

for gamma in gamma_list:
    model = freqprob.Lidstone(freqdist, gamma=gamma, bins=vocabulary_size, logprob=False)
    lidstone_models[gamma] = model

    # Track probabilities for plotting
    common_probs.append(model(common_word))
    unseen_probs.append(model(unseen_word))

print("Lidstone Smoothing with Different Gamma Values:")
for gamma in gamma_list:
    model = lidstone_models[gamma]
    print(
        f"gamma = {gamma}: P({common_word}) = {model(common_word):.6f}, P({unseen_word}) = {model(unseen_word):.6f}"
    )

# Plot the effect of gamma on probabilities
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.semilogx(gamma_list, common_probs, "o-", linewidth=2, markersize=8)
plt.title(f'Effect of gamma on P("{common_word}")')
plt.xlabel("Gamma (gamma)")
plt.ylabel("Probability")
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.semilogx(gamma_list, unseen_probs, "o-", linewidth=2, markersize=8, color="orange")
plt.title(f'Effect of gamma on P("{unseen_word}")')
plt.xlabel("Gamma (gamma)")
plt.ylabel("Probability")
plt.grid(True, alpha=0.3)

plt.tight_layout()

print("\nObservation: As gamma increases, probabilities become more uniform")

#' ## Expected Likelihood Estimation (ELE)
#'
#' ELE is a special case of Lidstone smoothing with γ = 0.5, which has theoretical justification.

# Create ELE model
ele = freqprob.ELE(freqdist, bins=vocabulary_size, logprob=False)

print("Expected Likelihood Estimation (gamma = 0.5):")
for word in test_words:
    prob = ele(word)
    print(f"P({word}) = {prob:.6f}")

# Compare all methods
methods = {"MLE": mle, "Laplace": laplace, "ELE": ele, "Lidstone (gamma=0.1)": lidstone_models[0.1]}

comparison_words = ["the", "cat", "garden", "unknown_word"]

plt.figure(figsize=(14, 8))
method_names = list(methods.keys())
word_positions = np.arange(len(comparison_words))
bar_width = 0.2

for i, (method_name, model) in enumerate(methods.items()):
    probs = [model(word) for word in comparison_words]
    offset = (i - len(methods) / 2 + 0.5) * bar_width
    plt.bar(word_positions + offset, probs, bar_width, label=method_name, alpha=0.8)

plt.title("Comparison of Basic Smoothing Methods")
plt.xlabel("Words")
plt.ylabel("Probability")
plt.xticks(word_positions, comparison_words)
plt.legend()
plt.yscale("log")  # Log scale to see differences better
plt.grid(True, alpha=0.3)
plt.tight_layout()

#' ## Model Evaluation with Perplexity
#'
#' Let's evaluate our models using perplexity on a held-out test set.

# Create a test set
test_corpus = ["the cat sleeps on the sofa", "a dog barks in the yard", "cats and dogs are friends"]

test_words = []
for sentence in test_corpus:
    test_words.extend(sentence.split())

print(f"Test set: {test_words}")
print(f"Test set size: {len(test_words)} words")

# Evaluate models using perplexity
print("\nModel Evaluation (Perplexity):")
print("Lower perplexity = better model")
print("-" * 40)

perplexities = {}
for method_name, model in methods.items():
    # Convert to log probabilities for perplexity calculation
    if hasattr(model, "logprob") and not model.logprob:
        # Create log version
        if method_name == "MLE":
            log_model = freqprob.MLE(freqdist, logprob=True)
        elif method_name == "Laplace":
            log_model = freqprob.Laplace(freqdist, bins=vocabulary_size, logprob=True)
        elif method_name == "ELE":
            log_model = freqprob.ELE(freqdist, bins=vocabulary_size, logprob=True)
        elif "Lidstone" in method_name:
            log_model = freqprob.Lidstone(freqdist, gamma=0.1, bins=vocabulary_size, logprob=True)

        try:
            pp = freqprob.perplexity(log_model, test_words)
            perplexities[method_name] = pp
            print(f"{method_name:<15}: {pp:.2f}")
        except Exception as e:
            print(f"{method_name:<15}: Error - {e!s}")

# Visualize perplexity comparison
if perplexities:
    plt.figure(figsize=(10, 6))
    methods_list = list(perplexities.keys())
    pp_values = list(perplexities.values())

    colors = plt.cm.Set3(np.linspace(0, 1, len(methods_list)))
    bars = plt.bar(methods_list, pp_values, color=colors, alpha=0.8)

    plt.title("Model Comparison: Perplexity on Test Set")
    plt.xlabel("Smoothing Method")
    plt.ylabel("Perplexity (lower is better)")
    plt.xticks(rotation=45)

    # Add value labels on bars
    for bar, value in zip(bars, pp_values, strict=False):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.1,
            f"{value:.1f}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()

    best_method = min(perplexities.items(), key=lambda x: x[1])
    print(f"\nBest performing method: {best_method[0]} (PP = {best_method[1]:.2f})")

#' ## Understanding the Trade-offs
#'
#' Let's visualize how different smoothing methods handle the trade-off between fitting training data and generalizing to unseen data.

# Analyze the probability mass allocated to unseen events
total_prob_mass = {}
unseen_prob_mass = {}

# Calculate total probability mass for observed words
observed_words = list(freqdist.keys())

for method_name, model in methods.items():
    observed_mass = sum(model(word) for word in observed_words)
    unseen_mass = model("unseen_word_example")  # Probability for any unseen word

    total_prob_mass[method_name] = observed_mass
    unseen_prob_mass[method_name] = unseen_mass

    print(f"{method_name}:")
    print(f"  Probability mass for observed words: {observed_mass:.4f}")
    print(f"  Probability for unseen words: {unseen_mass:.6f}")
    print(f"  Reserved mass for unseen events: {1 - observed_mass:.4f}")
    print()

# Visualize the trade-off
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Plot 1: Probability mass for observed vs unseen
methods_list = list(total_prob_mass.keys())
observed_masses = [total_prob_mass[m] for m in methods_list]
unseen_masses = [1 - total_prob_mass[m] for m in methods_list]

x = np.arange(len(methods_list))
ax1.bar(x, observed_masses, label="Observed words", alpha=0.8)
ax1.bar(x, unseen_masses, bottom=observed_masses, label="Reserved for unseen", alpha=0.8)

ax1.set_title("Probability Mass Distribution")
ax1.set_xlabel("Smoothing Method")
ax1.set_ylabel("Probability Mass")
ax1.set_xticks(x)
ax1.set_xticklabels(methods_list, rotation=45)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Individual unseen word probabilities
unseen_probs = [unseen_prob_mass[m] for m in methods_list]
bars = ax2.bar(x, unseen_probs, alpha=0.8, color="orange")

ax2.set_title("Probability for Individual Unseen Words")
ax2.set_xlabel("Smoothing Method")
ax2.set_ylabel("Probability")
ax2.set_xticks(x)
ax2.set_xticklabels(methods_list, rotation=45)
ax2.set_yscale("log")
ax2.grid(True, alpha=0.3)

# Add value labels
for bar, value in zip(bars, unseen_probs, strict=False):
    ax2.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() * 1.5,
        f"{value:.2e}",
        ha="center",
        va="bottom",
        rotation=45,
        fontsize=8,
    )

plt.tight_layout()

print("Key Insights:")
print("- MLE gives zero probability to unseen words (problematic)")
print("- Smoothing methods allocate probability mass to unseen events")
print("- Higher smoothing → more probability reserved for unseen words")
print("- Trade-off: fitting training data vs. handling unseen data")

#' ## Practical Tips and Conclusions
#'
#' Let's summarize what we've learned and provide practical guidance.

print("PRACTICAL RECOMMENDATIONS:")
print("=" * 50)
print()

print("1. NEVER use pure MLE for real applications")
print("   → Zero probabilities break many algorithms")
print()

print("2. Laplace smoothing (add-one) is a good baseline")
print("   → Simple, robust, works well for small datasets")
print()

print("3. Use ELE (gamma=0.5) for theoretically motivated smoothing")
print("   → Good balance between smoothing and data fidelity")
print()

print("4. Tune Lidstone gamma parameter using validation data")
print("   → Cross-validation to find optimal smoothing strength")
print()

print("5. Consider vocabulary size when setting 'bins' parameter")
print("   → Underestimating leads to over-smoothing")
print("   → Overestimating leads to under-smoothing")
print()

# Demonstrate vocabulary size effect
print("VOCABULARY SIZE EFFECT DEMONSTRATION:")
print("-" * 40)

vocab_sizes = [100, 1000, 10000]
for vocab_size in vocab_sizes:
    laplace_model = freqprob.Laplace(freqdist, bins=vocab_size, logprob=False)
    unseen_prob = laplace_model("unseen_word")
    print(f"Vocabulary size {vocab_size:5d}: P(unseen) = {unseen_prob:.6f}")

print("\nAs vocabulary size increases, unseen word probability decreases")
print("This affects the smoothing strength!")

#' ## Exercise: Try It Yourself!
#'
#' Now it's your turn to experiment with the concepts we've covered.

# EXERCISE 1: Create your own text corpus and compare smoothing methods
# TODO: Replace this with your own text data
your_corpus = [
    "machine learning is fascinating",
    "deep learning models are powerful",
    "natural language processing uses machine learning",
    "artificial intelligence and machine learning overlap",
    "learning algorithms improve with data",
]

# Create frequency distribution
your_words = []
for sentence in your_corpus:
    your_words.extend(sentence.split())

your_freqdist = Counter(your_words)
print("Your frequency distribution:")
print(your_freqdist)

# TODO: Create different smoothing models and compare them
# Hint: Use the code patterns from above

# EXERCISE 2: Find the optimal gamma for Lidstone smoothing
# TODO: Create a validation set and test different gamma values
# Hint: Use perplexity to evaluate performance

# EXERCISE 3: Analyze the effect of different vocabulary size estimates
# TODO: Try bins=[100, 500, 1000, 5000] and see how it affects smoothing

print("\nComplete the exercises above to deepen your understanding!")
print("Experiment with different:")
print("- Text corpora (different domains, sizes)")
print("- Smoothing parameters (gamma values)")
print("- Vocabulary size estimates")
print("- Evaluation metrics")

#' ## Summary
#'
#' In this tutorial, you learned:
#'
#' 1. **The zero probability problem** and why pure MLE fails
#' 2. **Laplace smoothing** as a simple solution (add-one)
#' 3. **Lidstone smoothing** for tunable smoothing strength
#' 4. **Expected Likelihood Estimation** as a theoretically motivated choice
#' 5. **Model evaluation** using perplexity
#' 6. **Trade-offs** between fitting training data and generalizing to unseen data
#' 7. **Practical considerations** for real-world applications
#'
#' **Next Steps:**
#' - Try Tutorial 2: Advanced Smoothing Methods (Kneser-Ney, Simple Good-Turing)
#' - Try Tutorial 3: Computational Efficiency and Memory Management
#' - Try Tutorial 4: Real-world NLP Applications
#'
#' **Key Takeaway:** Always use some form of smoothing in real applications. The choice of method depends on your data size, domain, and performance requirements.
