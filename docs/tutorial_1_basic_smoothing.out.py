# FreqProb Tutorial 1: Basic Smoothing Methods

This tutorial introduces the fundamental smoothing methods in FreqProb. You'll learn how to:

1. Create frequency distributions from text data
2. Apply different smoothing techniques
3. Compare model performance
4. Handle unseen elements

## Setup

First, let's import the necessary libraries and set up our environment.
```python

from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import freqprob

# Set style for better plots
plt.style.use("default")
sns.set_palette("husl")

print(f"FreqProb version: {freqprob.__version__ if hasattr(freqprob, '__version__') else 'dev'}")
```

Output:
```
FreqProb version: 0.4.0
```

## Creating a Frequency Distribution

Let's start with a simple text corpus and create a frequency distribution.
```python

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
```

Output:
```
Frequency Distribution:
the: 7
cat: 3
dog: 2
in: 2
a: 2
and: 2
sat: 1
on: 1
mat: 1
ran: 1
park: 1
played: 1
garden: 1
likes: 1
warm: 1
sun: 1
dogs: 1
cats: 1
are: 1
pets: 1

Total words: 32
Unique words: 20
```

## Maximum Likelihood Estimation (MLE)

Let's start with the simplest approach - Maximum Likelihood Estimation.
```python

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

plt.figure(figsize=(10, 5))
plt.bar(words, probs)
plt.title("MLE Probability Distribution")
plt.xlabel("Words")
plt.ylabel("Probability")
plt.xticks(rotation=45)
plt.tight_layout()

print(f"\nProbability sum: {sum(probs):.4f}")
print(f"Problem: P(unknown_word) = {mle('unknown_word'):.4f} (zero probability!)")
```

Output:
```
MLE Probabilities:
P(the) = 0.2188
P(cat) = 0.0938
P(dog) = 0.0625
P(garden) = 0.0312
P(unknown_word) = 0.0000

Probability sum: 1.0000
Problem: P(unknown_word) = 0.0000 (zero probability!)
```

![Figure](figures/figure_0.png)

## Laplace Smoothing (Add-One)

Laplace smoothing solves the zero probability problem by adding 1 to all counts.
```python

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

plt.figure(figsize=(10, 5))
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
```

Output:
```
Laplace Smoothing Probabilities:
P(the) = 0.007752
P(cat) = 0.003876
P(dog) = 0.002907
P(garden) = 0.001938
P(unknown_word) = 0.000969

Unseen word probability (Laplace): 0.000969
Unseen word probability (MLE): 0.000000
```

![Figure](figures/figure_1.png)

## Lidstone Smoothing (Add-k)

Lidstone smoothing is a generalization of Laplace smoothing where we can adjust the smoothing parameter.
```python

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
plt.figure(figsize=(11, 5))

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
```

Output:
```
Lidstone Smoothing with Different Gamma Values:
gamma = 0.01: P(the) = 0.166905, P(unknown_word) = 0.000238
gamma = 0.1: P(the) = 0.053788, P(unknown_word) = 0.000758
gamma = 0.5: P(the) = 0.014098, P(unknown_word) = 0.000940
gamma = 1.0: P(the) = 0.007752, P(unknown_word) = 0.000969
gamma = 2.0: P(the) = 0.004429, P(unknown_word) = 0.000984
gamma = 5.0: P(the) = 0.002385, P(unknown_word) = 0.000994

Observation: As gamma increases, probabilities become more uniform
```

![Figure](figures/figure_2.png)

## Expected Likelihood Estimation (ELE)

ELE is a special case of Lidstone smoothing with γ = 0.5, which has theoretical justification.
```python

# Create ELE model
ele = freqprob.ELE(freqdist, bins=vocabulary_size, logprob=False)

print("Expected Likelihood Estimation (gamma = 0.5):")
for word in test_words:
    prob = ele(word)
    print(f"P({word}) = {prob:.6f}")

# Compare all methods
methods = {"MLE": mle, "Laplace": laplace, "ELE": ele, "Lidstone (gamma=0.1)": lidstone_models[0.1]}

comparison_words = ["the", "cat", "garden", "unknown_word"]

plt.figure(figsize=(11, 6))
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
```

Output:
```
Expected Likelihood Estimation (gamma = 0.5):
P(the) = 0.014098
P(cat) = 0.006579
P(dog) = 0.004699
P(garden) = 0.002820
P(unknown_word) = 0.000940
```

![Figure](figures/figure_3.png)

## Model Evaluation with Perplexity

Let's evaluate our models using perplexity on a held-out test set.
```python

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
    plt.figure(figsize=(9, 5))
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
```

Output:
```
Test set: ['the', 'cat', 'sleeps', 'on', 'the', 'sofa', 'a', 'dog', 'barks', 'in', 'the', 'yard', 'cats', 'and', 'dogs', 'are', 'friends']
Test set size: 17 words

Model Evaluation (Perplexity):
Lower perplexity = better model
----------------------------------------
MLE            : 5696.44
Laplace        : 432.32
ELE            : 311.14
Lidstone (gamma=0.1): 141.26

Best performing method: Lidstone (gamma=0.1) (PP = 141.26)
```

![Figure](figures/figure_4.png)

## Understanding the Trade-offs

Let's visualize how different smoothing methods handle the trade-off between fitting training data and generalizing to unseen data.
```python

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

# Display results in a clean table format
methods_list = list(total_prob_mass.keys())

print("\n" + "="*75)
print("PROBABILITY MASS ALLOCATION: OBSERVED vs. UNSEEN WORDS")
print("="*75)
print(f"{'Method':<20} {'Observed Mass':<18} {'Reserved Mass':<18} {'Per Unseen Word':<18}")
print("-"*75)

for method_name in methods_list:
    observed = total_prob_mass[method_name]
    reserved = 1 - observed
    per_unseen = unseen_prob_mass[method_name]
    print(f"{method_name:<20} {observed:>12.4f} ({observed*100:>5.1f}%)  "
          f"{reserved:>12.4f} ({reserved*100:>5.1f}%)  {per_unseen:>17.2e}")

# Simple bar chart showing reserved probability mass
reserved_masses = [1 - total_prob_mass[m] for m in methods_list]
plt.figure(figsize=(8, 4))
plt.bar(methods_list, reserved_masses, alpha=0.8)
plt.title("Probability Mass Reserved for Unseen Events")
plt.ylabel("Reserved Probability")
plt.xticks(rotation=45)
plt.tight_layout()

print("\nKey Insights:")
print("- MLE gives zero probability to unseen words (problematic)")
print("- Smoothing methods allocate probability mass to unseen events")
print("- Higher smoothing → more probability reserved for unseen words")
print("- Trade-off: fitting training data vs. handling unseen data")
```

Output:
```
MLE:
  Probability mass for observed words: 1.0000
  Probability for unseen words: 0.000000
  Reserved mass for unseen events: 0.0000

Laplace:
  Probability mass for observed words: 0.0504
  Probability for unseen words: 0.000969
  Reserved mass for unseen events: 0.9496

ELE:
  Probability mass for observed words: 0.0789
  Probability for unseen words: 0.000940
  Reserved mass for unseen events: 0.9211

Lidstone (gamma=0.1):
  Probability mass for observed words: 0.2576
  Probability for unseen words: 0.000758
  Reserved mass for unseen events: 0.7424


===========================================================================
PROBABILITY MASS ALLOCATION: OBSERVED vs. UNSEEN WORDS
===========================================================================
Method               Observed Mass      Reserved Mass      Per Unseen Word   
---------------------------------------------------------------------------
MLE                        1.0000 (100.0%)        0.0000 (  0.0%)           0.00e+00
Laplace                    0.0504 (  5.0%)        0.9496 ( 95.0%)           9.69e-04
ELE                        0.0789 (  7.9%)        0.9211 ( 92.1%)           9.40e-04
Lidstone (gamma=0.1)       0.2576 ( 25.8%)        0.7424 ( 74.2%)           7.58e-04

Key Insights:
- MLE gives zero probability to unseen words (problematic)
- Smoothing methods allocate probability mass to unseen events
- Higher smoothing → more probability reserved for unseen words
- Trade-off: fitting training data vs. handling unseen data
```

![Figure](figures/figure_5.png)

## Practical Tips and Conclusions

Let's summarize what we've learned and provide practical guidance.
```python

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
```

Output:
```
PRACTICAL RECOMMENDATIONS:
==================================================

1. NEVER use pure MLE for real applications
   → Zero probabilities break many algorithms

2. Laplace smoothing (add-one) is a good baseline
   → Simple, robust, works well for small datasets

3. Use ELE (gamma=0.5) for theoretically motivated smoothing
   → Good balance between smoothing and data fidelity

4. Tune Lidstone gamma parameter using validation data
   → Cross-validation to find optimal smoothing strength

5. Consider vocabulary size when setting 'bins' parameter
   → Underestimating leads to over-smoothing
   → Overestimating leads to under-smoothing

VOCABULARY SIZE EFFECT DEMONSTRATION:
----------------------------------------
Vocabulary size   100: P(unseen) = 0.007576
Vocabulary size  1000: P(unseen) = 0.000969
Vocabulary size 10000: P(unseen) = 0.000100

As vocabulary size increases, unseen word probability decreases
This affects the smoothing strength!
```

## Exercise: Try It Yourself!

Now it's your turn to experiment with the concepts we've covered.
```python

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
```

Output:
```
Your frequency distribution:
Counter({'learning': 5, 'machine': 3, 'is': 1, 'fascinating': 1, 'deep': 1, 'models': 1, 'are': 1, 'powerful': 1, 'natural': 1, 'language': 1, 'processing': 1, 'uses': 1, 'artificial': 1, 'intelligence': 1, 'and': 1, 'overlap': 1, 'algorithms': 1, 'improve': 1, 'with': 1, 'data': 1})

Complete the exercises above to deepen your understanding!
Experiment with different:
- Text corpora (different domains, sizes)
- Smoothing parameters (gamma values)
- Vocabulary size estimates
- Evaluation metrics
```

## Summary

In this tutorial, you learned:

1. **The zero probability problem** and why pure MLE fails
2. **Laplace smoothing** as a simple solution (add-one)
3. **Lidstone smoothing** for tunable smoothing strength
4. **Expected Likelihood Estimation** as a theoretically motivated choice
5. **Model evaluation** using perplexity
6. **Trade-offs** between fitting training data and generalizing to unseen data
7. **Practical considerations** for real-world applications

**Next Steps:**
- Try Tutorial 2: Advanced Smoothing Methods (Kneser-Ney, Simple Good-Turing)
- Try Tutorial 3: Computational Efficiency and Memory Management
- Try Tutorial 4: Real-world NLP Applications

**Key Takeaway:** Always use some form of smoothing in real applications. The choice of method depends on your data size, domain, and performance requirements.
```python

```
