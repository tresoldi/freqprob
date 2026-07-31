# User Guide

FreqProb turns **counts into probabilities**. You give it a mapping of elements
to how often you observed them; it gives you back a smoothed probability
estimator that also assigns sensible probability to things you *haven't* seen.

That problem shows up far beyond text: species you didn't catch in a survey,
k-mers absent from a sample, product categories with no clicks yet. This guide
teaches the concepts, helps you pick a method, and works through examples in
several domains. Every code block on this page is executed in the test suite, so
the examples stay correct against the installed version.

> New here? Read **Core concepts** and **Choosing a method**, then jump to the
> worked example closest to your problem.

---

## Core concepts

### One contract for every estimator

Every estimator is built from a frequency distribution and then *called* to
score an element. That is the whole interface.

```python
import freqprob

counts = {"the": 30, "cat": 6, "sat": 3, "mat": 1}
mle = freqprob.MLE(counts, logprob=False)

assert round(mle("the"), 2) == 0.75  # 30 of 40 observations
assert round(mle("cat"), 2) == 0.15  # 6 of 40
```

Swapping methods means swapping the constructor — the call site never changes.

### The unseen-element problem

Plain relative frequency (`MLE`) gives anything unseen a probability of exactly
zero, which is usually wrong and breaks anything multiplicative (a single zero
zeroes the whole product). *Smoothing* moves a little mass onto the unseen.

```python
import freqprob

counts = {"the": 30, "cat": 6, "sat": 3, "mat": 1}
mle = freqprob.MLE(counts, logprob=False)
laplace = freqprob.Laplace(counts, bins=1000, logprob=False)

assert mle("mouse") == 0.0  # MLE: unseen -> zero
assert laplace("mouse") > 0.0  # Laplace: unseen -> small but non-zero
```

`bins` is the size of the space of *possible* elements (here, an assumed
vocabulary of 1000). It tells additive methods how much unseen mass to reserve;
see [Choosing `bins`](#choosing-bins).

### Log space and `logprob`

By default estimators return **log**-probabilities (`logprob=True`), because
real data multiplies many small probabilities and linear space underflows to
zero. Pass `logprob=False` when you want plain probabilities for a quick look.

```python
import math
import freqprob

counts = {"a": 3, "b": 1}
p = freqprob.MLE(counts, logprob=False)
logp = freqprob.MLE(counts, logprob=True)

assert math.isclose(math.exp(logp("a")), p("a"))
```

See [Working in log space](#working-in-log-space) for why this matters and how
to combine log-probabilities correctly.

---

## Choosing a method

Start from what your data looks like:

| Your situation | Method | Key parameter |
|----------------|--------|---------------|
| Just want relative frequencies, no smoothing | `MLE` | — |
| General-purpose additive smoothing | `Laplace` / `Lidstone` / `ELE` | `bins`, `gamma` |
| Heavy-tailed counts, many rare elements | `SimpleGoodTuring` | `p_value` |
| Parameter-free discounting by distinct-type count | `WittenBell` | `bins` |
| n-gram language models | `KneserNey` / `ModifiedKneserNey` | `discount` |
| Bigram discounting with unigram back-off | `AbsoluteDiscounting` / `PitmanYor` | `discount`, `strength` |
| Bigram back-off (normalised or fast) | `KatzBackoff` / `StupidBackoff` | `k`, `alpha` |
| You have a prior belief (Dirichlet) | `Bayesian` | `alpha` |
| Combine a specific and a general model | `Interpolated` / `JelinekMercer` | `lambda_weight` |
| Reserve mass by how fully the support is observed (experimental) | `CertaintyDegree` | `bins` |
| Non-informative baseline | `Uniform` / `Random` | — |

By data characteristics:

| Vocabulary | Data density | Good default |
|------------|--------------|--------------|
| Small | Dense (most elements seen often) | `MLE` or `Laplace` |
| Large | Sparse (many rare/unseen) | `SimpleGoodTuring` or `Lidstone` (small `gamma`) |
| Sequence contexts (n-grams) | Any | `KneserNey` |
| Any | You have a trusted background model | `Interpolated` |

When-to-use, in one line each:

- **`MLE`** — a reference point; not for sparse data (zeros unseen events).
- **`Laplace`** (add-one) — simple and robust, but over-smooths large vocabularies.
- **`Lidstone`** / **`ELE`** — add a fractional `gamma`; `ELE` (`gamma=0.5`) is a
  gentler default than add-one for large vocabularies.
- **`SimpleGoodTuring`** — the strongest general estimate of unseen mass for
  heavy-tailed data; needs a spread of low frequencies to fit.
- **`WittenBell`** — reserves mass in proportion to the number of *distinct*
  things seen; a solid, assumption-light choice.
- **`KneserNey`** — the standard for n-gram models; models how likely an element
  is to appear in a *novel* context.
- **`AbsoluteDiscounting`** / **`PitmanYor`** — bigram discounting that backs off
  to a unigram model; `PitmanYor` adds a concentration parameter (`strength`) and
  has absolute discounting as its `strength=0` case.
- **`KatzBackoff`** / **`StupidBackoff`** — bigram back-off; Katz is a proper
  normalized distribution (Good-Turing discounted), Stupid Back-off is a fast,
  unnormalized score for ranking at scale.
- **`Bayesian`** — smoothing as a Dirichlet(`alpha`) prior; principled and tunable.
- **`Interpolated`** — mix a high-order (specific) and low-order (general) model.
- **`JelinekMercer`** — interpolation whose weight is fit by EM on held-out data
  (falls back to a fixed weight when none is given).
- **`CertaintyDegree`** — reserves mass by how fully the possible support has been
  observed; the more of the space you have seen, the less is held back.
  *Experimental — not recommended as your only estimator yet.*

---

## The methods

Each estimator below takes a `freqdist` and returns log-probabilities unless you
pass `logprob=False`. The examples use `logprob=False` for readability.

### Baselines: MLE, Uniform, Random

`MLE` is relative frequency. `Uniform` ignores counts and splits mass evenly.
`Random` assigns reproducible pseudo-random probabilities — useful only as a
control.

```python
import freqprob

counts = {"a": 6, "b": 3, "c": 1}
assert round(freqprob.MLE(counts, logprob=False)("a"), 2) == 0.6
assert round(freqprob.Uniform(counts, logprob=False)("a"), 4) == round(1 / 3, 4)
```

### Additive smoothing: Laplace, Lidstone, ELE

Add a pseudo-count to every possible element. `Laplace` adds 1; `Lidstone` adds
a configurable `gamma`; `ELE` adds 0.5. Larger `gamma` and larger `bins` reserve
more mass for the unseen.

```python
import freqprob

counts = {"a": 6, "b": 3, "c": 1}
laplace = freqprob.Laplace(counts, bins=100, logprob=False)
lidstone = freqprob.Lidstone(counts, gamma=0.1, bins=100, logprob=False)

assert laplace("z") > 0  # unseen gets mass
assert lidstone("z") < laplace("z")  # smaller gamma -> less smoothing
```

### Discounting for heavy tails: Simple Good-Turing, Witten-Bell

For data with many rare elements, these estimate how much total probability
belongs to *everything you didn't see*. Good-Turing does it from the
frequency-of-frequencies; Witten-Bell from the count of distinct elements.

```python
import freqprob

# A Zipfian-ish distribution with several singletons.
counts = {"a": 10, "b": 5, "c": 4, "d": 3, "e": 2, "f": 2, "g": 1, "h": 1, "i": 1, "j": 1}
sgt = freqprob.SimpleGoodTuring(counts, logprob=False)

assert 0.0 < sgt.total_unseen_mass < 1.0  # reserved for unseen types
assert sgt("unseen") > 0.0

# Witten-Bell: parameter-free, reserves mass by the count of distinct types.
wb = freqprob.WittenBell(counts, logprob=False)
assert wb("a") > wb("g")  # observed elements ranked by frequency
assert wb("unseen") > 0.0  # unseen types share the reserved mass
```

### n-gram models: Kneser-Ney

The standard smoothing for language models. It discounts observed n-grams and
redistributes mass using how many *distinct contexts* an element completes —
capturing that "Francisco" is frequent but only after "San".

```python
import freqprob

bigrams = {("the", "cat"): 5, ("the", "dog"): 3, ("a", "cat"): 2}
kn = freqprob.KneserNey(bigrams, discount=0.75, logprob=False)

assert 0.0 < kn(("the", "cat")) <= 1.0
assert kn(("the", "bird")) > 0.0  # unseen bigram still gets mass
```

### Bigram back-off and discounting: AbsoluteDiscounting, PitmanYor, KatzBackoff, StupidBackoff

These take a distribution over `(context, word)` bigrams and model
`P(word | context)`, discounting observed bigrams and backing off to a unigram
model for unseen ones. `AbsoluteDiscounting` subtracts a fixed amount from each
count; `PitmanYor` generalises it with a concentration parameter (`strength`),
recovering absolute discounting at `strength=0`. `KatzBackoff` uses Good-Turing
discounting and normalised back-off weights, so it is a proper distribution;
`StupidBackoff` skips normalisation for speed and returns scores (not
probabilities) for ranking at scale.

```python
import freqprob

bigrams = {
    ("the", "cat"): 5,
    ("the", "dog"): 3,
    ("a", "cat"): 2,
    ("a", "dog"): 1,
    ("big", "cat"): 1,
    ("small", "dog"): 1,
}

ad = freqprob.AbsoluteDiscounting(bigrams, discount=0.75, logprob=False)
assert ad(("the", "cat")) > 0
assert ad(("new", "cat")) > 0  # unseen context backs off to the unigram model

sb = freqprob.StupidBackoff(bigrams, alpha=0.4, logprob=False)
assert sb(("the", "cat")) == 0.625  # observed bigram scores its relative frequency (5/8)
```

### Priors and interpolation: Bayesian, Interpolated, JelinekMercer

`Bayesian` treats smoothing as a Dirichlet(`alpha`) prior over categories.
`Interpolated` blends a high-order (specific) distribution with a low-order
(general) one via a fixed `lambda_weight`. `JelinekMercer` is the same
interpolation, but fits the weight by EM on a held-out distribution when one is
supplied (otherwise it uses the fixed weight).

```python
import freqprob

high = {("the", "big", "cat"): 1, ("a", "big", "dog"): 1}
low = {"cat": 3, "dog": 2, "the": 5}
interp = freqprob.Interpolated(high, low, lambda_weight=0.7, logprob=False)

assert 0.0 <= interp(("the", "big", "cat")) <= 1.0

# Jelinek-Mercer fits the interpolation weight from held-out data.
trigrams = {("the", "big", "cat"): 3, ("a", "big", "dog"): 2}
bigrams = {("big", "cat"): 5, ("big", "dog"): 3}
jm = freqprob.JelinekMercer(trigrams, bigrams, lambda_weight=0.6, logprob=False)
assert round(jm.estimated_lambda, 2) == 0.6  # no held-out data -> fixed weight
```

### Certainty-degree smoothing: CertaintyDegree

`CertaintyDegree` reserves mass for the unseen according to how much of the
possible support has already been observed: the more of the space you have seen,
the more certain it is that little remains unseen, so less mass is held back. The
remainder rescales the MLE of each observed element. Pass `bins` to declare the
size of the possible support.

```python
import freqprob

counts = {"apple": 8, "banana": 4, "cherry": 2, "date": 1}
cd = freqprob.CertaintyDegree(counts, logprob=False)

assert cd("apple") > cd("date")  # observed elements ranked by frequency
assert cd("kiwi") > 0.0  # unobserved element still gets mass
```

This is an **experimental** estimator still under development; it is documented
for completeness but is not recommended as your sole or primary method yet.

---

## Evaluating models

Score how well an estimator predicts held-out data. Lower perplexity and
cross-entropy are better. Build the estimator in log space for these.

```python
import freqprob

train = {"a": 8, "b": 4, "c": 2, "d": 1}
test = ["a", "b", "a", "c", "e"]  # note: 'e' is unseen in training

laplace = freqprob.Laplace(train, bins=100, logprob=True)

pp = freqprob.perplexity(laplace, test)
xe = freqprob.cross_entropy(laplace, test)
assert round(pp, 1) == 27.8
assert round(xe, 1) == 3.3
```

Compare several models at once with `model_comparison`:

```python
import freqprob

train = {"a": 8, "b": 4, "c": 2, "d": 1}
test = ["a", "b", "a", "c", "e"]

models = {
    "laplace": freqprob.Laplace(train, bins=100, logprob=True),
    "mle": freqprob.MLE(train, unobs_prob=0.01, logprob=True),
}
report = freqprob.model_comparison(models, test)
assert set(report) == {"laplace", "mle"}
```

### Coverage and entropy of a sample

These work directly on a frequency distribution rather than on a fitted model.
`sample_coverage` is the Good-Turing estimate of the observed probability mass
(one minus the singleton rate). `chao_shen_entropy` and `nsb_entropy` estimate
Shannon entropy while correcting the downward bias of the naive plug-in estimate
when the sample misses mass — useful for undersampled data. `nsb_entropy` takes
an optional `bins` to reserve entropy for types that could occur but were not
seen.

```python
import freqprob

counts = {"a": 8, "b": 1, "c": 1}  # two singletons out of ten observations

assert freqprob.sample_coverage(counts) == 0.8  # 1 - 2/10
# Both estimators exceed the (biased-low) plug-in entropy on undersampled data.
assert freqprob.chao_shen_entropy(counts) > 0.0
assert freqprob.nsb_entropy(counts, bins=100) > freqprob.nsb_entropy(counts)
```

---

## Worked examples by domain

The same counts-to-probabilities tool fits many fields. Each example is
self-contained.

### Text and language

Word or n-gram frequencies with additive smoothing — the classic case.

```python
import freqprob

text = "the cat sat on the mat the cat sat".split()
counts = freqprob.word_frequency(text)
model = freqprob.Laplace(counts, bins=10_000, logprob=False)

assert counts["the"] == 3
assert model("the") > model("mat")  # more frequent -> higher probability
assert model("dog") > 0.0  # unseen word still scored
```

Build n-grams with the text helpers:

```python
import freqprob

tokens = ["the", "cat", "sat"]
assert freqprob.generate_ngrams(tokens, 2) == [("the", "cat"), ("cat", "sat")]
```

### Ecology: species abundance and unseen species

Good-Turing was invented for exactly this — estimating the probability mass of
species you *didn't* observe in a survey (the "unseen species" problem).

```python
import freqprob

# Individuals counted per species in a survey (many rare species).
survey = {
    "sparrow": 40,
    "robin": 18,
    "finch": 12,
    "hawk": 3,
    "owl": 2,
    "crane": 1,
    "heron": 1,
    "egret": 1,
}
sgt = freqprob.SimpleGoodTuring(survey, logprob=False)

# Probability mass reserved for as-yet-unseen species:
assert sgt.total_unseen_mass > 0.0
assert round(sgt("sparrow"), 2) == 0.50  # the dominant species
```

### Genomics: k-mer frequencies

Count k-mers (length-k subsequences) in a sequence and smooth over the space of
all possible k-mers, so absent k-mers get non-zero probability.

```python
from collections import Counter

import freqprob

sequence = "ATGGCATGCA"
k = 3
kmers = Counter(sequence[i : i + k] for i in range(len(sequence) - k + 1))

# 4 nucleotides -> 4**3 = 64 possible 3-mers.
model = freqprob.Laplace(dict(kmers), bins=64, logprob=False)
assert kmers["ATG"] == 2
assert model("ATG") > model("TTT")  # observed vs absent k-mer
assert model("TTT") > 0.0
```

### Categorical features and analytics

Smoothed category priors keep a rare or brand-new category from collapsing to
zero probability — useful for ML features, event/click counts, and A/B data.

```python
import freqprob

category_counts = {"electronics": 120, "books": 80, "toys": 25, "garden": 5}
prior = freqprob.Laplace(category_counts, bins=10, logprob=False)

assert prior("electronics") > prior("garden")
assert prior("music") > 0.0  # a category with no events yet
```

---

## Scaling and performance

The default estimators are in-memory and eager, which is fine for most work. For
large or streaming data, FreqProb offers drop-in alternatives that share the
same call contract. Reach for these only when you actually hit a limit.

### Streaming updates

Update counts incrementally without rebuilding the model, for data that arrives
over time or doesn't fit in memory.

```python
import freqprob

model = freqprob.StreamingMLE(logprob=False)
model.update_batch(["x", "y", "x", "z"])
assert round(model("x"), 2) == 0.5  # 2 of 4 so far
model.update_single("x")
assert round(model("x"), 2) == 0.6  # 3 of 5 after one more
```

### Vectorized batch scoring

Score many elements at once as a NumPy array — much faster than a Python loop.

```python
import freqprob

scorer = freqprob.VectorizedScorer(freqprob.MLE({"a": 3, "b": 1}, logprob=False))
scores = scorer.score_batch(["a", "b", "c"])
assert [round(float(s), 2) for s in scores] == [0.75, 0.25, 0.0]
```

### Lazy evaluation

Defer per-element computation until an element is actually scored — helpful when
you only touch a small slice of a huge vocabulary.

```python
import freqprob

lazy = freqprob.create_lazy_mle({"a": 3, "b": 1}, logprob=False)
assert round(lazy("a"), 2) == 0.75
```

### Compact storage

`create_compressed_distribution` and `create_sparse_distribution` trade a little
speed for much lower memory on large, skewed count tables. `MemoryProfiler`
helps you measure the difference — see the API reference for details.

```python
import freqprob

compact = freqprob.create_compressed_distribution({"a": 100, "b": 50, "c": 1})
assert compact.get_memory_usage()["total"] > 0
```

---

## Practical notes

### Working in log space

Multiplying many probabilities underflows; adding log-probabilities does not.
Keep `logprob=True` (the default) for any real scoring pipeline, and combine
scores with `logsumexp` when you need to sum probabilities in log space.

```python
import math

# Product of many small probabilities underflows to 0.0 in linear space...
probs = [1e-30] * 20
assert math.prod(probs) == 0.0

# ...but the sum of their logs is finite and exact.
logs = [math.log(p) for p in probs]
assert math.isfinite(sum(logs))
```

To add probabilities that are stored as logs (e.g. mixing models), use a stable
log-sum-exp rather than exponentiating first:

```python
from scipy.special import logsumexp

log_a, log_b = -700.0, -701.0  # would underflow under exp()
combined = logsumexp([log_a, log_b])  # log(exp(log_a) + exp(log_b))
assert combined > log_a  # the sum is larger than either term
```

### Choosing `bins`

`bins` is the number of *possible* distinct elements (the support size),
including those you haven't observed. Additive methods spread reserved mass over
`bins - observed`. If you don't know it exactly, estimate a little above your
observed vocabulary — a rule of thumb is 20–50% more:

```python
counts = {"a": 5, "b": 3, "c": 1}
estimated_bins = int(len(counts) * 1.3)  # ~30% headroom for unseen types
assert estimated_bins >= len(counts)
```

### Save and load a fitted model

Persist a fitted estimator and restore it without re-fitting (pickle-based; only
load files you trust).

```python
import os
import tempfile

import freqprob

model = freqprob.Laplace({"a": 5, "b": 3}, bins=100, logprob=False)
path = os.path.join(tempfile.mkdtemp(), "model.pkl")
model.save(path)
restored = freqprob.Laplace.load(path)
assert restored("a") == model("a")
```

### scikit-learn-style API

Alongside the callable contract, every estimator offers `fit` / `predict` /
`score`. `score` is a single-element alias for calling the estimator; `predict`
scores an iterable in one call.

```python
import freqprob

model = freqprob.MLE({"a": 3, "b": 1}, logprob=False)
assert model.score("a") == model("a")
assert list(model.predict(["a", "b"])) == [model("a"), model("b")]
```

---

## Next steps

- **[API Reference](reference.md)** — every public class and function, generated
  from the source, with a runnable example for each.
- **[Home](index.md)** — the one-minute overview and the method-selection table.
