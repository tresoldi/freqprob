# Migration Guide

## Migrating to 0.9.0

### `kl_divergence` counts each element once

`kl_divergence(p, q, elements)` now sums over the distinct elements given. If
you passed a sample with repeats, such as a held-out corpus, earlier results
weighted each element by how often it occurred, and are not comparable. Pass
the support you want to compare the two models on; the result is the same as
before when `elements` has no repeats.

```python
kl_divergence(p, q, ["a", "b", "a", "c"])
# before: "a" counted twice
# now:    the same as kl_divergence(p, q, ["a", "b", "c"])
```

### n-gram methods return 0.0 for unknown words in plain space

With `logprob=False`, the n-gram methods now return `0.0`, not `1e-10`, for a
bigram whose word never occurs in the data. Log-space scores are unchanged.

## Migrating to 0.8.0

Version 0.8.0 shrinks the top-level `freqprob` namespace from 50 names to 41.
Code that uses only the estimators, the entropy and evaluation functions, and
the distribution classes needs no change.

### Removed names and their replacements

| Removed | Use instead |
|---------|-------------|
| `create_compressed_distribution(counts)` | `CompressedFrequencyDistribution.from_counts(counts)` |
| `create_sparse_distribution(counts)` | `SparseFrequencyDistribution.from_counts(counts)` |
| `create_vectorized_batch_scorer(scorers)` | `BatchScorer(scorers)` |
| `LazyBatchScorer` | The scorer's own `predict()`, or `BatchScorer` |
| `VectorizedScorer.score_parallel` | `VectorizedScorer.score_batch` (the old method ran sequentially anyway) |

### Moved out of the top level

These still exist; import them from their modules:

```python
from freqprob.performance.lazy import LazyScoringMethod
from freqprob.performance.profiling import (
    DistributionMemoryAnalyzer,
    MemoryMonitor,
    MemoryProfiler,
)
from freqprob.performance.streaming import StreamingDataProcessor
```

### Per-method config classes

The per-method config subclasses (`KneserNeyConfig`, `BayesianConfig`, and so
on) were never exported at the top level, and they are now gone. Pass a
method's parameters to its constructor, as the top-level API always did.
`ScoringMethodConfig` keeps `unobs_prob`, `bins` and `logprob`.

### Results that change

- **Kneser-Ney scores for unseen bigrams** now depend on the context's back-off
  weight instead of one flat average, and a word absent from the data scores at
  the floor.
- **`chao_shen_entropy` on all-singleton samples** returns a positive estimate
  instead of `0.0`.
- **An empty distribution scored by a log-scorer** returns `log(1e-10)` instead
  of `1e-10`.

See the [CHANGELOG](CHANGELOG.md) for the full list of fixes.

## Migrating to 0.6.0

Version 0.6.0 completes a multi-phase internal restructuring and introduces a
small, deliberate set of public-API changes. The **top-level `freqprob`
namespace is the stable, supported surface** — if you import everything from
`freqprob` directly, the only change you need is the two class renames below.

### Breaking changes

#### 1. Renamed estimator classes

The inconsistent `Smoothing` suffix was dropped so all estimator names are
uniform:

| Old name | New name |
|----------|----------|
| `BayesianSmoothing` | `Bayesian` |
| `InterpolatedSmoothing` | `Interpolated` |

```python
# Before
from freqprob import BayesianSmoothing, InterpolatedSmoothing

model = BayesianSmoothing(counts, alpha=1.0)

# After
from freqprob import Bayesian, Interpolated

model = Bayesian(counts, alpha=1.0)
```

Constructor parameters and behavior are unchanged.

#### 2. Internal module paths moved

If you imported from **internal module paths** rather than the top-level
namespace, those paths changed when the package was reorganized:

| Old import | New location |
|------------|--------------|
| `from freqprob.basic import MLE, Uniform, Random` | `from freqprob import ...` (or `freqprob.methods.baselines`) |
| `from freqprob.lidstone import Laplace, Lidstone, ELE` | `from freqprob import ...` (or `freqprob.methods.additive`) |
| `from freqprob.advanced import SimpleGoodTuring, WittenBell` | `from freqprob import ...` (or `freqprob.methods.goodturing`) |
| `from freqprob.smoothing import KneserNey, ...` | `from freqprob import ...` (or `freqprob.methods.kneser_ney` / `interpolated` / `bayesian`) |
| `from freqprob.utils import perplexity, ...` | `from freqprob import ...` (or `freqprob.metrics`) |
| `from freqprob.utils import generate_ngrams, ...` | `from freqprob import ...` (or `freqprob.text`) |
| `from freqprob.lazy / vectorized / streaming / memory_efficient / profiling import ...` | `from freqprob import ...` (or `freqprob.performance.*`) |

**Recommended:** import from the top-level `freqprob` namespace, which is
stable across these moves.

### New features (non-breaking)

- **scikit-learn-style aliases** on every estimator, alongside the existing
  callable contract:
  ```python
  scorer.fit(freqdist)  # returns self
  scorer.score(element)  # single element (alias for scorer(element))
  scorer.predict(elements)  # batch: list of scores
  ```
- **Serialization** — persist and restore a fitted estimator without re-fitting:
  ```python
  scorer.save("model.pkl")
  restored = freqprob.KneserNey.load("model.pkl")
  ```
  `load()` only accepts files you trust (unpickling executes code).
- **`freqprob.ScoringMethod`** — the estimator base type is now public, for
  `isinstance` checks and type annotations.
