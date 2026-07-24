# Migration Guide

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
  scorer.fit(freqdist)        # returns self
  scorer.score(element)       # single element (alias for scorer(element))
  scorer.predict(elements)    # batch: list of scores
  ```
- **Serialization** — persist and restore a fitted estimator without re-fitting:
  ```python
  scorer.save("model.pkl")
  restored = freqprob.KneserNey.load("model.pkl")
  ```
  `load()` only accepts files you trust (unpickling executes code).
- **`freqprob.ScoringMethod`** — the estimator base type is now public, for
  `isinstance` checks and type annotations.
