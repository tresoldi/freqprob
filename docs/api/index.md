# API reference

The reference is generated from the docstrings in the source, so it always
matches the installed version. Everything below is available from the
top-level package (`freqprob.Laplace`, `freqprob.perplexity`, ...):

```pycon
>>> import freqprob
>>> len(freqprob.__all__)
41
```

| Page | Contents |
|------|----------|
| [Estimators](estimators.md) | every smoothing method, and the `ScoringMethod` base class |
| [Metrics](metrics.md) | perplexity, cross-entropy, KL divergence, model comparison, coverage and entropy |
| [Text helpers](text.md) | word and n-gram counting |
| [Performance](performance.md) | streaming, vectorized and lazy estimators, compact storage, caches |

Each [method page](../methods/index.md) also includes the reference entries of
its estimators, next to their definition and examples.
