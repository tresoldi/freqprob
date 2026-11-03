# Scale to large data

The default estimators are in memory and eager, which is fine for most work.
For large or streaming data, FreqProb has alternatives with the same call
contract. Reach for them when you hit a limit.

## Streaming updates

`StreamingMLE` and `StreamingLaplace` update their counts incrementally, for
data that arrives over time or does not fit in memory:

```pycon
>>> import freqprob
>>> model = freqprob.StreamingMLE(logprob=False)
>>> model.update_batch(["x", "y", "x", "z"])
>>> model("x")
0.5
>>> model.update_single("x")
>>> model("x")
0.6
```

## Vectorized scoring

`VectorizedScorer` scores many elements at once into a NumPy array, which is
much faster than a Python loop. `BatchScorer` does the same for several
estimators side by side:

```pycon
>>> scorer = freqprob.VectorizedScorer(freqprob.MLE({"a": 3, "b": 1}, logprob=False))
>>> scorer.score_batch(["a", "b", "c"]).tolist()
[0.75, 0.25, 0.0]
```

## Lazy evaluation

A lazy estimator computes each element's probability only when it is first
scored, which helps when only a small part of a large vocabulary is queried:

```pycon
>>> lazy = freqprob.create_lazy_mle({"a": 3, "b": 1}, logprob=False)
>>> lazy("a")
0.75
```

## Compact storage

`CompressedFrequencyDistribution` and `SparseFrequencyDistribution` hold large,
skewed count tables in less memory, and `QuantizedProbabilityTable` stores
probabilities at reduced precision:

```pycon
>>> compact = freqprob.CompressedFrequencyDistribution.from_counts({"a": 100, "b": 50, "c": 1})
>>> compact.get_memory_usage()["total"] > 0
True
```

To measure the difference, `freqprob.performance.profiling` provides
`MemoryProfiler` and `DistributionMemoryAnalyzer`. They are development tools,
so they are imported from that module rather than from `freqprob`.

## Caches

Some estimators cache expensive computations, such as Simple Good–Turing's fit.
`freqprob.get_cache_stats()` reports the caches and `freqprob.clear_all_caches()`
empties them.
