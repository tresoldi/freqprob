# Performance

Alternatives to the default estimators for large or streaming data; see
[Scale to large data](../how-to/scale.md). Profiling tools live in
`freqprob.performance.profiling` and are not re-exported.

## Streaming

::: freqprob.performance.streaming
    options:
      show_root_heading: false
      heading_level: 3
      members:
        - StreamingFrequencyDistribution
        - StreamingMLE
        - StreamingLaplace

## Vectorized scoring

::: freqprob.performance.vectorized
    options:
      show_root_heading: false
      heading_level: 3
      members:
        - VectorizedScorer
        - BatchScorer

## Lazy evaluation

::: freqprob.performance.lazy
    options:
      show_root_heading: false
      heading_level: 3
      members:
        - create_lazy_mle
        - create_lazy_laplace

## Compact storage

::: freqprob.performance.memory_efficient
    options:
      show_root_heading: false
      heading_level: 3
      members:
        - CompressedFrequencyDistribution
        - SparseFrequencyDistribution
        - QuantizedProbabilityTable

## Caches

::: freqprob.cache
    options:
      show_root_heading: false
      heading_level: 3
      members:
        - get_cache_stats
        - clear_all_caches
