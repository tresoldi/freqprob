# How estimators work

## One contract for every method

Every estimator is built from a frequency distribution and then **called** to
score an element. That is the whole interface:

```pycon
>>> import freqprob
>>> counts = {"the": 30, "cat": 6, "sat": 3, "mat": 1}
>>> mle = freqprob.MLE(counts, logprob=False)
>>> mle("the"), mle("cat")
(0.75, 0.15)
```

Swapping methods means swapping the constructor; the call site never changes:

```pycon
>>> laplace = freqprob.Laplace(counts, bins=1000, logprob=False)
>>> round(laplace("the"), 4)
0.0298
```

Each method's own parameters are constructor arguments: `bins`, `gamma`,
`discount` and so on. Fitting happens in the constructor; `fit(freqdist)`
refits the same estimator to new counts.

## The unseen-element problem

A relative frequency gives anything unobserved a probability of exactly zero.
That is usually wrong (an element absent from a sample is rarely impossible),
and it breaks anything multiplicative: one zero makes the product of
probabilities zero, and its logarithm undefined. **Smoothing** moves a little
probability mass from the observed elements to the unseen ones.

```pycon
>>> mle("mouse")
0.0
>>> round(laplace("mouse"), 4)
0.001
```

The methods differ in how much mass they move and in how they share it: evenly
across all possible elements (additive smoothing), according to how many
elements were seen once (Good–Turing), according to the number of distinct
elements (Witten–Bell), or, for n-grams, through a lower-order model
(Kneser–Ney and back-off). [Choosing a method](choosing-a-method.md) compares
them.

## Log space and `logprob`

By default, estimators return **log**-probabilities (`logprob=True`), because
real data multiplies many small probabilities, and products of probabilities
underflow to zero. Pass `logprob=False` for plain probabilities:

```pycon
>>> import math
>>> logp = freqprob.MLE(counts)  # logprob=True by default
>>> round(logp("the"), 4), round(math.log(0.75), 4)
(-0.2877, -0.2877)
```

A log-scorer cannot return $\ln 0$, so an element with no probability scores
$\ln 10^{-10} \approx -23.03$ instead; see
[Pitfalls](pitfalls.md#a-log-scorer-never-returns-minus-infinity).

The [evaluation metrics](../evaluation/model-metrics.md) need log-scorers, and
[Work in log space](../how-to/log-space.md) shows how to combine
log-probabilities safely.

## Batch scoring and the scikit-learn-style API

`predict` scores many elements at once, and `score` is an alias for calling the
estimator:

```pycon
>>> mle.predict(["the", "mat", "mouse"])
[0.75, 0.025, 0.0]
>>> mle.score("cat") == mle("cat")
True
```

For large inputs, [Scale to large data](../how-to/scale.md) describes vectorized,
lazy and streaming estimators that keep the same contract.
