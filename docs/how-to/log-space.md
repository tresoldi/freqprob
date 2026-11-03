# Work in log space

Probabilities of long sequences are products of many small numbers, which
underflow to zero in floating point. Log-probabilities add instead, and stay
finite. Estimators return log-probabilities by default (`logprob=True`), and the
[evaluation metrics](../evaluation/model-metrics.md) require them.

## Products become sums

```pycon
>>> import math
>>> probs = [1e-30] * 20
>>> math.prod(probs)
0.0
>>> round(sum(math.log(p) for p in probs), 1)
-1381.6
```

To score a sequence, add the log-probabilities of its elements:

```pycon
>>> import freqprob
>>> model = freqprob.Laplace({"the": 3, "cat": 1, "sat": 1}, bins=10)
>>> round(sum(model(w) for w in ["the", "cat", "sat"]), 3)
-5.352
```

## Sums need log-sum-exp

Adding probabilities that are stored as logarithms, as when mixing two models,
needs a stable log-sum-exp rather than exponentiating first:

```pycon
>>> from scipy.special import logsumexp
>>> log_a, log_b = -800.0, -801.0
>>> math.exp(log_a)  # underflows
0.0
>>> round(float(logsumexp([log_a, log_b])), 4)  # log(exp(a) + exp(b))
-799.6867
```

## The floor for impossible elements

A log-scorer cannot return $\ln 0 = -\infty$, so elements with no probability
score $\ln 10^{-10} \approx -23.03$. The floor keeps sums finite; see
[Pitfalls](../concepts/pitfalls.md#a-log-scorer-never-returns-minus-infinity)
for its effect on the metrics.

```pycon
>>> round(freqprob.MLE({"a": 1})("b"), 2)
-23.03
```
