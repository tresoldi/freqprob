# Evaluating models

**`perplexity`**, **`cross_entropy`**, **`kl_divergence`**,
**`model_comparison`** · need log-scorers

The way to choose between estimators is to ask how well each predicts data it
was not fitted to. These metrics score a fitted model on a list of held-out
elements. All of them need models built with `logprob=True` (the default).

## Definition

For held-out elements $w_1, \ldots, w_n$ and a model $P$,

$$
H = -\frac{1}{n}\sum_{i=1}^{n} \ln P(w_i),
\qquad
\mathrm{PP} = e^{H}.
$$

`cross_entropy` returns $H$ in nats and `perplexity` returns $\mathrm{PP}$: the
effective number of equally likely choices the model faces at each element.
Lower is better for both. `kl_divergence(p, q, elements)` sums
$P(w) \ln \big(P(w)/Q(w)\big)$ over the given elements: pass each element of
the support once to obtain the Kullback–Leibler divergence of $Q$ from $P$ over
that support (an element listed twice is counted twice). `model_comparison`
returns the perplexity and cross-entropy of several models at once.

## Example

Fit on the [example counts](../concepts/notation.md#the-example-counts) and
evaluate on a held-out sample that includes an unseen word:

```pycon
>>> import freqprob
>>> counts = dict(the=10, cat=5, sat=4, on=3, mat=2, a=2, dog=1, ran=1, far=1, away=1)
>>> held_out = ["the", "cat", "sat", "on", "the", "rug"]
>>> models = {
...     "laplace": freqprob.Laplace(counts, bins=20),
...     "good_turing": freqprob.SimpleGoodTuring(counts),
...     "witten_bell": freqprob.WittenBell(counts, bins=20),
... }
>>> report = freqprob.model_comparison(models, held_out)
>>> for name, metrics in report.items():
...     print(name, round(metrics["perplexity"], 2))
laplace 10.12
good_turing 7.73
witten_bell 9.38
>>> round(freqprob.cross_entropy(models["good_turing"], held_out), 3)
2.045
```

## Properties

- **Hold data out.** Evaluated on its own training data, the unsmoothed `MLE`
  always wins, because it gives the training data the highest possible
  likelihood.
- **Unseen elements dominate.** One unseen element scored at the log floor of a
  model that reserves no mass adds about 23 nats to the sum; see
  [Pitfalls](../concepts/pitfalls.md#a-log-scorer-never-returns-minus-infinity).
- **Compare models on the same support.** A model that spreads mass over more
  possible elements gives each a lower probability; compare models with the
  same `bins`, or report it.
- **Only for probability models.** Stupid Back-off returns unnormalized scores.

## API

::: freqprob.perplexity
    options:
      show_root_heading: true
      heading_level: 3

::: freqprob.cross_entropy
    options:
      show_root_heading: true
      heading_level: 3

::: freqprob.kl_divergence
    options:
      show_root_heading: true
      heading_level: 3

::: freqprob.model_comparison
    options:
      show_root_heading: true
      heading_level: 3

## References

- Shannon (1948); Kullback & Leibler (1951). See
  [References](../references.md#entropy-coverage-and-evaluation).
