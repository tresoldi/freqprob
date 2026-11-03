# Baselines: MLE, Uniform, Random

**`MLE`**, **`Uniform`**, **`Random`** · no smoothing · reference points

These three estimators are the reference points against which smoothing is
judged. `MLE` uses the counts as they are, `Uniform` ignores them, and `Random`
replaces them with random ones.

## Definition

The maximum-likelihood estimate is the relative frequency:

$$
P_{\text{MLE}}(w) = (1 - p_0)\,\frac{c(w)}{N},
\qquad P_{\text{MLE}}(\text{unseen}) = p_0,
$$

with $p_0$ = `unobs_prob`, 0 by default. `Uniform` gives every observed element
$(1 - p_0)/V$, whatever its count. `Random` draws a random count between the
smallest and largest observed counts for each element, and normalizes those
like `MLE`; `seed` makes it reproducible.

## Example

With the [example counts](../concepts/notation.md#the-example-counts):

```pycon
>>> import freqprob
>>> counts = dict(the=10, cat=5, sat=4, on=3, mat=2, a=2, dog=1, ran=1, far=1, away=1)
>>> mle = freqprob.MLE(counts, logprob=False)
>>> round(mle("the"), 4), round(mle("dog"), 4), mle("zebra")
(0.3333, 0.0333, 0.0)
>>> freqprob.Uniform(counts, logprob=False)("the")
0.1
>>> random = freqprob.Random(counts, seed=0, logprob=False)
>>> round(sum(random(w) for w in counts), 6)
1.0
```

Reserving mass with `unobs_prob` rescales the observed elements:

```pycon
>>> mle = freqprob.MLE(counts, unobs_prob=0.1, logprob=False)
>>> round(mle("the"), 4), mle("zebra")
(0.3, 0.1)
```

## Properties

- **`MLE` is unbiased for the observed elements** and gives nothing to the
  rest, which makes it unsuitable whenever unseen elements can occur.
- **`unobs_prob` is a per-element score**: every unseen element receives $p_0$
  (see [Pitfalls](../concepts/pitfalls.md#unobs_prob-is-a-per-element-score)).
- **`Uniform` and `Random` are controls.** A smoothing method that does no
  better than them on held-out data is not using the counts well.

## API

::: freqprob.MLE
    options:
      show_root_heading: true
      heading_level: 3

::: freqprob.Uniform
    options:
      show_root_heading: true
      heading_level: 3

::: freqprob.Random
    options:
      show_root_heading: true
      heading_level: 3

## References

Standard baselines; see [References](../references.md#methods-without-a-separate-reference).
