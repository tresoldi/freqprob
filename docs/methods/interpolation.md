# Interpolation: Interpolated and Jelinek–Mercer

**`Interpolated`**, **`JelinekMercer`** · linear interpolation

Interpolation mixes a specific model with a general one: a trigram model with a
bigram model, say, or a small in-domain sample with a large background corpus.
The specific model is sharper where it has evidence; the general model covers
what it has not seen. Jelinek and Mercer (1980) fit the mixing weight on
held-out data.

## Definition

With $P_{\text{high}}$ and $P_{\text{low}}$ the relative frequencies of the two
distributions and $\lambda$ = `lambda_weight`,

$$
P(w) = \lambda\,P_{\text{high}}(w) + (1 - \lambda)\,P_{\text{low}}(\operatorname{ctx}(w)).
$$

When the high-order keys are longer tuples than the low-order ones, the
low-order key is the suffix of the high-order n-gram: bigram $(w_2, w_3)$ for
trigram $(w_1, w_2, w_3)$. When both have the same type of key, keys are matched
directly. `JelinekMercer` estimates $\lambda$ by expectation–maximization on a
held-out distribution (`held_out_dist`), maximizing its likelihood; without
held-out data it uses `lambda_weight`.

## Example

```pycon
>>> import freqprob
>>> trigrams = {("the", "big", "cat"): 3, ("a", "big", "dog"): 2}
>>> bigrams = {("big", "cat"): 5, ("big", "dog"): 3, ("small", "cat"): 2}
>>> interp = freqprob.Interpolated(trigrams, bigrams, lambda_weight=0.7, logprob=False)
>>> round(interp(("the", "big", "cat")), 3)  # 0.7 * 3/5 + 0.3 * 5/10
0.57
>>> round(interp(("one", "big", "cat")), 3)  # unseen trigram: 0.3 * 5/10
0.15
>>> held_out = {("the", "big", "cat"): 8, ("a", "big", "dog"): 6}
>>> jm = freqprob.JelinekMercer(trigrams, bigrams, held_out_dist=held_out, logprob=False)
>>> jm.estimated_lambda > 0.5
True
```

The held-out data repeats the trigrams the high-order model already knows, so EM
moves the weight towards it.

## Properties

- **Keeps specificity and coverage**: unseen high-order n-grams still get the
  low-order mass.
- **The weight matters.** Tune it on held-out data, which `JelinekMercer` does
  for you; the held-out data must be disjoint from the training data.
- **One global weight.** The original scheme uses weights per context bucket;
  `JelinekMercer` fits a single $\lambda$.
- **Relative frequencies on both levels.** To smooth each level as well, combine
  a smoothed estimator's scores yourself.

## API

::: freqprob.Interpolated
    options:
      show_root_heading: true
      heading_level: 3

::: freqprob.JelinekMercer
    options:
      show_root_heading: true
      heading_level: 3

## References

- Jelinek & Mercer (1980). See
  [References](../references.md#interpolation-and-bayesian-smoothing).
