# Choosing a method

Start from the shape of your data and what you need the estimates for.

| Your situation | Methods | Key parameters |
|----------------|---------|----------------|
| Relative frequencies, no smoothing | [`MLE`](../methods/baselines.md) | — |
| A simple, robust default that never returns zero | [`Laplace`, `Lidstone`, `ELE`](../methods/additive.md) | `bins`, `gamma` |
| Smoothing framed as a prior belief | [`Bayesian`](../methods/additive.md#bayesian-smoothing) | `alpha` |
| Heavy-tailed counts with many rare elements | [`SimpleGoodTuring`](../methods/simple-good-turing.md) | `p_value`, `bins` |
| A parameter-free estimate of the unseen mass | [`WittenBell`](../methods/witten-bell.md) | `bins` |
| n-gram language models | [`KneserNey`, `ModifiedKneserNey`](../methods/kneser-ney.md) | `discount` |
| Bigram discounting with back-off to unigrams | [`AbsoluteDiscounting`, `PitmanYor`](../methods/absolute-discounting.md) | `discount`, `strength` |
| Bigram back-off, normalized or fast | [`KatzBackoff`, `StupidBackoff`](../methods/backoff.md) | `k`, `alpha` |
| Combining a specific and a general model | [`Interpolated`, `JelinekMercer`](../methods/interpolation.md) | `lambda_weight` |
| Reserving mass by how fully the support is covered (experimental) | [`CertaintyDegree`](../methods/certainty-degree.md) | `bins` |
| A non-informative baseline | [`Uniform`, `Random`](../methods/baselines.md) | — |

By data characteristics:

| Vocabulary | Data | A good default |
|------------|------|----------------|
| Small | dense: most elements seen often | `MLE` or `Laplace` |
| Large | sparse: many rare or unseen elements | `SimpleGoodTuring`, or `Lidstone` with a small `gamma` |
| Sequence contexts (n-grams) | any | `KneserNey` or `ModifiedKneserNey` |
| Any | a trusted background model is available | `Interpolated` |

## Notes

**Know the number of possible elements.** The additive methods and Witten–Bell
spread the reserved mass over the `bins` possible elements, and with the
default (`bins` equal to the observed vocabulary) they reserve less than you
might expect. See [Choose `bins`](../how-to/choose-bins.md).

**Prefer Simple Good–Turing for heavy tails.** When many elements are seen once
or twice, the number of singletons is direct evidence of how much mass is
missing, and SGT uses it. It needs a spread of low counts to fit.

**Use the n-gram methods for conditional probabilities.** Kneser–Ney, absolute
discounting and back-off model $P(\text{word} \mid \text{context})$ from bigram
counts. For a flat distribution over n-grams as elements, any of the other
methods applies.

**Compare on held-out data.** The [evaluation metrics](../evaluation/model-metrics.md)
score each candidate on data it was not fitted to; that, not the shape of the
estimates, is the best guide.

The [References](../references.md) page gives the published source of each
method.
