# Coverage and entropy of a sample

**`sample_coverage`**, **`chao_shen_entropy`**, **`nsb_entropy`** · work on
counts directly

These functions estimate properties of the population a sample came from,
directly from the frequency distribution, without fitting a model: how much of
the population's probability the sample has already covered, and the entropy
of the population, corrected for the elements the sample missed. They are
widely used in ecology (species diversity) and in the information-theoretic
measures of [ASymCat](https://asymcat.tresoldi.org/).

## Definition

**Sample coverage** is the Good–Turing estimate of the observed probability
mass, the complement of the unseen mass that
[Simple Good–Turing](../methods/simple-good-turing.md) reserves:

$$
C = 1 - \frac{N_1}{N}.
$$

The plug-in entropy $\hat H = -\sum_w \hat p_w \ln \hat p_w$, with
$\hat p_w = c(w)/N$, underestimates the population entropy when the sample
misses elements. The **Chao–Shen** estimator (Chao & Shen, 2003) scales the
relative frequencies by the coverage and corrects each term for the chance that
its element was observed at all:

$$
\hat H_{\text{CS}} = -\sum_w \frac{C\hat p_w \ln (C\hat p_w)}{1 - (1 - C\hat p_w)^N}.
$$

When every observation is a singleton, $N_1$ is taken as $N - 1$, so that $C$
stays positive. The **NSB** estimator (Nemenman, Shafee & Bialek, 2002) is a
Bayesian estimate that averages the posterior-mean entropy of a Dirichlet model
over its concentration, with a prior chosen to be nearly uniform in entropy.
`bins` sets the number of possible elements, reserving entropy for elements
that could occur but were not observed.

## Example

With the [example counts](../concepts/notation.md#the-example-counts):

```pycon
>>> import math
>>> import freqprob
>>> counts = dict(the=10, cat=5, sat=4, on=3, mat=2, a=2, dog=1, ran=1, far=1, away=1)
>>> round(freqprob.sample_coverage(counts), 4)  # 1 - 4/30
0.8667
>>> n = sum(counts.values())
>>> round(-sum(c / n * math.log(c / n) for c in counts.values()), 3)  # plug-in
1.978
>>> round(freqprob.chao_shen_entropy(counts), 3)
2.22
>>> round(freqprob.nsb_entropy(counts), 3), round(freqprob.nsb_entropy(counts, bins=100), 3)
(2.089, 2.299)
```

Both corrected estimates exceed the plug-in entropy: with four singletons in
30 observations, the sample has likely missed some words. Allowing for 100
possible words raises the NSB estimate further.

## Properties

- **Coverage is a quick diagnostic.** Close to 1, the sample has seen almost
  everything that matters, and smoothing has little to do; far from 1, the
  choice of smoothing method matters.
- **Chao–Shen needs no parameters** and is a good default for undersampled
  data.
- **NSB uses the support size** when it is known, and is well suited to samples
  smaller than the number of possible elements. It is slower than Chao–Shen.

## API

::: freqprob.sample_coverage
    options:
      show_root_heading: true
      heading_level: 3

::: freqprob.chao_shen_entropy
    options:
      show_root_heading: true
      heading_level: 3

::: freqprob.nsb_entropy
    options:
      show_root_heading: true
      heading_level: 3

## References

- Good (1953); Chao & Shen (2003); Nemenman, Shafee & Bialek (2002); Wolpert &
  Wolf (1995). See [References](../references.md#entropy-coverage-and-evaluation).
