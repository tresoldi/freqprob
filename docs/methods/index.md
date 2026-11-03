# Methods

Every estimator is built from a frequency distribution and called to score an
element (see [How estimators work](../concepts/estimators.md)). Each page below
gives the definition as FreqProb computes it, a worked example on the shared
[example counts](../concepts/notation.md#the-example-counts) or
[bigrams](../concepts/notation.md#bigrams), its properties, the API entry and
the references.

| Page | Estimators | Input | Unseen mass from |
|------|------------|-------|------------------|
| [Baselines](baselines.md) | `MLE`, `Uniform`, `Random` | counts | `unobs_prob` (none by default) |
| [Additive smoothing](additive.md) | `Lidstone`, `Laplace`, `ELE`, `Bayesian` | counts | a pseudo-count per possible element |
| [Simple Good–Turing](simple-good-turing.md) | `SimpleGoodTuring` | counts | the share of singletons, $N_1/N$ |
| [Witten–Bell](witten-bell.md) | `WittenBell` | counts | the number of types, $V/(N + V)$ |
| [Kneser–Ney](kneser-ney.md) | `KneserNey`, `ModifiedKneserNey` | bigrams | discounts, shared by continuation counts |
| [Absolute discounting and Pitman–Yor](absolute-discounting.md) | `AbsoluteDiscounting`, `PitmanYor` | bigrams | discounts, shared by unigram frequency |
| [Katz and Stupid back-off](backoff.md) | `KatzBackoff`, `StupidBackoff` | bigrams | Good–Turing discounts (Katz); none (Stupid) |
| [Interpolation](interpolation.md) | `Interpolated`, `JelinekMercer` | two distributions | the lower-order model |
| [Certainty degree](certainty-degree.md) | `CertaintyDegree` (experimental) | counts | the plausibility of full coverage |

To decide which to use, see [Choosing a method](../concepts/choosing-a-method.md).
To compare them on your data, see [Evaluating models](../evaluation/model-metrics.md).
