# Estimators

All estimators derive from `ScoringMethod` and share its contract: build from a
frequency distribution, call to score. See
[How estimators work](../concepts/estimators.md).

## Base class

::: freqprob.ScoringMethod
    options:
      show_root_heading: true
      heading_level: 3

## Smoothing methods

::: freqprob.methods.baselines
    options:
      show_root_heading: false
      heading_level: 3

::: freqprob.methods.additive
    options:
      show_root_heading: false
      heading_level: 3

::: freqprob.methods.bayesian
    options:
      show_root_heading: false
      heading_level: 3

::: freqprob.methods.goodturing
    options:
      show_root_heading: false
      heading_level: 3

::: freqprob.methods.kneser_ney
    options:
      show_root_heading: false
      heading_level: 3

::: freqprob.methods.discounting
    options:
      show_root_heading: false
      heading_level: 3

::: freqprob.methods.backoff
    options:
      show_root_heading: false
      heading_level: 3

::: freqprob.methods.interpolated
    options:
      show_root_heading: false
      heading_level: 3
      members:
        - Interpolated
        - JelinekMercer

::: freqprob.methods.certainty
    options:
      show_root_heading: false
      heading_level: 3
