# Additive smoothing: Laplace, Lidstone, ELE, Bayesian

**`Lidstone`**, **`Laplace`**, **`ELE`**, **`Bayesian`** · additive smoothing

Additive smoothing adds a pseudo-count to every possible element before
normalizing, so that no element has probability zero. It is the oldest and
simplest form of smoothing (Laplace's rule of succession), and a robust
default when the number of possible elements is known.

## Definition

With pseudo-count $\gamma$ and $B$ possible elements (`bins`),

$$
P(w) = \frac{c(w) + \gamma}{N + B\gamma},
\qquad P(\text{unseen}) = \frac{\gamma}{N + B\gamma}.
$$

| Estimator | $\gamma$ |
|-----------|----------|
| `Lidstone(counts, gamma, bins)` | any $\gamma \geq 0$ |
| `Laplace(counts, bins)` | 1 ("add one") |
| `ELE(counts, bins)` | 0.5 (the expected likelihood estimate; Jeffreys' prior) |

This is the posterior mean of a multinomial under a symmetric Dirichlet prior
with concentration $\gamma$. `bins` defaults to the observed vocabulary $V$;
see the properties below.

### Bayesian smoothing

`Bayesian(counts, alpha)` computes the same estimate with $\gamma = \alpha$ and
$B = V$, framing $\alpha$ as the strength of a Dirichlet prior. It has no
`bins` argument.

## Example

With the [example counts](../concepts/notation.md#the-example-counts) and a
support of 100 possible words:

```pycon
>>> import freqprob
>>> counts = dict(the=10, cat=5, sat=4, on=3, mat=2, a=2, dog=1, ran=1, far=1, away=1)
>>> for name, model in [
...     ("Laplace", freqprob.Laplace(counts, bins=100, logprob=False)),
...     ("ELE", freqprob.ELE(counts, bins=100, logprob=False)),
...     ("Lidstone 0.1", freqprob.Lidstone(counts, gamma=0.1, bins=100, logprob=False)),
... ]:
...     print(name, [round(model(w), 4) for w in ("the", "dog", "zebra")])
Laplace [0.0846, 0.0154, 0.0077]
ELE [0.1313, 0.0187, 0.0063]
Lidstone 0.1 [0.2525, 0.0275, 0.0025]
```

With 30 observations and 100 possible words, add-one smoothing gives the 90
unseen words $90/130 \approx 69\%$ of the mass, and the most frequent word falls
from $1/3$ to $0.085$. A smaller $\gamma$ stays closer to the counts.

## Properties

- **Simple and always positive.** Every element gets at least
  $\gamma/(N + B\gamma)$.
- **Over-smooths large supports.** The reserved mass is
  $(B - V)\gamma / (N + B\gamma)$, which approaches 1 when $B$ is much larger
  than $N$. Add-one smoothing is known to perform poorly for language data
  for this reason; prefer a small $\gamma$ or
  [Simple Good–Turing](simple-good-turing.md).
- **`bins` must be the support size.** With the default $B = V$, the observed
  elements already sum to one, and each unseen element is scored on top; see
  [Pitfalls](../concepts/pitfalls.md#the-default-bins-reserves-nothing) and
  [Choose `bins`](../how-to/choose-bins.md).
- **Treats all unseen elements alike**, and all observed elements with the
  same count alike.

## API

::: freqprob.Lidstone
    options:
      show_root_heading: true
      heading_level: 3

::: freqprob.Laplace
    options:
      show_root_heading: true
      heading_level: 3

::: freqprob.ELE
    options:
      show_root_heading: true
      heading_level: 3

::: freqprob.Bayesian
    options:
      show_root_heading: true
      heading_level: 3

## References

- Laplace (1814); Lidstone (1920); Manning & Schütze (1999, §6.2); MacKay &
  Peto (1995) for the Dirichlet view. See
  [References](../references.md#additive-smoothing).
