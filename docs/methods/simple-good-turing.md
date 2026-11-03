# Simple Good–Turing

**`SimpleGoodTuring`** · discounting · estimates the unseen mass from the
singletons

Good–Turing estimation (Good, 1953, after Turing) uses the **frequencies of
frequencies** to decide how much probability belongs to elements not seen. Its
central insight is that the mass of everything unseen is estimated by the
share of observations that are singletons. Gale and Sampson's (1995) "simple"
version smooths the frequencies of frequencies with a log-linear fit, which
makes it practical for real data.

## Definition

Let $N_r$ be the number of elements seen exactly $r$ times. The total mass
reserved for unseen elements is

$$
p_0 = \frac{N_1}{N},
$$

and an element seen $r$ times is given the adjusted count

$$
r^* = (r + 1)\,\frac{S(N_{r+1})}{S(N_r)},
$$

where $S$ is the log-linear smoothing of $N_r$. For small $r$, the empirical
Turing estimate ($S(N_r) = N_r$) is used for as long as it differs
significantly from the smoothed one, at the level `p_value`. The adjusted
counts are renormalized so that the observed elements share $1 - p_0$.

An unseen element is given $p_0 / (B - V)$. The support size $B$ = `bins`
defaults to $V + N_1$: as many unseen elements as singletons.
`total_unseen_mass` reports $p_0$.

## Example

With the [example counts](../concepts/notation.md#the-example-counts), four
of the 30 observations are singletons:

```pycon
>>> import freqprob
>>> counts = dict(the=10, cat=5, sat=4, on=3, mat=2, a=2, dog=1, ran=1, far=1, away=1)
>>> sgt = freqprob.SimpleGoodTuring(counts, logprob=False)
>>> round(sgt.total_unseen_mass, 4)  # N1 / N = 4 / 30
0.1333
>>> [round(sgt(w), 4) for w in ("the", "dog", "zebra")]
[0.3066, 0.0247, 0.0333]
```

The four default unseen elements share $0.133$, so each gets $0.033$. Each
singleton gives up a quarter of its relative frequency ($0.033 \to 0.025$),
while the most frequent word gives up less ($0.333 \to 0.307$).

## Properties

- **Principled unseen mass.** $N_1/N$ is a direct estimate of the probability
  that the next observation is new, which is what most applications need.
- **Discounts rare elements most.** High counts are reliable and barely
  adjusted.
- **Needs a spread of low counts.** The log-linear fit needs several distinct
  values of $r$, and the method assumes singletons are present. By default it
  proceeds when these assumptions fail; `allow_fail=False` raises a
  `RuntimeWarning` instead. With no singletons, $p_0$ is `default_p0`, or
  $1/(N + 1)$ if that is not given.
- **`bins` only shares the unseen mass out.** It does not change $p_0$; pass the
  true support size when it is known.
- The same estimate of the unseen mass underlies
  [`sample_coverage`](../evaluation/sample-statistics.md).

## API

::: freqprob.SimpleGoodTuring
    options:
      show_root_heading: true
      heading_level: 3

## References

- Good (1953); Gale & Sampson (1995). See
  [References](../references.md#goodturing-and-wittenbell).
