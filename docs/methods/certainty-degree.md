# Certainty degree

**`CertaintyDegree`** · experimental

!!! warning "Experimental"
    `CertaintyDegree` is an estimator under development, without a published
    reference. It is documented for completeness; do not use it as your only or
    main estimator yet.

The certainty degree reserves mass for unseen elements according to how
plausible it is that the sample has already covered the support.

## Definition

With $N$ observations of $V$ distinct elements and a support of $B$ elements
(`bins`, $V$ by default), the mass reserved for unseen elements is

$$
p_0 = \max\!\Big(\big(\tfrac{V}{B + 1}\big)^{N},\ \texttt{unobs\_prob}\Big),
$$

the probability that $N$ draws from $B + 1$ equally likely elements all fall
among the $V$ observed ones. The observed elements share the rest in proportion
to their counts, $P(w) = (1 - p_0)\,c(w)/N$, and **each** unseen element scores
$p_0$.

## Example

With the [example counts](../concepts/notation.md#the-example-counts):

```pycon
>>> import freqprob
>>> counts = dict(the=10, cat=5, sat=4, on=3, mat=2, a=2, dog=1, ran=1, far=1, away=1)
>>> cd = freqprob.CertaintyDegree(counts, logprob=False)
>>> [round(cd(w), 4) for w in ("the", "dog", "zebra")]
[0.3142, 0.0314, 0.0573]
>>> round(freqprob.CertaintyDegree(counts, bins=100, logprob=False)("zebra"), 6)
0.0
```

With the default support, $p_0 = (10/11)^{30} \approx 0.057$. Declaring a
support of 100 elements makes $p_0$ vanish: if the 30 observations had been
spread over 101 equally likely elements, finding only ten of them would be very
unlikely.

## Properties

- **Shrinks quickly with $N$**: the reserved mass falls exponentially with the
  number of observations.
- **A larger `bins` reserves less mass**, unlike the additive methods, because
  it makes the observed concentration on few elements more telling.
- **The unseen score is per element**, as for `unobs_prob` in `MLE` (see
  [Pitfalls](../concepts/pitfalls.md#unobs_prob-is-a-per-element-score)).

## API

::: freqprob.CertaintyDegree
    options:
      show_root_heading: true
      heading_level: 3

## References

No published reference; see
[References](../references.md#methods-without-a-separate-reference).
