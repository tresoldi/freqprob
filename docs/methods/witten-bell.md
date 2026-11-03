# Witten–Bell

**`WittenBell`** · discounting · estimates the unseen mass from the number of
types

Witten–Bell smoothing (Witten & Bell, 1991) reserves mass for unseen elements
in proportion to how often a **new** element has been observed: each of the
$V$ distinct elements was new once, out of $N + V$ events of "seeing an
element" or "seeing a new element". It has no parameters beyond the support
size.

## Definition

$$
P(w) = \frac{c(w)}{N + V},
\qquad p_0 = \frac{V}{N + V},
\qquad P(\text{unseen}) = \frac{p_0}{B - V},
$$

where $B$ = `bins`. With the default (`bins` unset, or equal to $V$), the whole
reserved mass goes to a single unseen bin, so every unseen element scores
$p_0$.

## Example

With the [example counts](../concepts/notation.md#the-example-counts),
$V = 10$ and $N = 30$:

```pycon
>>> import freqprob
>>> counts = dict(the=10, cat=5, sat=4, on=3, mat=2, a=2, dog=1, ran=1, far=1, away=1)
>>> wb = freqprob.WittenBell(counts, bins=100, logprob=False)
>>> [round(wb(w), 4) for w in ("the", "dog", "zebra")]
[0.25, 0.025, 0.0028]
```

The observed words keep $30/40$ of the mass, and the 90 unseen words share the
remaining $10/40$.

## Properties

- **Parameter-free and assumption-light.** It needs no fit, so it works where
  Simple Good–Turing's assumptions fail.
- **Discounts all observed elements by the same factor**, $N/(N + V)$, unlike
  Good–Turing, which discounts rare elements more.
- **Reserves more mass when the vocabulary grows quickly**: a sample in which
  most observations are new types predicts more new types.
- **Pass `bins`** to share the unseen mass among the possible elements; see
  [Pitfalls](../concepts/pitfalls.md#the-default-bins-reserves-nothing).

## API

::: freqprob.WittenBell
    options:
      show_root_heading: true
      heading_level: 3

## References

- Witten & Bell (1991). See
  [References](../references.md#goodturing-and-wittenbell).
