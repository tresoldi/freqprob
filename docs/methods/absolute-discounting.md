# Absolute discounting and Pitman–Yor

**`AbsoluteDiscounting`**, **`PitmanYor`** · n-gram smoothing · bigram input

Absolute discounting (Ney, Essen & Kneser, 1994) subtracts a fixed discount
from every observed bigram count and gives the freed mass to a unigram model.
It is [Kneser–Ney](kneser-ney.md) without the continuation counts. The
Pitman–Yor estimator adds a concentration parameter, following the
Pitman–Yor process (Pitman & Yor, 1997) that underlies hierarchical Bayesian
language models (Teh, 2006).

## Definition

For a bigram $(v, w)$ with count $c(v, w)$ and context total $c(v)$, absolute
discounting with discount $d$ gives

$$
P(w \mid v) = \frac{\max\big(c(v, w) - d,\ 0\big)}{c(v)}
+ \frac{d\,N_{1+}(v)}{c(v)}\,P_{\text{uni}}(w),
$$

where $P_{\text{uni}}$ is the relative frequency of $w$ as the second element
of a bigram. With concentration $\theta$ = `strength` and $t = N_{1+}(v)$,
Pitman–Yor gives

$$
P(w \mid v) = \frac{\max\big(c(v, w) - d,\ 0\big) + (\theta + d\,t)\,P_{\text{uni}}(w)}{\theta + c(v)},
$$

which is absolute discounting when $\theta = 0$. Unseen contexts back off to
$P_{\text{uni}}(w)$.

## Example

With the [example bigrams](../concepts/notation.md#bigrams):

```pycon
>>> import freqprob
>>> bigrams = {
...     ("the", "cat"): 5,
...     ("the", "dog"): 3,
...     ("a", "cat"): 2,
...     ("a", "dog"): 1,
...     ("big", "cat"): 1,
...     ("small", "dog"): 1,
... }
>>> queries = [("the", "cat"), ("big", "dog"), ("new", "cat")]
>>> ad = freqprob.AbsoluteDiscounting(bigrams, discount=0.75, logprob=False)
>>> [round(ad(b), 3) for b in queries]
[0.647, 0.288, 0.615]
>>> py = freqprob.PitmanYor(bigrams, discount=0.75, strength=1.0, logprob=False)
>>> [round(py(b), 3) for b in queries]
[0.643, 0.337, 0.615]
```

"cat" is the more frequent word ($8/13$), so unlike under Kneser–Ney it keeps
that weight in a new context. The concentration parameter moves more mass to
the unigram model: "dog" after "big" rises from 0.288 to 0.337.

## Properties

- **Normalized** over the words of each context.
- **Uses raw unigram frequencies** for the back-off, so a frequent word keeps
  its weight even if it occurs in few contexts; Kneser–Ney corrects this.
- **`strength`** controls the mass given to the unigram model independently of
  the discount; it must exceed $-d$.
- **Bigrams only**, as for Kneser–Ney.

## API

::: freqprob.AbsoluteDiscounting
    options:
      show_root_heading: true
      heading_level: 3

::: freqprob.PitmanYor
    options:
      show_root_heading: true
      heading_level: 3

## References

- Ney, Essen & Kneser (1994); Pitman & Yor (1997); Teh (2006). See
  [References](../references.md#discounting-and-back-off-for-n-grams).
