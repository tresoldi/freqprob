# Katz and Stupid back-off

**`KatzBackoff`**, **`StupidBackoff`** · n-gram back-off · bigram input

Back-off models use the bigram estimate when the bigram has been seen, and
fall back ("back off") to the unigram model when it has not. Katz (1987)
discounts the observed counts with Good–Turing and normalizes the back-off, so
the result is a probability distribution. Stupid Back-off (Brants et al.,
2007) skips the discounting and normalization, which makes it very cheap at web
scale, but its scores are not probabilities.

## Definition

For a bigram $(v, w)$ with count $c = c(v, w)$ and context total $c(v)$, Katz
back-off gives

$$
P(w \mid v) =
\begin{cases}
d_c\,\dfrac{c}{c(v)} & \text{if } c > 0,\\[1ex]
\alpha(v)\,P_{\text{uni}}(w) & \text{otherwise,}
\end{cases}
$$

where $d_c$ is the Good–Turing discount ratio for counts $1 \leq c \leq k$
($d_c = 1$ above $k$) and $\alpha(v)$ makes each context's distribution sum
to one. Stupid Back-off scores

$$
S(w \mid v) =
\begin{cases}
c / c(v) & \text{if } c > 0,\\
\alpha\,P_{\text{uni}}(w) & \text{otherwise,}
\end{cases}
$$

with a fixed $\alpha$ = `alpha`, 0.4 by default.

## Example

Katz back-off needs a spread of low counts to estimate its discounts:

```pycon
>>> import freqprob
>>> bigrams = {
...     ("the", "cat"): 4,
...     ("the", "dog"): 3,
...     ("a", "cat"): 2,
...     ("a", "dog"): 2,
...     ("my", "cat"): 1,
...     ("my", "dog"): 1,
...     ("my", "bird"): 1,
...     ("his", "fish"): 1,
...     ("her", "fish"): 1,
...     ("its", "bird"): 1,
... }
>>> katz = freqprob.KatzBackoff(bigrams, logprob=False)
>>> [round(katz(b), 3) for b in [("the", "cat"), ("my", "cat"), ("my", "fish")]]
[0.571, 0.222, 0.333]
>>> sb = freqprob.StupidBackoff(bigrams, logprob=False)
>>> [round(sb(b), 3) for b in [("the", "cat"), ("my", "cat"), ("my", "fish")]]
[0.571, 0.333, 0.047]
```

Under Katz back-off, the singletons after "my" are discounted from $1/3$ to
$0.222$, and the freed mass goes to unseen words such as "fish". Stupid
Back-off leaves the observed scores alone and scores "fish" after "my" as
$0.4 \times P_{\text{uni}}(\text{fish}) = 0.4 \times 2/17$.

## Properties

- **Katz is normalized**; Stupid Back-off is not, so its scores are for ranking
  and not for perplexity (see
  [Pitfalls](../concepts/pitfalls.md#stupid-back-off-returns-scores-not-probabilities)).
- **Katz needs varied low counts.** With too few distinct counts, some
  discounts default to 1, leaving little mass for unseen bigrams.
- **Bigrams only.** Keys must be `(context, word)` pairs.

## API

::: freqprob.KatzBackoff
    options:
      show_root_heading: true
      heading_level: 3

::: freqprob.StupidBackoff
    options:
      show_root_heading: true
      heading_level: 3

## References

- Katz (1987); Brants et al. (2007). See
  [References](../references.md#discounting-and-back-off-for-n-grams).
