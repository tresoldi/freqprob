# Kneser–Ney

**`KneserNey`**, **`ModifiedKneserNey`** · n-gram smoothing · bigram input

Kneser–Ney smoothing (Kneser & Ney, 1995) is the standard classical method for
n-gram language models. It discounts every observed bigram by a fixed amount,
and gives the freed mass to a lower-order model in which a word's weight is the
number of **distinct contexts** it follows, not its frequency. A word that is
frequent but appears in only one context ("Francisco", nearly always after
"San") is then unlikely in a new context. Modified Kneser–Ney (Chen & Goodman,
1999) uses three discounts instead of one.

## Definition

For a bigram $(v, w)$ with count $c(v, w)$, context total $c(v)$ and discount
$d$,

$$
P(w \mid v) = \frac{\max\big(c(v, w) - d,\ 0\big)}{c(v)}
+ \lambda(v)\,P_{\text{cont}}(w),
\qquad
\lambda(v) = \frac{d\,N_{1+}(v)}{c(v)},
$$

where $N_{1+}(v)$ is the number of distinct words seen after $v$, and the
continuation probability is

$$
P_{\text{cont}}(w) = \frac{|\{v : c(v, w) > 0\}|}{|\{(v', w') : c(v', w') > 0\}|}.
$$

An unseen context backs off to $P_{\text{cont}}(w)$ alone. `ModifiedKneserNey`
uses discounts $d_1$, $d_2$ and $d_{3+}$ for bigrams seen once, twice and three
or more times, estimated from the counts of counts.

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
>>> kn = freqprob.KneserNey(bigrams, discount=0.75, logprob=False)
>>> [round(kn(b), 3) for b in [("the", "cat"), ("the", "dog"), ("big", "dog"), ("new", "cat")]]
[0.625, 0.375, 0.375, 0.5]
>>> mkn = freqprob.ModifiedKneserNey(bigrams, logprob=False)
>>> [round(mkn(b), 3) for b in [("the", "cat"), ("a", "cat"), ("big", "dog")]]
[0.625, 0.558, 0.3]
```

"cat" and "dog" each follow three distinct contexts, so their continuation
probabilities are equal (0.5), and a new context gives them the same
probability although "cat" is more frequent. "big" was followed only by "cat",
so "dog" after "big" gets the context's whole back-off weight,
$0.75 \times 0.5 = 0.375$.

## Properties

- **The strongest classical n-gram smoother**, especially in its modified form
  (Chen & Goodman, 1999).
- **Normalized**: the probabilities of the words after each context sum to one.
- **Bigrams only.** Keys must be `(context, word)` pairs; other keys are
  ignored. A word never seen in any bigram scores 0 (the floor, in log
  space).
- **The discount** is typically between 0.5 and 0.8; 0.75 is a robust default.
  `ModifiedKneserNey` estimates its discounts and needs enough variety in the
  counts to do so.

## API

::: freqprob.KneserNey
    options:
      show_root_heading: true
      heading_level: 3

::: freqprob.ModifiedKneserNey
    options:
      show_root_heading: true
      heading_level: 3

## References

- Kneser & Ney (1995); Chen & Goodman (1999). See
  [References](../references.md#discounting-and-back-off-for-n-grams).
