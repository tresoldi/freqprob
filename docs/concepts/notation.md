# Notation

This page fixes the notation used across the documentation and the example
counts that the method pages share.

## Counts and probabilities

FreqProb takes a **frequency distribution**: a mapping from elements to the
number of times each was observed. Elements can be strings, numbers, tuples
(such as n-grams) or frozensets.

| Symbol | Meaning |
|--------|---------|
| $c(w)$ | the count of element $w$ |
| $N$ | the total count, $\sum_w c(w)$ |
| $V$ | the number of distinct elements observed (the observed vocabulary) |
| $B$ | the number of **possible** elements, observed or not (`bins`) |
| $N_r$ | the number of elements observed exactly $r$ times ("frequency of frequencies"); $N_1$ counts the singletons |
| $P(w)$ | the estimated probability of $w$ |
| $p_0$ | probability mass reserved for elements not observed |

Logarithms are natural logarithms, and entropies are in nats.

## The example counts

The method pages use the same small, Zipf-like distribution: ten words with
$N = 30$, $V = 10$ and four singletons.

```pycon
>>> counts = dict(the=10, cat=5, sat=4, on=3, mat=2, a=2, dog=1, ran=1, far=1, away=1)
>>> sum(counts.values()), len(counts)
(30, 10)
>>> sum(1 for c in counts.values() if c == 1)
4
```

The maximum-likelihood estimate is the relative frequency, $c(w)/N$:

```pycon
>>> import freqprob
>>> mle = freqprob.MLE(counts, logprob=False)
>>> round(mle("the"), 4), round(mle("dog"), 4), mle("zebra")
(0.3333, 0.0333, 0.0)
```

Every other method answers the question this leaves open: how much
probability should `"zebra"`, never observed, receive, and where should it
come from?

## Bigrams

The n-gram methods ([Kneser–Ney](../methods/kneser-ney.md),
[absolute discounting](../methods/absolute-discounting.md),
[back-off](../methods/backoff.md)) take counts of `(context, word)` pairs and
estimate the conditional probability $P(\text{word} \mid \text{context})$. They
use this example:

```pycon
>>> bigrams = {
...     ("the", "cat"): 5,
...     ("the", "dog"): 3,
...     ("a", "cat"): 2,
...     ("a", "dog"): 1,
...     ("big", "cat"): 1,
...     ("small", "dog"): 1,
... }
```

For a bigram, $c(\text{context}, \text{word})$ is its count,
$c(\text{context})$ the total count of the context, and
$N_{1+}(\text{context})$ the number of distinct words seen after it.

## Unseen elements: per element and in total

A method reserves some total mass $p_0$ for elements it has not observed, and
gives each of them a share of it. When the number of possible elements $B$ is
known, an unseen element receives about $p_0 / (B - V)$. A call to the
estimator always returns the **per-element** probability;
[Simple Good–Turing](../methods/simple-good-turing.md) also reports the total
as `total_unseen_mass`.
