# Pitfalls

Smoothing is easy to apply and easy to misjudge. This page collects the
surprises that come up most often, with the
[example counts](notation.md#the-example-counts):

```pycon
>>> import freqprob
>>> counts = dict(the=10, cat=5, sat=4, on=3, mat=2, a=2, dog=1, ran=1, far=1, away=1)
```

## The default `bins` reserves nothing

The additive methods spread mass over `bins` possible elements. By default,
`bins` is the number of observed elements, so the smoothed probabilities of
the observed elements already sum to one, and every unseen element is scored
on top of that:

```pycon
>>> laplace = freqprob.Laplace(counts, logprob=False)
>>> round(sum(laplace(w) for w in counts), 6), laplace("zebra")
(1.0, 0.025)
```

Declare the number of possible elements, so that the observed ones give up the
mass the unseen ones receive:

```pycon
>>> laplace = freqprob.Laplace(counts, bins=20, logprob=False)
>>> round(sum(laplace(w) for w in counts), 6), laplace("zebra")
(0.8, 0.02)
```

With `bins=20`, the ten unseen elements share the remaining $0.2$. The same
applies to `Lidstone`, `ELE` and `WittenBell`; `Bayesian` always uses the
observed vocabulary. See [Choose `bins`](../how-to/choose-bins.md).

## `unobs_prob` is a per-element score

`MLE`, `Uniform`, `Random` and `CertaintyDegree` accept `unobs_prob`, which
rescales the observed elements to $1 - p_0$ and gives **each** unseen element
the score $p_0$. It is not shared among them:

```pycon
>>> mle = freqprob.MLE(counts, unobs_prob=0.1, logprob=False)
>>> mle("zebra"), mle("yak")
(0.1, 0.1)
```

Querying many unseen elements therefore gives a total above one. Use a method
that knows the support size when that matters.

## A log-scorer never returns minus infinity

A log-scorer replaces a probability of zero with the floor $10^{-10}$, so an
element that should be impossible scores about $-23.03$:

```pycon
>>> round(freqprob.MLE(counts)("zebra"), 2)
-23.03
```

This keeps sums of log-probabilities finite, but it also means that a model
which gives unseen elements no mass is not rejected outright by the metrics:
its perplexity is merely very large.

```pycon
>>> test = ["the", "cat", "zebra"]
>>> round(freqprob.perplexity(freqprob.MLE(counts), test))
5646
>>> round(freqprob.perplexity(freqprob.Laplace(counts, bins=20), test), 2)
12.37
```

## Stupid Back-off returns scores, not probabilities

`StupidBackoff` skips normalization for speed, so its scores need not sum to
one over the words of a context:

```pycon
>>> bigrams = {
...     ("the", "cat"): 5,
...     ("the", "dog"): 3,
...     ("a", "cat"): 2,
...     ("a", "dog"): 1,
...     ("big", "cat"): 1,
...     ("small", "dog"): 1,
... }
>>> sb = freqprob.StupidBackoff(bigrams, logprob=False)
>>> round(sb(("big", "cat")) + sb(("big", "dog")), 3)
1.154
```

Use it for ranking; perplexity and the other metrics assume a probability
distribution.

## The n-gram methods only read bigrams

`KneserNey`, `ModifiedKneserNey`, `AbsoluteDiscounting`, `PitmanYor`,
`KatzBackoff` and `StupidBackoff` take `(context, word)` pairs and ignore other
keys. A word never seen in any bigram has no lower-order probability to back
off to, so it scores 0 (or the log floor):

```pycon
>>> freqprob.KneserNey(bigrams, logprob=False)(("the", "fish"))
0.0
```

Add such words to the training bigrams, or interpolate with a smoothed unigram
model, if they must receive probability.

## Simple Good–Turing needs a spread of low counts

SGT fits a line to the frequencies of frequencies $N_r$. With few distinct
counts, or no singletons, the fit is unreliable, and by default SGT proceeds
silently. Pass `allow_fail=False` to raise a `RuntimeWarning` instead when its
assumptions are not met. Its default `bins` assumes that as many types are
unseen as were seen once; pass the true support size when you know it.

## Random is not smoothing

`Random` assigns reproducible random probabilities, as a control for
comparisons. It does not estimate anything from the counts beyond their range.
