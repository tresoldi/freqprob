# Tutorial

This tutorial goes from raw text to smoothed probabilities, a comparison of
methods on held-out data, and estimates for n-grams. It takes about ten minutes,
and every example can be pasted into a Python session. The examples use words,
but FreqProb works on counts of anything: species, k-mers, categories, events.

## From counts to probabilities

FreqProb starts from a **frequency distribution**: a mapping from elements to
how often each was observed. `word_frequency` builds one from a list of tokens:

```pycon
>>> import freqprob
>>> text = "the cat sat on the mat and the dog sat on the log".split()
>>> counts = freqprob.word_frequency(text)
>>> counts["the"], counts["sat"], len(counts)
(4, 2, 8)
```

An estimator is built from the counts and then called to score an element:

```pycon
>>> mle = freqprob.MLE(counts, logprob=False)
>>> mle("the"), mle("cat"), mle("bird")
(0.3076923076923077, 0.07692307692307693, 0.0)
```

`MLE` is the relative frequency: "the" is 4 of the 13 words. The unseen "bird"
gets probability 0, which is the problem smoothing solves.

## Smooth the estimate

A smoothing method moves some probability to elements that were not observed.
Laplace smoothing adds one pseudo-count to every **possible** element, so it
needs to know how many there are (`bins`):

```pycon
>>> laplace = freqprob.Laplace(counts, bins=1000, logprob=False)
>>> round(laplace("the"), 5), round(laplace("bird"), 5)
(0.00494, 0.00099)
```

With 1000 possible words and only 13 observations, add-one smoothing gives
almost all the mass to the unseen words. Simple Good–Turing instead estimates
the unseen mass from the number of words seen once:

```pycon
>>> sgt = freqprob.SimpleGoodTuring(counts, logprob=False)
>>> round(sgt.total_unseen_mass, 3)  # 5 singletons in 13 words
0.385
>>> round(sgt("the"), 3)
0.22
```

The [method pages](../methods/index.md) explain each estimator, and
[Choosing a method](../concepts/choosing-a-method.md) compares them.

## Work in log space

By default, estimators return log-probabilities, so that scores can be added
instead of multiplied without underflowing:

```pycon
>>> log_sgt = freqprob.SimpleGoodTuring(counts)  # logprob=True by default
>>> round(log_sgt("the"), 3)
-1.516
>>> round(sum(log_sgt(w) for w in "the cat sat".split()), 3)  # log P(the) + log P(cat) + log P(sat)
-7.059
```

## Compare methods on held-out data

To choose a method, score each on data it was not fitted to. Lower perplexity
is better:

```pycon
>>> held_out = "the cat sat on the rug".split()
>>> models = {
...     "laplace": freqprob.Laplace(counts, bins=50),
...     "good_turing": freqprob.SimpleGoodTuring(counts),
...     "witten_bell": freqprob.WittenBell(counts, bins=50),
... }
>>> for name, metrics in freqprob.model_comparison(models, held_out).items():
...     print(name, round(metrics["perplexity"], 2))
laplace 22.76
good_turing 9.44
witten_bell 13.84
```

See [Evaluating models](../evaluation/model-metrics.md) for the metrics.

## Estimate n-grams

For sequences, `ngram_frequency` counts n-grams, and the n-gram methods estimate
the probability of a word given its context:

```pycon
>>> bigrams = freqprob.ngram_frequency(text, 2)
>>> bigrams[("the", "cat")], bigrams[("sat", "on")]
(1, 2)
>>> kn = freqprob.KneserNey(bigrams, discount=0.75, logprob=False)
>>> round(kn(("sat", "on")), 3), round(kn(("the", "mat")), 3)
(0.662, 0.138)
```

After "sat", "on" always followed, so it keeps most of the mass; after "the",
four different words were seen, and "mat" gets a smaller share. See [Kneser–Ney](../methods/kneser-ney.md).

## Next steps

- [How estimators work](../concepts/estimators.md) and
  [Notation](../concepts/notation.md) explain the contract and the symbols.
- [Pitfalls](../concepts/pitfalls.md) lists the surprises to watch for.
- The [how-to guides](../how-to/index.md) cover choosing `bins`, scaling to
  large data, saving models and worked examples by domain.
- The [API reference](../api/index.md) documents every function.
