# Worked examples by domain

Estimating probabilities from counts, including for what has not been observed,
is the same problem in many fields. Each example below is self-contained.

## Ecology: species abundance and unseen species

Good–Turing estimation was developed for this problem: how much of a
population belongs to species not yet seen in a survey.

```pycon
>>> import freqprob
>>> survey = {
...     "sparrow": 40,
...     "robin": 18,
...     "finch": 12,
...     "hawk": 3,
...     "owl": 2,
...     "crane": 1,
...     "heron": 1,
...     "egret": 1,
... }
>>> round(freqprob.sample_coverage(survey), 3)  # share of individuals from species already seen
0.962
>>> sgt = freqprob.SimpleGoodTuring(survey, logprob=False)
>>> round(sgt.total_unseen_mass, 3)  # chance that the next individual is a new species
0.038
>>> round(freqprob.chao_shen_entropy(survey), 3)  # diversity, corrected for unseen species
1.463
```

## Genomics: k-mer frequencies

Count the k-mers of a sequence and smooth over all $4^k$ possible k-mers, so
that absent k-mers keep a non-zero probability:

```pycon
>>> from collections import Counter
>>> sequence = "ATGGCATGCAATGC"
>>> k = 3
>>> kmers = Counter(sequence[i : i + k] for i in range(len(sequence) - k + 1))
>>> model = freqprob.Lidstone(kmers, gamma=0.5, bins=4**k, logprob=False)
>>> round(model("ATG"), 4), round(model("TTT"), 4)
(0.0795, 0.0114)
```

## Categorical data: priors for rare categories

A smoothed category prior keeps a rare or new category from getting
probability zero, as for click counts, product categories or the categories of
a feature in a classifier:

```pycon
>>> clicks = {"electronics": 120, "books": 80, "toys": 25, "garden": 5}
>>> prior = freqprob.Laplace(clicks, bins=10, logprob=False)
>>> round(prior("electronics"), 3), round(prior("music"), 4)
(0.504, 0.0042)
```

## Text: word and n-gram models

See the [Tutorial](../getting-started/tutorial.md) for word counts, held-out
evaluation and a bigram model with [Kneser–Ney](../methods/kneser-ney.md).
