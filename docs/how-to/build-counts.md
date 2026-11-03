# Build frequency distributions

Every estimator takes a **frequency distribution**: a mapping from elements to
counts. Any `dict` or `collections.Counter` works. Elements can be strings,
numbers, tuples or frozensets.

## Words

`word_frequency` counts the words of a list of tokens, or of a string split on
whitespace:

```pycon
>>> import freqprob
>>> counts = freqprob.word_frequency("the cat sat on the mat")
>>> counts["the"], counts["cat"]
(2, 1)
```

With `normalize=True`, it returns relative frequencies instead; estimators
expect raw counts.

## n-grams

`generate_ngrams` lists the n-grams of a sequence, and `ngram_frequency`
counts them. A list is a sequence of tokens; a string is a sequence of
characters:

```pycon
>>> tokens = "the cat sat on the mat".split()
>>> freqprob.generate_ngrams(tokens, 2)[:3]
[('the', 'cat'), ('cat', 'sat'), ('sat', 'on')]
>>> freqprob.ngram_frequency("banana", 2)
{('b', 'a'): 1, ('a', 'n'): 2, ('n', 'a'): 2}
```

Bigram counts are the input of the [n-gram methods](../methods/kneser-ney.md).

## Anything else

For other data, count with `collections.Counter`. Here, the 3-mers of a DNA
sequence:

```pycon
>>> from collections import Counter
>>> sequence = "ATGGCATGCA"
>>> kmers = Counter(sequence[i : i + 3] for i in range(len(sequence) - 2))
>>> kmers.most_common(2)
[('ATG', 2), ('GCA', 2)]
>>> freqprob.Laplace(kmers, bins=4**3, logprob=False)("TTT") > 0
True
```

With pandas, `df["column"].value_counts().to_dict()` gives the counts of a
column.
