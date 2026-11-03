# Plot a smoothed distribution

Plotting observed relative frequencies against a smoothed estimate shows where
a method takes mass from. This example uses the word counts of *The Zen of
Python*, as the figure on the [home page](../index.md) does, and matplotlib,
which is not a dependency of FreqProb (`pip install matplotlib`).

## The counts

```pycon
>>> import codecs
>>> import contextlib
>>> import io
>>> import re
>>> import freqprob
>>> with contextlib.redirect_stdout(io.StringIO()):  # `import this` prints the Zen
...     import this
>>> zen = codecs.decode(this.s, "rot13")
>>> counts = freqprob.word_frequency(re.findall(r"[a-z']+", zen.lower()))
>>> n = sum(counts.values())
>>> n, len(counts), sum(1 for c in counts.values() if c == 1)
(143, 85, 60)
>>> sgt = freqprob.SimpleGoodTuring(counts, logprob=False)
>>> round(sgt.total_unseen_mass, 2)
0.42
```

With 60 of 85 words seen once, Simple Good–Turing reserves 42% of the mass for
words not in the text.

## The figure

```pycon
>>> import matplotlib
>>> matplotlib.use("Agg")
>>> import matplotlib.pyplot as plt
>>> top = sorted(counts, key=lambda w: (-counts[w], w))[:11]
>>> fig, ax = plt.subplots(figsize=(8, 3.5), constrained_layout=True)
>>> _ = ax.bar(top, [counts[w] / n for w in top], color="0.7", label="relative frequency (MLE)")
>>> _ = ax.plot(top, [sgt(w) for w in top], "o-", color="black", label="Simple Good–Turing")
>>> _ = ax.set_ylabel("probability")
>>> _ = ax.legend(frameon=False)
>>> fig.savefig("zen_sgt.png", dpi=150)
```

![Relative frequencies of the eleven most frequent words of The Zen of Python as bars, with their lower Simple Good–Turing estimates as a line.](../figures/zen_sgt.png)

The estimates lie below the bars: every observed word gives up some mass to the
unseen ones, the rarer words proportionally more. The figure is produced by
this code when the documentation is built.
