# Choose `bins`

`bins` is the number of **possible** elements, observed or not: the size of the
vocabulary, the number of species in a region, $4^k$ for DNA k-mers. The
additive methods and Witten–Bell share the reserved mass among the
`bins - V` unseen elements, and Simple Good–Turing uses it to divide its unseen
mass. It is the most consequential parameter of those methods.

## When you know the support

Use the true number of possible elements:

```pycon
>>> import freqprob
>>> kmers = {"ATG": 2, "TGG": 1, "GGC": 1, "GCA": 2, "CAT": 1, "TGC": 1}
>>> model = freqprob.Laplace(kmers, bins=4**3, logprob=False)  # 64 possible 3-mers
>>> round(model("ATG"), 4), round(model("TTT"), 4)
(0.0417, 0.0139)
```

## When you do not

Leaving `bins` at its default (the observed vocabulary) reserves nothing: the
observed elements already sum to one, and each unseen element is scored on top
(see [Pitfalls](../concepts/pitfalls.md#the-default-bins-reserves-nothing)).
Instead:

- **Estimate the support from the data.** Simple Good–Turing's default assumes
  as many unseen elements as singletons, $V + N_1$, which is a reasonable
  starting point for any method.
- **Leave headroom.** For a vocabulary that keeps growing, 20–50% more than
  observed is a common rule of thumb.
- **Compare on held-out data.** Try a few values and keep the one with the lowest
  [perplexity](../evaluation/model-metrics.md), comparing models with the same
  `bins`.

```pycon
>>> counts = dict(the=10, cat=5, sat=4, on=3, mat=2, a=2, dog=1, ran=1, far=1, away=1)
>>> singletons = sum(1 for c in counts.values() if c == 1)
>>> len(counts) + singletons  # V + N1
14
```

## How much mass it reserves

For additive smoothing with pseudo-count $\gamma$, the unseen elements receive
$(B - V)\gamma / (N + B\gamma)$ in total, which grows towards 1 as $B$ grows.
A very large `bins` with add-one smoothing can give most of the mass to
elements never seen; use a smaller `gamma` ([`Lidstone`](../methods/additive.md))
or [Simple Good–Turing](../methods/simple-good-turing.md) then.
