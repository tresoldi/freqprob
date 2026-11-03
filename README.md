# FreqProb

[![PyPI version](https://img.shields.io/pypi/v/freqprob.svg)](https://pypi.org/project/freqprob/)
[![Python versions](https://img.shields.io/pypi/pyversions/freqprob.svg)](https://pypi.org/project/freqprob/)
[![CI](https://github.com/tresoldi/freqprob/actions/workflows/quality.yml/badge.svg)](https://github.com/tresoldi/freqprob/actions/workflows/quality.yml)
[![Docs](https://img.shields.io/badge/docs-freqprob.tresoldi.org-blue.svg)](https://freqprob.tresoldi.org/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23917393.svg)](https://doi.org/10.5281/zenodo.23917393)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](https://github.com/tresoldi/freqprob/blob/main/LICENSE)

**Turn frequency counts into probability estimates.**

FreqProb converts a mapping of elements to observed counts into smoothed
probabilities that handle unseen elements sensibly. It is a general-purpose
statistical tool: natural language processing is one use among many, alongside
information retrieval, ecology, genomics and categorical data.

If you use FreqProb in research, please cite it:
[doi:10.5281/zenodo.23917394](https://doi.org/10.5281/zenodo.23917394)
(version 0.8.1; see [Citation](#citation)).

## Install

```bash
pip install freqprob
```

FreqProb needs only NumPy and SciPy.

## Quick start

```pycon
>>> import freqprob
>>> counts = {"the": 100, "cat": 50, "dog": 30, "bird": 10}
>>> freqprob.MLE(counts, logprob=False)("elephant")  # raw relative frequency
0.0
>>> laplace = freqprob.Laplace(counts, bins=10_000, logprob=False)
>>> round(laplace("cat"), 4)  # an observed element
0.005
>>> round(laplace("elephant"), 4)  # an unseen element still gets probability mass
0.0001
```

A raw relative-frequency estimate gives the unseen `"elephant"` probability 0,
which breaks any model that multiplies probabilities or takes their logarithm.
Smoothing reserves some mass for what has not been seen yet, and FreqProb offers
a dozen well-tested ways to do it behind one interface.

## Scoring and evaluating

Every estimator is built from a frequency distribution and then called to score
an element. `predict` scores many at once, metrics such as `perplexity`
evaluate a model, and sample-based estimators work directly on the counts:

```pycon
>>> import freqprob
>>> bigrams = {("the", "cat"): 5, ("the", "dog"): 3, ("a", "cat"): 2}
>>> kneser_ney = freqprob.KneserNey(bigrams, discount=0.75, logprob=False)
>>> [round(p, 3) for p in kneser_ney.predict([("the", "cat"), ("a", "dog")])]
[0.656, 0.125]
>>> round(
...     freqprob.perplexity(freqprob.KneserNey(bigrams, discount=0.75), [("the", "cat"), ("a", "cat")]),
...     3,
... )
1.32
>>> species = {"oak": 12, "pine": 7, "birch": 3, "elm": 1, "ash": 1, "yew": 1}
>>> round(freqprob.sample_coverage(species), 3)  # share of the population already observed
0.88
>>> round(freqprob.chao_shen_entropy(species), 3)  # entropy, corrected for unseen species
1.56
```

Estimators also offer `fit`/`predict`/`score` aliases for scikit-learn
familiarity, and can be saved with `.save(path)` and reloaded with
`.load(path)`.

## Choosing a method

| Method | Use it for | Key parameter |
|--------|------------|---------------|
| `MLE` | raw relative frequencies (no smoothing) | — |
| `Laplace` / `Lidstone` / `ELE` | simple, robust additive smoothing | `bins`, `gamma` |
| `SimpleGoodTuring` | heavy-tailed count data (many rare items) | `p_value` |
| `KneserNey` / `ModifiedKneserNey` | n-gram language models | `discount` |
| `AbsoluteDiscounting` / `PitmanYor` | bigram discounting with unigram back-off | `discount`, `strength` |
| `KatzBackoff` / `StupidBackoff` | bigram back-off models | `k`, `alpha` |
| `WittenBell` | parameter-free discounting by distinct-type count | `bins` |
| `Bayesian` | Dirichlet-prior smoothing | `alpha` |
| `Interpolated` / `JelinekMercer` | combining models of different orders (fixed or EM-fit weight) | `lambda_weight` |
| `CertaintyDegree` | reserving mass by how fully the support is observed (experimental) | `bins` |
| `Uniform` / `Random` | non-informative baselines | — |

For large or streaming data, FreqProb also provides vectorized batch scoring,
lazy evaluation, incremental (streaming) estimators and memory-efficient
representations. The [User Guide](https://freqprob.tresoldi.org/USER_GUIDE/)
explains each method and when to use it.

## Highlights

- **One interface for every method:** swap estimators without rewriting your
  code.
- **Handles the unseen:** every smoothing method reserves probability mass for
  elements absent from the data.
- **Beyond smoothing:** sample coverage, bias-corrected entropy (Chao–Shen,
  NSB), perplexity, cross-entropy and KL divergence.
- **Validated:** estimators are checked against NLTK's implementations and with
  property-based tests.
- **Typed and tested:** full type hints (`py.typed`), with tests on Python
  3.10–3.12 across Linux, macOS and Windows.

## Documentation

- **[User Guide](https://freqprob.tresoldi.org/USER_GUIDE/):** concepts,
  choosing a method, and worked examples across text, ecology, genomics and
  categorical data.
- **[API Reference](https://freqprob.tresoldi.org/reference/):** every public
  class and function, generated from the source with a runnable example.
- **[References](https://freqprob.tresoldi.org/references/):** the published
  source of each method, with a BibTeX file.
- **[Changelog](https://github.com/tresoldi/freqprob/blob/main/CHANGELOG.md)**
  and **[Migration guide](https://github.com/tresoldi/freqprob/blob/main/MIGRATION.md)**.

## Related projects

- **[NLTK](https://www.nltk.org/)'s `nltk.probability`** module includes several
  of the same estimators (Laplace, Lidstone, ELE, Simple Good–Turing,
  Kneser–Ney, Witten–Bell) as part of a broader NLP toolkit. FreqProb is a
  standalone library for estimation from any frequency counts, and its
  estimators are tested against NLTK's.
- **[ASymCat](https://asymcat.tresoldi.org/)** uses FreqProb for smoothing and
  bias-corrected entropy in its measures of directional association between
  categorical variables.

## Citation

If you use FreqProb in published work, please cite the version you used:

```bibtex
@software{tresoldi_freqprob,
  author  = {Tresoldi, Tiago},
  title   = {{FreqProb}: A {Python} library for probability smoothing and
             frequency-based estimation},
  year    = {2026},
  version = {0.8.1},
  doi     = {10.5281/zenodo.23917394},
  url     = {https://freqprob.tresoldi.org/}
}
```

The DOI [10.5281/zenodo.23917394](https://doi.org/10.5281/zenodo.23917394)
identifies version 0.8.1;
[10.5281/zenodo.23917393](https://doi.org/10.5281/zenodo.23917393) always
resolves to the latest version. See
[How to cite](https://freqprob.tresoldi.org/citing/) for APA and classic BibTeX
forms, and [References](https://freqprob.tresoldi.org/references/) for the papers
behind each method. Citation metadata is also in
[`CITATION.cff`](https://github.com/tresoldi/freqprob/blob/main/CITATION.cff).

## License

MIT; see [LICENSE](https://github.com/tresoldi/freqprob/blob/main/LICENSE).
