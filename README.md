# FreqProb

[![CI](https://github.com/tresoldi/freqprob/actions/workflows/quality.yml/badge.svg)](https://github.com/tresoldi/freqprob/actions/workflows/quality.yml)
[![codecov](https://codecov.io/gh/tresoldi/freqprob/branch/main/graph/badge.svg)](https://codecov.io/gh/tresoldi/freqprob)
[![Docs](https://img.shields.io/badge/docs-mkdocs-blue.svg)](https://freqprob.tresoldi.org/)
[![PyPI version](https://badge.fury.io/py/freqprob.svg)](https://badge.fury.io/py/freqprob)
[![Python versions](https://img.shields.io/pypi/pyversions/freqprob.svg)](https://pypi.org/project/freqprob/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

**Turn frequency counts into probability estimates.**

FreqProb converts a mapping of elements to observed counts into smoothed
probabilities that handle unseen elements sensibly. It's a general-purpose
statistical tool — natural language processing is one consumer among many
(information retrieval, ecology, genomics, categorical analytics, ML features).

```python
import freqprob

counts = {"the": 100, "cat": 50, "dog": 30, "bird": 10}

laplace = freqprob.Laplace(counts, bins=10_000, logprob=False)
laplace("cat")  # 0.0050  — an observed element
laplace("elephant")  # 0.0001  — an unseen element still gets non-zero mass
```

That last line is the whole point: a raw relative-frequency estimate would
assign probability **0** to `"elephant"` and break any model that multiplies or
takes logs of probabilities. Smoothing reserves a bit of mass for what you
haven't seen yet — and FreqProb gives you a dozen well-tested ways to do it
behind one consistent interface.

## Install

```bash
pip install freqprob
```

## The interface

Every estimator follows the same contract: construct it with a frequency
distribution, then **call it** to score an element.

```python
import freqprob

bigram_counts = {("the", "cat"): 5, ("the", "dog"): 3, ("a", "cat"): 2}
scorer = freqprob.KneserNey(bigram_counts, discount=0.75)

scorer(("the", "cat"))  # score one element
scorer.predict([("the", "cat"), ("a", "dog")])  # score many (scikit-learn-style)
freqprob.perplexity(scorer, [("the", "cat"), ("a", "cat")])  # evaluate a model
```

`fit`/`predict`/`score` aliases are available for scikit-learn familiarity, and
any fitted estimator can be saved with `.save(path)` and reloaded with
`.load(path)` without re-fitting.

## Choosing a method

| Method | Use it for | Key parameter |
|--------|------------|---------------|
| `MLE` | raw relative frequencies (no smoothing) | — |
| `Laplace` / `Lidstone` / `ELE` | simple, robust additive smoothing | `bins`, `gamma` |
| `SimpleGoodTuring` | heavy-tailed count data (many rare items) | `p_value` |
| `KneserNey` / `ModifiedKneserNey` | n-gram language models | `discount` |
| `Bayesian` | Dirichlet-prior smoothing | `alpha` |
| `Interpolated` | combining models of different orders | `lambda_weight` |
| `WittenBell`, `CertaintyDegree`, `Uniform`, `Random` | baselines & specialized cases | — |

For large or streaming data, FreqProb also provides vectorized batch scoring,
lazy evaluation, streaming (incremental) estimators, and memory-efficient
compressed/sparse representations.

## Why FreqProb

- **One consistent API** across a dozen smoothing methods — swap estimators
  without rewriting your code.
- **Mathematically validated** against reference implementations (NLTK, SciPy)
  and checked with property-based tests.
- **Typed and production-ready** — full type hints (`py.typed`), strict linting
  and type-checking, and a test suite run across Python 3.10–3.12 on Linux,
  macOS, and Windows.

## Documentation

- **[Documentation site](https://freqprob.tresoldi.org/)** — user guide and
  full API reference.
- **[User Guide](docs/USER_GUIDE.md)** — concepts, choosing a method, and worked
  examples across text, ecology, genomics, and categorical data.
- **[API Reference](https://freqprob.tresoldi.org/reference/)** — every
  public class and function, generated from the source with a runnable example.

## Citation

If you use FreqProb in academic research, please cite:

```bibtex
@software{tresoldi_freqprob_2026,
  author = {Tresoldi, Tiago},
  title = {FreqProb: A Python library for probability smoothing and frequency-based estimation},
  url = {https://github.com/tresoldi/freqprob},
  version = {0.6.2},
  year = {2026}
}
```

## License

MIT — see [LICENSE](LICENSE).
