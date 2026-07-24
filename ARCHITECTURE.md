# FreqProb Architecture

> Status: **describes the delivered architecture as of 0.6.0.**
> This document was originally the anchor for a five-phase revision; that
> revision is complete, and the file now reflects how FreqProb is actually
> built. Key design decisions are recorded in §9.

This document defines the structure, boundaries, and design principles of
FreqProb. New modules should fit within it; changes that move away from it
should update it in the same PR.

---

## 1. Purpose and scope

FreqProb turns **frequency counts into probability estimates**. Given a mapping
of elements to observed counts, it produces smoothed probabilities that handle
unseen elements sensibly.

This is a **general-purpose** statistical tool, not a computational-linguistics
library. NLP is one important consumer, but the core makes no assumption that
elements are words. Representative domains:

| Domain | Elements | Counts |
|--------|----------|--------|
| NLP / language modeling | words, n-grams | corpus frequencies |
| Information retrieval | terms | document/term frequencies |
| Ecology | species | individuals observed |
| Genomics | k-mers | occurrences in reads |
| Analytics | categories | event counts |
| ML features | categorical values | value counts |

**Design consequence:** the core vocabulary is *element*, *category*,
*observation*, *count* — never *word* or *vocabulary*. NLP-specific
convenience (n-gram generation, perplexity) lives in a clearly separated layer
so that a genomics or analytics user never trips over linguistic terminology.

---

## 2. Design principles

1. **Domain-neutral core, separated domain layers.** The estimator core depends
   only on `numpy`/`scipy`. Text/NLP helpers live in a separate module
   (`text.py`), installed by default but importable in isolation. No linguistic
   assumptions leak into `base` or `methods`.
2. **One estimator contract.** Every method is a `ScoringMethod`: construct
   with a frequency distribution, then call it to score an element. There is a
   single, predictable lifecycle. Performance variants (lazy, streaming,
   vectorized) are *adapters* over this contract, not parallel hierarchies.
3. **KISS.** Prefer a small, obvious API over configurability nobody asked for.
   A method's constructor takes plain keyword arguments.
4. **DRY.** No per-method boilerplate that merely restates the base. Shared
   validation and configuration live in exactly one place (`base`).
5. **YAGNI.** Do not add abstraction, config surface, or "extension points"
   ahead of a concrete need.
6. **Namespace-preserving refactors.** Internal reorganization must not break
   `import freqprob; freqprob.Laplace(...)`. The public surface is defined by
   `freqprob/__init__.py`, independent of file layout.
7. **Typed and validated.** Full type hints, `py.typed`, strict mypy;
   parameter validation raises early with clear messages.

---

## 3. Package structure

`src/` layout; the public API is re-exported from `__init__.py`, so file layout
is transparent to users.

```
src/freqprob/
├── __init__.py          # public API surface (stable import path)
├── base.py              # ScoringMethod (ABC), ScoringMethodConfig, MIN_PROBABILITY, type aliases
├── cache.py             # memoization used by the estimators (shared foundation)
├── metrics.py           # perplexity, cross_entropy, kl_divergence, model_comparison
├── text.py              # n-gram / word-frequency helpers  (NLP-specific layer)
├── methods/             # the estimators, grouped by family
│   ├── baselines.py     # Uniform, Random, MLE
│   ├── additive.py      # Lidstone, Laplace, ELE
│   ├── goodturing.py    # SimpleGoodTuring, WittenBell
│   ├── certainty.py     # CertaintyDegree
│   ├── kneser_ney.py    # KneserNey, ModifiedKneserNey
│   ├── interpolated.py  # Interpolated
│   └── bayesian.py      # Bayesian
├── performance/         # infrastructure, not estimators
│   ├── lazy.py
│   ├── vectorized.py
│   ├── streaming.py
│   ├── memory_efficient.py
│   └── profiling.py
└── py.typed
```

**Rationale for the boundaries**

- `methods/` groups estimators by mathematical family under intuitive names,
  replacing the earlier arbitrary split across `basic`/`lidstone`/`advanced`/
  `smoothing`. Module boundaries are a maintainer concern only (users import
  from the top level), so they read as a clean taxonomy.
- `metrics.py` and `text.py` split the former `utils.py` junk drawer into its
  two real concerns: **model evaluation** (general) and **text preprocessing**
  (domain-specific).
- `performance/` isolates the lazy/streaming/vectorized/memory/profiling
  machinery — none of which are smoothing methods — from the estimator core, so
  the package's conceptual surface stays small.
- `base.py` and `cache.py` are top-level **foundations**. `cache.py` is a
  memoization utility the *methods* depend on (Simple Good-Turing, Kneser-Ney),
  so it lives alongside `base` rather than under `performance/` — otherwise
  `methods/` would have to reach into `performance/`, against the dependency
  rule below. (This is a deliberate deviation from the original plan, which had
  sketched `cache.py` under `performance/`.)

### Dependency rule

Dependencies point **inward**: `methods/`, `metrics`, `text`, and
`performance/` may import from `base` (and, where needed, the top-level
`cache`); `base` imports from nothing else in the package. `text` is the only
module allowed to encode NLP concepts. Nothing in `base`/`methods` may import
`text`.

---

## 4. Public API contract

The estimator lifecycle:

```python
scorer = freqprob.Laplace(freqdist, bins=..., logprob=True)
p = scorer(element)          # __call__ scores one element
```

- Construction fits the model; `__call__(element)` returns a probability or
  log-probability.
- `logprob` is a per-instance choice, defaulting to `True`.
- Unobserved elements return the method's reserved/smoothed mass.
- Re-fitting (`scorer.fit(other_freqdist)`) resets state and behaves like a
  fresh fit.

**scikit-learn-style aliases** sit alongside the callable contract for the
data-science audience:

- `fit(freqdist)` — returns `self`.
- `predict(elements)` — scores an iterable, returning one value per element.
- `score(element)` — single-element alias for `__call__`.

These are conveniences over the one estimator contract, not a second hierarchy.
Full `BaseEstimator` compliance (`get_params`/`set_params`, cloning, pipeline
support) is **out of scope** (YAGNI) — revisit only if a concrete
pipeline/grid-search need appears.

**Persistence.** A fitted estimator can be saved and restored without
re-fitting:

```python
scorer.save("model.pkl")
restored = freqprob.KneserNey.load("model.pkl")
```

`ScoringMethod` is exported publicly for `isinstance` checks and type
annotations.

### Configuration

The base `ScoringMethodConfig` validates `unobs_prob`, `gamma`, `bins`,
`logprob`. Per-method config subclasses exist **only** where a method adds a
genuine parameter (`RandomConfig.seed`, `KneserNeyConfig.discount`,
`BayesianConfig.alpha`, `InterpolatedConfig.lambda_weight`,
`SimpleGoodTuringConfig.p_value`/`default_p0`/`allow_fail`). Constructors accept
plain keyword arguments; the config object is an internal detail users never
touch.

---

## 5. Terminology

To keep the library legible across domains, user-facing docs, examples, and
docstrings use neutral terms:

| Prefer | Avoid |
|--------|-------|
| element, category | word, token (except in `text.py`) |
| number of elements, support size | vocabulary size |
| observation, sample | corpus |
| count / frequency | word count |

Existing parameter names (`bins`, `freqdist`) and the public identifiers that
contain `vocabulary` (`max_vocabulary_size`, `get_vocabulary_size`) stay for
compatibility; the shift is in prose and messages.

---

## 6. Documentation architecture

- **Autogenerated API reference** from docstrings via `mkdocstrings` (no
  hand-maintained reference to drift).
- **Narrative guide** (hand-written): concepts, choosing a method, examples.
- **Executable tutorials** kept as source only; rendered output is a build
  artifact, never committed.
- **Tooling:** MkDocs-Material + mkdocstrings, published to GitHub Pages from
  CI (see `mkdocs.yml`; build with `make site`).

Generated artifacts (tutorial HTML, figures) are kept out of version control.

---

## 7. Versioning & compatibility

The revision culminated in a **0.6.0 release with a clean break** — the
inconsistent estimator names were renamed with no deprecation aliases, since
the project was pre-1.0 and a single well-communicated break is cleaner than
carrying shims. `MIGRATION.md` maps old → new names, and the `CHANGELOG` has a
breaking-changes section.

The public API is the top-level `freqprob` namespace. After 0.6.0, breaking
changes to it are expected to be rare and clearly flagged; the project follows
semantic versioning.

---

## 8. How it was built (history)

The revision shipped as five independent, reviewable PRs. Phases 1–4 were
namespace-preserving (no user-visible change); phase 5 was the deliberate
public break.

1. **`src/` layout + artifact cleanup** — moved `freqprob/` → `src/freqprob/`,
   dropped committed tutorial HTML/figures, hardened the toolchain.
2. **Internal reorg** — introduced `methods/`, `performance/`, `metrics.py`,
   `text.py`; `__init__.py` re-exports kept the public API identical.
3. **Config collapse (DRY)** — removed the redundant per-method `*Config`
   classes.
4. **Docs site + README** — MkDocs site, autogenerated reference, domain-neutral
   README.
5. **Naming/terminology pass → 0.6.0** — consistent naming, neutral
   terminology, `fit`/`predict`/`score` aliases, `save`/`load`, migration guide,
   and the clean-break release.

---

## 9. Decisions (resolved)

1. **API compatibility:** clean break at **0.6.0**, no deprecation aliases;
   `MIGRATION.md` documents old → new names (§7).
2. **NLP helpers:** live in `freqprob/text.py`, **installed by default** (not a
   separate extra); the core stays domain-neutral by module boundary, not by
   install gating (§3).
3. **scikit-learn API:** thin `fit`/`predict`/`score` aliases over the callable
   contract; full `BaseEstimator` compliance is out of scope (§4).
4. **Docs tooling:** MkDocs-Material + mkdocstrings, published to GitHub Pages
   (§6).
5. **`cache.py` placement:** kept top-level (a foundation the methods depend on)
   rather than under `performance/`, to preserve the inward dependency rule
   (§3).
