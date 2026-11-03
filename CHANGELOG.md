# CHANGELOG

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Zenodo DOIs: 10.5281/zenodo.23917394 for version 0.8.1, and 10.5281/zenodo.23917393 for all versions. They are in
  `CITATION.cff`, the README (with a DOI badge), the home page and the How to
  cite page.

### Changed

- **README rewritten as the project's front door**, with the same structure as
  ASymCat's: badges, a short description with the DOI, install, a runnable
  quick start, the selection table, highlights, links into the documentation,
  related projects, citation and licence. Every example now uses inline data
  and shows its real output, and a new test (`tests/test_readme.py`)
  runs them so they cannot drift from the code. All links are absolute, so they
  also work on PyPI. The badges are reduced to version, Python versions, CI,
  docs, DOI and licence.
- **Landing page refreshed.** The masthead shows the version and DOI and a row
  of links into the documentation; four entry cards lead to getting started, the
  methods, the how-to guides and the API reference. The quick start is the
  README's, and a new test (`tests/test_landing_example.py`) runs it. The
  illustrative figure is replaced by a figure computed with FreqProb on the word
  counts of *The Zen of Python* (observed relative frequencies against Simple
  Good–Turing estimates, with the 42% of probability mass it reserves for unseen
  words). The page also gains Related projects and links to How to cite and the
  migration guide.
- **Documentation restructured by purpose**, on the same template as ASymCat's.
  The User Guide is replaced by a tutorial; concept pages on how estimators
  work, notation, choosing a method and pitfalls; one page per method family
  with its definition as FreqProb computes it, a worked example on shared
  counts, its properties, its API entries and its references; evaluation pages
  for the model metrics and for sample coverage and entropy; how-to guides for
  building counts, choosing `bins`, log space, scaling, persistence, worked
  examples by domain and plotting; and an API reference by module. Old URLs
  redirect to their new pages, and the method pages render their formulas with
  MathJax.
- **Every documentation example checks its output.** Examples are `pycon` blocks
  with their real output, run as doctests by `tests/test_docs_examples.py`,
  which replaces the test that only checked that `python` blocks ran. The
  plotting guide's figure is drawn from its own example when the site is built
  (`scripts/docs_figures.py`).
- **Plain-probability scores are Python floats.** `SimpleGoodTuring` returned
  NumPy scalars for observed elements with `logprob=False`; every estimator now
  returns `float`. Values are unchanged.
- **Docstrings.** `SimpleGoodTuring`'s `allow_fail` was documented the wrong way
  round: the default (`True`) proceeds when the assumptions fail, and `False`
  raises `RuntimeWarning`. Sphinx roles in the docstrings are replaced by plain
  code, which the API reference renders.
- The `docs` extra adds `mkdocs-redirects` and `matplotlib`.

### Fixed

- **`kl_divergence` counted repeated elements more than once.** It summed `P(w)
  log(P(w)/Q(w))` over every element of `test_data`, so an element listed twice
  contributed twice and the result was not a divergence between the two
  distributions. It now sums over the distinct elements given. **This changes
  results** when `test_data` has repeats.
- **The n-gram methods returned `1e-10` instead of `0.0` with `logprob=False`**
  for a bigram whose word never occurs in the data. `KneserNey`,
  `ModifiedKneserNey`, `AbsoluteDiscounting`, `PitmanYor`, `KatzBackoff` and
  `StupidBackoff` now return `0.0` in plain space, like the other estimators; in
  log space they still return the floor, `log(1e-10)`.

## [0.8.1] - 2026-11-03

Documentation and citation-metadata release, the first archived on Zenodo.
No code changes.

### Added

- **References** page on the documentation site, with the published source of
  every method and a `references.bib` file.
- **How to cite** page with APA and BibTeX (`@software` and `@misc`) entries.

### Changed

- Citation metadata: `CITATION.cff` now uses the same title as the README and
  site ("…probability smoothing and frequency-based estimation"), adds an
  ORCID iD, the Uppsala University affiliation, the PyPI artifact and the
  documentation site as its URL. The maintainer contact returns to
  `tiago.tresoldi@lingfil.uu.se` in `pyproject.toml`, the package metadata and
  `SECURITY.md`, matching ASymCat.
- PyPI project links now point to the documentation site
  (`https://freqprob.tresoldi.org/`) instead of the retired GitHub Pages URL.
- Dropped the `Implementation :: PyPy` classifier: CI tests CPython only.
- README and home-page BibTeX entries now use a version-independent key and the
  documentation URL.

## [0.8.0] - 2026-11-03

Internal-architecture release. The estimators and their mathematics are
unchanged except where noted; what changed is where the code lives and what the
package promises.

### Changed

- **Estimators expose an interface for indexing them.** `support()`,
  `unobserved_score()` and `revision` give a caller the facts it needs to build
  its own index over a scorer. `performance/` now uses these instead of reading
  the private probability table, which fixes four defects: `VectorizedScorer`
  served stale scores after a re-fit and returned all zeros when wrapping a lazy
  scorer, `StreamingMLE.fit()` destroyed the model instead of replacing it, and
  `StreamingLaplace.update_single` ran the MLE increment while `update_batch`
  ran Laplace.
- **One place converts a probability into a score.** A method implements
  `_estimate`, returning a probability table and a reserved mass in plain
  probability space; `fit` applies the floor and the log conversion. The 17
  per-method `logprob` branches are gone.
- **The bigram estimators share one seam.** Kneser-Ney and Modified Kneser-Ney
  join absolute discounting, Pitman-Yor and the back-off methods on
  `methods/_bigram.py`.
- **Per-method config subclasses are gone.** A method's parameters are typed
  attributes on the estimator, validated in its constructor;
  `ScoringMethodConfig` keeps `unobs_prob`, `bins` and `logprob`.
- **`nsb_entropy` is several times faster.** It sums over distinct counts
  rather than over types (types with equal counts contribute identically),
  computes the integration grid's fixed terms once, and reuses the terms that
  depend only on the alphabet size across calls with the same `bins`. Results
  are unchanged up to floating-point rounding; working memory stays bounded for
  inputs with many distinct counts.

### Fixed

- Pickling a lazy or streaming scorer silently dropped the state that is not in
  `__slots__`, producing an object that scored correctly and then failed on
  first use.
- `str()` on a lazy scorer reported zero elements regardless of what it was
  fitted to.
- Modified Kneser-Ney estimated its discounts from all keys, including the
  non-bigram ones it documents as ignored.
- An unseen bigram under Kneser-Ney is scored through its context's back-off
  weight rather than a single flat average, so it is now context-sensitive. A
  word absent from the data scores at the floor instead of borrowing the
  average.
- An empty distribution scored by a log-scorer returns `log(1e-10)` rather than
  the raw `1e-10`, which as a log-probability meant `p == 1.0`.
- `SimpleGoodTuring` validates its `p_value`.
- `chao_shen_entropy` no longer returns `0.0` when every observation is a
  singleton. Zero coverage used to short-circuit the estimate, placing it below
  even the plug-in entropy in exactly the undersampled regime the estimator is
  for; it now takes the singleton count as `N - 1`, as in Chao and Shen (2003)
  and R's `entropy::entropy.ChaoShen`.

### Removed

The top-level namespace goes from 50 names to 41. Every removed name was a
pass-through, a false promise, or development tooling; nothing lost a
capability.

- Deleted: `LazyBatchScorer`, `create_vectorized_batch_scorer`,
  `VectorizedScorer.score_parallel` (whose body was sequential),
  `create_compressed_distribution`, `create_sparse_distribution`,
  `memory_usage_comparison` (no callers anywhere), and `MemoizedProperty`
  (no production callers; it raised on any `__slots__` class).
- Use `CompressedFrequencyDistribution.from_counts` and
  `SparseFrequencyDistribution.from_counts` in place of the two deleted
  factories.
- No longer re-exported at the top level, still importable from their modules:
  `MemoryProfiler`, `MemoryMonitor`, `DistributionMemoryAnalyzer`
  (`freqprob.performance.profiling`), `StreamingDataProcessor`
  (`freqprob.performance.streaming`), and `LazyScoringMethod`
  (`freqprob.performance.lazy`; build one with `create_lazy_mle` or
  `create_lazy_laplace`).

## [0.7.0] - 2026-07-31

Feature release: new smoothing estimators and entropy metrics, together with the
license/metadata fixes, documentation rebuild, and landing page that had been
prepared for 0.6.2. Version 0.6.2 was never published, so PyPI goes from 0.6.1
directly to 0.7.0, which carries all of that work.

### Added

- **New bigram smoothing methods.** `AbsoluteDiscounting` and `PitmanYor`
  (absolute discounting and its Pitman-Yor generalization, backing off to a
  unigram model), and `KatzBackoff` and `StupidBackoff` (Good-Turing-normalized
  and fast unnormalized back-off). They share internal machinery in
  `methods/_bigram.py` and model `P(word | context)` over `(context, word)`
  bigram distributions.
- **`JelinekMercer` smoothing.** Linear interpolation whose weight is fit by EM
  (deleted interpolation) on a held-out distribution, exposed via
  `estimated_lambda`; falls back to a fixed weight when no held-out data is given.
- **Sample-based entropy estimators** in `metrics`: `sample_coverage` (the
  Good-Turing observed-mass estimate), `chao_shen_entropy` (Chao-Shen
  bias-corrected entropy), and `nsb_entropy` (the Nemenman-Shafee-Bialek Bayesian
  entropy estimator, with an optional `bins` for the assumed alphabet size).
- All new estimators and functions are exported from the top-level `freqprob`
  namespace and covered by `tests/test_new_methods.py`.
- **Documentation landing page.** The docs site now opens with a dedicated,
  self-contained landing page — a dry, academic layout (neutral palette,
  monochrome figure and code) with a figure illustrating smoothing — served as
  a MkDocs custom home template; the User Guide and API Reference keep the
  standard Material navigation behind it.
- **Custom domain.** The site is served from `https://freqprob.tresoldi.org`
  (via a `docs/CNAME` file and updated `site_url`); in-repo documentation links
  were updated accordingly.

### Fixed

- **Method coverage in the docs.** `CertaintyDegree` was missing from the User
  Guide entirely (all method tables, the one-line summaries, and the per-method
  sections) and from the landing page's methods table; `WittenBell` was absent
  from the "choosing a method" tables and had no code example. Added
  `CertaintyDegree` (marked experimental) and `WittenBell` throughout, and
  reorganised the README method table so both are described accurately rather
  than lumped under "baselines."
- **License file corrected to MIT.** The `LICENSE` file contained the full GNU
  GPL v3 text, contradicting the MIT license declared in `pyproject.toml`,
  `CITATION.cff`, and the README. Replaced it with the MIT License text so all
  license metadata agrees.

### Changed

- Removed the University of Uppsala affiliation/publisher references (README
  citation, `CITATION.cff` affiliation) and updated the maintainer contact email
  to `freqprob@tresoldi.org` across `pyproject.toml`, `CITATION.cff`, the package
  metadata, and `SECURITY.md`.
- **Documentation rebuilt from scratch.** The docs site is now a focused
  three-page set — Home, a single flowing **User Guide** (concepts → choosing a
  method → the methods → evaluation → worked examples in text, ecology, genomics,
  and categorical data → a "Scaling & performance" section → practical notes),
  and an **API Reference generated entirely from docstrings**. Every public
  symbol's docstring was converted to Google style and given a correct, runnable
  example (fixing, among others, `Interpolated`'s wrong numbers and the fake
  performance/validation APIs the old guide referenced).
- **All documentation examples are now executed in CI.** New tests run every
  docstring example (`tests/test_doctests.py`) and every Python code block in the
  README and docs (`tests/test_docs_examples.py`), so an example can no longer
  drift from the code.
- Bumped pinned dev tooling in lockstep across `pyproject.toml` and
  `.pre-commit-config.yaml`: **ruff** `0.15.8` → `0.16.0` and **mypy**
  `1.19.1` → `2.3.0`. Applied ruff 0.16.0's updated formatting (Markdown
  embedded code blocks).

### Removed

- Deleted the hand-written `docs/API_REFERENCE.md` (superseded by the
  docstring-generated reference), `docs/LLM_DOCUMENTATION.md` (stale, with many
  non-existent APIs), and the four `docs/tutorial_*.py` files. Dropped the
  now-unused `nhandu`/`matplotlib`/`seaborn` dev dependencies and the Makefile
  `docs`/`docs-clean` tutorial targets.

## [0.6.1] - 2026-07-28

Maintenance release: test-coverage and CI hardening only. No library code or
public API changes since 0.6.0.

### Added

- **Test coverage for the performance helpers**: new `tests/test_profiling.py`
  and `tests/test_performance_edge_cases.py` cover the previously under-tested
  branches of `performance/profiling.py` (69% → 96%), `performance/vectorized.py`
  (70% → 100%), and `performance/lazy.py` (78% → 99%), raising overall coverage
  from ~84% to ~91%.
- **CI security scanning**: the quality workflow now runs `bandit` against
  `src/` on every push and pull request (configuration already lived in
  `pyproject.toml`).
- **CI coverage reporting**: the test workflow uploads coverage to Codecov once
  per run (from the Ubuntu / Python 3.12 matrix cell), using the existing
  `codecov.yml` configuration.

### Changed

- Raised the enforced coverage floor from 80% to 88% (in both `pyproject.toml`
  and the CI workflow) to lock in the higher coverage.

## [0.6.0] - 2026-07-24

### Fixed

- **Re-fitting an estimator**: Calling `fit()` a second time on an already-fitted
  estimator now behaves identically to a fresh fit. Previously, log-space methods
  (`MLE`, `Uniform`, `Random`, `CertaintyDegree`) could raise a `math domain error`
  because `self._unobs` — which holds the previous *output* — was reused as an epsilon
  floor; and methods that build `self._prob` incrementally (`KneserNey`,
  `SimpleGoodTuring`) left stale keys from the previous data. `fit()` now resets state
  first, and the additive-floor methods use a fixed `MIN_PROBABILITY` constant.

### Changed - BREAKING

- **Renamed estimator classes** (dropped the inconsistent `Smoothing` suffix):
  `BayesianSmoothing` → `Bayesian`, `InterpolatedSmoothing` → `Interpolated`.
  Constructor parameters and behavior are unchanged. Additionally, direct imports of
  the old internal module paths changed during the internal reorganization; import
  from the top-level `freqprob` namespace instead. Full details in `MIGRATION.md`.

### Added

- **scikit-learn-style API**: `fit()` / `predict()` / `score()` methods on every
  estimator, alongside the existing callable contract. `score` is a single-element
  alias for `__call__`; `predict` scores an iterable in one call.
- **Model serialization**: `save(path)` / `load(path)` on `ScoringMethod` to persist
  and restore a fitted estimator without re-fitting (pickle-based). `load()` validates
  the object type and documents the trust requirement of unpickling.
- **Public `ScoringMethod` base type**: exported from `freqprob` for `isinstance`
  checks and type annotations.
- **Documentation site** (MkDocs-Material + mkdocstrings): `mkdocs.yml`, a landing
  page, and an API reference generated from docstrings (so it can't drift from the
  code). Build with `make site` / preview with `make site-serve`; a new `docs` extra
  provides the toolchain. Intended to be published to GitHub Pages.

### Changed

- **README**: Rewritten to lead with the core value (counts → probabilities) and a
  single motivating example, add a method-selection table, frame the library as
  general-purpose (not NLP-only), and collapse the previously repeated testing
  sections. Documentation links now point to the docs site.

- **Collapsed redundant configuration classes**: Removed eight per-method `*Config`
  dataclasses that only restated fields already defined on the base
  `ScoringMethodConfig` (`UniformConfig`, `MLEConfig`, `LidstoneConfig`,
  `LaplaceConfig`, `ELEConfig`, `WittenBellConfig`, `CertaintyDegreeConfig`,
  `ModifiedKneserNeyConfig` — the last three of which were never even instantiated).
  Those methods now build the base `ScoringMethodConfig` directly. The five configs
  that add a genuine parameter are kept but trimmed to just that field
  (`RandomConfig.seed`, `KneserNeyConfig.discount`, `BayesianConfig.alpha`,
  `InterpolatedConfig.lambda_weight`, `SimpleGoodTuringConfig.p_value`/`default_p0`/
  `allow_fail`). No public API or behavior change — all constructor parameters work
  exactly as before.

- **mypy target**: Set mypy `python_version` to `3.12` so it can parse modern
  dependency stubs (e.g. numpy's PEP 695 `type` statements). Source-level 3.10
  compatibility is still enforced by ruff (`target-version = "py310"`) and validated
  by the Python 3.10 CI test-matrix leg.
- **Constrained type-check dependencies**: Bounded `numpy` (`<2.6`) and `scipy`
  (`<1.19`) in the `dev` extra so a new release can't silently change mypy's view of
  their type stubs and break the type-check job with no code change. The runtime
  dependency and the `test` matrix stay unpinned to catch real regressions.

- **Internal module organization**: Regrouped the package internals for
  maintainability, with **no change to the public API** (`import freqprob` and every
  `freqprob.<Name>` continue to work unchanged). Estimators now live under
  `freqprob/methods/` grouped by family (`baselines`, `additive`, `goodturing`,
  `certainty`, `kneser_ney`, `interpolated`, `bayesian`); the lazy/streaming/
  vectorized/memory/profiling infrastructure moved under `freqprob/performance/`;
  and the old `utils.py` junk drawer was split into `metrics.py` (model evaluation)
  and `text.py` (n-gram/frequency helpers). `base.py` and `cache.py` remain
  top-level as shared foundations. Direct imports of the old internal module paths
  (e.g. `from freqprob.smoothing import KneserNey`) have changed; import from the
  top-level `freqprob` namespace instead.
- **Project layout**: Moved the package to a `src/` layout (`src/freqprob/`). This
  prevents accidental imports of the working-tree package instead of the installed
  one and is the modern packaging standard. The import path is unchanged
  (`import freqprob`); only the on-disk location moved. Build/type-check/test configs
  updated accordingly (`packages.find` → `src`, `mypy` uses `[tool.mypy] files`).
- **Pinned linters**: `ruff` and `mypy` are now pinned to exact versions in the `dev`
  extra and in `.pre-commit-config.yaml`, so local, CI, and pre-commit run identical
  versions. Unpinned versions had drifted (e.g. a newer `ruff` began formatting code
  blocks inside Markdown), silently changing check results between runs.

### Removed

- **Committed documentation artifacts**: Removed the generated tutorial HTML and
  figure PNGs (~17 MB) from version control. They are build outputs (`make docs`)
  and are now git-ignored; rendered docs will be published to the documentation site.

### Added

- **CI test matrix**: Tests now run across Python 3.10/3.11/3.12 on Linux, macOS, and
  Windows, matching the platforms and versions advertised in the metadata.
- **Automated release workflow**: Tag-triggered GitHub Actions workflow that builds and
  publishes to PyPI via trusted publishing (OIDC), removing the need for stored tokens.
- **Pre-commit configuration** (`.pre-commit-config.yaml`) mirroring the CI quality gates
  (ruff lint, ruff format, mypy).
- **`test` optional-dependency extra**: Lightweight testing dependency set used by the CI
  test matrix (`pip install -e ".[test]"`); `dev` now includes it.
- **`CITATION.cff`**: Machine-readable citation metadata, enabling GitHub's native
  "Cite this repository" support.
- **`CONTRIBUTING.md`** and **`SECURITY.md`**: Contributor setup/quality-check guide and
  a vulnerability reporting policy.
- **Codecov integration**: `codecov.yml` configuration and a coverage badge in the README.
- **`make security` target**: Runs the (already configured) `bandit` static security
  analysis; `bandit[toml]` added to the `dev` extra.

### Changed

- **CI**: The quality job now installs `.[dev]` so the Hypothesis, NLTK, and scikit-learn
  test suites actually execute (previously skipped because those dependencies were not
  installed), and type checking now also covers `scripts/`.

### Fixed

- **Documentation URL** in package metadata pointed to `docs/user_guide.md` (wrong case);
  corrected to `docs/USER_GUIDE.md`.
- **Dead code**: Removed the unreachable `HAS_VALIDATION` block in `freqprob/__init__.py`
  that referenced a removed module.

## [0.4.0] - 2025-10-04

### Changed - BREAKING

- **SimpleGoodTuring**: Changed unseen word probability semantics from total mass to per-word probability
  - Previously: `sgt('unseen')` returned p₀ (total probability mass for ALL unseen words, e.g., ~0.07)
  - Now: `sgt('unseen')` returns per-word probability (p₀ / estimated_unseen_types, e.g., ~0.00012)
  - This makes SGT consistent with other smoothing methods and enables meaningful probability arithmetic
  - **Migration**: Use `sgt.total_unseen_mass` property to access the old p₀ value

### Added

- **SimpleGoodTuring bins parameter**: Controls total vocabulary size for per-word probability calculation
  - Default heuristic: `bins = V_observed + N₁` (observed vocabulary + singleton count)
  - Allows explicit control over estimated vocabulary size
  - Theoretically motivated default provides sensible out-of-box behavior
- **SimpleGoodTuring.total_unseen_mass property**: Read-only property providing access to p₀ (total unseen mass)
- **ScoringMethod._total_unseen_mass**: Base class support for methods that track total unseen mass

### Fixed

- **InterpolatedSmoothing**: Fixed n-gram interpolation to properly extract lower-order context
  - Previously: trigram+bigram interpolation returned zero probabilities (looked for trigram keys in bigram model)
  - Now: automatically detects n-gram orders and extracts appropriate context (e.g., extracts bigram `('big', 'cat')` from trigram `('the', 'big', 'cat')`)
  - Supports two modes: n-gram interpolation (different orders) and same-type interpolation (same element types)
  - All probabilities floored at `1e-10` for numerical stability
  - Unseen n-grams backoff to lower-order model: `(1-λ) * P_low(context)`

### Improved

- **InterpolatedSmoothing**: Enhanced with automatic n-gram mode detection
  - Validates that high-order n ≥ low-order n for tuple distributions
  - Provides helpful error messages suggesting how to fix order issues
  - Added `_detect_order()` and `_extract_lower_context()` helper methods
  - Dual-mode support for both n-gram and same-type interpolation
- Tutorial 2: Comprehensive explanation of SGT's per-word vs total mass semantics
- Tutorial 2: Demonstration of bins parameter effects on unseen probabilities
- Tutorial 2: Example showing SGT compatibility with perplexity calculation
- Tutorial 2: Updated interpolated smoothing section with n-gram mode explanation
- API Reference: Complete documentation of new bins parameter and migration guide
- API Reference: Updated InterpolatedSmoothing documentation with dual-mode examples
- Test suite: Added 5 new tests for bins parameter, total_unseen_mass, and perplexity compatibility
- Test suite: Added 7 new tests for n-gram interpolation modes

### Migration Guide

For users upgrading from v0.3.x:

```python
# v0.3.x code:
sgt = SimpleGoodTuring(freqdist)
p_total_unseen = sgt("unseen_word")  # Returned p₀ ≈ 0.07

# v0.4.0 equivalent:
sgt = SimpleGoodTuring(freqdist)
p_per_word = sgt("unseen_word")  # Returns per-word prob ≈ 0.00012
p_total_unseen = sgt.total_unseen_mass  # Access p₀ ≈ 0.07

# If you need the old behavior (not recommended):
# The old behavior was mathematically inconsistent. If you truly need it,
# multiply per-word probability by estimated unseen types:
estimated_unseen = int(sgt.total_unseen_mass / sgt("unseen_word"))
p_approx_old = sgt("unseen_word") * estimated_unseen
```

**Why this change?**
- **Consistency**: All smoothing methods now return per-word probabilities
- **Composability**: Probabilities can be meaningfully added/compared: P(word₁) + P(word₂)
- **Perplexity**: Enables direct use with perplexity and other evaluation metrics
- **Semantics**: Returns what users actually expect: P(this specific unseen word)

## [0.3.1] - 2025-07-21

### Fixed
- Type compatibility issues with mypy --strict mode in vectorized.py
- Pre-commit formatting issues in documentation files
- Trailing whitespace and end-of-file formatting

## [0.3.0] - 2025-07-21

### Added
- Comprehensive LLM coding agent documentation (LLM_DOCUMENTATION.md)
- Enhanced AGENTS.md with detailed development commands and architecture overview

### Fixed
- Removed unnecessary mypy type ignore comments in validation.py and test files

## [0.2.1] - 2025-01-14

### Fixed
- Windows timing precision issue in MemoryProfiler for cross-platform compatibility
- Type import organization in test files following Python best practices
- Pre-commit hook issues with proper TYPE_CHECKING block usage

### Improved
- Documentation structure and organization
- README.md streamlined and focused on quick start
- Enhanced performance comparison guide with Hatch integration
- Advanced features documentation in user guide
- Cross-referencing between documentation files

### Changed
- Moved detailed benchmarking instructions to docs/performance_comparison.md
- Moved advanced features examples to docs/user_guide.md
- Updated performance comparison table for quick reference

## [0.2.0] - 2025-01-13

### Added
- Complete project modernization with current best practices
- Modern build system using `pyproject.toml` and Hatch
- Pre-commit hooks for code quality (black, isort, ruff, mypy, bandit)
- GitHub Actions CI/CD pipeline with comprehensive testing
- Automated releases and semantic versioning workflows
- Code coverage reporting and quality badges
- Memory profiling and optimization features
- Vectorized operations for batch processing
- Streaming/incremental updates for large datasets
- Comprehensive documentation suite including:
  - User guide with mathematical background
  - 4 interactive Jupyter notebook tutorials
  - Complete API reference documentation
  - Performance benchmarks and comparison guide
  - Development workflow documentation
- Advanced smoothing methods (Kneser-Ney, Modified Kneser-Ney, Bayesian)
- Utility functions for model evaluation (perplexity, cross-entropy, KL divergence)
- Caching and lazy evaluation for performance optimization
- Memory-efficient representations (compressed, sparse, quantized)
- Type hints throughout the codebase
- Extensive test suite with >85% coverage requirement

### Changed
- Migrated from `setup.py` to modern `pyproject.toml` configuration
- Replaced `flake8` with `ruff` for faster linting
- Updated dependency management with optional extras
- Improved package structure and organization
- Enhanced error handling and validation
- Modernized development tooling and workflows

### Removed
- Legacy `setup.py` and `requirements.txt` files
- Old `MANIFEST.in` configuration

## [0.1.0] - 2023-02-18

### Added
- Initial release, importing and changing code from the `lpngram` package
- Basic smoothing methods (MLE, Laplace, Lidstone, ELE)
- Simple Good-Turing smoothing implementation
- Core frequency distribution handling
