# CHANGELOG

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.6.2] - 2026-07-28

Maintenance release: license/metadata fixes and a full documentation rebuild.
No library code or public API changes since 0.6.1.

### Fixed

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
