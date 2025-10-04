# CHANGELOG

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
p_total_unseen = sgt('unseen_word')  # Returned p₀ ≈ 0.07

# v0.4.0 equivalent:
sgt = SimpleGoodTuring(freqdist)
p_per_word = sgt('unseen_word')      # Returns per-word prob ≈ 0.00012
p_total_unseen = sgt.total_unseen_mass  # Access p₀ ≈ 0.07

# If you need the old behavior (not recommended):
# The old behavior was mathematically inconsistent. If you truly need it,
# multiply per-word probability by estimated unseen types:
estimated_unseen = int(sgt.total_unseen_mass / sgt('unseen_word'))
p_approx_old = sgt('unseen_word') * estimated_unseen
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
- Enhanced CLAUDE.md with detailed development commands and architecture overview

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
