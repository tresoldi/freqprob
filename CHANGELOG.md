# CHANGELOG

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
