# CHANGELOG

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
