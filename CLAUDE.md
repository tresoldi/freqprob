# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Testing
- `hatch run test` - Run the full test suite
- `hatch run test-fast` - Run tests in parallel with pytest-xdist
- `hatch run test-cov` - Run tests with coverage reporting
- `hatch run test-numerical` - Run numerical stability tests
- `hatch run test-statistical` - Run statistical correctness tests
- `hatch run test-regression` - Run regression tests against reference implementations
- `hatch run test-property` - Run property-based tests with Hypothesis

### Code Quality
- `hatch run lint:all` - Run all linting (style, typing, security)
- `hatch run lint:style` - Run ruff for style checking
- `hatch run lint:format` - Auto-format code with ruff
- `hatch run lint:typing` - Run mypy type checking
- `hatch run lint:security` - Run bandit security checks

### Build and Development
- `hatch run build` - Build the package
- `hatch run clean` - Clean build artifacts
- `hatch run install-dev` - Install in development mode with dev dependencies
- `hatch run precommit` - Run pre-commit hooks on all files

### Benchmarking and Validation
- `hatch run bench` - Quick performance benchmarks
- `hatch run bench-all` - Comprehensive benchmark suite
- `hatch run validate` - Full validation against reference implementations
- `hatch run validate-quick` - Quick validation checks

## Architecture Overview

### Core Structure
The library is built around abstract base classes in `base.py` that define the `ScoringMethod` interface. All smoothing methods implement this interface for consistent API design.

### Key Modules
- **base.py**: Abstract base classes, type definitions, and configuration
- **basic.py**: Simple methods (MLE, Uniform, Random)  
- **lidstone.py**: Additive smoothing methods (Laplace, Lidstone, ELE)
- **advanced.py**: Sophisticated methods (Good-Turing, Witten-Bell, Certainty Degree)
- **smoothing.py**: Advanced n-gram methods (Kneser-Ney, Bayesian, Interpolated)
- **vectorized.py**: High-performance batch scoring implementations
- **streaming.py**: Memory-efficient streaming algorithms
- **memory_efficient.py**: Compressed and sparse representations
- **lazy.py**: Lazy evaluation for expensive computations
- **cache.py**: Intelligent caching system
- **profiling.py**: Memory and performance profiling tools
- **validation.py**: Validation framework and benchmarking
- **utils.py**: Common utilities (perplexity, cross-entropy, n-gram generation)

### Design Patterns
- **Abstract Factory**: Base classes provide consistent interface
- **Strategy Pattern**: Interchangeable smoothing algorithms
- **Decorator Pattern**: Caching and lazy evaluation wrappers
- **Observer Pattern**: Memory profiling and monitoring

### Type System
The library uses strict typing with custom type aliases:
- `Element`: Union type for hashable items being scored
- `FrequencyDistribution`: Mapping from elements to counts
- `Probability` / `LogProbability`: Float aliases for clarity
- All methods support both regular and log-probability modes

### Testing Strategy
- **Property-based testing** with Hypothesis for mathematical correctness
- **Regression testing** against NLTK and SciPy reference implementations
- **Numerical stability** testing for edge cases and extreme inputs
- **Statistical correctness** validation against known distributions
- **Memory profiling** tests to prevent regressions
- **Performance benchmarking** with timing and scaling analysis

### Performance Optimization
- Vectorized operations with NumPy for batch processing
- Streaming algorithms for large datasets
- Compressed representations for sparse data
- Intelligent caching with configurable policies
- Lazy evaluation for expensive computations

## Development Notes

- Uses Hatch for environment and dependency management
- Requires Python 3.10+ with modern type hints
- All code must pass ruff, mypy, and bandit checks
- Test coverage target is 80% minimum
- Mathematical implementations are validated against academic literature
- Memory usage is actively profiled and optimized
