# FreqProb

[![CI](https://github.com/tresoldi/freqprob/actions/workflows/ci.yml/badge.svg)](https://github.com/tresoldi/freqprob/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/tresoldi/freqprob/branch/main/graph/badge.svg?token=YOUR_TOKEN_HERE)](https://codecov.io/gh/tresoldi/freqprob)
[![PyPI version](https://badge.fury.io/py/freqprob.svg)](https://badge.fury.io/py/freqprob)
[![Python versions](https://img.shields.io/pypi/pyversions/freqprob.svg)](https://pypi.org/project/freqprob/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)

A modern Python library for scoring observation probabilities from frequency counts, with multiple smoothing methods and advanced features for NLP applications.

## Features

✨ **Comprehensive Smoothing Methods**: MLE, Laplace, Lidstone, Simple Good-Turing, Kneser-Ney, and more  
🚀 **High Performance**: Vectorized operations, caching, and lazy evaluation  
💾 **Memory Efficient**: Compressed representations and streaming updates  
🔧 **Developer Friendly**: Type hints, comprehensive documentation, and modern tooling  
📊 **Rich Analytics**: Model comparison, perplexity calculation, and performance benchmarks  
🧪 **Well Tested**: Extensive test suite with >85% coverage  
🔬 **Rigorous Validation**: Numerical stability, statistical correctness, and regression testing

## Installation

FreqProb requires Python 3.8+ and can be installed via pip:

```bash
pip install freqprob
```

For additional features:

```bash
# Memory profiling capabilities
pip install freqprob[memory]

# Jupyter notebook support for tutorials
pip install freqprob[notebook]

# Validation and testing capabilities
pip install freqprob[validation]

# All optional dependencies
pip install freqprob[all]
```

## Quick Start

```python
import freqprob

# Create a frequency distribution
freqdist = {'cat': 3, 'dog': 2, 'bird': 1}

# Basic smoothing methods
mle = freqprob.MLE(freqdist, logprob=False)
laplace = freqprob.Laplace(freqdist, bins=1000, logprob=False)

print(f"MLE P(cat) = {mle('cat'):.4f}")      # 0.5000
print(f"Laplace P(cat) = {laplace('cat'):.4f}")  # 0.0040
print(f"Laplace P(mouse) = {laplace('mouse'):.4f}")  # 0.0010 (unseen word)

# Advanced features
vectorized = freqprob.VectorizedScorer(laplace)
batch_scores = vectorized.score_batch(['cat', 'dog', 'bird'])
```

## Documentation

- 📖 [User Guide](docs/user_guide.md) - Comprehensive guide with mathematical background
- 🎓 [Tutorials](docs/) - Interactive Jupyter notebooks
- 📚 [API Reference](docs/api_reference.md) - Complete API documentation
- ⚡ [Performance Comparison](docs/performance_comparison.md) - Benchmarks and optimization tips
- 🔬 [Validation Guide](docs/validation_guide.md) - Testing and validation procedures

## Development

FreqProb uses modern Python tooling and best practices:

### Setup Development Environment

```bash
git clone https://github.com/tresoldi/freqprob.git
cd freqprob
pip install hatch
```

### Available Commands

```bash
# Run tests
hatch run test

# Run tests with coverage
hatch run test-cov

# Format code
hatch run lint:format

# Check code quality
hatch run lint:all

# Run benchmarks
hatch run bench

# Run validation suite
hatch run validate

# Run specific validation tests
hatch run test-numerical      # Numerical stability
hatch run test-statistical    # Statistical correctness
hatch run test-regression     # Reference implementation compatibility
hatch run test-property       # Property-based testing

# Build documentation
hatch run docs:build
```

### Pre-commit Hooks

Install pre-commit hooks for automatic code quality checks:

```bash
pip install pre-commit
pre-commit install
```

## Contributing

Contributions are welcome! Please see our [development workflow](docs/development.md) for details.

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the test suite and linting
6. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use FreqProb in your research, please cite:

```bibtex
@software{tresoldi_freqprob,
  author = {Tresoldi, Tiago},
  title = {FreqProb: A Python library for probability smoothing},
  url = {https://github.com/tresoldi/freqprob},
  version = {0.1.0},
  year = {2024}
}
```