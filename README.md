# FreqProb

[![CI](https://github.com/tresoldi/freqprob/actions/workflows/quality.yml/badge.svg)](https://github.com/tresoldi/freqprob/actions/workflows/quality.yml)
[![PyPI version](https://badge.fury.io/py/freqprob.svg)](https://badge.fury.io/py/freqprob)
[![Python versions](https://img.shields.io/pypi/pyversions/freqprob.svg)](https://pypi.org/project/freqprob/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

**A modern, high-performance Python library for probability smoothing and frequency-based language modeling.**

FreqProb provides state-of-the-art smoothing techniques for converting frequency counts into probability estimates, with applications in natural language processing, information retrieval, and statistical modeling.

## 🌟 Why FreqProb?

### **🎯 Comprehensive & Accurate**
- **10+ smoothing methods**: From basic Laplace to advanced Kneser-Ney and Simple Good-Turing
- **Mathematically rigorous**: Implementations validated against reference sources (NLTK, SciPy)
- **Production-ready**: Extensive testing with 400+ test cases and property-based validation

### **⚡ High Performance**
- **Vectorized operations**: Batch processing with NumPy acceleration
- **Memory efficient**: Compressed representations and streaming algorithms  
- **Lazy evaluation**: Compute probabilities only when needed
- **Caching system**: Intelligent memoization for expensive operations

### **🔧 Developer Experience**
- **Type safety**: Full type hints with mypy validation
- **Modern Python**: Requires Python 3.10+, uses latest language features
- **Rich documentation**: Mathematical background, tutorials, and API reference
- **Easy integration**: Clean, intuitive API design

## 🚀 Quick Start

### Installation

```bash
pip install freqprob
```

For additional features:
```bash
pip install freqprob[all]  # All optional dependencies
```

### Basic Usage

```python
import freqprob

# Create a frequency distribution
word_counts = {'the': 100, 'cat': 50, 'dog': 30, 'bird': 10}

# Basic smoothing - handles zero probabilities
laplace = freqprob.Laplace(word_counts, bins=10000)
print(f"P(cat) = {laplace('cat'):.4f}")      # 0.0053
print(f"P(elephant) = {laplace('elephant'):.6f}")  # 0.000105 (unseen word)

# Advanced smoothing for n-gram models
bigrams = {('the', 'cat'): 25, ('the', 'dog'): 20, ('a', 'cat'): 15}
kneser_ney = freqprob.KneserNey(bigrams, discount=0.75)

# Model evaluation
test_data = ['cat', 'dog', 'bird'] * 10
perplexity = freqprob.perplexity(laplace, test_data)
print(f"Perplexity: {perplexity:.2f}")
```

### High-Performance Operations

```python
# Vectorized batch processing
vectorized = freqprob.VectorizedScorer(laplace)
words = ['cat', 'dog', 'bird', 'fish'] * 1000
scores = vectorized.score_batch(words)  # Fast batch scoring

# Memory-efficient streaming for large datasets
streaming = freqprob.StreamingMLE(max_vocabulary_size=100000)
for word in massive_text_stream:
    streaming.update_single(word)

# Compare multiple models
models = {
    'laplace': freqprob.Laplace(word_counts, bins=10000),
    'kneser_ney': freqprob.KneserNey(bigrams),
    'simple_gt': freqprob.SimpleGoodTuring(word_counts)
}
comparison = freqprob.model_comparison(models, test_data)
```

## 🎓 Smoothing Methods

### Basic Methods
- **MLE (Maximum Likelihood)**: Unsmoothed relative frequencies
- **Laplace (Add-One)**: Classic add-one smoothing
- **Lidstone (Add-k)**: Generalized additive smoothing
- **ELE (Expected Likelihood)**: Lidstone with γ=0.5

### Advanced Methods  
- **Simple Good-Turing**: Frequency-of-frequency based smoothing
- **Kneser-Ney**: State-of-the-art for n-gram language models
- **Modified Kneser-Ney**: Improved version with automatic parameter estimation
- **Bayesian**: Dirichlet prior-based smoothing
- **Interpolated**: Linear combination of multiple models

### Specialized Features
- **Streaming algorithms**: Real-time updates for large datasets
- **Memory optimization**: Compressed and sparse representations
- **Performance profiling**: Built-in benchmarking and validation tools

## 📊 Use Cases

### **Natural Language Processing**
```python
# Language modeling
bigrams = freqprob.ngram_frequency(tokens, n=2)
lm = freqprob.KneserNey(bigrams, discount=0.75)

# Text classification with smoothed features
doc_features = freqprob.word_frequency(document_tokens)
classifier_probs = freqprob.Laplace(doc_features, bins=vocab_size)
```

### **Information Retrieval**
```python
# Document scoring with term frequency smoothing
term_counts = compute_term_frequencies(document)
smoothed_tf = freqprob.BayesianSmoothing(term_counts, alpha=0.5)

# Query likelihood with unseen term handling
query_prob = sum(smoothed_tf(term) for term in query_terms)
```

### **Data Science & Analytics**
```python
# Probability estimation for sparse categorical data
category_counts = {cat: count for cat, count in data.value_counts().items()}
estimator = freqprob.SimpleGoodTuring(category_counts)

# Handle zero frequencies in statistical analysis
smoothed_dist = freqprob.ELE(observed_frequencies, bins=total_categories)
```

## 🔬 Quality & Reliability

### **Rigorous Testing**
- **400+ test cases** covering edge cases and normal operations
- **Property-based testing** with Hypothesis for mathematical correctness
- **Regression testing** against reference implementations (NLTK, SciPy)
- **Numerical stability** validation for extreme inputs

### **Performance Validated**
- **Benchmarking framework** for performance regression detection
- **Memory profiling** to ensure efficient resource usage
- **Scaling analysis** from small to large vocabulary sizes
- **Cross-platform testing** on Linux, Windows, and macOS

### **Mathematical Accuracy**
- **Formula verification** against academic literature
- **Statistical correctness** validation with known distributions
- **Precision testing** for floating-point edge cases
- **Reference compatibility** with established libraries

## 📚 Documentation & Learning

- **[User Guide](docs/user_guide.md)**: Mathematical foundations and usage patterns
- **[API Reference](docs/api_reference.md)**: Complete function and class documentation
- **[Performance Guide](docs/performance_comparison.md)**: Optimization tips and benchmarks
- **[Development Guide](docs/development.md)**: Contributing and development setup

### 📖 Interactive Tutorials

Learn FreqProb through comprehensive, executable tutorials with visualizations:

1. **[Basic Smoothing Methods](docs/tutorial_1_basic_smoothing.py)** ([View HTML](https://htmlpreview.github.io/?https://github.com/tresoldi/freqprob/blob/main/docs/tutorial_1_basic_smoothing.html))
   - Introduction to probability smoothing
   - MLE, Laplace, Lidstone, and ELE methods
   - Model evaluation with perplexity

2. **[Advanced Methods](docs/tutorial_2_advanced_methods.py)** ([View HTML](https://htmlpreview.github.io/?https://github.com/tresoldi/freqprob/blob/main/docs/tutorial_2_advanced_methods.html))
   - Simple Good-Turing smoothing
   - Kneser-Ney and Modified Kneser-Ney
   - Bayesian and interpolated methods

3. **[Efficiency & Memory](docs/tutorial_3_efficiency_memory.py)** ([View HTML](https://htmlpreview.github.io/?https://github.com/tresoldi/freqprob/blob/main/docs/tutorial_3_efficiency_memory.html))
   - Vectorized batch processing
   - Streaming algorithms
   - Memory-efficient representations

4. **[Real-World Applications](docs/tutorial_4_real_world_applications.py)** ([View HTML](https://htmlpreview.github.io/?https://github.com/tresoldi/freqprob/blob/main/docs/tutorial_4_real_world_applications.html))
   - Language modeling
   - Text classification
   - Information retrieval

*Tutorials are written using [Nhandu](https://github.com/tresoldi/nhandu) literate programming format.*

## 🏁 Performance Benchmarking

FreqProb provides a comprehensive benchmarking suite to evaluate the speed, memory usage, and accuracy of all smoothing methods.

### Quick Benchmark

```bash
# Using Makefile
make bench

# Direct Python invocation
python docs/benchmarks.py --quick
```

### Detailed Analysis

```bash
# Full benchmarks
make bench-all

# Custom benchmarks
python scripts/run_benchmarks.py --output results --format all
```

Results are saved in timestamped directories with comprehensive analysis reports.

*See the [Performance Guide](docs/performance_comparison.md) for detailed comparisons and optimization tips.*

## 🤝 Contributing & Development

We welcome contributions! For detailed developer instructions—including environment setup, running the full CI suite locally, making releases, troubleshooting, and best practices—please see our [Development Guide](docs/development.md).

### Quick Development Setup

```bash
git clone https://github.com/tresoldi/freqprob.git
cd freqprob
make install-dev  # Install with dev dependencies
make quality      # Run code quality checks
make test         # Run test suite
```

## 📜 License

FreqProb is released under the [GNU General Public License v3.0](LICENSE). See the license file for details.

## 🎓 Citation

If you use FreqProb in academic research, please cite:

```bibtex
@software{tresoldi_freqprob_2025,
  author = {Tresoldi, Tiago},
  title = {FreqProb: A Python library for probability smoothing and frequency-based language modeling},
  url = {https://github.com/tresoldi/freqprob},
  version = {0.3.1},
  publisher = {Department of Linguistics and Philology, Uppsala University},
  address = {Uppsala},
  year = {2025}
}
```
