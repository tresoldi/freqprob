# Development Guide

This guide covers the development workflow, tools, and best practices for FreqProb.

## Development Environment Setup

### Prerequisites

- Python 3.10+
- Git
- Modern terminal/shell

### Quick Setup

```bash
# Clone the repository
git clone https://github.com/tresoldi/freqprob.git
cd freqprob

# Install Hatch (modern Python project manager)
pip install hatch

# Install pre-commit hooks
hatch run precommit-install
```

### Available Hatch Commands

FreqProb uses Hatch environments to organize development tasks. Here are the main commands:

```bash
# Show available environments
hatch env show

# Run commands in default environment
hatch run <command>

# Run commands in specific environment
hatch run lint:<command>
hatch run docs:<command>
hatch run ci:<command>
```

## Project Structure

```
freqprob/
├── freqprob/              # Main package
│   ├── __init__.py        # Package exports
│   ├── base.py            # Base classes
│   ├── basic.py           # Basic smoothing methods
│   ├── advanced.py        # Advanced smoothing methods
│   ├── smoothing.py       # Additional smoothing methods
│   ├── utils.py           # Utility functions
│   ├── cache.py           # Caching functionality
│   ├── vectorized.py      # Vectorized operations
│   ├── lazy.py            # Lazy evaluation
│   ├── streaming.py       # Streaming/incremental updates
│   ├── memory_efficient.py # Memory optimization
│   └── profiling.py       # Memory profiling
├── tests/                 # Test suite
├── docs/                  # Documentation
├── .github/               # GitHub workflows
├── .pre-commit-config.yaml # Pre-commit configuration
└── pyproject.toml         # Project configuration
```

## Development Workflow

### 1. Feature Development

```bash
# Create a new branch for your feature
git checkout -b feature/your-feature-name

# Make your changes
# ...

# Run tests to ensure everything works
hatch run test

# Format and lint your code
hatch run lint:format
hatch run lint:all

# Commit your changes (pre-commit hooks will run automatically)
git add .
git commit -m "feat: add your feature description"

# Push your branch
git push origin feature/your-feature-name
```

### 2. Essential Development Commands

#### Quick Start Commands
```bash
# Install project locally in development mode
hatch run install-dev

# Run all tests
hatch run test

# Run tests with coverage
hatch run test-cov

# Run linting and formatting
hatch run lint:all

# Clean build artifacts
hatch run clean
```

#### Build and Release Commands
```bash
# Build package for distribution
hatch run build

# Run full CI pipeline locally (mimics GitHub Actions)
hatch run ci:ci-full

# Quick CI check
hatch run ci:ci-quick

# Prepare for release (full check + build)
hatch run ci:release-check
```

### 3. Testing

We use pytest with comprehensive coverage requirements:

```bash
# Run all tests
hatch run test

# Run tests with coverage report
hatch run test-cov

# Run tests in parallel (faster)
hatch run test-fast

# Run specific test suites
hatch run test-numerical     # Numerical stability tests
hatch run test-statistical   # Statistical correctness tests
hatch run test-regression    # Regression tests
hatch run test-property      # Property-based tests

# Run specific test file
hatch run test tests/test_basic.py

# Run tests matching a pattern
hatch run test -k "test_laplace"
```

### 4. Code Quality

#### Formatting and Linting

```bash
# Format code (black + isort)
hatch run lint:format

# Check style without modifying files
hatch run lint:style

# Run type checking
hatch run lint:typing

# Run security checks
hatch run lint:security

# Run all quality checks
hatch run lint:all
```

#### Pre-commit Hooks

Pre-commit hooks run automatically on `git commit` and include:

- **black**: Code formatting
- **isort**: Import sorting
- **ruff**: Fast linting (replaces flake8, pycodestyle, etc.)
- **mypy**: Type checking
- **bandit**: Security linting
- **various**: YAML/JSON validation, trailing whitespace, etc.

```bash
# Install pre-commit hooks
hatch run precommit-install

# Run pre-commit on all files manually
hatch run precommit

# Run pre-commit on staged files only
hatch run precommit-staged

# Update pre-commit hook versions
hatch run precommit-update
```

### 5. Documentation

#### Building Documentation

```bash
# Start documentation server (auto-reload)
hatch run docs:serve

# Build documentation
hatch run docs:build

# Test Jupyter notebooks
hatch run docs:test-notebooks

# Launch Jupyter for editing notebooks
hatch run docs:notebooks
```

#### Writing Documentation

- **User Guide**: High-level explanations and tutorials in `docs/user_guide.md`
- **API Reference**: Auto-generated from docstrings in `docs/api_reference.md`
- **Tutorials**: Interactive Jupyter notebooks in `docs/`
- **Docstrings**: Follow Google style conventions

Example docstring:
```python
def example_function(param1: str, param2: int = 5) -> bool:
    """Brief description of the function.

    Longer description explaining the function's purpose,
    behavior, and any important details.

    Args:
        param1: Description of the first parameter.
        param2: Description of the second parameter. Defaults to 5.

    Returns:
        Description of the return value.

    Raises:
        ValueError: When param1 is empty.

    Example:
        >>> result = example_function("test", 10)
        >>> print(result)
        True
    """
```

### 6. Performance Testing

```bash
# Run quick benchmarks
hatch run bench

# Run comprehensive benchmarks
hatch run bench-all

# Run validation tests
hatch run validate

# Run quick validation
hatch run validate-quick

# Profile memory usage
python -c "
import freqprob
profiler = freqprob.MemoryProfiler()
with profiler.profile_operation('test'):
    # Your code here
    pass
print(profiler.get_latest_metrics())
"
```

## Release Process

### 1. Version Bumping

We use semantic versioning (MAJOR.MINOR.PATCH):

```bash
# Check current version
hatch run ci:version-show

# Bump version using Hatch
hatch run ci:version-patch  # 0.1.0 -> 0.1.1
hatch run ci:version-minor  # 0.1.0 -> 0.2.0  
hatch run ci:version-major  # 0.1.0 -> 1.0.0

# Or trigger version bump workflow (requires repository access)
gh workflow run version-bump.yml --ref main -f bump_type=patch
gh workflow run version-bump.yml --ref main -f bump_type=minor  
gh workflow run version-bump.yml --ref main -f bump_type=major
```

### 2. Pre-release Validation

```bash
# Run full release checks locally
hatch run ci:release-check

# This runs:
# - Clean build artifacts
# - Full CI pipeline
# - Package build
# - Package validation with twine
```

### 3. Creating Releases

```bash
# After version bump PR is merged, create and push a tag
git tag v1.0.0
git push origin v1.0.0

# This triggers the release workflow which:
# 1. Runs full test suite
# 2. Builds distribution packages
# 3. Creates GitHub release
# 4. Publishes to PyPI
```

## Continuous Integration

### Running GitHub Actions Locally

You can run the equivalent of GitHub Actions locally using Hatch:

```bash
# Run full CI pipeline (equivalent to GitHub Actions)
hatch run ci:ci-full

# Run quick CI checks
hatch run ci:ci-quick

# Run individual components
hatch run lint:all        # Linting and formatting
hatch run test-cov        # Tests with coverage
hatch run docs:build      # Documentation build
hatch run bench --quick   # Quick benchmarks
```

### GitHub Actions Workflows

- **CI** (`ci.yml`): Runs on every push/PR
  - Linting and formatting checks
  - Tests on multiple Python versions and OS
  - Security scans
  - Build verification
  - Documentation tests

- **Release** (`release.yml`): Runs on version tags
  - Creates GitHub releases
  - Publishes to PyPI

- **Version Bump** (`version-bump.yml`): Manual trigger
  - Automated version bumping
  - Creates PR with changes

### Quality Gates

All PRs must pass:
- ✅ All tests passing on Python 3.10-3.12
- ✅ Code coverage ≥85%
- ✅ All linting checks passing
- ✅ Type checking with mypy
- ✅ Security scan with bandit
- ✅ Documentation builds successfully

## Code Standards

### Style Guidelines

- **Line length**: 100 characters (configured in black)
- **Imports**: Sorted with isort, grouped by type
- **Type hints**: Required for all public functions
- **Docstrings**: Google style for all public APIs
- **Variable names**: Clear, descriptive names

### Performance Guidelines

- Use numpy for vectorized operations when possible
- Implement caching for expensive computations
- Provide lazy evaluation options for large datasets
- Include memory-efficient alternatives
- Profile performance-critical code

### Testing Guidelines

- **Coverage**: Aim for >90% line coverage
- **Test types**: Unit tests, integration tests, property-based tests
- **Fixtures**: Use pytest fixtures for common test data
- **Markers**: Use `@pytest.mark.slow` for expensive tests
- **Assertions**: Descriptive assertion messages

Example test:
```python
def test_laplace_smoothing_basic():
    """Test basic Laplace smoothing functionality."""
    freqdist = {'cat': 3, 'dog': 2}
    laplace = freqprob.Laplace(freqdist, bins=100, logprob=False)

    # Test known word probability
    expected_cat = (3 + 1) / (5 + 100)
    assert abs(laplace('cat') - expected_cat) < 1e-10

    # Test unknown word probability  
    expected_unknown = 1 / (5 + 100)
    assert abs(laplace('bird') - expected_unknown) < 1e-10
```

## Debugging and Profiling

### Common Debug Commands

```bash
# Run tests with debugging
hatch run test --pdb

# Profile test execution time
hatch run test --durations=10

# Run with verbose output
hatch run test -v -s

# Memory profiling with memory_profiler
pip install memory_profiler
python -m memory_profiler your_script.py
```

### Memory Debugging

```python
import freqprob

# Use built-in memory profiler
profiler = freqprob.MemoryProfiler()
with profiler.profile_operation("model_creation"):
    model = freqprob.SimpleGoodTuring(large_dataset)

metrics = profiler.get_latest_metrics()
print(f"Memory used: {metrics.memory_delta_mb:.2f} MB")
```

## Troubleshooting

### Common Issues

1. **Import errors**: Make sure you're in the correct environment
   ```bash
   hatch env show  # Show current environment
   hatch shell     # Activate environment shell
   ```

2. **Pre-commit failures**: Update hooks and retry
   ```bash
   hatch run precommit-update
   hatch run precommit
   ```

3. **Test failures**: Check for missing dependencies
   ```bash
   hatch run pip list  # Show installed packages
   hatch env create --force  # Recreate environment
   ```

4. **Type checking errors**: Install missing type stubs
   ```bash
   hatch run lint:typing  # Will install missing stubs
   ```

### Hatch Environment Issues

1. **Environment creation failures**: Force recreate environments
   ```bash
   hatch env prune        # Remove all environments
   hatch env create       # Recreate default environment
   ```

2. **Missing dependencies**: Check environment dependencies
   ```bash
   hatch dep show         # Show dependency tree
   hatch env show         # Show environment info
   ```

3. **Script not found**: Ensure you're using the correct environment
   ```bash
   hatch env show         # List available environments
   hatch run lint:all     # Use environment:script syntax
   ```

### Getting Help

- **Documentation**: Check docs/ directory
- **Issues**: Search GitHub issues
- **Discussions**: Use GitHub discussions for questions
- **Code Review**: All PRs receive thorough review

## Best Practices Summary

1. **Always** run tests before committing
2. **Always** run linting and formatting
3. **Write** comprehensive docstrings
4. **Add** tests for new functionality  
5. **Update** documentation for user-facing changes
6. **Follow** semantic versioning for changes
7. **Profile** performance-critical code
8. **Consider** memory efficiency in implementations
9. **Use** type hints consistently
10. **Keep** commits focused and well-described

This development guide ensures consistent, high-quality contributions to FreqProb. For questions or suggestions, please open an issue or discussion on GitHub.
