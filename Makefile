# FreqProb Makefile
# POSIX-compatible development commands

.PHONY: help quality format test test-cov test-fast bump-version build build-release clean install install-dev validate validate-quick bench bench-all docs docs-clean

# Default target: show help
.DEFAULT_GOAL := help

# Python interpreter
PYTHON := python3
PIP := $(PYTHON) -m pip

# Version bump type (patch, minor, major)
TYPE ?= patch

help: ## Show this help message
	@echo "FreqProb Development Commands"
	@echo "=============================="
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Usage examples:"
	@echo "  make quality              # Run all quality checks"
	@echo "  make test-cov             # Run tests with coverage"
	@echo "  make bump-version TYPE=minor  # Bump minor version"
	@echo "  make build-release        # Full release build"

quality: ## Run code quality checks (ruff format --check, ruff check, mypy)
	@echo "==> Checking code formatting..."
	ruff format --check .
	@echo "==> Running ruff linter..."
	ruff check .
	@echo "==> Running mypy type checker..."
	mypy freqprob/ tests/
	@echo "✓ All quality checks passed!"

format: ## Auto-format code with ruff
	@echo "==> Formatting code with ruff..."
	ruff format .
	@echo "✓ Code formatted!"

test: ## Run test suite
	@echo "==> Running tests..."
	pytest tests/
	@echo "✓ Tests passed!"

test-cov: ## Run tests with coverage (HTML report in tests/htmlcov/, fails if <80%)
	@echo "==> Running tests with coverage..."
	pytest --cov=freqprob --cov-report=html:tests/htmlcov --cov-report=term-missing --cov-fail-under=80 tests/
	@echo "✓ Coverage report generated in tests/htmlcov/"

test-fast: ## Run tests in parallel (faster)
	@echo "==> Running tests in parallel..."
	pytest -n auto tests/
	@echo "✓ Tests passed!"

bump-version: ## Bump version (TYPE=patch|minor|major), commit, and tag
	@CURRENT=$$(grep -o "__version__ = \"[^\"]*\"" freqprob/__init__.py | cut -d'"' -f2); \
	echo "==> Current version: $$CURRENT"; \
	IFS='.' read -r major minor patch <<< "$$CURRENT"; \
	if [ "$(TYPE)" = "major" ]; then NEW="$$((major + 1)).0.0"; \
	elif [ "$(TYPE)" = "minor" ]; then NEW="$$major.$$((minor + 1)).0"; \
	elif [ "$(TYPE)" = "patch" ]; then NEW="$$major.$$minor.$$((patch + 1))"; \
	else echo "Error: TYPE must be patch, minor, or major"; exit 1; fi; \
	echo "==> Bumping $(TYPE) version to $$NEW..."; \
	sed -i "s/__version__ = \"$$CURRENT\"/__version__ = \"$$NEW\"/" freqprob/__init__.py; \
	echo ""; \
	echo "⚠️  Please update CHANGELOG.md manually before committing!"; \
	echo ""; \
	read -p "Press Enter to commit and tag, or Ctrl+C to cancel..."; \
	git add freqprob/__init__.py; \
	git commit -m "chore: bump version to $$NEW"; \
	git tag -a "v$$NEW" -m "Release v$$NEW"; \
	echo "✓ Version bumped to $$NEW and tagged!"; \
	echo ""; \
	echo "Next steps:"; \
	echo "  1. Update CHANGELOG.md"; \
	echo "  2. git add CHANGELOG.md && git commit --amend --no-edit"; \
	echo "  3. git push && git push --tags"

build: ## Build package (creates dist/)
	@echo "==> Building package..."
	$(PYTHON) -m build
	@echo "✓ Package built in dist/"

build-release: clean quality test build ## Full release build (clean → quality → test → build)
	@echo "✓ Release build complete!"
	@echo ""
	@echo "Package ready in dist/"
	@ls -lh dist/

clean: ## Remove build artifacts, caches, and coverage reports
	@echo "==> Cleaning build artifacts..."
	rm -rf dist/ build/ *.egg-info
	rm -rf .coverage htmlcov/ tests/htmlcov/ coverage.xml
	rm -rf .pytest_cache .ruff_cache .mypy_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@echo "✓ Cleaned!"

install: ## Install package in development mode
	@echo "==> Installing package..."
	$(PIP) install -e .
	@echo "✓ Package installed!"

install-dev: ## Install package with development dependencies (includes all Makefile tools)
	@echo "==> Installing package with dev dependencies..."
	$(PIP) install -e .[dev]
	@echo "✓ Package installed with dev dependencies!"
	@echo ""
	@echo "Installed tools for Makefile:"
	@echo "  - pytest, pytest-cov, pytest-xdist (testing)"
	@echo "  - ruff, mypy (code quality)"
	@echo "  - build, twine (build/release)"
	@echo "  - nhandu (documentation generation)"

validate: ## Run full validation suite
	@echo "==> Running validation suite..."
	$(PYTHON) scripts/validation_report.py --output-dir validation_results
	@echo "✓ Validation complete! Results in validation_results/"

validate-quick: ## Run quick validation checks
	@echo "==> Running quick validation..."
	$(PYTHON) scripts/validation_report.py --quick --output-dir quick_validation
	@echo "✓ Quick validation complete!"

bench: ## Run quick performance benchmarks
	@echo "==> Running quick benchmarks..."
	$(PYTHON) docs/benchmarks.py --quick
	@echo "✓ Benchmarks complete!"

bench-all: ## Run comprehensive benchmark suite
	@echo "==> Running comprehensive benchmarks..."
	$(PYTHON) scripts/run_benchmarks.py
	@echo "✓ Comprehensive benchmarks complete!"

docs: ## Generate HTML documentation from Nhandu tutorial sources
	@echo "==> Generating tutorial documentation..."
	@for f in docs/tutorial_*.py; do \
		echo "  Generating $$(basename $$f .py).html..."; \
		nhandu "$$f" -o "docs/$$(basename $$f .py).html"; \
	done
	@echo "✓ Documentation generated in docs/"

docs-clean: ## Remove generated HTML documentation
	@echo "==> Cleaning generated documentation..."
	rm -f docs/tutorial_*.html
	@echo "✓ Documentation cleaned!"
