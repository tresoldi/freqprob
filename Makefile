PYTHON_BINARY := python3
VIRTUAL_ENV := venv
VIRTUAL_BIN := $(VIRTUAL_ENV)/bin
PROJECT_NAME := freqprob
TEST_DIR := tests
SCRIPTS_DIR := scripts
DOCS_DIR := docs

## help - Display help about make targets for this Makefile
help:
	@cat Makefile | grep '^## ' --color=never | cut -c4- | sed -e "`printf 's/ - /\t- /;'`" | column -s "`printf '\t'`" -t

## build - Builds the project in preparation for release
build:
	$(VIRTUAL_BIN)/python -m build

## coverage - Test the project and generate an HTML coverage report
coverage:
	$(VIRTUAL_BIN)/pytest --cov=$(PROJECT_NAME) --cov-branch --cov-report=html --cov-report=lcov --cov-report=term-missing

## clean - Remove the virtual environment and clear out .pyc files
clean:
	rm -rf $(VIRTUAL_ENV) dist *.egg-info .coverage
	find . -name '*.pyc' -delete

## format - Runs all formatting tools against the project
format: ruff lint mypy

## format-quick - Runs just ruff formatting (for pre-push)
format-quick: ruff

## format-check - Checks if the project is formatted correctly against all formatting rules
format-check: ruff-check lint mypy

## install - Install the project locally
install:
	$(PYTHON_BINARY) -m venv $(VIRTUAL_ENV)
	$(VIRTUAL_BIN)/pip install -e ."[dev]"

## lint - Lint the project
lint:
	$(VIRTUAL_BIN)/ruff check $(PROJECT_NAME)/ $(TEST_DIR)/ $(SCRIPTS_DIR)/ $(DOCS_DIR)/

## ruff - Run ruff linting and formatting (including notebooks)
ruff:
	$(VIRTUAL_BIN)/ruff check $(PROJECT_NAME)/ $(TEST_DIR)/ $(SCRIPTS_DIR)/ $(DOCS_DIR)/ --fix
	$(VIRTUAL_BIN)/ruff format $(PROJECT_NAME)/ $(TEST_DIR)/ $(SCRIPTS_DIR)/ $(DOCS_DIR)/

## ruff-check - Check ruff linting and formatting without fixing (including notebooks)
ruff-check:
	$(VIRTUAL_BIN)/ruff check $(PROJECT_NAME)/ $(TEST_DIR)/ $(SCRIPTS_DIR)/ $(DOCS_DIR)/
	$(VIRTUAL_BIN)/ruff format $(PROJECT_NAME)/ $(TEST_DIR)/ $(SCRIPTS_DIR)/ $(DOCS_DIR)/ --check

## mypy - Run mypy type checking on the project
mypy:
	$(VIRTUAL_BIN)/mypy --strict $(PROJECT_NAME)/ $(TEST_DIR)/ $(SCRIPTS_DIR)/


## precommit - Run pre-commit hooks on all files
precommit:
	$(VIRTUAL_BIN)/pre-commit run --all-files

## precommit-check - Run pre-commit hooks on staged files only
precommit-check:
	$(VIRTUAL_BIN)/pre-commit run

## precommit-install - Install pre-commit hooks (not recommended for local development)
precommit-install:
	@echo "Warning: This will install pre-commit hooks to run automatically on commits."
	@echo "For local development, use 'make precommit' instead."
	@read -p "Are you sure you want to install Git hooks? [y/N]: " confirm && [ "$$confirm" = "y" ] || [ "$$confirm" = "Y" ]
	$(VIRTUAL_BIN)/pre-commit install

## precommit-update - Update pre-commit hooks to latest versions
precommit-update:
	$(VIRTUAL_BIN)/pre-commit autoupdate

## test - Test the project
test:
	$(VIRTUAL_BIN)/pytest

.PHONY: help build coverage clean format format-quick format-check install lint mypy precommit precommit-check precommit-update ruff ruff-check test
