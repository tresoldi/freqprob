PYTHON_BINARY := python3
PROJECT_NAME := freqprob
TEST_DIR := tests

## help - Display help about make targets for this Makefile
help:
	@cat Makefile | grep '^## ' --color=never | cut -c4- | sed -e "`printf 's/ - /\t- /;'`" | column -s "`printf '\t'`" -t

## build - Builds the project in preparation for release
build:
	$(PYTHON_BINARY) -m build

## coverage - Test the project and generate an HTML coverage report
coverage:
	pytest --cov=$(PROJECT_NAME) --cov-branch --cov-report=html --cov-report=lcov --cov-report=term-missing

## clean - Remove build artifacts and clear out .pyc files
clean:
	rm -rf dist *.egg-info .coverage
	find . -name '*.pyc' -delete

## black - Runs the Black Python formatter against the project
black:
	black $(PROJECT_NAME)/ $(TEST_DIR)/

## black-check - Checks if the project is formatted correctly against the Black rules
black-check:
	black $(PROJECT_NAME)/ $(TEST_DIR)/ --check

## format - Runs all formatting tools against the project
format: black isort lint mypy

## format-quick - Runs just isort and black formatting (for pre-push)
format-quick: isort black

## format-check - Checks if the project is formatted correctly against all formatting rules
format-check: black-check isort-check lint mypy

## install - Install the project locally
install:
	pip install -e ."[dev]"

## isort - Sorts imports throughout the project
isort:
	isort $(PROJECT_NAME)/ $(TEST_DIR)/

## isort-check - Checks that imports throughout the project are sorted correctly
isort-check:
	isort $(PROJECT_NAME)/ $(TEST_DIR)/ --check-only

## lint - Lint the project
lint:
	flake8 $(PROJECT_NAME)/ $(TEST_DIR)/

## mypy - Run mypy type checking on the project
mypy:
	mypy $(PROJECT_NAME)/ $(TEST_DIR)/

## test - Test the project
test:
	pytest

.PHONY: help build coverage clean black black-check format format-quick format-check install isort isort-check lint mypy test
