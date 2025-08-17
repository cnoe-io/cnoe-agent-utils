# Makefile

.PHONY: build setup-venv activate-venv install run run-acp run-client langgraph-dev help

add-copyright-license-headers:
	@echo "Adding copyright license headers..."
	docker run --rm -v $(shell pwd)/openapi_mcp_codegen:/workspace ghcr.io/google/addlicense:latest -c "CNOE" -l apache -s=only -v /workspace

setup-venv:
	@echo "======================================="
	@echo " Setting up the Virtual Environment   "
	@echo "======================================="
	@if [ ! -d ".venv" ]; then \
		python3 -m venv .venv; \
		echo "Virtual environment created."; \
	else \
		echo "Virtual environment already exists."; \
	fi

	@echo "======================================="
	@echo " Activating virtual environment       "
	@echo "======================================="
	@echo "To activate venv manually, run: source .venv/bin/activate"
	. .venv/bin/activate

	@echo "======================================="
	@echo " Adding pip as a Poetry dependency    "
	@echo "======================================="
	. .venv/bin/activate && poetry add pip --dev

	@echo "======================================="
	@echo " Installing dependencies with Poetry  "
	@echo "======================================="
	. .venv/bin/activate && poetry install

activate-venv:
	@echo "Activating virtual environment..."
	@if [ -d "venv" ]; then \
		. venv/bin/activate; \
	else \
		echo "Virtual environment not found. Please run 'make setup-venv' first."; \
	fi

install:
	@echo "======================================="
	@echo " Activating virtual environment and    "
	@echo " Installing poetry the current package "
	@echo "======================================="
	. .venv/bin/activate && poetry install

lint: setup-venv
	@echo "Running ruff..."
	. .venv/bin/activate && ruff check openapi_mcp_codegen tests

ruff-fix: setup-venv
	@echo "Running ruff and fix lint errors..."
	. .venv/bin/activate && ruff check openapi_mcp_codegen tests --fix

generate: setup-venv install
	@echo "Running the application with arguments: $(filter-out $@,$(MAKECMDGOALS))"
	@echo "Sourcing .env with set +a"
	@set +a; [ -f .env ] && . .env || true
	. .venv/bin/activate && poetry run python -m openapi_mcp_codegen $(filter-out $@,$(MAKECMDGOALS))

# This rule allows passing arguments to the run target
%:
	@:

cz-changelog: setup-venv
	@echo "======================================="
	@echo " Checking and installing commitizen    "
	@echo "======================================="
	. .venv/bin/activate && pip show commitizen >/dev/null 2>&1 || . .venv/bin/activate && pip install -U commitizen
	@echo "======================================="
	@echo " Generating changelog with cz-changelog"
	@echo "======================================="
	. .venv/bin/activate && cz bump --changelog

# test_Makefile

.PHONY: test test-venv

test-venv:
	@echo "======================================="
	@echo " Setting up test virtual environment   "
	@echo "======================================="
	@if [ ! -d ".venv" ]; then \
		python3 -m venv .venv; \
		echo "Test virtual environment created."; \
	else \
		echo "Test virtual environment already exists."; \
	fi
	@echo "======================================="
	@echo " Installing test dependencies         "
	@echo "======================================="
	. .venv/bin/activate && pip install -U pip pytest && poetry install

test: test-venv
	@echo "======================================="
	@echo " Running pytest on tests directory     "
	@echo "======================================="
	. .venv/bin/activate && pytest tests -v

examples: test-venv
	@echo "======================================="
	@echo " Running all example scripts           "
	@echo "======================================="
	@echo "Sourcing .env and running examples..."
	@if [ -f $(PWD)/.env ]; then \
		set -a; \
		. $(PWD)/.env; \
		set +a; \
	fi; \
	total=0; \
	passed=0; \
	failed=0; \
	for f in examples/*.py; do \
		if [ -f "$$f" ]; then \
			echo "Running $$f..."; \
			total=$$((total + 1)); \
			. .venv/bin/activate && python "$$f"; \
			status=$$?; \
			if [ $$status -eq 0 ]; then \
				echo "✅ $$f passed"; \
				passed=$$((passed + 1)); \
			else \
				echo "❌ $$f failed (exit code $$status)"; \
				failed=$$((failed + 1)); \
			fi; \
		fi; \
	done; \
	echo "======================================="; \
	echo " Test Results Summary:"; \
	echo " Total examples: $$total"; \
	echo " ✅ Passed: $$passed"; \
	echo " ❌ Failed: $$failed"; \
	echo "======================================="

test-all: test-venv
	@echo "======================================="
	@echo " Running all tests and examples"
	@echo "======================================="
	@echo "Running pytest tests..."
	. .venv/bin/activate && pytest tests/ -v
	@echo "======================================="
	@echo "Running example tests..."
	$(MAKE) examples

## ========== Release & Versioning ==========
release: setup-venv  ## Bump version and create a release
	@. .venv/bin/activate; poetry install
	@. .venv/bin/activate; poetry add commitizen --dev
	@. .venv/bin/activate; cz changelog
	@git add CHANGELOG.md
	@git commit -m "docs: update changelog"
	@. .venv/bin/activate; cz bump --increment PATCH
	@echo "Version bumped and stable tag updated successfully."

help:
	@echo "Available targets:"
	@echo "  add-copyright-license-headers  Add copyright license headers to source files"
	@echo "  setup-venv                     Create virtual environment in .venv and install dependencies"
	@echo "  activate-venv                  Activate the virtual environment"
	@echo "  install                        Install the package"
	@echo "  lint                           Run ruff linter on codebase"
	@echo "  ruff-fix                       Run ruff and fix lint errors"
	@echo "  generate [ARGS]                Build, install, and run the application with optional arguments"
	@echo "  cz-changelog                   Generate changelog using commitizen"
	@echo "  test                           Run tests using pytest"
	@echo "  test-venv                      Set up test virtual environment and install test dependencies"
	@echo "  test-examples                  Run all example scripts and show test results"
	@echo "  test-all                       Run all tests (unit, integration, examples)"
	@echo "  help                           Show this help message"
