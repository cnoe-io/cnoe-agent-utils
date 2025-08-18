# Makefile

.PHONY: setup-venv activate-venv help test test-venv test-examples test-all

default: test-all

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
	@echo " Installing dependencies with uv       "
	@echo "======================================="
	. .venv/bin/activate && uv sync --dev

activate-venv:
	@echo "Activating virtual environment..."
	@if [ -d "venv" ]; then \
		. venv/bin/activate; \
	else \
		echo "Virtual environment not found. Please run 'make setup-venv' first."; \
	fi

lint: setup-venv
	@echo "Running ruff..."
	. .venv/bin/activate && ruff check openapi_mcp_codegen tests

ruff-fix: setup-venv
	@echo "Running ruff and fix lint errors..."
	. .venv/bin/activate && ruff check openapi_mcp_codegen tests --fix

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
	. .venv/bin/activate && uv sync --dev

test: test-venv
	@echo "======================================="
	@echo " Running pytest on tests directory     "
	@echo "======================================="
	. .venv/bin/activate && pytest tests -v --cov=cnoe_agent_utils --cov-report=term-missing --cov-report=html

test-examples: test-venv
	@echo "======================================="
	@echo " Running examples with detailed output "
	@echo "======================================="
	@echo "Loading .env file for environment variables..."
	@if [ -f .env ]; then \
		echo ".env file found, will be loaded by individual scripts."; \
	else \
		echo "No .env file found. Proceeding without loading .env."; \
	fi
	@echo "Testing Azure OpenAI consolidated script..."
	. .venv/bin/activate && python3 examples/azure_openai_example.py || echo "Azure OpenAI consolidated test failed"
	@echo ""
	@echo "Testing OpenAI consolidated script..."
	. .venv/bin/activate && python3 examples/openai_example.py || echo "OpenAI consolidated test failed"
	@echo ""
	#@echo "Testing Anthropic..."
	# @echo "Testing Anthropic..."
	# . .venv/bin/activate && set -a && [ -f .env ] && . .env || true && set +a && python3 examples/anthropic_stream.py || echo "Anthropic streaming failed"
	@echo ""
	@echo "Testing AWS Bedrock..."
	. .venv/bin/activate && python3 examples/aws_bedrock_stream.py || echo "AWS Bedrock failed"
	@echo ""
	@echo "Testing Google Gemini..."
	. .venv/bin/activate && python3 examples/google_gemini_stream.py || echo "Google Gemini failed"
	@echo ""
	@echo "Testing Google Vertex AI..."
	. .venv/bin/activate && python3 examples/gcp_vertex_stream.py || echo "Google Vertex AI failed"
	@echo ""
	@echo "======================================="
	@echo " All example tests completed"
	@echo "======================================="

test-all: test-venv
	@echo "======================================="
	@echo " Running all tests and examples"
	@echo "======================================="
	@echo "Running pytest tests..."
	. .venv/bin/activate && pytest tests/ -v
	@echo "======================================="
	@echo "Running example tests..."
	$(MAKE) test
	$(MAKE) test-examples

coverage: test-venv
	@echo "======================================="
	@echo " Running tests with detailed coverage   "
	@echo "======================================="
	. .venv/bin/activate && pytest tests/ --cov=cnoe_agent_utils --cov-report=term-missing --cov-report=html --cov-report=xml
	@echo "======================================="
	@echo " Coverage report generated in htmlcov/ "
	@echo " Open htmlcov/index.html in your browser"

## ========== Release & Versioning ==========
release: setup-venv  ## Bump version and create a release
	@. .venv/bin/activate; uv sync --dev
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
	@echo "  test                           Run tests using pytest with coverage"
	@echo "  test-venv                      Set up test virtual environment and install test dependencies"
	@echo "  test-examples                  Run all example scripts and show test results"
	@echo "  test-all                       Run all tests (unit, integration, examples)"
	@echo "  coverage                       Run tests with detailed coverage reports"
	@echo "  help                           Show this help message"
