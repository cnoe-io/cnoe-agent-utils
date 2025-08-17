# Test Directory Structure

This directory contains all tests and examples for the `cnoe-agent-utils` library.

## Directory Organization

### `tests/` (Main Directory)
Contains all test files following the `test_*` pattern:
- `test_*.py` - Pytest test files for different LLM providers and functionality
- Fast, focused tests that may or may not require external API calls

### `tests/examples/`
Contains example scripts that demonstrate library usage:
- `*_stream.py` - Streaming examples for different LLM providers
- `*_invoke*.py` - Invoke examples for different LLM providers
- `README_TIMING.md` - Documentation for timing features
- `_archive/` - Archived examples

## Running Tests

### Run All Tests
```bash
make test-all
```

### Run Specific Test Types
```bash
# All pytest tests
make test

# Examples only
make examples

# All tests and examples
make test-all
```

### Run Individual Tests
```bash
# Activate virtual environment first
source .venv/bin/activate

# Run specific test file
python tests/examples/azure_openai_stream_gpt5.py

# Run with pytest
pytest tests/test_openai_gpt5.py -v
```

## Environment Setup

Most tests require environment variables to be set. You can:

1. **Use a .env file** (recommended):
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

2. **Set environment variables manually**:
   ```bash
   export OPENAI_API_KEY="your-key"
   export AZURE_OPENAI_API_KEY="your-key"
   # etc.
   ```

3. **Use the Makefile** (automatically sources .env):
   ```bash
   make test-examples
   ```

## Test Categories

### Pytest Tests (`tests/`)
- Fast execution for most tests
- May or may not require external dependencies
- Test individual functions, classes, and workflows
- Run with: `make test` or `pytest tests/`

### Example Tests (`tests/examples/`)
- Variable execution time
- Require API keys and external services
- Demonstrate real-world usage
- Run with: `make test-examples`

## Adding New Tests

### New Pytest Test
```bash
# Create in tests/
touch tests/test_new_feature.py
```

### New Example
```bash
# Create in tests/examples/
touch tests/examples/new_provider_stream.py
```

## Best Practices

1. **Unit tests** should be fast and not require external services
2. **Integration tests** should test real workflows but can be slower
3. **Examples** should demonstrate practical usage patterns
4. **Use environment variables** for API keys and configuration
5. **Include timing information** using the built-in timing utilities
6. **Add spinners** for long-running operations using `stream_with_spinner` or `invoke_with_spinner`
