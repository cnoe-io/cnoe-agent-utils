# Test Directory Structure

This directory contains all tests and examples for the `cnoe-agent-utils` library.

## Directory Organization

### `tests/unit/`
Contains unit tests that test individual components in isolation:
- `test_*.py` - Unit tests for specific LLM providers
- Fast, focused tests that don't require external API calls

### `tests/integration/`
Contains integration tests that test the full workflow:
- `test_*.py` - Integration tests for complete LLM workflows
- May require API keys and external services
- Tests the interaction between components

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
# Unit tests only
make test

# Examples only
make test-examples

# Integration tests only
pytest tests/integration/ -v
```

### Run Individual Tests
```bash
# Activate virtual environment first
source .venv/bin/activate

# Run specific test file
python tests/examples/azure_openai_stream_gpt5.py

# Run with pytest
pytest tests/unit/test_openai_gpt5.py -v
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

### Unit Tests (`tests/unit/`)
- Fast execution
- No external dependencies
- Test individual functions and classes
- Run with: `pytest tests/unit/`

### Integration Tests (`tests/integration/`)
- Medium execution time
- May require API keys
- Test complete workflows
- Run with: `pytest tests/integration/`

### Example Tests (`tests/examples/`)
- Variable execution time
- Require API keys and external services
- Demonstrate real-world usage
- Run with: `make test-examples`

## Adding New Tests

### New Unit Test
```bash
# Create in tests/unit/
touch tests/unit/test_new_feature.py
```

### New Integration Test
```bash
# Create in tests/integration/
touch tests/integration/test_new_workflow.py
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
