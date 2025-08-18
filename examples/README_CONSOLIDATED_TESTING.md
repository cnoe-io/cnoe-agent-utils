# Consolidated Testing Approach

This document describes the new consolidated testing approach for LLM provider examples.

## Overview

Instead of running individual scripts for each model and mode, we now have consolidated test scripts that cover multiple models with both invoke and streaming modes in a single execution.

## Available Consolidated Tests

### 1. Azure OpenAI (`azure_openai_consolidated_test.py`)
- **Purpose**: Comprehensive testing of all Azure OpenAI models
- **Models Tested**: gpt-4o, gpt-4.1, gpt-5
- **Modes Tested**: Invoke and Streaming
- **Total Tests**: 6 (3 models × 2 modes)

### 2. OpenAI (`openai_consolidated_test.py`)
- **Purpose**: Comprehensive testing of all OpenAI models
- **Models Tested**: gpt-4o, gpt-4o-mini, gpt-5
- **Modes Tested**: Invoke and Streaming
- **Total Tests**: 6 (3 models × 2 modes)

## Features

### ✅ Comprehensive Coverage
- Tests multiple models in a single run
- Covers both invoke and streaming modes
- Consistent prompt and response handling across all tests
- Handles different API response formats automatically

### 🔧 Smart Error Handling
- Individual test failures don't stop the entire suite
- Detailed error reporting for each test
- Graceful degradation when specific models fail

### 📊 Detailed Reporting
- Real-time progress indicators
- Individual test results with pass/fail status
- Overall success rate calculation
- Timing information for each test

### 🚀 Easy Execution
- Single command to run all tests
- Environment variable validation
- Clear output formatting

## Usage

### Local Testing
```bash
# From project root
python examples/azure_openai_consolidated_test.py
python examples/openai_consolidated_test.py

# Or from examples directory
cd examples
python azure_openai_consolidated_test.py
python openai_consolidated_test.py
```

### CI/CD Integration
The scripts are integrated into the GitHub Actions workflow:
- **Automatic**: Runs on all pushes and PRs
- **Manual**: Can be triggered via workflow dispatch
- **Selective**: Choose between different test types

## Test Results Example

```
🚀 Starting Consolidated Azure OpenAI Tests
============================================================

📝 Testing INVOKE mode for all models...
✅ GPT-4o invoke test PASSED
✅ GPT-4.1 invoke test PASSED
✅ GPT-5 invoke test PASSED

🌊 Testing STREAMING mode for all models...
✅ GPT-4o stream test PASSED
✅ GPT-4.1 stream test PASSED
✅ GPT-5 stream test PASSED

📊 TEST RESULTS SUMMARY
============================================================
Overall: 6/6 tests passed (100.0%)
🎉 All tests passed!
```

## Environment Variables Required

### Azure OpenAI
```bash
AZURE_OPENAI_ENDPOINT=your-endpoint
AZURE_OPENAI_API_KEY=your-api-key
AZURE_OPENAI_API_VERSION=2025-03-01-preview
AZURE_OPENAI_DEPLOYMENT=any-deployment-name
```

### OpenAI
```bash
OPENAI_API_KEY=your-api-key
OPENAI_MODEL_NAME=any-model-name  # Will be overridden during testing
```

## Benefits Over Individual Scripts

1. **Faster Execution**: Single script startup vs. multiple script launches
2. **Better Reporting**: Consolidated results and success rates
3. **Easier Debugging**: All failures in one place
4. **CI/CD Friendly**: Single exit code for overall success/failure
5. **Consistent Testing**: Same prompt and validation across all models
6. **API Format Handling**: Automatically handles different response formats

## Migration from Individual Scripts

**Note**: Individual scripts have been removed and replaced with consolidated tests.

The following individual scripts were removed:
- `azure_openai_invoke_gpt41.py`
- `azure_openai_invoke_gpt4o.py`
- `azure_openai_invoke_gpt5.py`
- `azure_openai_stream_gpt5.py`
- `openai_invoke_gpt5.py`
- `openai_stream_gpt5.py`

## Other Provider Scripts

The following provider scripts remain for individual testing:
- `anthropic_stream.py` - Anthropic Claude streaming
- `bedrock_stream.py` - AWS Bedrock streaming
- `gemini_stream.py` - Google Gemini streaming
- `vertex_stream.py` - Google Vertex AI streaming

## Response Format Handling

### Azure OpenAI
- Handles standard Azure OpenAI responses
- Extracts content from `result.content` or `result.text`
- Manages deployment-specific configurations

### OpenAI
- Supports both standard OpenAI API and responses API
- Automatically detects and handles different chunk formats
- Handles structured responses with type and text fields

## Future Enhancements

- Add performance benchmarking
- Include response quality validation
- Support for custom prompts via command line
- Integration with other LLM providers
- Export results to various formats (JSON, CSV, etc.)
- Support for function calling and tools
- Multimodal input testing
- Consolidated tests for other providers (Anthropic, Google, AWS)

## Workflow Integration

The consolidated tests are integrated into the GitHub Actions workflow with:
- **Automatic execution** on all pushes and PRs
- **Manual triggering** via workflow dispatch
- **Selective testing** based on test type selection
- **Error handling** that continues execution even if individual tests fail
- **Comprehensive reporting** of all test results
