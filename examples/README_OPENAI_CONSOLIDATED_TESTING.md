# Consolidated OpenAI Testing

This document describes the new consolidated testing approach for OpenAI examples.

## Overview

Instead of running individual scripts for each model and mode, we now have a single consolidated test script that covers all OpenAI models with both invoke and streaming modes.

## Files

### `openai_consolidated_test.py`
- **Purpose**: Comprehensive testing of all OpenAI models
- **Models Tested**: gpt-4o, gpt-4o-mini, gpt-5
- **Modes Tested**: Invoke and Streaming
- **Total Tests**: 6 (3 models × 2 modes)

## Features

### ✅ Comprehensive Coverage
- Tests all three OpenAI models in a single run
- Covers both invoke and streaming modes
- Consistent prompt and response handling across all tests
- Handles both standard OpenAI and responses API formats

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
python examples/openai_consolidated_test.py

# Or from examples directory
cd examples
python openai_consolidated_test.py
```

### CI/CD Integration
The script is integrated into the GitHub Actions workflow:
- **Automatic**: Runs on all pushes and PRs
- **Manual**: Can be triggered via workflow dispatch
- **Selective**: Choose between consolidated or individual testing

## Test Results Example

```
🚀 Starting Consolidated OpenAI Tests
============================================================

📝 Testing INVOKE mode for all models...
✅ GPT-4o invoke test PASSED
✅ GPT-4o Mini invoke test PASSED
✅ GPT-5 invoke test PASSED

🌊 Testing STREAMING mode for all models...
✅ GPT-4o stream test PASSED
✅ GPT-4o Mini stream test PASSED
✅ GPT-5 stream test PASSED

📊 TEST RESULTS SUMMARY
============================================================
Overall: 6/6 tests passed (100.0%)
🎉 All tests passed!
```

## Environment Variables Required

```bash
OPENAI_API_KEY=your-api-key
OPENAI_MODEL_NAME=any-model-name  # Will be overridden during testing
```

## Model Details

### GPT-4o
- **Type**: Multimodal model
- **Best for**: General purpose tasks, image understanding
- **Performance**: Fast and efficient

### GPT-4o Mini
- **Type**: Lightweight version of GPT-4o
- **Best for**: Cost-effective general tasks
- **Performance**: Faster than GPT-4o, good quality

### GPT-5
- **Type**: Latest generation model
- **Best for**: Advanced reasoning, complex tasks
- **Performance**: Highest quality, uses responses API

## Benefits Over Individual Scripts

1. **Faster Execution**: Single script startup vs. multiple script launches
2. **Better Reporting**: Consolidated results and success rates
3. **Easier Debugging**: All failures in one place
4. **CI/CD Friendly**: Single exit code for overall success/failure
5. **Consistent Testing**: Same prompt and validation across all models
6. **API Format Handling**: Automatically handles different response formats

## Migration from Individual Scripts

The individual scripts are still available for backward compatibility:
- `openai_invoke_gpt5.py`
- `openai_stream_gpt5.py`

However, for comprehensive testing, use the consolidated script.

## Response Format Handling

The script automatically handles different OpenAI response formats:

### Standard OpenAI API
- Extracts content from `result.content` or `result.text`
- Handles callable attributes gracefully

### Responses API (GPT-5)
- Handles structured responses with type and text fields
- Extracts text content from response objects

### Streaming Responses
- Supports both standard streaming and responses API streaming
- Automatically detects and handles different chunk formats

## Future Enhancements

- Add performance benchmarking
- Include response quality validation
- Support for custom prompts via command line
- Integration with other LLM providers
- Export results to various formats (JSON, CSV, etc.)
- Support for function calling and tools
- Multimodal input testing (for GPT-4o)
