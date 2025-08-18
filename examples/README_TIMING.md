# LLM Timing Features

This document describes the timing features available in the `cnoe_agent_utils` library.

## Environment Variables

### `LLM_SHOW_TIMING`
- **Default**: `true`
- **Values**: `true` or `false`
- **Purpose**: Controls whether timing information is displayed for LLM operations

## Available Timing Functions

### 1. `stream_with_spinner()`
Automatically times streaming operations and shows:
- Start time (🕐)
- End time (🕐)
- Total duration (⏱️)

### 2. `invoke_with_spinner()`
Automatically times invoke operations and shows:
- Start time (🕐)
- End time (🕐)
- Total duration (⏱️)

### 3. `time_llm_operation()`
Context manager for manual timing of custom operations:

```python
from cnoe_agent_utils import time_llm_operation

with time_llm_operation("Custom LLM call"):
    result = llm.invoke(prompt)
    # ... other operations ...
```

## Usage Examples

### Enable Timing (Default)
```bash
export LLM_SHOW_TIMING=true
# or
LLM_SHOW_TIMING=true python examples/azure_openai_stream_gpt5.py
```

### Disable Timing
```bash
export LLM_SHOW_TIMING=false
# or
LLM_SHOW_TIMING=false python examples/azure_openai_stream_gpt5.py
```

### In .env File
```bash
# Add to your .env file
LLM_SHOW_TIMING=true
```

## Output Format

When timing is enabled, you'll see output like:
```
🕐 Started at: 16:45:30
=== Azure OpenAI (stream) ===
The Moon's surface is a gray, dusty landscape...
⏱️  Total time: 5.23 seconds
🕐 Finished at: 16:45:35
=== done ===
```

## Benefits

1. **Performance Monitoring**: Track how long LLM operations take
2. **Debugging**: Identify slow operations or API issues
3. **User Experience**: Show progress and completion times
4. **Configurable**: Can be disabled in production if needed
5. **Non-intrusive**: Timing is displayed without affecting the main output
