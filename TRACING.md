# CNOE Agent Tracing

**NEW in v0.2.0**: Centralized tracing utilities that eliminate 350+ lines of code duplication across CNOE agents.

## Problem Solved

Before v0.2.0, every CNOE agent had identical duplicated code:

- **Conditional imports** - Each agent repeated `if ENABLE_TRACING: import langfuse`
- **A2A disabling** - 70+ lines of identical monkey patching in every agent
- **Stream tracing logic** - 100+ lines of span management in every stream method
- **Environment checking** - Repeated environment variable handling

## Solution

This package now provides centralized tracing utilities alongside the existing `LLMFactory`.

## Installation

```bash
pip install "cnoe-agent-utils>=0.2.0"
```

**Note**: Langfuse is now included as a standard dependency. Tracing is controlled purely by the `ENABLE_TRACING` environment variable.

## Usage

### 1. Replace Conditional Import Duplication

**Before (duplicated in every agent):**
```python
# 23-29 lines repeated everywhere
if os.getenv("ENABLE_TRACING", "false").lower() == "true":
    from langfuse import get_client
    from langfuse.langchain import CallbackHandler
    langfuse_handler = CallbackHandler()
else:
    langfuse_handler = None
```

**After (single import):**
```python
from cnoe_agent_utils import LLMFactory
from cnoe_agent_utils.tracing import TracingManager

class SlackAgent:
    def __init__(self):
        self.llm = LLMFactory().get_llm()    # Existing pattern
        self.tracing = TracingManager()      # New tracing utility
```

### 2. Replace A2A Disabling Duplication

**Before (70+ lines in every `__main__.py`):**
```python
from agent_slack.tracing import disable_a2a_tracing
disable_a2a_tracing()
```

**After (single import, auto-disabled):**
```python
from cnoe_agent_utils.tracing import disable_a2a_tracing
disable_a2a_tracing()  # Or import automatically disables A2A
```

### 3. Replace Stream Tracing Duplication

**Before (100+ lines duplicated per agent):**
```python
async def stream(self, query, context_id, trace_id=None):
    # 50+ lines of identical tracing setup
    if langfuse_handler:
        with langfuse.start_as_current_span(...) as span:
            # stream logic
            span.update_trace(...)
    else:
        # stream logic (duplicated)
```

**After (single decorator):**
```python
from cnoe_agent_utils.tracing import trace_agent_stream

@trace_agent_stream("slack")  # or "argocd", "jira", etc.
async def stream(self, query, context_id, trace_id=None):
    # Agent keeps ORIGINAL logic - just remove tracing setup:

    inputs = {'messages': [HumanMessage(content=query)]}
    config = self.tracing.create_config(context_id)  # Handles callbacks automatically

    # Original graph.astream() logic - NO CHANGES:
    async for item in self.graph.astream(inputs, config, stream_mode='values'):
        message = item.get('messages', [])[-1] if item.get('messages') else None
        if isinstance(message, AIMessage) and message.tool_calls:
            yield {'is_task_complete': False, 'content': 'Processing...'}
        # ... rest of original logic

    yield self.get_agent_response(config)
```

## Complete Agent Example

```python
# agent_slack/__main__.py
from cnoe_agent_utils.tracing import disable_a2a_tracing
disable_a2a_tracing()

# agent_slack/agent.py
from cnoe_agent_utils import LLMFactory
from cnoe_agent_utils.tracing import TracingManager, trace_agent_stream

class SlackAgent:
    def __init__(self):
        self.llm = LLMFactory().get_llm()
        self.tracing = TracingManager()

    @trace_agent_stream("slack")
    async def stream(self, query: str, context_id: str, trace_id: str = None):
        """Stream method with automatic tracing - zero duplication."""

        inputs = {'messages': [HumanMessage(content=query)]}
        config = self.tracing.create_config(context_id)

        # Your original agent logic - NO CHANGES needed
        async for item in self.graph.astream(inputs, config, stream_mode='values'):
            # ... your existing message processing logic
            yield event

        yield self.get_agent_response(config)
```

## Migration from v0.1.x

### Update Dependencies

```toml
# pyproject.toml - SIMPLE VERSION UPDATE:
dependencies = [
    "cnoe-agent-utils>=0.2.0",  # Was: cnoe-agent-utils>=0.1.4
    # Remove: langfuse>=3.0.0  (now included in cnoe-agent-utils)
]
```

### What to Change per Agent

1. ➕ **Add decorator**: `@trace_agent_stream("agent_name")`
2. ➖ **Delete conditional import block** (23-29 lines)
3. 🔄 **Replace config setup**: `config = self.tracing.create_config(context_id)`
4. ➖ **Delete all tracing setup/span creation code** (50+ lines)

### What Stays the Same

- ✅ Original `graph.astream()` loops
- ✅ Original message processing logic
- ✅ Original `get_agent_response()` calls
- ✅ Original error handling

## Environment Variables

- `ENABLE_TRACING=true` - Enable tracing (requires langfuse)
- `ENABLE_TRACING=false` - Disable tracing (default)

## Features

- **🔄 Graceful Degradation** - Works with or without langfuse
- **🐳 Container Ready** - Designed for multi-container deployments
- **⚡ Zero Overhead** - No performance impact when tracing disabled
- **🧪 Backward Compatible** - Existing agents work unchanged
- **📝 Type Safe** - Full mypy type checking

## Architecture

The tracing module eliminates duplication by providing:

1. **TracingManager** - Singleton handling conditional imports and configuration
2. **@trace_agent_stream** - Decorator wrapping existing stream methods
3. **disable_a2a_tracing()** - Unified A2A framework disabling
4. **Context Management** - Automatic trace ID propagation

## Benefits

- **Eliminates 350+ lines of duplicated code** across agents
- **Single source of truth** for all tracing logic
- **Automatic environment-based toggling**
- **Unified A2A disabling**
- **Consistent tracing behavior** across all containers
- **Easy maintenance** - fix once, works everywhere

## Compatibility

- **Python**: >=3.13
- **Langfuse**: >=3.0.0 (optional)
- **Existing agents**: Backward compatible
- **New agents**: Can use tracing from day one