# CNOE Agent Base Classes

This module provides base classes and utilities for building CNOE agents with different frameworks.

## Features

- **Framework Support**: Base classes for LangGraph and Strands agent frameworks
- **A2A Protocol Integration**: Seamless integration with A2A (Agent-to-Agent) protocol for agent execution
- **Context Management**: Automatic context window management with token counting and message trimming
- **Streaming Support**: Built-in streaming capabilities for real-time agent responses
- **Optional Dependencies**: Graceful handling of missing dependencies with informative error messages

## Installation

### Core Agent Utilities

```bash
pip install cnoe-agent-utils[agents]
```

### LangGraph Agents

```bash
pip install cnoe-agent-utils[langgraph]
```

### Strands Agents

```bash
pip install cnoe-agent-utils[strands]
```

### A2A Protocol Support

```bash
pip install cnoe-agent-utils[a2a]
```

### All Agent Features

```bash
pip install cnoe-agent-utils[agents-all]
```

## Quick Start

### LangGraph Agent

```python
from cnoe_agent_utils.agents import BaseLangGraphAgent, BaseLangGraphAgentExecutor
from pydantic import BaseModel

class ResponseFormat(BaseModel):
    message: str

class MyAgent(BaseLangGraphAgent):
    def get_agent_name(self) -> str:
        return "MyAgent"

    def get_system_instruction(self) -> str:
        return "You are a helpful assistant."

    def get_response_format_instruction(self) -> str:
        return "Respond in the specified format."

    def get_response_format_class(self) -> type[BaseModel]:
        return ResponseFormat

    def get_mcp_config(self, server_path: str) -> dict:
        return {
            "command": "python",
            "args": [server_path],
            "env": {}
        }

    def get_tool_working_message(self) -> str:
        return "MyAgent is working with tools..."

    def get_tool_processing_message(self) -> str:
        return "MyAgent is processing results..."

# Usage
agent = MyAgent()
async for response in agent.stream("Hello!", "session-123"):
    print(response['content'])
```

### Strands Agent

```python
from cnoe_agent_utils.agents import BaseStrandsAgent, BaseStrandsAgentExecutor
from strands.tools.mcp import MCPClient
from typing import List, Tuple

class MyStrandsAgent(BaseStrandsAgent):
    def get_agent_name(self) -> str:
        return "MyStrandsAgent"

    def get_system_prompt(self) -> str:
        return "You are a helpful assistant using Strands."

    def create_mcp_clients(self) -> List[Tuple[str, MCPClient]]:
        # Return empty list for no MCP clients, or create clients as needed
        return []

    def get_model_config(self):
        # Return your model configuration
        from strands.models import BedrockModel
        return BedrockModel("claude-3-5-sonnet-20241022")

# Usage
agent = MyStrandsAgent()
response = agent.chat("Hello!")
print(response['answer'])
```

### A2A Agent Executor

```python
from cnoe_agent_utils.agents import BaseLangGraphAgentExecutor

class MyAgentExecutor(BaseLangGraphAgentExecutor):
    def __init__(self):
        super().__init__(MyAgent())

# Usage with A2A protocol
executor = MyAgentExecutor()
# executor.execute(context, event_queue) - used by A2A framework
```

## Context Configuration

The agents module derives its context-window limit from the LLM instance
itself, not a hand-maintained per-provider table. Every supported chat model
(`langchain-aws`, `langchain-anthropic`, `langchain-openai`,
`langchain-google-genai`/`vertexai`, `langchain-groq`) resolves a `.profile`
dict from its own model catalog, so the limit always matches the model
actually configured - including new model releases, with no code change here:

```python
from cnoe_agent_utils.agents import get_context_limit_for_model

# 85% of the model's advertised max_input_tokens, or a generic fallback
# if the model has no resolvable profile (e.g. Strands models, which
# aren't LangChain BaseChatModels).
max_tokens = get_context_limit_for_model(llm)
```

`BaseLangGraphAgent` calls this automatically with its constructed model.
`BaseStrandsAgent` calls it with `None`, since Strands models don't expose a
LangChain `.profile`, and gets the generic fallback (200,000 tokens).

## Environment Variables

### Context Management

- `MIN_MESSAGES_TO_KEEP`: Minimum recent messages to preserve (default: 10)
- `ENABLE_AUTO_COMPRESSION`: Enable automatic message trimming (default: true)

### Agent Behavior

- `ENABLE_STREAMING`: Enable token-by-token streaming (default: true)
- `STREAM_TOOL_OUTPUT`: Stream intermediate tool outputs (default: false)
- `MAX_TOOL_OUTPUT_LENGTH`: Maximum tool output length in stream (default: 2000)
- `TOOL_OUTPUT_CHUNK_THRESHOLD`: Large output chunking threshold (default: 50000)
- `TOOL_OUTPUT_CHUNK_SIZE`: Chunk size for large outputs (default: 10000)

### MCP Configuration

- `MCP_MODE`: Transport mode ("stdio" or "http", default: "stdio")
- `MCP_HOST`: MCP server host for HTTP mode (default: "localhost")
- `MCP_PORT`: MCP server port for HTTP mode (default: "3000")

### Debug

- `ACP_SERVER_DEBUG`: Enable debug output (default: false)

## Architecture

### Base Classes

1. **BaseLangGraphAgent**: Abstract base for LangGraph-based agents
   - Handles LLM initialization, MCP setup, and streaming
   - Automatic context management with token counting
   - Support for both stdio and HTTP MCP modes

2. **BaseLangGraphAgentExecutor**: A2A protocol executor for LangGraph agents
   - Bridges agent streaming to A2A event queue
   - Handles task state transitions (working → completed)
   - Tool call notifications and status updates

3. **BaseStrandsAgent**: Abstract base for Strands-based agents
   - Multi-server MCP support with parallel initialization
   - Conversation state management
   - Async streaming and sync chat interfaces

4. **BaseStrandsAgentExecutor**: A2A protocol executor for Strands agents
   - Converts sync Strands streaming to async A2A events
   - Error handling and resource cleanup
   - Task cancellation support

### Context Management

The context management system automatically:
- Counts tokens in conversation history
- Trims old messages when approaching limits
- Preserves system messages and recent conversation
- Derives its safety margin from the configured model's own advertised context window

### Optional Dependencies

The module gracefully handles missing dependencies:
- Core utilities (context config) always available
- LangGraph classes only available if `langgraph` installed
- Strands classes only available if `strands` installed
- A2A executors only available if `a2a-sdk` installed

## Examples

See the `examples/` directory for complete working examples:

- `examples/langgraph_agent.py` - Complete LangGraph agent implementation
- `examples/strands_agent.py` - Complete Strands agent implementation
- `examples/a2a_integration.py` - A2A protocol integration examples
- `examples/context_management.py` - Context management examples

## Contributing

When adding new agent frameworks:

1. Create base classes following the existing patterns
2. Add optional dependencies to pyproject.toml
3. Update the __init__.py imports with graceful error handling
4. Add comprehensive tests and documentation
5. Include examples demonstrating usage

## License

Apache 2.0 - see LICENSE file for details.
