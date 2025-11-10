from .llm_factory import LLMFactory

# Import tracing utilities (always available since langfuse is now a standard dependency)
from .tracing import TracingManager, trace_agent_stream, disable_a2a_tracing, is_a2a_disabled

# Import utility functions
from .utils import Spinner, stream_with_spinner, invoke_with_spinner, time_llm_operation

# Import agent utilities (conditionally available based on installed dependencies)
try:
    from .agents import (
        # Context configuration (always available)
        get_context_limit_for_provider, # noqa: F401
        get_min_messages_to_keep, # noqa: F401
        is_auto_compression_enabled, # noqa: F401
        get_context_config, # noqa: F401
        log_context_config, # noqa: F401
    )
    _AGENTS_BASE_AVAILABLE = True
except ImportError:
    _AGENTS_BASE_AVAILABLE = False

# Try to import LangGraph agent classes
try:
    from .agents import BaseLangGraphAgent, BaseLangGraphAgentExecutor # noqa: F401
    _LANGGRAPH_AGENTS_AVAILABLE = True
except ImportError:
    _LANGGRAPH_AGENTS_AVAILABLE = False

# Try to import Strands agent classes
try:
    from .agents import BaseStrandsAgent, BaseStrandsAgentExecutor # noqa: F401
    _STRANDS_AGENTS_AVAILABLE = True
except ImportError:
    _STRANDS_AGENTS_AVAILABLE = False

__all__ = [
    # Core utilities
    'LLMFactory',
    'TracingManager',
    'trace_agent_stream',
    'disable_a2a_tracing',
    'is_a2a_disabled',
    'Spinner',
    'stream_with_spinner',
    'invoke_with_spinner',
    'time_llm_operation'
]

# Add agent context configuration if available
if _AGENTS_BASE_AVAILABLE:
    __all__.extend([
        'get_context_limit_for_provider',
        'get_min_messages_to_keep',
        'is_auto_compression_enabled',
        'get_context_config',
        'log_context_config',
    ])

# Add LangGraph classes if available
if _LANGGRAPH_AGENTS_AVAILABLE:
    __all__.extend([
        'BaseLangGraphAgent',
        'BaseLangGraphAgentExecutor',
    ])

# Add Strands classes if available
if _STRANDS_AGENTS_AVAILABLE:
    __all__.extend([
        'BaseStrandsAgent',
        'BaseStrandsAgentExecutor',
    ])