# Copyright 2025 CNOE
# SPDX-License-Identifier: Apache-2.0

"""Agent base classes and utilities for CNOE agents."""

# Context configuration utilities
from .context_config import (
    get_context_limit_for_provider,
    get_min_messages_to_keep,
    is_auto_compression_enabled,
    get_context_config,
    log_context_config,
)

# Base agent classes (imported conditionally based on available dependencies)
try:
    from .base_langgraph_agent import BaseLangGraphAgent # noqa: F401
    from .base_langgraph_agent_executor import BaseLangGraphAgentExecutor # noqa: F401
    _LANGGRAPH_AVAILABLE = True
except ImportError:
    _LANGGRAPH_AVAILABLE = False

try:
    from .base_strands_agent import BaseStrandsAgent # noqa: F401
    from .base_strands_agent_executor import BaseStrandsAgentExecutor # noqa: F401
    _STRANDS_AVAILABLE = True
except ImportError:
    _STRANDS_AVAILABLE = False

# Export what's available
__all__ = [
    # Context config (always available)
    "get_context_limit_for_provider",
    "get_min_messages_to_keep",
    "is_auto_compression_enabled",
    "get_context_config",
    "log_context_config",
]

# Add LangGraph classes if available
if _LANGGRAPH_AVAILABLE:
    __all__.extend([
        "BaseLangGraphAgent",
        "BaseLangGraphAgentExecutor",
    ])

# Add Strands classes if available
if _STRANDS_AVAILABLE:
    __all__.extend([
        "BaseStrandsAgent",
        "BaseStrandsAgentExecutor",
    ])
