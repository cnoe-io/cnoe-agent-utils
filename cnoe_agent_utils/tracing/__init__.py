# Copyright 2025 CNOE
# SPDX-License-Identifier: Apache-2.0

"""
CNOE Agent Tracing Module

Centralized tracing utilities that eliminate code duplication across CNOE agents.
Part of cnoe-agent-utils package alongside LLMFactory.

Key features:
- Eliminates conditional import duplication (langfuse)
- Unified A2A framework disabling
- Stream tracing decorator to remove repetitive code
- Graceful degradation when tracing dependencies unavailable
- Multi-container deployment ready

Usage:
    from cnoe_agent_utils.tracing import TracingManager, trace_agent_stream, disable_a2a_tracing
    
    # Disable A2A framework interference
    disable_a2a_tracing()
    
    class MyAgent:
        def __init__(self):
            self.tracing = TracingManager()  # Handles conditional imports
        
        @trace_agent_stream("myagent")  # Default trace name: "ai-platform-engineer"
        async def stream(self, query, context_id, trace_id=None):
            # Just agent logic - tracing handled automatically
            pass
            
        @trace_agent_stream("myagent", trace_name="Custom Workflow")  # Custom trace name
        async def stream_custom(self, query, context_id, trace_id=None):
            # Agent logic with custom trace name
            pass
"""

from .manager import TracingManager
from .a2a_disabler import disable_a2a_tracing, is_a2a_disabled
from .decorators import trace_agent_stream
from .a2a_utils import extract_trace_id_from_context

# Auto-disable A2A on import to prevent any interference
disable_a2a_tracing()

__all__ = [
    "TracingManager",
    "disable_a2a_tracing",
    "is_a2a_disabled",
    "trace_agent_stream",
    "extract_trace_id_from_context",
]
