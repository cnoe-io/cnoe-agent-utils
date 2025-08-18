# Copyright 2025 CNOE
# SPDX-License-Identifier: Apache-2.0

"""
Tracing Decorators - Eliminates Stream Method Duplication

This module provides decorators that eliminate the 100+ lines of repetitive
tracing logic that was duplicated across every CNOE agent stream method.

Before:
    # DUPLICATED in every agent stream method (100+ lines each):
    async def stream(self, query, context_id, trace_id=None):
        # 50+ lines of tracing setup
        if langfuse_handler:
            with langfuse.start_as_current_span(...) as span:
                # agent logic
                span.update_trace(...)
        else:
            # agent logic (duplicated)

After:
    @trace_agent_stream("slack")  # Single decorator
    async def stream(self, query, context_id, trace_id=None):
        # Just agent logic - tracing handled automatically
        config = self.tracing.create_config(context_id)
        async for item in self.graph.astream(inputs, config, stream_mode='values'):
            # original logic
"""

import logging
from functools import wraps
from typing import Any, AsyncIterable, Callable, Optional, TypeVar, cast

from .manager import TracingManager

logger = logging.getLogger(__name__)

# Type variable for async generator functions
AsyncStreamFunc = TypeVar('AsyncStreamFunc', bound=Callable[..., AsyncIterable[dict[str, Any]]])

def trace_agent_stream(
    agent_name: str,
    trace_name: Optional[str] = "ai-platform-engineer",
    update_input: bool = False
) -> Callable[[AsyncStreamFunc], AsyncStreamFunc]:
    """
    Decorator that replaces ALL the duplicated tracing code in agent stream methods.
    
    This decorator eliminates:
    - 23-29 lines of conditional langfuse imports (handled by TracingManager)
    - 15+ lines of trace ID logging and validation
    - 30+ lines of span creation and management
    - 20+ lines of duplicated non-tracing execution path
    - Environment checking and configuration setup
    
    MINIMAL CHANGES: Agent keeps their existing stream logic, just add the decorator.
    
    Args:
        agent_name: Name of the agent (e.g., "argocd", "jira", "slack")
        trace_name: Custom name for the trace (defaults to "ai-platform-engineer")
        update_input: Whether to re-include the original input when updating trace with output
                     (defaults to False). Set to True to preserve input throughout trace lifecycle.
        
    Returns:
        Decorated function with automatic tracing
        
    Usage:
        # Basic usage (uses default trace name "ai-platform-engineer")
        @trace_agent_stream("slack")
        async def stream(self, query: str, context_id: str, trace_id: str = None):
        
        # With custom trace name
        @trace_agent_stream("slack", trace_name="Custom Workflow")
        async def stream(self, query: str, context_id: str, trace_id: str = None):
        
        # Preserve input throughout trace lifecycle
        @trace_agent_stream("supervisor", update_input=True)
        async def stream(self, query: str, context_id: str, trace_id: str = None):
            # Agent keeps ORIGINAL logic - just remove duplicated tracing setup:
            
            inputs = {'messages': [HumanMessage(content=query)]}
            config = self.tracing.create_config(context_id)  # Replaces conditional callback setup
            
            # Original graph.astream() calls - NO CHANGES:
            async for item in self.graph.astream(inputs, config, stream_mode='values'):
                message = item.get('messages', [])[-1] if item.get('messages') else None
                if isinstance(message, AIMessage) and message.tool_calls:
                    yield {'is_task_complete': False, 'content': 'Processing...'}
                # ... rest of original logic
            
            yield self.get_agent_response(config)
    """
    def decorator(stream_func: AsyncStreamFunc) -> AsyncStreamFunc:
        @wraps(stream_func)
        async def wrapper(
            self: Any,
            query: str,
            context_id: str,
            trace_id: Optional[str] = None
        ) -> AsyncIterable[dict[str, Any]]:
            
            # Replace ALL the duplicated setup code with unified manager
            tracing = TracingManager()
            
            # Add tracing manager to self so agent can use it
            self.tracing = tracing
            
            # Unified logging (replaces duplicated logging in every agent)
            logger.info(f"🔍 {agent_name.title()} Agent stream started - query: {query}, context_id: {context_id}")
            logger.info(f"🔍 Tracing enabled: {tracing.is_enabled}")
            
            # Unified trace ID handling (replaces duplicated validation everywhere)
            if not trace_id:
                logger.warning(f"🔍 {agent_name.title()} Agent - NO trace_id provided from supervisor for context_id: {context_id}")
                logger.warning("🔍 This indicates a problem with trace ID propagation")
            else:
                logger.info(f"🔍 {agent_name.title()} Agent - using SUPERVISOR trace_id: {trace_id} for context_id: {context_id}")
            
            # Set trace_id in context variable for tools to access
            tracing.set_trace_id(trace_id)
            
            if tracing.is_enabled:
                # Unified span management (replaces duplicated span logic)
                span_name = f"🤖-{agent_name}-agent"
                
                with tracing.start_span(
                    name=span_name,
                    agent_type=agent_name,
                    query=query,
                    context_id=context_id,
                    trace_id=trace_id,
                    trace_name=trace_name,
                    update_input=update_input
                ) as span:
                    # Agent executes with original logic - capture final response
                    final_response_content = None
                    async for event in stream_func(self, query, context_id, trace_id):
                        # Capture any content response
                        if event.get('content'):
                            final_response_content = event.get('content')
                        yield event
                    
                    # Update span with final output
                    if final_response_content:
                        span.update_trace(output=final_response_content)
                    else:
                        span.update_trace(output=f"{agent_name.title()} agent execution completed")
            else:
                # Non-tracing path - just run original agent logic
                async for event in stream_func(self, query, context_id, trace_id):
                    yield event
                    
        return cast(AsyncStreamFunc, wrapper)
    return decorator