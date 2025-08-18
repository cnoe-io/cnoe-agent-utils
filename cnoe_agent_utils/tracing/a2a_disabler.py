# Copyright 2025 CNOE
# SPDX-License-Identifier: Apache-2.0

"""
Unified A2A Tracing Disabler

This module centralizes the A2A framework tracing disabling logic that was
duplicating 70+ lines of identical code across all CNOE agent containers.

Before:
    # DUPLICATED in every agent __main__.py (70+ lines each):
    import sys
    import types
    # ... monkey patching code ...
    from agent_x.tracing import disable_a2a_tracing
    disable_a2a_tracing()

After:
    from cnoe_agent_utils.tracing import disable_a2a_tracing
    disable_a2a_tracing()  # Or automatically disabled on import

The A2A framework's built-in telemetry can interfere with custom Langfuse
tracing, so this module monkey patches the telemetry module with no-op
implementations before any A2A imports occur.
"""

import sys
import types
import logging
from typing import Any, Callable, TypeVar, Union

logger = logging.getLogger(__name__)

# Track if A2A has already been disabled to prevent multiple patches
_a2a_disabled = False

F = TypeVar('F', bound=Callable[..., Any])
C = TypeVar('C', bound=type)

def disable_a2a_tracing() -> bool:
    """
    Disable A2A framework tracing by monkey patching the telemetry module.
    
    This function replaces A2A's trace decorators with no-op implementations
    to prevent interference with custom Langfuse tracing in CNOE agents.
    
    This MUST be called before any A2A framework imports to be effective.
    If A2A modules are already imported, this will have no effect.
    
    Returns:
        bool: True if successful or already disabled, False if failed
        
    Note:
        This function is idempotent - safe to call multiple times.
    """
    global _a2a_disabled
    
    if _a2a_disabled:
        logger.debug("✅ CNOE Agent Tracing: A2A already disabled")
        return True
    
    try:
        # Create no-op decorators to replace a2a's trace decorators
        def noop_trace_function(
            func: Union[F, None] = None,
            **_kwargs: Any
        ) -> Union[F, Callable[[F], F]]:
            """No-op replacement for trace_function decorator."""
            if func is None:
                return lambda f: f  # Return decorator that does nothing
            return func  # Return function unchanged
        
        def noop_trace_class(
            cls: Union[C, None] = None,
            **_kwargs: Any
        ) -> Union[C, Callable[[C], C]]:
            """No-op replacement for trace_class decorator."""
            if cls is None:
                return lambda c: c  # Return decorator that does nothing
            return cls  # Return class unchanged
        
        # Create a dummy SpanKind class with required OpenTelemetry attributes
        class DummySpanKind:
            """Dummy SpanKind class to replace OpenTelemetry SpanKind."""
            INTERNAL = 'INTERNAL'
            SERVER = 'SERVER'
            CLIENT = 'CLIENT'
            PRODUCER = 'PRODUCER'
            CONSUMER = 'CONSUMER'
        
        # Monkey patch the a2a telemetry module before it's imported anywhere
        # This intercepts any 'from a2a.utils.telemetry import ...' statements
        telemetry_module = types.ModuleType('a2a.utils.telemetry')
        telemetry_module.trace_function = noop_trace_function
        telemetry_module.trace_class = noop_trace_class
        telemetry_module.SpanKind = DummySpanKind
        
        # Insert into sys.modules to intercept future imports
        sys.modules['a2a.utils.telemetry'] = telemetry_module
        
        _a2a_disabled = True
        logger.debug("✅ CNOE Agent Tracing: A2A framework tracing disabled via monkey patching")
        return True
        
    except Exception as e:
        logger.error(f"❌ CNOE Agent Tracing: A2A monkey patching failed: {e}")
        return False


def is_a2a_disabled() -> bool:
    """
    Check if A2A tracing has been disabled.
    
    Returns:
        bool: True if A2A tracing is disabled, False otherwise
    """
    return _a2a_disabled