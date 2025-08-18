# Copyright 2025 CNOE
# SPDX-License-Identifier: Apache-2.0

"""
TracingManager - Eliminates Conditional Import Duplication

This module provides a centralized TracingManager that handles all the
conditional langfuse imports and environment checking that was duplicated
across every CNOE agent.

Before:
    # DUPLICATED in every agent (23-29 lines each):
    if os.getenv("ENABLE_TRACING", "false").lower() == "true":
        from langfuse import get_client
        from langfuse.langchain import CallbackHandler
        langfuse_handler = CallbackHandler()
    else:
        langfuse_handler = None

After:
    from cnoe_agent_utils.tracing import TracingManager
    tracing = TracingManager()  # All conditional logic handled internally
    config = tracing.create_config(context_id)  # Includes callbacks if enabled
"""

import os
import logging
from typing import Optional, Any, Dict
from contextvars import ContextVar

logger = logging.getLogger(__name__)

class TracingManager:
    """
    Centralized manager for all tracing functionality.

    Eliminates the conditional import duplication that was repeated
    across every CNOE agent by handling:
    - Conditional langfuse imports based on environment
    - Unified callback handler management
    - Trace context management
    - Environment-based feature toggling
    - Graceful degradation when dependencies unavailable
    """

    _instance: Optional['TracingManager'] = None

    def __new__(cls) -> 'TracingManager':
        """Singleton pattern to prevent multiple langfuse initializations."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        """Initialize tracing manager (only once due to singleton)."""
        if hasattr(self, '_initialized') and self._initialized:
            return

        self._langfuse_handler: Optional[Any] = None
        self._langfuse_client: Optional[Any] = None
        self._is_enabled: bool = False
        self._current_trace_id: ContextVar[Optional[str]] = ContextVar(
            'current_trace_id', default=None
        )

        self._initialize_tracing()
        self._initialized = True

    def _initialize_tracing(self) -> None:
        """
        Handle all conditional imports and environment checking.

        This replaces the conditional import blocks that were duplicated
        across every agent container.
        """
        try:
            # Single place for environment checking (eliminates duplication)
            self._is_enabled = os.getenv("ENABLE_TRACING", "false").lower() == "true"

            if self._is_enabled:
                # Only import langfuse if tracing is enabled AND available
                try:
                    from langfuse import get_client
                    from langfuse.langchain import CallbackHandler

                    self._langfuse_client = get_client()
                    self._langfuse_handler = CallbackHandler()

                    logger.info("✅ CNOE Agent Tracing: Langfuse initialized successfully")

                except ImportError as import_err:
                    logger.error(
                        f"❌ CNOE Agent Tracing: Langfuse import failed ({import_err}). "
                        "This should not happen since langfuse is a standard dependency."
                    )
                    self._is_enabled = False

            else:
                logger.debug(
                    "🔍 CNOE Agent Tracing: Disabled via ENABLE_TRACING environment variable"
                )

        except Exception as e:
            logger.error(f"❌ CNOE Agent Tracing: Initialization failed: {e}")
            self._is_enabled = False
            self._langfuse_handler = None
            self._langfuse_client = None

    @property
    def is_enabled(self) -> bool:
        """Check if tracing is enabled and available."""
        return self._is_enabled

    @property
    def langfuse_handler(self) -> Optional[Any]:
        """Get the Langfuse callback handler if available."""
        return self._langfuse_handler

    @property
    def langfuse_client(self) -> Optional[Any]:
        """Get the Langfuse client if available."""
        return self._langfuse_client

    def create_config(self, context_id: str) -> Dict[str, Any]:
        """
        Create a LangChain runnable config with tracing if enabled.

        This replaces the repeated config creation logic that was
        duplicated across every agent.

        Args:
            context_id: The context/thread ID for this execution

        Returns:
            Configuration dict with callbacks if tracing is enabled
        """
        config = {'configurable': {'thread_id': context_id}}

        if self.is_enabled and self._langfuse_handler:
            config['callbacks'] = [self._langfuse_handler]
            logger.debug(f"🔍 CNOE Agent Tracing: Added callback for context_id: {context_id}")

        return config

    def set_trace_id(self, trace_id: Optional[str]) -> None:
        """Set the current trace ID in context for tools to access."""
        self._current_trace_id.set(trace_id)

    def get_trace_id(self) -> Optional[str]:
        """Get the current trace ID from context."""
        return self._current_trace_id.get()

    def start_span(
        self,
        name: str,
        agent_type: str,
        query: str,
        context_id: str,
        trace_id: Optional[str] = None,
        trace_name: Optional[str] = None,
        update_input: bool = True
    ) -> 'LangfuseSpanContextManager | NoOpSpanContextManager':
        """
        Start a new trace span if tracing is enabled.

        This eliminates the span creation logic that was duplicated
        across every agent stream method.

        Args:
            name: Span name (e.g., "🤖-slack-agent")
            agent_type: Type of agent (slack, jira, argocd, etc.)
            query: User query being processed
            context_id: Context/thread ID
            trace_id: Optional trace ID from supervisor
            trace_name: Optional custom name for the trace (defaults to "ai-platform-engineer")
            update_input: Whether to set input in the trace (defaults to True)

        Returns:
            Context manager for the span (no-op if tracing disabled)
        """
        if not self.is_enabled or not self._langfuse_client:
            return NoOpSpanContextManager()

        try:
            # Set trace ID in context for tools to access
            self.set_trace_id(trace_id)

            logger.info(f"🔍 CNOE Agent Tracing: Started span '{name}' for {agent_type}")

            # Return a wrapper that handles the span properly
            return LangfuseSpanContextManager(
                langfuse_client=self._langfuse_client,
                name=name,
                query=query,
                agent_type=agent_type,
                context_id=context_id,
                trace_id=trace_id,
                trace_name=trace_name,
                update_input=update_input
            )

        except Exception as e:
            logger.error(f"❌ CNOE Agent Tracing: Failed to start span: {e}")
            return NoOpSpanContextManager()


class LangfuseSpanContextManager:
    """Context manager wrapper for Langfuse span context."""

    def __init__(self, langfuse_client: Any, name: str, query: str, agent_type: str,
                 context_id: str, trace_id: Optional[str], trace_name: Optional[str] = None,
                 update_input: bool = True) -> None:
        self.langfuse_client = langfuse_client
        self.name = name
        self.query = query
        self.agent_type = agent_type
        self.context_id = context_id
        self.trace_id = trace_id
        self.trace_name = trace_name or "ai-platform-engineer"
        self.update_input = update_input
        self._context = None
        self._span = None

    def __enter__(self) -> 'LangfuseSpanContextManager':
        # Create the context manager
        trace_context = {"trace_id": self.trace_id} if self.trace_id else {}
        self._context = self.langfuse_client.start_as_current_span(
            name=self.name,
            trace_context=trace_context
        )

        # Enter the context and get the span
        self._span = self._context.__enter__()

        # Update trace with initial metadata
        try:
            update_kwargs = {
                "name": self.trace_name,
                "metadata": {
                    "agent_type": self.agent_type,
                    "context_id": self.context_id,
                    "trace_id": self.trace_id,
                    "cnoe_agent_utils_version": "0.2.0"
                }
            }
            # Only set input if update_input flag is True
            if self.update_input:
                update_kwargs["input"] = self.query

            self._span.update_trace(**update_kwargs)
        except Exception as e:
            logger.error(f"Failed to update trace metadata: {e}")

        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit the Langfuse span context."""
        if self._context:
            self._context.__exit__(exc_type, exc_val, exc_tb)

    def update_trace(self, **kwargs: Any) -> None:
        """Update trace with additional metadata."""
        try:
            if self._span:
                self._span.update_trace(**kwargs)
        except Exception as e:
            logger.error(f"Failed to update trace: {e}")


class NoOpSpanContextManager:
    """No-operation span context manager for when tracing is disabled."""

    def __enter__(self) -> 'NoOpSpanContextManager':
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        pass

    def update_trace(self, **kwargs: Any) -> None:
        """No-op update trace method."""
        pass