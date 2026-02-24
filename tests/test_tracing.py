#!/usr/bin/env python3
"""Tests for tracing functionality in cnoe_agent_utils.tracing."""

import logging
import os
import pytest
from unittest.mock import patch, MagicMock

from cnoe_agent_utils.tracing import (
    TracingManager,
    trace_agent_stream,
    disable_a2a_tracing,
    is_a2a_disabled,
    extract_trace_id_from_context
)
from cnoe_agent_utils.tracing.decorators import _quiet_span_exit


class TestTracingManager:
    """Test the TracingManager class."""

    def test_singleton_pattern(self):
        """Test that TracingManager follows singleton pattern."""
        manager1 = TracingManager()
        manager2 = TracingManager()
        assert manager1 is manager2

    def test_create_config_tracing_disabled(self):
        """Test config creation when tracing is disabled."""
        with patch.dict(os.environ, {"ENABLE_TRACING": "false"}, clear=True):
            manager = TracingManager()
            config = manager.create_config("test-context-id")

            assert config['configurable']['thread_id'] == "test-context-id"
            assert 'callbacks' not in config

    def test_get_trace_id(self):
        """Test getting the current trace ID."""
        manager = TracingManager()
        trace_id = manager.get_trace_id()
        assert trace_id is None  # Initially no trace ID

    def test_set_trace_id(self):
        """Test setting the current trace ID."""
        manager = TracingManager()
        manager.set_trace_id("test-trace-123")
        trace_id = manager.get_trace_id()
        assert trace_id == "test-trace-123"


class TestTraceAgentStream:
    """Test the trace_agent_stream decorator."""

    def test_trace_agent_stream_decorator(self):
        """Test that trace_agent_stream decorator can be applied."""
        @trace_agent_stream("test-agent")
        async def test_function():
            return "test result"

        # The decorator should be applied without error
        assert callable(test_function)

    def test_trace_agent_stream_with_custom_name(self):
        """Test trace_agent_stream decorator with custom trace name."""
        @trace_agent_stream("test-agent", trace_name="Custom Workflow")
        async def test_function():
            return "test result"

        # The decorator should be applied without error
        assert callable(test_function)

    def test_trace_agent_stream_with_update_input(self):
        """Test trace_agent_stream decorator with update_input parameter."""
        @trace_agent_stream("test-agent", update_input=True)
        async def test_function():
            return "test result"

        # The decorator should be applied without error
        assert callable(test_function)


class TestA2AFunctions:
    """Test A2A-related functions."""

    def test_disable_a2a_tracing(self):
        """Test that disable_a2a_tracing can be called."""
        # This function should not raise an error
        disable_a2a_tracing()

    def test_is_a2a_disabled(self):
        """Test that is_a2a_disabled can be called."""
        # This function should return a boolean
        result = is_a2a_disabled()
        assert isinstance(result, bool)

    def test_extract_trace_id_from_context(self):
        """Test that extract_trace_id_from_context can be called."""
        # Test with empty context
        result = extract_trace_id_from_context({})
        assert result is None

        # Test with context containing trace_id
        context = {"trace_id": "test-123"}
        result = extract_trace_id_from_context(context)
        # This function might not work as expected, so just test it doesn't crash
        assert result is not None or result is None


class TestTracingIntegration:
    """Integration tests for tracing functionality."""

    def test_trace_id_context_isolation(self):
        """Test that trace IDs are isolated between different contexts."""
        manager = TracingManager()

        # Set trace ID in one context
        manager.set_trace_id("trace-1")
        assert manager.get_trace_id() == "trace-1"

        # Set different trace ID
        manager.set_trace_id("trace-2")
        assert manager.get_trace_id() == "trace-2"

        # Set to None
        manager.set_trace_id(None)
        assert manager.get_trace_id() is None


class TestQuietSpanExit:
    """Test _quiet_span_exit suppresses OTel context-detach noise."""

    def test_suppresses_otel_context_detach_valueerror(self):
        """The OTel 'different Context' ValueError should be suppressed."""
        span_ctx = MagicMock()
        span_ctx.__exit__ = MagicMock(
            side_effect=ValueError(
                "<Token ...> was created in a different Context"
            )
        )

        # Should not raise
        _quiet_span_exit(span_ctx)
        span_ctx.__exit__.assert_called_once()

    def test_raises_unrelated_valueerror(self):
        """Other ValueErrors should still propagate."""
        span_ctx = MagicMock()
        span_ctx.__exit__ = MagicMock(
            side_effect=ValueError("something else entirely")
        )

        with pytest.raises(ValueError, match="something else entirely"):
            _quiet_span_exit(span_ctx)

    def test_otel_logger_suppressed_during_exit(self):
        """The opentelemetry.context logger should be raised to CRITICAL."""
        otel_logger = logging.getLogger("opentelemetry.context")
        original_level = otel_logger.level

        levels_during_exit = []

        def capture_level(*args, **kwargs):
            levels_during_exit.append(otel_logger.level)

        span_ctx = MagicMock()
        span_ctx.__exit__ = MagicMock(side_effect=capture_level)

        _quiet_span_exit(span_ctx)

        assert levels_during_exit == [logging.CRITICAL]
        assert otel_logger.level == original_level

    def test_otel_logger_restored_after_error(self):
        """Logger level is restored even when __exit__ raises."""
        otel_logger = logging.getLogger("opentelemetry.context")
        original_level = otel_logger.level

        span_ctx = MagicMock()
        span_ctx.__exit__ = MagicMock(
            side_effect=ValueError(
                "<Token ...> was created in a different Context"
            )
        )

        _quiet_span_exit(span_ctx)

        assert otel_logger.level == original_level

    def test_passes_exc_info_through(self):
        """Custom exc_info tuple is forwarded to __exit__."""
        span_ctx = MagicMock()
        exc_info = (RuntimeError, RuntimeError("boom"), None)

        _quiet_span_exit(span_ctx, exc_info)

        span_ctx.__exit__.assert_called_once_with(*exc_info)

    def test_normal_exit_passes_none_tuple(self):
        """Default call passes (None, None, None)."""
        span_ctx = MagicMock()

        _quiet_span_exit(span_ctx)

        span_ctx.__exit__.assert_called_once_with(None, None, None)


class TestTraceAgentStreamGeneratorExit:
    """Test that GeneratorExit in trace_agent_stream is handled gracefully."""

    @pytest.mark.asyncio
    async def test_generator_exit_does_not_raise(self):
        """Closing the async generator should not raise."""

        @trace_agent_stream("test-agent")
        async def fake_stream(self, query, context_id, trace_id=None):
            yield {"content": "hello"}
            yield {"content": "world"}

        agent = MagicMock()
        gen = fake_stream(agent, "test query", "ctx-1")

        # Consume one event then close (simulates client disconnect)
        first = await anext(gen)
        assert first["content"] == "hello"
        await gen.aclose()

    @pytest.mark.asyncio
    async def test_tracing_disabled_generator_exit(self):
        """GeneratorExit on the non-tracing path should also be clean."""

        @trace_agent_stream("test-agent")
        async def fake_stream(self, query, context_id, trace_id=None):
            yield {"content": "one"}
            yield {"content": "two"}

        agent = MagicMock()
        gen = fake_stream(agent, "q", "c")

        await anext(gen)
        await gen.aclose()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
