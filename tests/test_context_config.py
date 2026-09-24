#!/usr/bin/env python3
"""Tests for model-profile-derived context limit configuration."""

import os
from types import SimpleNamespace
from unittest.mock import patch

from cnoe_agent_utils.agents.context_config import (
    DEFAULT_CONTEXT_LIMIT_FALLBACK,
    DEFAULT_CONTEXT_LIMIT_FRACTION,
    get_context_limit_for_model,
    get_min_messages_to_keep,
    is_auto_compression_enabled,
)


class TestGetContextLimitForModel:
    def test_uses_model_profile_max_input_tokens(self):
        model = SimpleNamespace(profile={"max_input_tokens": 200_000})
        assert get_context_limit_for_model(model) == int(200_000 * DEFAULT_CONTEXT_LIMIT_FRACTION)

    def test_custom_fraction_and_fallback_are_respected(self):
        model = SimpleNamespace(profile={"max_input_tokens": 100_000})
        assert get_context_limit_for_model(model, fraction=0.5) == 50_000
        assert get_context_limit_for_model(None, fallback=42) == 42

    def test_no_model_uses_default_fallback(self):
        assert get_context_limit_for_model(None) == DEFAULT_CONTEXT_LIMIT_FALLBACK

    def test_model_with_no_profile_attribute_uses_fallback(self):
        model = SimpleNamespace()
        assert get_context_limit_for_model(model) == DEFAULT_CONTEXT_LIMIT_FALLBACK

    def test_profile_missing_max_input_tokens_uses_fallback(self):
        model = SimpleNamespace(profile={"name": "some-model"})
        assert get_context_limit_for_model(model) == DEFAULT_CONTEXT_LIMIT_FALLBACK

    def test_non_mapping_profile_uses_fallback(self):
        model = SimpleNamespace(profile="not-a-mapping")
        assert get_context_limit_for_model(model) == DEFAULT_CONTEXT_LIMIT_FALLBACK

    def test_non_int_max_input_tokens_uses_fallback(self):
        model = SimpleNamespace(profile={"max_input_tokens": "200000"})
        assert get_context_limit_for_model(model) == DEFAULT_CONTEXT_LIMIT_FALLBACK

    def test_bool_max_input_tokens_uses_fallback(self):
        # isinstance(True, int) is True in Python - must be excluded explicitly.
        model = SimpleNamespace(profile={"max_input_tokens": True})
        assert get_context_limit_for_model(model) == DEFAULT_CONTEXT_LIMIT_FALLBACK

    def test_zero_or_negative_max_input_tokens_uses_fallback(self):
        assert get_context_limit_for_model(SimpleNamespace(profile={"max_input_tokens": 0})) == DEFAULT_CONTEXT_LIMIT_FALLBACK
        assert get_context_limit_for_model(SimpleNamespace(profile={"max_input_tokens": -5})) == DEFAULT_CONTEXT_LIMIT_FALLBACK

    def test_tiny_profile_still_returns_at_least_one_token(self):
        model = SimpleNamespace(profile={"max_input_tokens": 1})
        assert get_context_limit_for_model(model, fraction=0.01) == 1


class TestGetMinMessagesToKeep:
    def test_default(self):
        with patch.dict(os.environ, {}, clear=True):
            assert get_min_messages_to_keep() == 10

    def test_env_override(self):
        with patch.dict(os.environ, {"MIN_MESSAGES_TO_KEEP": "5"}):
            assert get_min_messages_to_keep() == 5

    def test_invalid_env_value_falls_back_to_default(self):
        with patch.dict(os.environ, {"MIN_MESSAGES_TO_KEEP": "not-a-number"}):
            assert get_min_messages_to_keep() == 10


class TestIsAutoCompressionEnabled:
    def test_default_enabled(self):
        with patch.dict(os.environ, {}, clear=True):
            assert is_auto_compression_enabled() is True

    def test_explicit_disable(self):
        with patch.dict(os.environ, {"ENABLE_AUTO_COMPRESSION": "false"}):
            assert is_auto_compression_enabled() is False
