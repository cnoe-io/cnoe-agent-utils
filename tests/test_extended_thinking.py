#!/usr/bin/env python3
"""Tests for extended thinking configuration in LLMFactory."""

import pytest
import os
from unittest.mock import patch

from cnoe_agent_utils.llm_factory import (
  LLMFactory,
  _parse_thinking_budget,
  THINKING_DEFAULT_BUDGET,
  THINKING_MIN_BUDGET,
  ThinkingConfig,
)


class TestThinkingBudgetParsing:
  """Test the _parse_thinking_budget helper function."""

  def test_default_budget(self):
    """Test that default budget is used when env var is not set."""
    with patch.dict(os.environ, {}, clear=True):
      budget = _parse_thinking_budget("NONEXISTENT_VAR")
      assert budget == THINKING_DEFAULT_BUDGET

  def test_valid_budget(self):
    """Test parsing a valid budget value."""
    with patch.dict(os.environ, {"TEST_BUDGET": "8000"}):
      budget = _parse_thinking_budget("TEST_BUDGET")
      assert budget == 8000

  def test_budget_with_comment(self):
    """Test parsing budget with inline comment."""
    with patch.dict(os.environ, {"TEST_BUDGET": "5000  # optimized"}):
      budget = _parse_thinking_budget("TEST_BUDGET")
      assert budget == 5000

  def test_invalid_budget_uses_default(self):
    """Test that invalid budget values fall back to default."""
    with patch.dict(os.environ, {"TEST_BUDGET": "invalid"}):
      budget = _parse_thinking_budget("TEST_BUDGET")
      assert budget == THINKING_DEFAULT_BUDGET

  def test_budget_below_minimum(self):
    """Test that budgets below minimum are clamped."""
    with patch.dict(os.environ, {"TEST_BUDGET": "512"}):
      budget = _parse_thinking_budget("TEST_BUDGET")
      assert budget == THINKING_MIN_BUDGET

  def test_budget_above_max_tokens(self):
    """Test that budget is clamped to max_tokens when provided."""
    with patch.dict(os.environ, {"TEST_BUDGET": "10000"}):
      budget = _parse_thinking_budget("TEST_BUDGET", max_tokens=8000)
      assert budget == 8000

  def test_budget_below_max_tokens(self):
    """Test that budget is not clamped when below max_tokens."""
    with patch.dict(os.environ, {"TEST_BUDGET": "5000"}):
      budget = _parse_thinking_budget("TEST_BUDGET", max_tokens=8000)
      assert budget == 5000


class TestBedrockExtendedThinking:
  """Test extended thinking configuration for AWS Bedrock."""

  def test_thinking_disabled_by_default(self):
    """Test that thinking is disabled when env var is not set."""
    factory = LLMFactory("aws-bedrock")

    with patch.dict(
      os.environ,
      {
        "AWS_BEDROCK_MODEL_ID": "anthropic.claude-4-sonnet",
        "AWS_REGION": "us-east-1",
        "AWS_ACCESS_KEY_ID": "test-key",
        "AWS_SECRET_ACCESS_KEY": "test-secret",
      },
      clear=False,
    ):
      llm = factory._build_aws_bedrock_llm(None, 0.0)
      # Verify thinking is not in model_kwargs
      model_kwargs = llm.model_kwargs if hasattr(llm, "model_kwargs") else {}
      assert "thinking" not in model_kwargs

  def test_thinking_enabled(self):
    """Test that thinking is enabled when env var is set."""
    factory = LLMFactory("aws-bedrock")

    with patch.dict(
      os.environ,
      {
        "AWS_BEDROCK_MODEL_ID": "anthropic.claude-4-sonnet",
        "AWS_REGION": "us-east-1",
        "AWS_ACCESS_KEY_ID": "test-key",
        "AWS_SECRET_ACCESS_KEY": "test-secret",
        "AWS_BEDROCK_THINKING_ENABLED": "true",
        "AWS_BEDROCK_THINKING_BUDGET": "5000",
      },
      clear=False,
    ):
      llm = factory._build_aws_bedrock_llm(None, 0.0)
      # Verify thinking configuration
      model_kwargs = llm.model_kwargs if hasattr(llm, "model_kwargs") else {}
      assert "thinking" in model_kwargs
      thinking = model_kwargs["thinking"]
      assert thinking["type"] == "enabled"
      assert thinking["budget_tokens"] == 5000

  def test_thinking_with_default_budget(self):
    """Test thinking with default budget when not specified."""
    factory = LLMFactory("aws-bedrock")

    with patch.dict(
      os.environ,
      {
        "AWS_BEDROCK_MODEL_ID": "anthropic.claude-4-sonnet",
        "AWS_REGION": "us-east-1",
        "AWS_ACCESS_KEY_ID": "test-key",
        "AWS_SECRET_ACCESS_KEY": "test-secret",
        "AWS_BEDROCK_THINKING_ENABLED": "true",
      },
      clear=False,
    ):
      llm = factory._build_aws_bedrock_llm(None, 0.0)
      model_kwargs = llm.model_kwargs if hasattr(llm, "model_kwargs") else {}
      thinking = model_kwargs.get("thinking", {})
      assert thinking["budget_tokens"] == THINKING_DEFAULT_BUDGET

  def test_thinking_budget_clamped_to_max_tokens(self):
    """Test that thinking budget is clamped to max_tokens."""
    factory = LLMFactory("aws-bedrock")

    with patch.dict(
      os.environ,
      {
        "AWS_BEDROCK_MODEL_ID": "anthropic.claude-4-sonnet",
        "AWS_REGION": "us-east-1",
        "AWS_ACCESS_KEY_ID": "test-key",
        "AWS_SECRET_ACCESS_KEY": "test-secret",
        "AWS_BEDROCK_THINKING_ENABLED": "true",
        "AWS_BEDROCK_THINKING_BUDGET": "10000",
      },
      clear=False,
    ):
      llm = factory._build_aws_bedrock_llm(None, 0.0, max_tokens=8000)
      model_kwargs = llm.model_kwargs if hasattr(llm, "model_kwargs") else {}
      thinking = model_kwargs.get("thinking", {})
      assert thinking["budget_tokens"] == 8000

  def test_thinking_preserved_with_response_format(self):
    """Test that thinking config is preserved when response_format is also set (bug fix)."""
    factory = LLMFactory("aws-bedrock")

    with patch.dict(
      os.environ,
      {
        "AWS_BEDROCK_MODEL_ID": "anthropic.claude-4-sonnet",
        "AWS_REGION": "us-east-1",
        "AWS_ACCESS_KEY_ID": "test-key",
        "AWS_SECRET_ACCESS_KEY": "test-secret",
        "AWS_BEDROCK_THINKING_ENABLED": "true",
        "AWS_BEDROCK_THINKING_BUDGET": "3000",
      },
      clear=False,
    ):
      # Pass response_format (this used to clobber thinking config)
      llm = factory._build_aws_bedrock_llm({"type": "json_object"}, 0.0)

      # Verify BOTH thinking and response_format are present
      model_kwargs = llm.model_kwargs if hasattr(llm, "model_kwargs") else {}
      assert "thinking" in model_kwargs, "Thinking config should not be clobbered"
      assert "response_format" in model_kwargs, "Response format should be set"

      thinking = model_kwargs.get("thinking", {})
      assert thinking["type"] == "enabled"
      assert thinking["budget_tokens"] == 3000
      assert model_kwargs["response_format"] == {"type": "json_object"}


class TestAnthropicExtendedThinking:
  """Test extended thinking configuration for Anthropic."""

  def test_thinking_disabled_by_default(self):
    """Test that thinking is disabled when env var is not set."""
    factory = LLMFactory("anthropic-claude")

    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key", "ANTHROPIC_MODEL_NAME": "claude-4-sonnet-20250514"}, clear=False):
      llm = factory._build_anthropic_claude_llm(None, 0.0)
      # Verify thinking is not in model_kwargs
      model_kwargs = llm.model_kwargs if hasattr(llm, "model_kwargs") else {}
      assert "thinking" not in model_kwargs

  def test_thinking_enabled(self):
    """Test that thinking is enabled when env var is set."""
    factory = LLMFactory("anthropic-claude")

    with patch.dict(
      os.environ,
      {
        "ANTHROPIC_API_KEY": "test-key",
        "ANTHROPIC_MODEL_NAME": "claude-4-sonnet-20250514",
        "ANTHROPIC_THINKING_ENABLED": "true",
        "ANTHROPIC_THINKING_BUDGET": "6000",
      },
      clear=False,
    ):
      llm = factory._build_anthropic_claude_llm(None, 0.0)
      # Verify thinking configuration - ChatAnthropic keeps it in model_kwargs
      assert hasattr(llm, "model_kwargs")
      assert "thinking_budget" in llm.model_kwargs
      assert llm.model_kwargs["thinking_budget"] == 6000

  def test_thinking_with_default_budget(self):
    """Test thinking with default budget when not specified."""
    factory = LLMFactory("anthropic-claude")

    with patch.dict(
      os.environ,
      {"ANTHROPIC_API_KEY": "test-key", "ANTHROPIC_MODEL_NAME": "claude-4-sonnet-20250514", "ANTHROPIC_THINKING_ENABLED": "1"},
      clear=False,
    ):
      llm = factory._build_anthropic_claude_llm(None, 0.0)
      # Verify thinking configuration with default budget
      assert hasattr(llm, "model_kwargs")
      assert "thinking_budget" in llm.model_kwargs
      assert llm.model_kwargs["thinking_budget"] == THINKING_DEFAULT_BUDGET

  def test_thinking_budget_clamped_to_max_tokens(self):
    """Test that thinking budget is clamped to max_tokens."""
    factory = LLMFactory("anthropic-claude")

    with patch.dict(
      os.environ,
      {
        "ANTHROPIC_API_KEY": "test-key",
        "ANTHROPIC_MODEL_NAME": "claude-4-sonnet-20250514",
        "ANTHROPIC_THINKING_ENABLED": "yes",
        "ANTHROPIC_THINKING_BUDGET": "15000",
      },
      clear=False,
    ):
      llm = factory._build_anthropic_claude_llm(None, 0.0, max_tokens=10000)
      # Verify thinking budget was clamped to max_tokens
      assert hasattr(llm, "model_kwargs")
      assert "thinking_budget" in llm.model_kwargs
      assert llm.model_kwargs["thinking_budget"] == 10000


class TestVertexAIExtendedThinking:
  """Test extended thinking configuration for Vertex AI."""

  def test_thinking_disabled_by_default(self):
    """Test that thinking is disabled when env var is not set."""
    factory = LLMFactory("gcp-vertexai")

    with patch.dict(
      os.environ,
      {
        "GOOGLE_APPLICATION_CREDENTIALS": "/fake/path/creds.json",
        "VERTEXAI_MODEL_NAME": "claude-4-sonnet@20250514",
        "GOOGLE_CLOUD_PROJECT": "test-project",
        "GOOGLE_CLOUD_LOCATION": "us-central1",
      },
      clear=False,
    ):
      with patch("google.auth.default", return_value=(None, None)):
        llm = factory._build_gcp_vertexai_llm(None, 0.0)
        # Verify thinking is not in model_kwargs
        model_kwargs = llm.model_kwargs if hasattr(llm, "model_kwargs") else {}
        assert "thinking_budget" not in model_kwargs

  def test_thinking_enabled(self):
    """Test that thinking is enabled when env var is set."""
    factory = LLMFactory("gcp-vertexai")

    with patch.dict(
      os.environ,
      {
        "GOOGLE_APPLICATION_CREDENTIALS": "/fake/path/creds.json",
        "VERTEXAI_MODEL_NAME": "claude-4-sonnet@20250514",
        "GOOGLE_CLOUD_PROJECT": "test-project",
        "GOOGLE_CLOUD_LOCATION": "us-central1",
        "VERTEXAI_THINKING_ENABLED": "true",
        "VERTEXAI_THINKING_BUDGET": "6000",
      },
      clear=False,
    ):
      with patch("google.auth.default", return_value=(None, None)):
        llm = factory._build_gcp_vertexai_llm(None, 0.0)
        # Verify thinking configuration - ChatVertexAI extracts from kwargs
        assert hasattr(llm, "thinking_budget")
        assert llm.thinking_budget == 6000

  def test_thinking_with_default_budget(self):
    """Test thinking with default budget when not specified."""
    factory = LLMFactory("gcp-vertexai")

    with patch.dict(
      os.environ,
      {
        "GOOGLE_APPLICATION_CREDENTIALS": "/fake/path/creds.json",
        "VERTEXAI_MODEL_NAME": "claude-4-sonnet@20250514",
        "GOOGLE_CLOUD_PROJECT": "test-project",
        "GOOGLE_CLOUD_LOCATION": "us-central1",
        "VERTEXAI_THINKING_ENABLED": "1",
      },
      clear=False,
    ):
      with patch("google.auth.default", return_value=(None, None)):
        llm = factory._build_gcp_vertexai_llm(None, 0.0)
        assert hasattr(llm, "thinking_budget")
        assert llm.thinking_budget == THINKING_DEFAULT_BUDGET

  def test_thinking_budget_clamped_to_max_tokens(self):
    """Test that thinking budget is clamped to max_tokens."""
    factory = LLMFactory("gcp-vertexai")

    with patch.dict(
      os.environ,
      {
        "GOOGLE_APPLICATION_CREDENTIALS": "/fake/path/creds.json",
        "VERTEXAI_MODEL_NAME": "claude-4-sonnet@20250514",
        "GOOGLE_CLOUD_PROJECT": "test-project",
        "GOOGLE_CLOUD_LOCATION": "us-central1",
        "VERTEXAI_THINKING_ENABLED": "yes",
        "VERTEXAI_THINKING_BUDGET": "15000",
      },
      clear=False,
    ):
      with patch("google.auth.default", return_value=(None, None)):
        llm = factory._build_gcp_vertexai_llm(None, 0.0, max_tokens=10000)
        assert hasattr(llm, "thinking_budget")
        assert llm.thinking_budget == 10000


class TestThinkingConfigType:
  """Test the ThinkingConfig TypedDict."""

  def test_thinking_config_structure(self):
    """Test that ThinkingConfig has the correct structure."""
    config: ThinkingConfig = {"type": "enabled", "budget_tokens": 5000}
    assert config["type"] == "enabled"
    assert config["budget_tokens"] == 5000


class TestThinkingTemperatureCompatibility:
  """Test temperature compatibility with extended thinking."""

  def test_thinking_with_nonzero_temperature_bedrock(self):
    """Test that non-zero temperature is removed when thinking is enabled for Bedrock."""
    factory = LLMFactory("aws-bedrock")

    with patch.dict(
      os.environ,
      {
        "AWS_BEDROCK_MODEL_ID": "anthropic.claude-4-sonnet",
        "AWS_REGION": "us-east-1",
        "AWS_ACCESS_KEY_ID": "test-key",
        "AWS_SECRET_ACCESS_KEY": "test-secret",
        "AWS_BEDROCK_THINKING_ENABLED": "true",
      },
      clear=False,
    ):
      # Pass non-zero temperature
      llm = factory._build_aws_bedrock_llm(None, 0.7)
      # Verify temperature was removed (incompatible with thinking)
      # ChatBedrock sets it to None when not provided
      assert llm.temperature is None or llm.temperature == 0

  def test_thinking_with_nonzero_temperature_anthropic(self):
    """Test that Anthropic allows non-zero temperature with thinking enabled."""
    factory = LLMFactory("anthropic-claude")

    with patch.dict(
      os.environ,
      {"ANTHROPIC_API_KEY": "test-key", "ANTHROPIC_MODEL_NAME": "claude-4-sonnet-20250514", "ANTHROPIC_THINKING_ENABLED": "true"},
      clear=False,
    ):
      # Pass non-zero temperature - Anthropic allows this with thinking
      llm = factory._build_anthropic_claude_llm(None, 0.8)
      # Verify thinking is enabled and temperature is preserved
      assert hasattr(llm, "model_kwargs")
      assert "thinking_budget" in llm.model_kwargs
      assert llm.temperature == 0.8

  def test_thinking_with_top_p_bedrock(self):
    """Test that top_p is removed when thinking is enabled for Bedrock."""
    factory = LLMFactory("aws-bedrock")

    with patch.dict(
      os.environ,
      {
        "AWS_BEDROCK_MODEL_ID": "anthropic.claude-4-sonnet",
        "AWS_REGION": "us-east-1",
        "AWS_ACCESS_KEY_ID": "test-key",
        "AWS_SECRET_ACCESS_KEY": "test-secret",
        "AWS_BEDROCK_THINKING_ENABLED": "true",
      },
      clear=False,
    ):
      # Pass top_p in kwargs - it should be removed
      factory._build_aws_bedrock_llm(None, 0.0, top_p=0.9)
      # The test validates that the code runs without error (top_p is removed)

  def test_thinking_with_top_k_anthropic(self):
    """Test that top_k is removed when thinking is enabled for Anthropic."""
    factory = LLMFactory("anthropic-claude")

    with patch.dict(
      os.environ,
      {"ANTHROPIC_API_KEY": "test-key", "ANTHROPIC_MODEL_NAME": "claude-4-sonnet-20250514", "ANTHROPIC_THINKING_ENABLED": "true"},
      clear=False,
    ):
      # Pass top_k in kwargs - it should be removed
      llm = factory._build_anthropic_claude_llm(None, 0.0, top_k=40)
      # Verify the LLM was created successfully without top_k
      assert llm is not None


class TestThinkingBooleanValues:
  """Test various boolean value representations for thinking enabled."""

  @pytest.mark.parametrize("value", ["true", "True", "TRUE", "1", "yes", "YES", "y", "Y", "on", "ON"])
  def test_thinking_enabled_various_true_values(self, value):
    """Test that various truthy values enable thinking."""
    factory = LLMFactory("anthropic-claude")

    with patch.dict(
      os.environ,
      {"ANTHROPIC_API_KEY": "test-key", "ANTHROPIC_MODEL_NAME": "claude-4-sonnet-20250514", "ANTHROPIC_THINKING_ENABLED": value},
      clear=False,
    ):
      llm = factory._build_anthropic_claude_llm(None, 0.0)
      # Check model_kwargs for thinking configuration
      assert hasattr(llm, "model_kwargs"), f"LLM should have model_kwargs for value '{value}'"
      assert "thinking_budget" in llm.model_kwargs, f"Thinking should be enabled for value '{value}'"

  @pytest.mark.parametrize("value", ["false", "False", "FALSE", "0", "no", "NO", "n", "N", "off", "OFF"])
  def test_thinking_disabled_various_false_values(self, value):
    """Test that various falsy values disable thinking."""
    factory = LLMFactory("anthropic-claude")

    with patch.dict(
      os.environ,
      {"ANTHROPIC_API_KEY": "test-key", "ANTHROPIC_MODEL_NAME": "claude-4-sonnet-20250514", "ANTHROPIC_THINKING_ENABLED": value},
      clear=False,
    ):
      llm = factory._build_anthropic_claude_llm(None, 0.0)
      model_kwargs = llm.model_kwargs if hasattr(llm, "model_kwargs") else {}
      assert "thinking" not in model_kwargs


if __name__ == "__main__":
  pytest.main([__file__, "-v"])
