"""
Tests for AWS Bedrock prompt caching functionality.
"""

import os
import pytest
from unittest.mock import patch, MagicMock
from cnoe_agent_utils.llm_factory import LLMFactory


class TestBedrockPromptCaching:
    """Test suite for AWS Bedrock prompt caching support."""

    def test_get_cache_supported_models(self):
        """Test that cache supported models list is not empty and contains expected models."""
        supported = LLMFactory._get_cache_supported_models()

        assert isinstance(supported, set)
        assert len(supported) > 0

        # Check for expected model families
        assert any("claude-3-5-sonnet" in model for model in supported)
        assert any("claude-3-7-sonnet" in model for model in supported)
        assert any("claude-3-5-haiku" in model for model in supported)
        assert any("nova" in model for model in supported)

    def test_is_cache_supported_exact_match(self):
        """Test cache support detection with exact model ID match."""
        factory = LLMFactory.__new__(LLMFactory)

        # Test exact matches
        assert factory._is_cache_supported("anthropic.claude-3-5-sonnet-20241022-v2:0")
        assert factory._is_cache_supported("amazon.nova-micro-v1:0")
        assert factory._is_cache_supported("amazon.nova-lite-v1:0")

    def test_is_cache_supported_prefix_match(self):
        """Test cache support detection with prefix matching."""
        factory = LLMFactory.__new__(LLMFactory)

        # Test prefix matches (for model versions we might encounter)
        assert factory._is_cache_supported("anthropic.claude-3-5-sonnet-20241022")
        assert factory._is_cache_supported("amazon.nova-pro-v1")

    def test_is_cache_not_supported(self):
        """Test that unsupported models are correctly identified."""
        factory = LLMFactory.__new__(LLMFactory)

        # Test models that don't support caching
        assert not factory._is_cache_supported("anthropic.claude-2.1")
        assert not factory._is_cache_supported("anthropic.claude-instant-v1")
        assert not factory._is_cache_supported("meta.llama2-70b-v1")
        assert not factory._is_cache_supported("amazon.titan-text-express-v1")
        assert not factory._is_cache_supported("unsupported-model")

    @patch.dict(os.environ, {
        "LLM_PROVIDER": "aws-bedrock",
        "AWS_BEDROCK_MODEL_ID": "anthropic.claude-3-5-sonnet-20241022-v2:0",
        "AWS_REGION": "us-east-1",
        "AWS_ACCESS_KEY_ID": "test_key",
        "AWS_SECRET_ACCESS_KEY": "test_secret",
        "AWS_BEDROCK_ENABLE_PROMPT_CACHE": "true"
    })
    @patch("cnoe_agent_utils.llm_factory.ChatBedrock")
    def test_cache_enabled_for_supported_model(self, mock_chatbedrock):
        """Test that caching is enabled for supported models."""
        mock_instance = MagicMock()
        mock_chatbedrock.return_value = mock_instance

        factory = LLMFactory("aws-bedrock")
        with patch.object(factory, '_is_cache_supported', return_value=True):
            llm = factory.get_llm()

            # Verify ChatBedrock was instantiated
            assert mock_chatbedrock.called
            assert llm == mock_instance

    @patch.dict(os.environ, {
        "LLM_PROVIDER": "aws-bedrock",
        "AWS_BEDROCK_MODEL_ID": "unsupported.model-v1:0",
        "AWS_REGION": "us-east-1",
        "AWS_ACCESS_KEY_ID": "test_key",
        "AWS_SECRET_ACCESS_KEY": "test_secret",
        "AWS_BEDROCK_ENABLE_PROMPT_CACHE": "true"
    })
    @patch("cnoe_agent_utils.llm_factory.ChatBedrock")
    def test_cache_warning_for_unsupported_model(self, mock_chatbedrock, caplog):
        """Test that a warning is logged when caching is enabled for unsupported model."""
        mock_instance = MagicMock()
        mock_chatbedrock.return_value = mock_instance

        factory = LLMFactory("aws-bedrock")
        llm = factory.get_llm()

        # Check that warning was logged
        assert any("Prompt caching requested but not supported" in record.message
                   for record in caplog.records)
        assert llm == mock_instance

    @patch.dict(os.environ, {
        "LLM_PROVIDER": "aws-bedrock",
        "AWS_BEDROCK_MODEL_ID": "anthropic.claude-3-5-sonnet-20241022-v2:0",
        "AWS_REGION": "us-east-1",
        "AWS_ACCESS_KEY_ID": "test_key",
        "AWS_SECRET_ACCESS_KEY": "test_secret",
        "AWS_BEDROCK_ENABLE_PROMPT_CACHE": "false"
    })
    @patch("cnoe_agent_utils.llm_factory.ChatBedrock")
    def test_cache_disabled_by_default(self, mock_chatbedrock, caplog):
        """Test that caching is not enabled when AWS_BEDROCK_ENABLE_PROMPT_CACHE is false."""
        mock_instance = MagicMock()
        mock_chatbedrock.return_value = mock_instance

        factory = LLMFactory("aws-bedrock")
        llm = factory.get_llm()

        # Check that cache enabled message was NOT logged
        assert not any("Prompt caching enabled" in record.message
                       for record in caplog.records)
        assert llm == mock_instance

    @patch.dict(os.environ, {
        "LLM_PROVIDER": "aws-bedrock",
        "AWS_BEDROCK_MODEL_ID": "anthropic.claude-3-7-sonnet-20250219",
        "AWS_REGION": "us-west-2",
        "AWS_ACCESS_KEY_ID": "test_key",
        "AWS_SECRET_ACCESS_KEY": "test_secret",
        "AWS_BEDROCK_ENABLE_PROMPT_CACHE": "true"
    })
    @patch("cnoe_agent_utils.llm_factory.ChatBedrock")
    def test_cache_enabled_log_message(self, mock_chatbedrock, caplog):
        """Test that appropriate log message is shown when caching is enabled."""
        import logging
        caplog.set_level(logging.INFO)

        mock_instance = MagicMock()
        mock_chatbedrock.return_value = mock_instance

        factory = LLMFactory("aws-bedrock")
        llm = factory.get_llm()

        # Check that cache enabled message was logged
        log_messages = [record.message for record in caplog.records]
        assert any("Prompt caching enabled" in msg and "anthropic.claude-3-7-sonnet-20250219" in msg
                   for msg in log_messages), f"Expected cache enabled message in logs: {log_messages}"
        assert llm == mock_instance

    def test_supported_models_include_nova_premier(self):
        """Test that Nova Premier model is included in supported models."""
        supported = LLMFactory._get_cache_supported_models()
        assert "us.amazon.nova-premier-v1:0" in supported

    def test_supported_models_are_documented(self):
        """Test that all supported models follow AWS naming conventions."""
        supported = LLMFactory._get_cache_supported_models()

        for model in supported:
            # Check format: provider.model-name-version
            assert "." in model, f"Model {model} should contain provider prefix"
            assert "-" in model or "nova" in model, f"Model {model} should contain version info"

            # Check known providers
            providers = ["anthropic", "amazon", "us.amazon"]
            assert any(model.startswith(p) for p in providers), \
                f"Model {model} should start with known provider"

    def test_normalize_model_id_removes_regional_prefix(self):
        """Test that normalize_model_id removes regional prefixes."""
        factory = LLMFactory.__new__(LLMFactory)

        # Test US prefix
        assert factory._normalize_model_id("us.anthropic.claude-3-5-sonnet-20241022-v2:0") == \
            "anthropic.claude-3-5-sonnet-20241022-v2"

        # Test EU prefix
        assert factory._normalize_model_id("eu.anthropic.claude-3-7-sonnet-20250219-v1:0") == \
            "anthropic.claude-3-7-sonnet-20250219-v1"

        # Test no regional prefix
        assert factory._normalize_model_id("anthropic.claude-3-5-haiku-20241022-v1:0") == \
            "anthropic.claude-3-5-haiku-20241022-v1"

        # Test Amazon model with US prefix
        assert factory._normalize_model_id("us.amazon.nova-pro-v1:0") == \
            "amazon.nova-pro-v1"

    def test_normalize_model_id_removes_version_suffix(self):
        """Test that normalize_model_id removes version suffixes."""
        factory = LLMFactory.__new__(LLMFactory)

        assert factory._normalize_model_id("anthropic.claude-3-5-sonnet-20241022-v2:0") == \
            "anthropic.claude-3-5-sonnet-20241022-v2"
        assert factory._normalize_model_id("amazon.nova-lite-v1:0") == \
            "amazon.nova-lite-v1"

    def test_cache_supported_with_regional_prefix(self):
        """Test that cache support works with regional prefixes."""
        factory = LLMFactory.__new__(LLMFactory)

        # US prefixed Claude model
        assert factory._is_cache_supported("us.anthropic.claude-3-5-sonnet-20241022-v2:0")

        # EU prefixed Claude model
        assert factory._is_cache_supported("eu.anthropic.claude-3-7-sonnet-20250219-v1:0")

        # AP prefixed Nova model
        assert factory._is_cache_supported("ap.amazon.nova-pro-v1:0")

    def test_infer_provider_from_model_id(self):
        """Test that provider is correctly inferred from model ID."""
        factory = LLMFactory.__new__(LLMFactory)

        # Anthropic models
        assert factory._infer_provider("anthropic.claude-3-5-sonnet-20241022-v2:0") == "anthropic"
        assert factory._infer_provider("us.anthropic.claude-3-7-sonnet-20250219-v1:0") == "anthropic"

        # Amazon models
        assert factory._infer_provider("amazon.nova-pro-v1:0") == "amazon"
        assert factory._infer_provider("us.amazon.nova-lite-v1:0") == "amazon"

        # Unknown provider
        assert factory._infer_provider("unknown.model-v1:0") is None

    @patch.dict(os.environ, {
        "LLM_PROVIDER": "aws-bedrock",
        "AWS_BEDROCK_MODEL_ID": "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
        "AWS_REGION": "us-east-1",
        "AWS_ACCESS_KEY_ID": "test_key",
        "AWS_SECRET_ACCESS_KEY": "test_secret"
    })
    @patch("cnoe_agent_utils.llm_factory.ChatBedrock")
    def test_provider_inference_with_regional_model(self, mock_chatbedrock, caplog):
        """Test that provider is correctly inferred for regional model IDs."""
        import logging
        caplog.set_level(logging.INFO)

        mock_instance = MagicMock()
        mock_chatbedrock.return_value = mock_instance

        factory = LLMFactory("aws-bedrock")
        llm = factory.get_llm()

        # Check that provider inference message was logged
        log_messages = [record.message for record in caplog.records]
        assert any("Inferred provider 'anthropic'" in msg for msg in log_messages)

        # Verify ChatBedrock was called with inferred provider
        call_kwargs = mock_chatbedrock.call_args.kwargs
        assert call_kwargs.get("provider") == "anthropic"
        assert llm == mock_instance