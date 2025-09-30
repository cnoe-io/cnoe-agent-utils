"""
Tests for AWS Bedrock prompt caching functionality.
"""

import os
from unittest.mock import patch, MagicMock
from cnoe_agent_utils.llm_factory import LLMFactory


class TestBedrockPromptCaching:
    """Test suite for AWS Bedrock prompt caching support."""

    @patch.dict(os.environ, {
        "LLM_PROVIDER": "aws-bedrock",
        "AWS_BEDROCK_MODEL_ID": "anthropic.claude-3-5-sonnet-20241022-v2:0",
        "AWS_REGION": "us-east-1",
        "AWS_ACCESS_KEY_ID": "test_key",
        "AWS_SECRET_ACCESS_KEY": "test_secret",
        "AWS_BEDROCK_ENABLE_PROMPT_CACHE": "true"
    })
    @patch("cnoe_agent_utils.llm_factory.ChatBedrockConverse")
    def test_cache_enabled_uses_converse(self, mock_chatbedrock_converse):
        """Test that ChatBedrockConverse is used when caching is enabled."""
        mock_instance = MagicMock()
        mock_chatbedrock_converse.return_value = mock_instance

        factory = LLMFactory("aws-bedrock")
        llm = factory.get_llm()

        # Verify ChatBedrockConverse was instantiated
        assert mock_chatbedrock_converse.called
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
    def test_cache_disabled_uses_chatbedrock(self, mock_chatbedrock):
        """Test that ChatBedrock is used when caching is disabled."""
        mock_instance = MagicMock()
        mock_chatbedrock.return_value = mock_instance

        factory = LLMFactory("aws-bedrock")
        llm = factory.get_llm()

        # Verify ChatBedrock was instantiated
        assert mock_chatbedrock.called
        assert llm == mock_instance

    @patch.dict(os.environ, {
        "LLM_PROVIDER": "aws-bedrock",
        "AWS_BEDROCK_MODEL_ID": "anthropic.claude-3-7-sonnet-20250219",
        "AWS_REGION": "us-west-2",
        "AWS_ACCESS_KEY_ID": "test_key",
        "AWS_SECRET_ACCESS_KEY": "test_secret",
        "AWS_BEDROCK_ENABLE_PROMPT_CACHE": "true"
    })
    @patch("cnoe_agent_utils.llm_factory.ChatBedrockConverse")
    def test_cache_enabled_log_message(self, mock_chatbedrock_converse, caplog):
        """Test that appropriate log message is shown when caching is enabled."""
        import logging
        caplog.set_level(logging.INFO)

        mock_instance = MagicMock()
        mock_chatbedrock_converse.return_value = mock_instance

        factory = LLMFactory("aws-bedrock")
        llm = factory.get_llm()

        # Check that cache enabled message was logged
        log_messages = [record.message for record in caplog.records]
        assert any("Prompt caching enabled" in msg and "anthropic.claude-3-7-sonnet-20250219" in msg
                   for msg in log_messages), f"Expected cache enabled message in logs: {log_messages}"
        assert any("Using ChatBedrockConverse" in msg for msg in log_messages), \
            f"Expected ChatBedrockConverse message in logs: {log_messages}"
        assert llm == mock_instance

    @patch.dict(os.environ, {
        "LLM_PROVIDER": "aws-bedrock",
        "AWS_BEDROCK_MODEL_ID": "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
        "AWS_REGION": "us-east-1",
        "AWS_ACCESS_KEY_ID": "test_key",
        "AWS_SECRET_ACCESS_KEY": "test_secret",
        "AWS_BEDROCK_ENABLE_PROMPT_CACHE": "true"
    })
    @patch("cnoe_agent_utils.llm_factory.ChatBedrockConverse")
    def test_regional_model_id_with_caching(self, mock_chatbedrock_converse):
        """Test that regional model IDs (us. prefix) work with caching enabled."""
        mock_instance = MagicMock()
        mock_chatbedrock_converse.return_value = mock_instance

        factory = LLMFactory("aws-bedrock")
        llm = factory.get_llm()

        # Verify ChatBedrockConverse was called with the full model ID (unchanged)
        call_kwargs = mock_chatbedrock_converse.call_args.kwargs
        assert call_kwargs.get("model_id") == "us.anthropic.claude-3-5-sonnet-20241022-v2:0"
        assert llm == mock_instance

    @patch.dict(os.environ, {
        "LLM_PROVIDER": "aws-bedrock",
        "AWS_BEDROCK_MODEL_ID": "us.amazon.nova-premier-v1:0",
        "AWS_REGION": "us-east-1",
        "AWS_ACCESS_KEY_ID": "test_key",
        "AWS_SECRET_ACCESS_KEY": "test_secret",
        "AWS_BEDROCK_ENABLE_PROMPT_CACHE": "true"
    })
    @patch("cnoe_agent_utils.llm_factory.ChatBedrockConverse")
    def test_amazon_model_with_caching(self, mock_chatbedrock_converse):
        """Test that Amazon Nova models work with caching enabled."""
        mock_instance = MagicMock()
        mock_chatbedrock_converse.return_value = mock_instance

        factory = LLMFactory("aws-bedrock")
        llm = factory.get_llm()

        # Verify ChatBedrockConverse was called with the full model ID (unchanged)
        call_kwargs = mock_chatbedrock_converse.call_args.kwargs
        assert call_kwargs.get("model_id") == "us.amazon.nova-premier-v1:0"
        assert llm == mock_instance

    @patch.dict(os.environ, {
        "LLM_PROVIDER": "aws-bedrock",
        "AWS_BEDROCK_MODEL_ID": "anthropic.claude-3-5-sonnet-20241022-v2:0",
        "AWS_REGION": "us-east-1",
        "AWS_ACCESS_KEY_ID": "test_key",
        "AWS_SECRET_ACCESS_KEY": "test_secret",
        "AWS_BEDROCK_PROVIDER": "anthropic"
    })
    @patch("cnoe_agent_utils.llm_factory.ChatBedrock")
    def test_explicit_provider_passed_through(self, mock_chatbedrock):
        """Test that explicitly set AWS_BEDROCK_PROVIDER is passed through."""
        mock_instance = MagicMock()
        mock_chatbedrock.return_value = mock_instance

        factory = LLMFactory("aws-bedrock")
        llm = factory.get_llm()

        # Verify ChatBedrock was called with the explicit provider
        call_kwargs = mock_chatbedrock.call_args.kwargs
        assert call_kwargs.get("provider") == "anthropic"
        assert llm == mock_instance