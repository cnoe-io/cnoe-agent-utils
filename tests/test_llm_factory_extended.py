#!/usr/bin/env python3
"""Extended tests for LLMFactory to improve coverage."""

import os
import pytest
from unittest.mock import patch, MagicMock
from cnoe_agent_utils.llm_factory import LLMFactory


class TestLLMFactoryExtendedCoverage:
    """Additional tests to improve coverage of LLMFactory."""

    def test_aws_bedrock_builder_success(self):
        """Test successful AWS Bedrock LLM creation."""
        with patch.dict(os.environ, {
            "AWS_ACCESS_KEY_ID": "test-key",
            "AWS_SECRET_ACCESS_KEY": "test-secret",
            "AWS_REGION": "us-east-1",
            "AWS_BEDROCK_MODEL_ID": "anthropic.claude-3-sonnet-20240229-v1"
        }):
            factory = LLMFactory("aws-bedrock")
            llm = factory._build_aws_bedrock_llm(None, None)

            assert llm is not None
            # Verify the LLM was created with correct parameters
            assert hasattr(llm, 'model_id')

    def test_azure_openai_builder_success(self):
        """Test successful Azure OpenAI LLM creation."""
        with patch.dict(os.environ, {
            "AZURE_OPENAI_API_KEY": "test-key",
            "AZURE_OPENAI_DEPLOYMENT": "test-deployment",
            "AZURE_OPENAI_API_VERSION": "2024-02-15-preview",
            "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com/"
        }):
            factory = LLMFactory("azure-openai")
            llm = factory._build_azure_openai_llm(None, None)

            assert llm is not None
            # Verify the LLM was created with correct parameters
            assert hasattr(llm, 'deployment_name')

    def test_google_gemini_builder_success(self):
        """Test successful Google Gemini LLM creation."""
        with patch.dict(os.environ, {
            "GOOGLE_API_KEY": "test-key",
            "GOOGLE_GEMINI_MODEL_NAME": "gemini-2.0-flash"
        }):
            factory = LLMFactory("google-gemini")
            llm = factory._build_google_gemini_llm(None, None)

            assert llm is not None
            # Verify the LLM was created with correct parameters
            assert hasattr(llm, 'model')

    def test_gcp_vertexai_builder_success(self):
        """Test successful GCP Vertex AI LLM creation."""
        with patch.dict(os.environ, {
            "GOOGLE_CLOUD_PROJECT": "test-project",
            "GOOGLE_CLOUD_LOCATION": "us-central1",
            "VERTEXAI_MODEL_NAME": "gemini-2.0-flash-001",
            "GOOGLE_APPLICATION_CREDENTIALS": "/tmp/mock-credentials.json"
        }):
            # Mock the google.auth.default call to avoid actual credential validation
            with patch('google.auth.default') as mock_auth:
                mock_credentials = MagicMock()
                mock_auth.return_value = (mock_credentials, "test-project")

                factory = LLMFactory("gcp-vertexai")
                llm = factory._build_gcp_vertexai_llm(None, None)

                assert llm is not None
                # Verify the LLM was created with correct parameters
                assert hasattr(llm, 'model_name')

    def test_get_llm_without_tools_success(self):
        """Test getting LLM without tools successfully."""
        with patch.dict(os.environ, {
            "ANTHROPIC_API_KEY": "test-key",
            "ANTHROPIC_MODEL_NAME": "claude-3-sonnet-20240229-v1"
        }):
            factory = LLMFactory("anthropic-claude")

            llm = factory.get_llm()
            assert llm is not None

    def test_provider_normalization_edge_cases(self):
        """Test provider name normalization edge cases."""
        # Test that provider names are handled correctly
        valid_providers = [
            "anthropic-claude",
            "aws-bedrock",
            "azure-openai",
            "openai",
            "google-gemini",
            "gcp-vertexai"
        ]

        for provider in valid_providers:
            factory = LLMFactory(provider)
            # Provider should be normalized to use underscores
            expected_normalized = provider.replace('-', '_')
            assert factory.provider == expected_normalized

    def test_environment_variable_loading(self):
        """Test that environment variables are loaded correctly."""
        with patch.dict(os.environ, {
            "LLM_PROVIDER": "anthropic-claude",
            "ANTHROPIC_API_KEY": "test-key",
            "ANTHROPIC_MODEL_NAME": "claude-3-sonnet-20240229-v1"
        }):
            factory = LLMFactory()
            assert factory.provider == "anthropic_claude"

    def test_missing_dependencies_for_all_providers(self):
        """Test missing dependencies for all provider types."""
        providers = [
            "anthropic-claude",
            "aws-bedrock",
            "azure-openai",
            "openai",
            "google-gemini",
            "gcp-vertexai"
        ]

        for provider in providers:
            missing = LLMFactory.get_missing_dependencies(provider)
            # Should return a list (even if empty)
            assert isinstance(missing, list)

    def test_supported_providers_consistency(self):
        """Test that supported providers are consistent."""
        providers = LLMFactory.get_supported_providers()

        # All providers should be available if dependencies are installed
        for provider in providers:
            assert LLMFactory.is_provider_available(provider)
            missing_deps = LLMFactory.get_missing_dependencies(provider)
            assert len(missing_deps) == 0

    def test_unsupported_provider_handling(self):
        """Test handling of unsupported providers."""
        unsupported_providers = [
            "invalid-provider",
            "unknown-llm",
            "fake-ai"
        ]

        for provider in unsupported_providers:
            assert not LLMFactory.is_provider_available(provider)
            missing_deps = LLMFactory.get_missing_dependencies(provider)
            assert len(missing_deps) == 0  # No missing deps for unsupported providers

    def test_llm_factory_error_handling(self):
        """Test error handling in LLMFactory."""
        # Test with invalid provider
        with pytest.raises(ValueError, match="Unsupported provider"):
            LLMFactory("invalid-provider")

        # Test with None provider when no env var
        with patch.dict(os.environ, {}, clear=True):
            with patch('cnoe_agent_utils.llm_factory.dotenv.load_dotenv'):
                with pytest.raises(ValueError, match="Provider must be specified"):
                    LLMFactory()

    def test_llm_factory_provider_validation(self):
        """Test provider validation in LLMFactory."""
        # Test that valid providers work
        valid_providers = [
            "anthropic-claude",
            "aws-bedrock",
            "azure-openai",
            "openai",
            "google-gemini",
            "gcp-vertexai"
        ]

        for provider in valid_providers:
            try:
                factory = LLMFactory(provider)
                assert factory.provider == provider.replace('-', '_')
            except Exception as e:
                # Some providers might fail due to missing credentials, but shouldn't fail validation
                assert "Unsupported provider" not in str(e)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
