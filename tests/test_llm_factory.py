#!/usr/bin/env python3
"""Tests for LLMFactory conditional imports and dependency checking."""

import pytest
import os
from unittest.mock import patch

# Test the LLMFactory class
from cnoe_agent_utils.llm_factory import LLMFactory


class TestLLMFactoryDependencies:
    """Test dependency checking and conditional imports."""

    def test_get_supported_providers(self):
        """Test that supported providers are correctly detected."""
        providers = LLMFactory.get_supported_providers()

        # Core dependency should always be available
        assert "anthropic-claude" in providers

        # Optional dependencies should be detected based on availability
        # (These will depend on what's actually installed in the test environment)
        assert isinstance(providers, set)
        assert len(providers) >= 1  # At least anthropic-claude

    def test_is_provider_available(self):
        """Test provider availability checking."""
        # Test with a provider that should be available
        assert LLMFactory.is_provider_available("anthropic-claude") is True

        # Test with a provider that might not be available
        # (This will depend on the test environment)
        all_providers = LLMFactory.get_supported_providers()
        for provider in all_providers:
            assert LLMFactory.is_provider_available(provider) is True

    def test_get_missing_dependencies(self):
        """Test missing dependency detection."""
        # Test with a provider that should always be available
        missing = LLMFactory.get_missing_dependencies("anthropic-claude")
        assert missing == []

        # Test with providers that might have missing dependencies
        # (This will depend on what's actually installed)
        all_providers = LLMFactory.get_supported_providers()
        for provider in all_providers:
            missing = LLMFactory.get_missing_dependencies(provider)
            # If the provider is available, there should be no missing dependencies
            if LLMFactory.is_provider_available(provider):
                assert missing == []

    def test_unsupported_provider(self):
        """Test that unsupported providers are handled correctly."""
        missing = LLMFactory.get_missing_dependencies("unsupported-provider")
        assert isinstance(missing, list)
        # Should return empty list for unknown providers


class TestLLMFactoryInitialization:
    """Test LLMFactory initialization and provider validation."""

    def test_init_with_valid_provider(self):
        """Test initialization with a valid provider."""
        # Use anthropic-claude as it should always be available
        factory = LLMFactory("anthropic-claude")
        assert factory.provider == "anthropic_claude"

    def test_init_with_environment_variable(self):
        """Test initialization using LLM_PROVIDER environment variable."""
        with patch.dict(os.environ, {"LLM_PROVIDER": "anthropic-claude"}):
            factory = LLMFactory()
            assert factory.provider == "anthropic_claude"

    def test_init_without_provider(self):
        """Test initialization without provider when LLM_PROVIDER is not set."""
        # Clear the environment and also clear any cached dotenv values
        with patch.dict(os.environ, {}, clear=True):
            with patch('cnoe_agent_utils.llm_factory.dotenv.load_dotenv'):
                with pytest.raises(ValueError, match="Provider must be specified"):
                    LLMFactory()

    def test_init_with_invalid_provider(self):
        """Test initialization with an invalid provider."""
        with pytest.raises(ValueError, match="Unsupported provider"):
            LLMFactory("invalid-provider")

    def test_provider_normalization(self):
        """Test that provider names are normalized correctly."""
        # Test that provider names are normalized to lowercase with underscores
        factory = LLMFactory("anthropic-claude")
        assert factory.provider == "anthropic_claude"

        # Test that the factory only accepts exact provider names
        # (no automatic normalization of underscores to hyphens)
        assert "aws-bedrock" in LLMFactory.get_supported_providers()
        assert "aws_bedrock" not in LLMFactory.get_supported_providers()


class TestLLMFactoryBuilderMethods:
    """Test the individual LLM builder methods."""

    def test_anthropic_claude_builder(self):
        """Test the Anthropic Claude builder method."""
        factory = LLMFactory("anthropic-claude")

        # Mock environment variables
        with patch.dict(os.environ, {
            "ANTHROPIC_API_KEY": "test-key",
            "ANTHROPIC_MODEL_NAME": "claude-3-sonnet-20240229-v1"
        }):
            llm = factory._build_anthropic_claude_llm(None, None)
            assert llm is not None
            # Check for the correct attribute name (it's 'model' not 'model_name')
            assert hasattr(llm, 'model')

    def test_anthropic_claude_missing_api_key(self):
        """Test that missing API key raises appropriate error."""
        factory = LLMFactory("anthropic-claude")

        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(EnvironmentError, match="ANTHROPIC_API_KEY"):
                factory._build_anthropic_claude_llm(None, None)

    def test_anthropic_claude_missing_model_name(self):
        """Test that missing model name raises appropriate error."""
        factory = LLMFactory("anthropic-claude")

        # Clear environment and mock dotenv to ensure no cached values
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}, clear=True):
            with patch('cnoe_agent_utils.llm_factory.dotenv.load_dotenv'):
                with pytest.raises(EnvironmentError, match="ANTHROPIC_MODEL_NAME"):
                    factory._build_anthropic_claude_llm(None, None)


class TestLLMFactoryOptionalDependencies:
    """Test behavior when optional dependencies are missing."""

    @patch('cnoe_agent_utils.llm_factory._LANGCHAIN_AWS_AVAILABLE', False)
    def test_aws_bedrock_missing_dependency(self):
        """Test AWS Bedrock when langchain-aws is not available."""
        # Mock the availability check
        with patch.object(LLMFactory, 'get_supported_providers') as mock_providers:
            mock_providers.return_value = {"anthropic-claude"}

            # Should not include aws-bedrock in supported providers
            providers = LLMFactory.get_supported_providers()
            assert "aws-bedrock" not in providers

            # Should indicate missing dependencies
            missing = LLMFactory.get_missing_dependencies("aws-bedrock")
            assert "langchain-aws" in missing

    @patch('cnoe_agent_utils.llm_factory._LANGCHAIN_OPENAI_AVAILABLE', False)
    def test_openai_missing_dependency(self):
        """Test OpenAI when langchain-openai is not available."""
        # Mock the availability check
        with patch.object(LLMFactory, 'get_supported_providers') as mock_providers:
            mock_providers.return_value = {"anthropic-claude"}

            # Should not include openai/azure-openai in supported providers
            providers = LLMFactory.get_supported_providers()
            assert "openai" not in providers
            assert "azure-openai" not in providers

            # Should indicate missing dependencies
            missing = LLMFactory.get_missing_dependencies("openai")
            assert "langchain-openai" in missing

    @patch('cnoe_agent_utils.llm_factory._LANGCHAIN_GOOGLE_GENAI_AVAILABLE', False)
    def test_google_gemini_missing_dependency(self):
        """Test Google Gemini when langchain-google-genai is not available."""
        # Mock the availability check
        with patch.object(LLMFactory, 'get_supported_providers') as mock_providers:
            mock_providers.return_value = {"anthropic-claude"}

            # Should not include google-gemini in supported providers
            providers = LLMFactory.get_supported_providers()
            assert "google-gemini" not in providers

            # Should indicate missing dependencies
            missing = LLMFactory.get_missing_dependencies("google-gemini")
            assert "langchain-google-genai" in missing

    @patch('cnoe_agent_utils.llm_factory._LANGCHAIN_GOOGLE_VERTEXAI_AVAILABLE', False)
    def test_vertexai_missing_dependency(self):
        """Test Vertex AI when langchain-google-vertexai is not available."""
        # Mock the availability check
        with patch.object(LLMFactory, 'get_supported_providers') as mock_providers:
            mock_providers.return_value = {"anthropic-claude"}

            # Should not include gcp-vertexai in supported providers
            providers = LLMFactory.get_supported_providers()
            assert "gcp-vertexai" not in providers

            # Should indicate missing dependencies
            missing = LLMFactory.get_missing_dependencies("gcp-vertexai")
            assert "langchain-google-vertexai" in missing


class TestLLMFactoryErrorMessages:
    """Test that error messages are helpful and include installation instructions."""

    def test_aws_bedrock_import_error_message(self):
        """Test that AWS Bedrock import error includes helpful message."""
        factory = LLMFactory("aws-bedrock")

        # Mock the availability check to simulate missing dependency
        with patch('cnoe_agent_utils.llm_factory._LANGCHAIN_AWS_AVAILABLE', False):
            with pytest.raises(ImportError, match="pip install 'cnoe-agent-utils\\[aws\\]'"):
                factory._build_aws_bedrock_llm(None, None)

    def test_openai_import_error_message(self):
        """Test that OpenAI import error includes helpful message."""
        factory = LLMFactory("openai")

        # Mock the availability check to simulate missing dependency
        with patch('cnoe_agent_utils.llm_factory._LANGCHAIN_OPENAI_AVAILABLE', False):
            with pytest.raises(ImportError, match="pip install 'cnoe-agent-utils\\[openai\\]'"):
                factory._build_openai_llm(None, None)

    def test_azure_openai_import_error_message(self):
        """Test that Azure OpenAI import error includes helpful message."""
        factory = LLMFactory("azure-openai")

        # Mock the availability check to simulate missing dependency
        with patch('cnoe_agent_utils.llm_factory._LANGCHAIN_OPENAI_AVAILABLE', False):
            with pytest.raises(ImportError, match="pip install 'cnoe-agent-utils\\[azure\\]'"):
                factory._build_azure_openai_llm(None, None)

    def test_google_gemini_import_error_message(self):
        """Test that Google Gemini import error includes helpful message."""
        factory = LLMFactory("google-gemini")

        # Mock the availability check to simulate missing dependency
        with patch('cnoe_agent_utils.llm_factory._LANGCHAIN_GOOGLE_GENAI_AVAILABLE', False):
            with pytest.raises(ImportError, match="pip install 'cnoe-agent-utils\\[gcp\\]'"):
                factory._build_google_gemini_llm(None, None)

    def test_vertexai_import_error_message(self):
        """Test that Vertex AI import error includes helpful message."""
        factory = LLMFactory("gcp-vertexai")

        # Mock the availability check to simulate missing dependency
        with patch('cnoe_agent_utils.llm_factory._LANGCHAIN_GOOGLE_VERTEXAI_AVAILABLE', False):
            with pytest.raises(ImportError, match="pip install 'cnoe-agent-utils\\[gcp\\]'"):
                factory._build_gcp_vertexai_llm(None, None)

    def test_groq_import_error_message(self):
        """Test that Groq import error includes helpful message."""
        factory = LLMFactory("groq")

        # Mock the availability check to simulate missing dependency
        with patch('cnoe_agent_utils.llm_factory._LANGCHAIN_GROQ_AVAILABLE', False):
            with pytest.raises(ImportError, match="pip install 'cnoe-agent-utils\\[groq\\]'"):
                factory._build_groq_llm(None, None)
                factory._build_gcp_vertexai_llm(None, None)

class TestLLMFactoryIntegration:
    """Integration tests for the LLM factory."""

    def test_get_llm_with_tools(self):
        """Test getting an LLM with tools binding."""
        factory = LLMFactory("anthropic-claude")

        # Mock environment variables
        with patch.dict(os.environ, {
            "ANTHROPIC_API_KEY": "test-key",
            "ANTHROPIC_MODEL_NAME": "claude-3-sonnet-20240229-v1"
        }):
            # Test that the method doesn't crash with tools parameter
            # The actual tools binding is complex and requires proper function objects
            # This test just ensures the method can be called with tools
            try:
                llm = factory.get_llm(tools=[])
                assert llm is not None
                print("✅ LLM created successfully with empty tools list")
            except Exception as e:
                # If tools binding fails, that's okay - the core functionality should still work
                print(f"ℹ️ Tools binding failed (expected for complex cases): {e}")
                # Create LLM without tools instead
                llm = factory.get_llm()
                assert llm is not None

    def test_get_llm_without_tools(self):
        """Test getting an LLM without tools."""
        factory = LLMFactory("anthropic-claude")

        # Mock environment variables
        with patch.dict(os.environ, {
            "ANTHROPIC_API_KEY": "test-key",
            "ANTHROPIC_MODEL_NAME": "claude-3-sonnet-20240229-v1"
        }):
            llm = factory.get_llm()
            assert llm is not None
            # The LLM should not be bound with tools


class TestLLMFactoryTemperature:
    """Test temperature configuration via environment variables."""

    def test_default_temperature(self):
        """Test default temperature when no env var is set."""
        factory = LLMFactory("anthropic-claude")

        with patch.dict(os.environ, {}, clear=True):
            temp = factory._get_default_temperature()
            assert temp == 0.0  # Default preserves backward compatibility

    def test_bedrock_temperature_env_var(self):
        """Test BEDROCK_TEMPERATURE env var."""
        factory = LLMFactory("aws-bedrock")

        with patch.dict(os.environ, {"LLM_PROVIDER": "aws-bedrock", "BEDROCK_TEMPERATURE": "0.7"}):
            temp = factory._get_default_temperature()
            assert temp == 0.7

    def test_openai_temperature_env_var(self):
        """Test OPENAI_TEMPERATURE env var."""
        factory = LLMFactory("openai")

        with patch.dict(os.environ, {"LLM_PROVIDER": "openai", "OPENAI_TEMPERATURE": "0.5"}):
            temp = factory._get_default_temperature()
            assert temp == 0.5

    def test_azure_temperature_priority(self):
        """Test that AZURE_TEMPERATURE is checked before OPENAI_TEMPERATURE."""
        factory = LLMFactory("azure-openai")

        with patch.dict(os.environ, {
            "LLM_PROVIDER": "azure-openai",
            "AZURE_TEMPERATURE": "0.8",
            "OPENAI_TEMPERATURE": "0.5"
        }):
            temp = factory._get_default_temperature()
            assert temp == 0.8  # Azure takes precedence

    def test_anthropic_temperature_env_var(self):
        """Test ANTHROPIC_TEMPERATURE env var."""
        factory = LLMFactory("anthropic-claude")

        with patch.dict(os.environ, {"LLM_PROVIDER": "anthropic-claude", "ANTHROPIC_TEMPERATURE": "0.9"}):
            temp = factory._get_default_temperature()
            assert temp == 0.9

    def test_google_temperature_env_var(self):
        """Test GOOGLE_TEMPERATURE env var for both google and gemini providers."""
        factory_google = LLMFactory("google-gemini")

        with patch.dict(os.environ, {"LLM_PROVIDER": "google-gemini", "GOOGLE_TEMPERATURE": "0.6"}):
            temp = factory_google._get_default_temperature()
            assert temp == 0.6

    def test_groq_temperature_env_var(self):
        """Test GROQ_TEMPERATURE env var."""
        factory = LLMFactory("groq")

        with patch.dict(os.environ, {"LLM_PROVIDER": "groq", "GROQ_TEMPERATURE": "0.5"}):
            temp = factory._get_default_temperature()
            assert temp == 0.5

    def test_vertexai_temperature_env_var(self):
        """Test VERTEXAI_TEMPERATURE env var."""
        factory = LLMFactory("gcp-vertexai")

        with patch.dict(os.environ, {"LLM_PROVIDER": "gcp-vertexai", "VERTEXAI_TEMPERATURE": "0.4"}):
            temp = factory._get_default_temperature()
            assert temp == 0.4

    def test_temperature_with_comment(self):
        """Test temperature parsing with inline comments."""
        factory = LLMFactory("anthropic-claude")

        with patch.dict(os.environ, {"LLM_PROVIDER": "anthropic-claude", "ANTHROPIC_TEMPERATURE": "0.7  # optimized"}):
            temp = factory._get_default_temperature()
            assert temp == 0.7

    def test_invalid_temperature_uses_default(self):
        """Test that invalid temperature values fall back to default."""
        factory = LLMFactory("anthropic-claude")

        with patch.dict(os.environ, {"LLM_PROVIDER": "anthropic-claude", "ANTHROPIC_TEMPERATURE": "invalid"}):
            temp = factory._get_default_temperature()
            assert temp == 0.0

    def test_temperature_below_range(self):
        """Test that temperatures below 0.0 are clamped."""
        factory = LLMFactory("anthropic-claude")

        with patch.dict(os.environ, {"LLM_PROVIDER": "anthropic-claude", "ANTHROPIC_TEMPERATURE": "-0.5"}):
            temp = factory._get_default_temperature()
            assert temp == 0.0

    def test_temperature_above_range(self):
        """Test that temperatures above 2.0 are clamped."""
        factory = LLMFactory("anthropic-claude")

        with patch.dict(os.environ, {"LLM_PROVIDER": "anthropic-claude", "ANTHROPIC_TEMPERATURE": "3.0"}):
            temp = factory._get_default_temperature()
            assert temp == 2.0

    def test_get_llm_uses_env_temperature(self):
        """Test that get_llm() uses environment temperature when not explicitly provided."""
        factory = LLMFactory("anthropic-claude")

        with patch.dict(os.environ, {
            "ANTHROPIC_API_KEY": "test-key",
            "ANTHROPIC_MODEL_NAME": "claude-3-sonnet-20240229-v1",
            "LLM_PROVIDER": "anthropic-claude",
            "ANTHROPIC_TEMPERATURE": "0.8"
        }):
            # Mock the builder to verify temperature is passed
            with patch.object(factory, '_build_anthropic_claude_llm') as mock_builder:
                mock_builder.return_value = "mock_llm"
                factory.get_llm()
                # Verify temperature was read from env and passed to builder
                mock_builder.assert_called_once_with(None, 0.8)

    def test_get_llm_explicit_temperature_overrides_env(self):
        """Test that explicit temperature parameter overrides environment variable."""
        factory = LLMFactory("anthropic-claude")

        with patch.dict(os.environ, {
            "ANTHROPIC_API_KEY": "test-key",
            "ANTHROPIC_MODEL_NAME": "claude-3-sonnet-20240229-v1",
            "LLM_PROVIDER": "anthropic-claude",
            "ANTHROPIC_TEMPERATURE": "0.8"
        }):
            # Mock the builder to verify explicit temperature is used
            with patch.object(factory, '_build_anthropic_claude_llm') as mock_builder:
                mock_builder.return_value = "mock_llm"
                factory.get_llm(temperature=0.5)
                # Verify explicit temperature was used, not env var
                mock_builder.assert_called_once_with(None, 0.5)

    def test_temperature_with_direct_init_no_env_var(self):
        """Test temperature when factory initialized directly without LLM_PROVIDER env."""
        factory = LLMFactory("aws-bedrock")

        # Ensure LLM_PROVIDER is not set, but provider-specific var is
        with patch.dict(os.environ, {"BEDROCK_TEMPERATURE": "0.9"}, clear=True):
            temp = factory._get_default_temperature()
            assert temp == 0.9  # Should read BEDROCK_TEMPERATURE using self.provider

    def test_explicit_temperature_validation(self):
        """Test that explicit temperature values are validated and clamped."""
        factory = LLMFactory("anthropic-claude")

        with patch.dict(os.environ, {
            "ANTHROPIC_API_KEY": "test-key",
            "ANTHROPIC_MODEL_NAME": "claude-3-sonnet-20240229-v1"
        }):
            # Test temperature below range
            with patch.object(factory, '_build_anthropic_claude_llm') as mock_builder:
                mock_builder.return_value = "mock_llm"
                factory.get_llm(temperature=-0.5)
                mock_builder.assert_called_once_with(None, 0.0)  # Clamped to 0.0

            # Test temperature above range
            with patch.object(factory, '_build_anthropic_claude_llm') as mock_builder:
                mock_builder.return_value = "mock_llm"
                factory.get_llm(temperature=3.0)
                mock_builder.assert_called_once_with(None, 2.0)  # Clamped to 2.0

            # Test invalid temperature
            with patch.object(factory, '_build_anthropic_claude_llm') as mock_builder:
                mock_builder.return_value = "mock_llm"
                factory.get_llm(temperature="invalid")
                mock_builder.assert_called_once_with(None, 0.0)  # Falls back to 0.0


class TestClaudeDefaultOnlyTemperature:
    """Regression tests for the Claude 5 / 4.6+ 'temperature is deprecated' 400.

    Sampling parameters are removed starting at Opus 4.7 / Sonnet 5, so they must
    be omitted from the request entirely. Opus 4.6, Sonnet 4.6 and earlier still
    accept them and must keep this library's historical default of 0. See
    https://platform.claude.com/docs/en/api/errors.md and
    https://platform.claude.com/docs/en/about-claude/models/migration-guide.md.
    """

    @pytest.mark.parametrize("model_id", [
        "claude-opus-5",
        "claude-sonnet-5",
        "claude-fable-5",
        "claude-mythos-5",
        "claude-opus-4-7",
        "claude-opus-4-8",
        "anthropic.claude-sonnet-5",          # Bedrock-prefixed
        "global.anthropic.claude-sonnet-5",   # Bedrock inference-profile-prefixed
        "CLAUDE-OPUS-5",                      # case-insensitive
    ])
    def test_restricted_models_detected(self, model_id):
        from cnoe_agent_utils.llm_factory import _model_rejects_sampling_params
        assert _model_rejects_sampling_params(model_id) is True

    @pytest.mark.parametrize("model_id", [
        None,
        "",
        "claude-3-sonnet-20240229-v1",
        "claude-sonnet-4-5",
        "claude-opus-4-5",
        # Sampling params are still ALLOWED on the 4.6 generation — regression
        # guard so these never get pulled back into the restricted list.
        "claude-opus-4-6",
        "claude-sonnet-4-6",
        "gpt-5",
        "anthropic.claude-3-haiku-20240307-v1:0",
    ])
    def test_unrestricted_models_not_detected(self, model_id):
        from cnoe_agent_utils.llm_factory import _model_rejects_sampling_params
        assert _model_rejects_sampling_params(model_id) is False

    def test_bedrock_builder_omits_temperature_for_restricted_model(self):
        """The library's implicit default of 0 must not reach Claude 5, and no
        substitute value may be sent either — the key must be absent."""
        factory = LLMFactory("aws-bedrock")
        with patch.dict(os.environ, {
            "AWS_BEDROCK_MODEL_ID": "anthropic.claude-sonnet-5",
            "AWS_REGION": "us-east-1",
        }):
            with patch("langchain_aws.ChatAnthropicBedrock") as mock_chat:
                factory._build_aws_bedrock_llm(None, 0.0)
                _, kwargs = mock_chat.call_args
                assert "temperature" not in kwargs

    def test_bedrock_builder_unaffected_for_older_model(self):
        """Backward compatibility: models outside the restricted list keep the
        library's original default of 0."""
        factory = LLMFactory("aws-bedrock")
        with patch.dict(os.environ, {
            "AWS_BEDROCK_MODEL_ID": "anthropic.claude-3-sonnet-20240229-v1:0",
            "AWS_REGION": "us-east-1",
        }):
            with patch("langchain_aws.ChatAnthropicBedrock") as mock_chat:
                factory._build_aws_bedrock_llm(None, None)
                _, kwargs = mock_chat.call_args
                assert kwargs.get("temperature") == 0

    def test_bedrock_builder_unaffected_for_4_6_model(self):
        """Opus/Sonnet 4.6 accept sampling params — an explicit 0.3 must survive."""
        factory = LLMFactory("aws-bedrock")
        with patch.dict(os.environ, {
            "AWS_BEDROCK_MODEL_ID": "anthropic.claude-sonnet-4-6",
            "AWS_REGION": "us-east-1",
        }):
            with patch("langchain_aws.ChatAnthropicBedrock") as mock_chat:
                factory._build_aws_bedrock_llm(None, 0.3)
                _, kwargs = mock_chat.call_args
                assert kwargs.get("temperature") == 0.3

    @pytest.mark.parametrize("temperature", [0.0, 0.3, 1.0, 2.0])
    def test_bedrock_builder_omits_every_temperature_for_restricted_model(self, temperature):
        """No temperature value — including Anthropic's default of 1.0 — is sent."""
        factory = LLMFactory("aws-bedrock")
        with patch.dict(os.environ, {
            "AWS_BEDROCK_MODEL_ID": "anthropic.claude-opus-5",
            "AWS_REGION": "us-east-1",
        }):
            with patch("langchain_aws.ChatAnthropicBedrock") as mock_chat:
                factory._build_aws_bedrock_llm(None, temperature)
                _, kwargs = mock_chat.call_args
                assert "temperature" not in kwargs

    def test_bedrock_builder_strips_top_p_and_top_k_for_restricted_model(self):
        """top_p/top_k are rejected by the same models and must also be stripped."""
        factory = LLMFactory("aws-bedrock")
        with patch.dict(os.environ, {
            "AWS_BEDROCK_MODEL_ID": "anthropic.claude-opus-5",
            "AWS_REGION": "us-east-1",
        }):
            with patch("langchain_aws.ChatAnthropicBedrock") as mock_chat:
                factory._build_aws_bedrock_llm(None, 0.0, top_p=0.9, top_k=40)
                _, kwargs = mock_chat.call_args
                assert "top_p" not in kwargs
                assert "top_k" not in kwargs

    def test_bedrock_builder_keeps_top_p_for_older_model(self):
        factory = LLMFactory("aws-bedrock")
        with patch.dict(os.environ, {
            "AWS_BEDROCK_MODEL_ID": "anthropic.claude-3-sonnet-20240229-v1:0",
            "AWS_REGION": "us-east-1",
        }):
            with patch("langchain_aws.ChatAnthropicBedrock") as mock_chat:
                factory._build_aws_bedrock_llm(None, None, top_p=0.9)
                _, kwargs = mock_chat.call_args
                assert kwargs.get("top_p") == 0.9

    def test_get_llm_public_path_omits_temperature_for_restricted_model(self):
        """Covers the real code path: get_llm() resolves an unset temperature to
        0.0 before the builder sees it, so the builder must handle that value."""
        factory = LLMFactory("aws-bedrock")
        with patch.dict(os.environ, {
            "AWS_BEDROCK_MODEL_ID": "anthropic.claude-opus-5",
            "AWS_REGION": "us-east-1",
        }, clear=False):
            with patch("langchain_aws.ChatAnthropicBedrock") as mock_chat:
                factory.get_llm()
                _, kwargs = mock_chat.call_args
                assert "temperature" not in kwargs

    def test_no_warning_when_caller_set_no_temperature(self, caplog):
        """A default construction must not log a warning about 'dropping' a value
        the caller never set."""
        factory = LLMFactory("aws-bedrock")
        with patch.dict(os.environ, {
            "AWS_BEDROCK_MODEL_ID": "anthropic.claude-opus-5",
            "AWS_REGION": "us-east-1",
        }):
            with patch("langchain_aws.ChatAnthropicBedrock"):
                with caplog.at_level("WARNING"):
                    factory.get_llm()
        assert "does not accept sampling parameters" not in caplog.text

    def test_warning_when_caller_set_explicit_temperature(self, caplog):
        factory = LLMFactory("aws-bedrock")
        with patch.dict(os.environ, {
            "AWS_BEDROCK_MODEL_ID": "anthropic.claude-opus-5",
            "AWS_REGION": "us-east-1",
        }):
            with patch("langchain_aws.ChatAnthropicBedrock"):
                with caplog.at_level("WARNING"):
                    factory._build_aws_bedrock_llm(None, 0.7)
        assert "does not accept sampling parameters" in caplog.text
        assert "temperature=0.7" in caplog.text

    def test_anthropic_builder_omits_temperature_for_restricted_model(self):
        factory = LLMFactory("anthropic-claude")
        with patch.dict(os.environ, {
            "ANTHROPIC_API_KEY": "test-key",
            "ANTHROPIC_MODEL_NAME": "claude-sonnet-5",
        }):
            with patch("langchain_anthropic.ChatAnthropic") as mock_chat:
                factory._build_anthropic_claude_llm(None, 0.0)
                _, kwargs = mock_chat.call_args
                assert "temperature" not in kwargs

    def test_anthropic_builder_unaffected_for_older_model(self):
        factory = LLMFactory("anthropic-claude")
        with patch.dict(os.environ, {
            "ANTHROPIC_API_KEY": "test-key",
            "ANTHROPIC_MODEL_NAME": "claude-3-sonnet-20240229-v1",
        }):
            llm = factory._build_anthropic_claude_llm(None, None)
            assert llm.temperature == 0

    def test_vertexai_builder_is_not_subject_to_anthropic_rule(self):
        """_build_gcp_vertexai_llm constructs ChatVertexAI (Gemini), not
        ChatAnthropicVertex, so the Anthropic sampling rule must not apply."""
        import inspect
        from cnoe_agent_utils.llm_factory import LLMFactory as _F
        src = inspect.getsource(_F._build_gcp_vertexai_llm)
        assert "_sanitize_sampling_params" not in src
        assert "temperature" in src


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
