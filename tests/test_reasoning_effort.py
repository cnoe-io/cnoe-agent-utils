"""Tests for portable reasoning-effort configuration."""

import json
import os
from unittest.mock import MagicMock, patch

import pytest

from cnoe_agent_utils.llm_factory import LLMFactory, resolve_reasoning_effort


@pytest.mark.parametrize(
    ("provider", "model", "effort", "native", "budget"),
    [
        ("openai", "gpt-5.6", "max", "max", None),
        ("openai", "gpt-5.5", "max", "xhigh", None),
        ("azure-openai", "gpt-5", "max", "high", None),
        ("azure-openai", "gpt-5-pro", "max", "high", None),
        ("anthropic-claude", "claude-sonnet-4-5", "medium", None, 4096),
        ("anthropic-claude", "claude-opus-4-5", "max", "high", 16384),
        ("anthropic-claude", "claude-sonnet-4-6", "max", "max", None),
        ("anthropic-claude", "claude-mythos-preview", "max", "max", None),
        ("aws-bedrock", "anthropic.claude-sonnet-4-5", "max", None, 16384),
        ("aws-bedrock", "anthropic.claude-sonnet-5", "max", "high", None),
        ("aws-bedrock", "anthropic.claude-sonnet-4-6", "max", "max", None),
        ("google-gemini", "gemini-3-pro", "max", "high", None),
        ("gcp-vertexai", "gemini-2.5-pro", "low", None, 1024),
        ("gcp-vertexai", "gemini-2.5-pro", "medium", None, 8192),
        ("gcp-vertexai", "gemini-2.5-pro", "max", None, 32768),
        ("gcp-vertexai", "gemini-2.5-flash", "max", None, 24576),
        ("groq", "openai/gpt-oss-120b", "max", "high", None),
    ],
)
def test_resolves_portable_effort(provider, model, effort, native, budget):
    resolution = resolve_reasoning_effort(provider, model, effort)

    assert resolution.supported is True
    assert resolution.native_effort == native
    assert resolution.thinking_budget == budget


def test_unknown_model_is_not_assumed_to_support_effort():
    resolution = resolve_reasoning_effort("openai", "gpt-4.1", "medium")

    assert resolution.supported is False
    assert resolution.reason


def test_older_claude_is_not_assumed_to_support_thinking():
    resolution = resolve_reasoning_effort(
        "anthropic-claude",
        "claude-3-5-sonnet-latest",
        "medium",
    )

    assert resolution.supported is False


@pytest.mark.parametrize(
    ("provider", "model", "effort"),
    [
        ("openai", "gpt-5-pro", "medium"),
        ("azure-openai", "o1-mini", "high"),
    ],
)
def test_model_specific_unsupported_effort(provider, model, effort):
    resolution = resolve_reasoning_effort(provider, model, effort)

    assert resolution.supported is False
    assert resolution.reason


def test_longest_custom_model_prefix_wins():
    mappings = {
        "openai:private-": {"max": "high"},
        "openai:private-reasoner": {"max": "xhigh"},
    }
    with patch.dict(os.environ, {"LLM_REASONING_EFFORT_MAP_JSON": json.dumps(mappings)}):
        resolution = resolve_reasoning_effort("openai", "private-reasoner-v2", "max")

    assert resolution.supported is True
    assert resolution.native_effort == "xhigh"


def test_explicit_effort_is_passed_to_builder():
    factory = object.__new__(LLMFactory)
    factory.provider = "openai"
    factory._get_default_temperature = MagicMock(return_value=0.0)
    factory._build_openai_llm = MagicMock(return_value=MagicMock())

    factory.get_llm(model="gpt-5.5", reasoning_effort="high")

    factory._build_openai_llm.assert_called_once_with(
        None, 0.0, model_override="gpt-5.5", reasoning_effort="high"
    )


def test_omitted_effort_preserves_builder_contract():
    factory = object.__new__(LLMFactory)
    factory.provider = "openai"
    factory._get_default_temperature = MagicMock(return_value=0.0)
    factory._build_openai_llm = MagicMock(return_value=MagicMock())

    factory.get_llm(model="gpt-5.5")

    factory._build_openai_llm.assert_called_once_with(None, 0.0, model_override="gpt-5.5")


@patch("cnoe_agent_utils.llm_factory._LANGCHAIN_OPENAI_AVAILABLE", True)
def test_azure_passes_explicit_reasoning_to_client():
    client = MagicMock()
    environment = {
        "AZURE_OPENAI_DEPLOYMENT": "gpt-5.5",
        "AZURE_OPENAI_API_VERSION": "2025-04-01-preview",
        "AZURE_OPENAI_ENDPOINT": "https://example.openai.azure.com",
        "AZURE_OPENAI_API_KEY": "test-key",
    }
    with (
        patch.dict(os.environ, environment, clear=True),
        patch("langchain_openai.AzureChatOpenAI", client),
    ):
        LLMFactory("azure-openai")._build_azure_openai_llm(
            None, 0.0, reasoning_effort="max"
        )

    assert client.call_args.kwargs["reasoning_effort"] == "xhigh"
    assert client.call_args.kwargs["use_responses_api"] is True


@patch("cnoe_agent_utils.llm_factory._LANGCHAIN_ANTHROPIC_AVAILABLE", True)
def test_anthropic_adaptive_model_uses_native_effort():
    client = MagicMock()
    environment = {
        "ANTHROPIC_API_KEY": "test-key",
        "ANTHROPIC_MODEL_NAME": "claude-sonnet-4-6",
    }
    with (
        patch.dict(os.environ, environment, clear=True),
        patch("langchain_anthropic.ChatAnthropic", client),
    ):
        LLMFactory("anthropic-claude")._build_anthropic_claude_llm(
            None,
            0.0,
            reasoning_effort="max",
        )

    assert client.call_args.kwargs["thinking"] == {"type": "adaptive"}
    assert client.call_args.kwargs["effort"] == "max"


@patch("cnoe_agent_utils.llm_factory._LANGCHAIN_ANTHROPIC_AVAILABLE", True)
def test_anthropic_opus_45_combines_effort_with_manual_budget():
    client = MagicMock()
    environment = {
        "ANTHROPIC_API_KEY": "test-key",
        "ANTHROPIC_MODEL_NAME": "claude-opus-4-5",
    }
    with (
        patch.dict(os.environ, environment, clear=True),
        patch("langchain_anthropic.ChatAnthropic", client),
    ):
        LLMFactory("anthropic-claude")._build_anthropic_claude_llm(
            None,
            0.0,
            reasoning_effort="max",
        )

    assert client.call_args.kwargs["effort"] == "high"
    assert client.call_args.kwargs["model_kwargs"]["thinking_budget"] == 16384
    assert "thinking" not in client.call_args.kwargs


@patch("cnoe_agent_utils.llm_factory._LANGCHAIN_GOOGLE_GENAI_AVAILABLE", True)
def test_gemini_25_uses_documented_thinking_budget():
    client = MagicMock()
    environment = {
        "GOOGLE_API_KEY": "test-key",
        "GOOGLE_GEMINI_MODEL_NAME": "gemini-2.5-pro",
    }
    with (
        patch.dict(os.environ, environment, clear=True),
        patch("langchain_google_genai.ChatGoogleGenerativeAI", client),
    ):
        LLMFactory("google-gemini")._build_google_gemini_llm(
            None,
            0.0,
            reasoning_effort="medium",
        )

    assert client.call_args.kwargs["thinking_budget"] == 8192


@patch("cnoe_agent_utils.llm_factory._LANGCHAIN_GROQ_AVAILABLE", True)
def test_groq_uses_native_reasoning_effort():
    client = MagicMock()
    environment = {
        "GROQ_API_KEY": "test-key",
        "GROQ_MODEL_NAME": "openai/gpt-oss-120b",
    }
    with (
        patch.dict(os.environ, environment, clear=True),
        patch("langchain_groq.ChatGroq", client),
    ):
        LLMFactory("groq")._build_groq_llm(
            None,
            0.0,
            reasoning_effort="max",
        )

    assert client.call_args.kwargs["reasoning_effort"] == "high"


@patch("cnoe_agent_utils.llm_factory._LANGCHAIN_AWS_AVAILABLE", True)
def test_bedrock_converse_uses_adaptive_effort_fields():
    client = MagicMock()
    environment = {
        "AWS_ACCESS_KEY_ID": "test-key",
        "AWS_SECRET_ACCESS_KEY": "test-secret",
        "AWS_BEDROCK_MODEL_ID": "global.anthropic.claude-sonnet-4-6",
        "AWS_REGION": "us-east-1",
    }
    with (
        patch.dict(os.environ, environment, clear=True),
        patch("langchain_aws.ChatBedrockConverse", client),
        patch(
            "cnoe_agent_utils.llm_factory.resolve_bedrock_client",
            return_value="converse",
        ),
    ):
        LLMFactory("aws-bedrock")._build_aws_bedrock_llm(
            None,
            0.0,
            reasoning_effort="max",
        )

    request_fields = client.call_args.kwargs["additional_model_request_fields"]
    assert request_fields["thinking"] == {"type": "adaptive"}
    assert request_fields["output_config"] == {"effort": "max"}
