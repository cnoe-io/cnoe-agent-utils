"""Tests for graceful degradation when bedrock:GetInferenceProfile is denied."""

import os
from unittest.mock import MagicMock, patch

from botocore.exceptions import ClientError

from cnoe_agent_utils.llm_factory import LLMFactory

ARN_MODEL_ID = "arn:aws:bedrock:us-west-2:123456789012:application-inference-profile/abc123"

BASE_ENV = {
    "LLM_PROVIDER": "aws-bedrock",
    "AWS_BEDROCK_MODEL_ID": ARN_MODEL_ID,
    "AWS_REGION": "us-east-1",
    "AWS_ACCESS_KEY_ID": "test_key",
    "AWS_SECRET_ACCESS_KEY": "test_secret",
    "AWS_BEDROCK_CLIENT": "converse",
}


def _access_denied_error() -> ClientError:
    return ClientError(
        error_response={"Error": {"Code": "AccessDeniedException", "Message": "denied"}},
        operation_name="GetInferenceProfile",
    )


class TestBedrockAipProfileDegradation:
    @patch.dict(os.environ, BASE_ENV)
    @patch("langchain_aws.ChatBedrockConverse")
    def test_retries_with_empty_base_model_id_on_access_denied(self, mock_chat):
        """First construction raises AccessDenied; retry with base_model_id='' succeeds."""
        mock_instance = MagicMock()
        mock_chat.side_effect = [_access_denied_error(), mock_instance]

        factory = LLMFactory("aws-bedrock")
        llm = factory.get_llm()

        assert llm is mock_instance
        assert mock_chat.call_count == 2
        first_kwargs = mock_chat.call_args_list[0].kwargs
        second_kwargs = mock_chat.call_args_list[1].kwargs
        assert "base_model_id" not in first_kwargs
        assert second_kwargs["base_model_id"] == ""
        # Retry preserves every other constructor argument unchanged.
        unrelated_keys = set(first_kwargs) - {"base_model_id"}
        assert all(first_kwargs[k] == second_kwargs[k] for k in unrelated_keys)

    @patch.dict(os.environ, BASE_ENV)
    @patch("langchain_aws.ChatBedrockConverse")
    def test_non_access_denied_client_error_propagates(self, mock_chat):
        """A ClientError unrelated to permissions must not be swallowed."""
        throttling_error = ClientError(
            error_response={"Error": {"Code": "ThrottlingException", "Message": "slow down"}},
            operation_name="GetInferenceProfile",
        )
        mock_chat.side_effect = throttling_error

        factory = LLMFactory("aws-bedrock")
        try:
            factory.get_llm()
            raise AssertionError("expected ClientError to propagate")
        except ClientError as exc:
            assert exc.response["Error"]["Code"] == "ThrottlingException"
        assert mock_chat.call_count == 1

    @patch.dict(os.environ, {**BASE_ENV, "AWS_BEDROCK_MODEL_ID": "us.anthropic.claude-sonnet-4-5-v1:0"})
    @patch("langchain_aws.ChatBedrockConverse")
    def test_non_aip_model_id_never_retries(self, mock_chat):
        """A plain regional model ID never hits the AIP retry path, even on ClientError."""
        mock_chat.side_effect = _access_denied_error()

        factory = LLMFactory("aws-bedrock")
        try:
            factory.get_llm()
            raise AssertionError("expected ClientError to propagate")
        except ClientError:
            pass
        assert mock_chat.call_count == 1

    @patch.dict(os.environ, {**BASE_ENV, "AWS_BEDROCK_BASE_MODEL_ID": "anthropic.claude-sonnet-4-5-v1:0"})
    @patch("langchain_aws.ChatBedrockConverse")
    def test_explicit_base_model_id_never_retries(self, mock_chat):
        """An explicit AWS_BEDROCK_BASE_MODEL_ID skips the retry path entirely."""
        mock_instance = MagicMock()
        mock_chat.return_value = mock_instance

        factory = LLMFactory("aws-bedrock")
        llm = factory.get_llm()

        assert llm is mock_instance
        assert mock_chat.call_count == 1
        assert mock_chat.call_args.kwargs["base_model_id"] == "anthropic.claude-sonnet-4-5-v1:0"
