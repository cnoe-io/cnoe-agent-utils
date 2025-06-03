# Copyright 2025 CNOE
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import os
from typing import Any, Iterable
import dotenv

from langchain_aws import ChatBedrock
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_vertexai import ChatVertexAI
from langchain_openai import AzureChatOpenAI, ChatOpenAI
import subprocess
import json

logging.basicConfig(
  level=logging.INFO,
  format="%(asctime)s %(levelname)s [llm_factory] %(message)s",
  datefmt="%Y-%m-%d %H:%M:%S",
)

class LLMFactory:
  """Factory that returns a *ready‑to‑use* LangChain chat model.

  Parameters
  ----------
  provider : str, optional
      Which LLM backend to use. See SUPPORTED_PROVIDERS for the list of
      supported providers. If not specified, the provider is read from
      the environment variable ``LLM_PROVIDER``. If that variable is not
      set, a ``ValueError`` is raised.

  Raises
  ------
  ValueError
      If the specified provider is not supported or if no provider is
      specified and the environment variable ``LLM_PROVIDER`` is not set.
  EnvironmentError
      If the required environment variables for the selected provider are
      not set (e.g., API keys, deployment names, etc.).
  """

  SUPPORTED_PROVIDERS = {
    "anthropic-claude",
    "aws-bedrock",
    "azure-openai",
    "google-gemini",
    "gcp-vertexai",
    "openai",
    "mistral"
  }

  # ------------------------------------------------------------------ #
  # Construction helpers
  # ------------------------------------------------------------------ #

  def __init__(self, provider: str | None = None) -> None:
    dotenv.load_dotenv()
    if provider is None:
      provider = os.getenv("LLM_PROVIDER")
      if provider is None:
        raise ValueError(
          "Provider must be specified as one of: azure-openai, openai, anthropic-claude, "
          "or set the LLM_PROVIDER environment variable"
        )
    if provider not in self.SUPPORTED_PROVIDERS:
      raise ValueError(
        f"Unsupported provider: {self.provider}. Supported providers are: {self.SUPPORTED_PROVIDERS}"
      )
    self.provider = provider.lower().replace("-", "_")

  # ------------------------------------------------------------------ #
  # Public helpers
  # ------------------------------------------------------------------ #

  def get_llm(
    self,
    response_format: str | dict | None = None,
    tools: Iterable[Any] | None = None,
    strict_tools: bool = True,
    temperature: float | None = None,
    **kwargs,
  ):
    """Return a LangChain chat model, optionally bound to *tools*.

    The returned object is an instance of ``ChatOpenAI``,
    ``AzureChatOpenAI`` or ``ChatAnthropic`` depending on the selected
    *provider*.
    """

    builder = getattr(self, f"_build_{self.provider}_llm")
    llm = builder(response_format, temperature, **kwargs)
    return llm.bind_tools(tools, strict=strict_tools) if tools else llm

  # ------------------------------------------------------------------ #
  # Internal builders (one per provider)
  # ------------------------------------------------------------------ #

  def _build_aws_bedrock_llm(
    self,
    response_format: str | dict | None,
    temperature: float | None,
    **kwargs,
  ):
    credentials_profile = os.getenv("AWS_PROFILE") or None
    model_id = os.getenv("AWS_BEDROCK_MODEL_ID")
    provider = os.getenv("AWS_BEDROCK_PROVIDER")
    region_name = os.getenv("AWS_REGION")

    missing_vars = []
    if not model_id:
      missing_vars.append("AWS_BEDROCK_MODEL_ID")
    if not region_name:
      missing_vars.append("AWS_REGION")
    if missing_vars:
      raise EnvironmentError(
        f"Missing the following AWS Bedrock environment variable(s): {', '.join(missing_vars)}."
      )
    logging.info(f"[LLM] Bedrock model={model_id} profile={credentials_profile} region={region_name}")

    model_kwargs = {"response_format": response_format} if response_format else {}

    bedrock_args = {
      "model_id": model_id,
      "provider": provider if provider else "amazon",
      "region_name": region_name,
      "temperature": temperature if temperature is not None else 0,
      "streaming": True,
      "beta_use_converse_api": True,  # Use Converse API for better performance
      "model_kwargs": model_kwargs,
      **kwargs,
    }
    if credentials_profile:
      bedrock_args["credentials_profile_name"] = credentials_profile
    if region_name:
      bedrock_args["region_name"] = region_name

    return ChatBedrock(**bedrock_args)

  def _build_anthropic_claude_llm(
    self,
    response_format: str | dict | None,
    temperature: float | None,
    **kwargs,
  ):
    api_key = os.getenv("ANTHROPIC_API_KEY")
    model_name = os.getenv("ANTHROPIC_MODEL_NAME")

    if not api_key:
      raise EnvironmentError("ANTHROPIC_API_KEY environment variable is required")

    if not model_name:
      raise EnvironmentError("ANTHROPIC_MODEL_NAME environment variable is required")

    logging.info(f"[LLM] Anthropic model={model_name}")

    model_kwargs = {"response_format": response_format} if response_format else {}
    return ChatAnthropic(
      model_name=model_name,
      anthropic_api_key=api_key,
      temperature=temperature if temperature is not None else 0,
      model_kwargs=model_kwargs,
      **kwargs,
    )

  def _build_azure_openai_llm(
    self,
    response_format: str | dict | None,
    temperature: float | None,
    **kwargs,
  ):
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")

    missing_vars = []
    if not deployment:
      missing_vars.append("AZURE_OPENAI_DEPLOYMENT")
    if not api_version:
      missing_vars.append("AZURE_OPENAI_API_VERSION")
    if not endpoint:
      missing_vars.append("AZURE_OPENAI_ENDPOINT")
    if not api_key:
      missing_vars.append("AZURE_OPENAI_API_KEY")
    if missing_vars:
      raise EnvironmentError(
        f"Missing the following Azure OpenAI environment variable(s): {', '.join(missing_vars)}."
      )

    logging.info(
      f"[LLM] AzureOpenAI deployment={deployment} api_version={api_version}"
    )

    model_kwargs = {"response_format": response_format} if response_format else {}
    return AzureChatOpenAI(
      azure_endpoint=endpoint,
      azure_deployment=deployment,
      openai_api_key=api_key,
      api_version=api_version,
      temperature=temperature if temperature is not None else 0,
      max_tokens=None,
      timeout=None,
      max_retries=5,
      model_kwargs=model_kwargs,
      **kwargs,
    )

  def _build_openai_llm(
    self,
    response_format: str | dict | None,
    temperature: float | None,
    **kwargs,
  ):
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_ENDPOINT", "https://api.openai.com/v1")
    model_name = os.getenv("OPENAI_MODEL_NAME")

    missing_vars = []
    if not api_key:
      missing_vars.append("OPENAI_API_KEY")
    if not base_url:
      missing_vars.append("OPENAI_ENDPOINT")
    if not model_name:
      missing_vars.append("OPENAI_MODEL_NAME")
    if missing_vars:
      raise EnvironmentError(
      f"Missing the following OpenAI environment variable(s): {', '.join(missing_vars)}."
      )

    logging.info(f"[LLM] OpenAI model={model_name} endpoint={base_url}")

    model_kwargs = {"response_format": response_format} if response_format else {}
    return ChatOpenAI(
      model_name=model_name,
      api_key=api_key,
      base_url=base_url,
      temperature=temperature if temperature is not None else 0,
      model_kwargs=model_kwargs,
      **kwargs,
    )


  def _build_google_gemini_llm(
    self,
    response_format: str | dict | None,
    temperature: float | None,
    **kwargs,
  ):

    api_key = os.getenv("GOOGLE_API_KEY")
    model_name = os.getenv("GOOGLE_GEMINI_MODEL_NAME", "gemini-2.0-flash")

    if not api_key:
      raise EnvironmentError("GOOGLE_API_KEY environment variable is required")

    logging.info(f"[LLM] Google Gemini model={model_name}")

    model_kwargs = {"response_format": response_format} if response_format else {}
    return ChatGoogleGenerativeAI(
      model=model_name,
      google_api_key=api_key,
      temperature=temperature if temperature is not None else 0,
      model_kwargs=model_kwargs,
      **kwargs,
    )



  def _build_gcp_vertexai_llm(
    self,
    response_format: str | dict | None,
    temperature: float | None,
    **kwargs,
  ):
    import google.auth

    # Check for credentials
    credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    try:
      credentials, _ = google.auth.default()
      logging.info(f"[LLM] Google VertexAI credentials loaded successfully")
    except Exception as e:
      raise EnvironmentError(
        "Could not load Google Cloud credentials. "
        "Set the GOOGLE_APPLICATION_CREDENTIALS environment variable to the path of your service account JSON file. "
        f"Original error: {e}"
      )

    model_name = os.getenv("VERTEXAI_MODEL_NAME")
    if not model_name:
      raise EnvironmentError("VERTEXAI_MODEL_NAME environment variable is required")

    logging.info(f"[LLM] Google VertexAI model={model_name}")

    model_kwargs = {"response_format": response_format} if response_format else {}
    return ChatVertexAI(
      model=model_name,
      credentials=credentials,
      temperature=temperature if temperature is not None else 0,
      max_tokens=None,
      max_retries=6,
      stop=None,
      model_kwargs=model_kwargs,
      **kwargs,
    )