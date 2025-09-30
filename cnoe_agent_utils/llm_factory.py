# Copyright 2025 CNOE
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import json
import os
from typing import Any, Iterable, Optional, Dict
import dotenv


# Conditional imports for optional dependencies
try:
    from langchain_aws import ChatBedrock
    _LANGCHAIN_AWS_AVAILABLE = True
except ImportError:
    _LANGCHAIN_AWS_AVAILABLE = False
    ChatBedrock = None

try:
    from langchain_anthropic import ChatAnthropic
    _LANGCHAIN_ANTHROPIC_AVAILABLE = True
except ImportError:
    _LANGCHAIN_ANTHROPIC_AVAILABLE = False
    ChatAnthropic = None

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    _LANGCHAIN_GOOGLE_GENAI_AVAILABLE = True
except ImportError:
    _LANGCHAIN_GOOGLE_GENAI_AVAILABLE = False
    ChatGoogleGenerativeAI = None

try:
    from langchain_google_vertexai import ChatVertexAI
    _LANGCHAIN_GOOGLE_VERTEXAI_AVAILABLE = True
except ImportError:
    _LANGCHAIN_GOOGLE_VERTEXAI_AVAILABLE = False
    ChatVertexAI = None

try:
    from langchain_openai import AzureChatOpenAI, ChatOpenAI
    _LANGCHAIN_OPENAI_AVAILABLE = True
except ImportError:
    _LANGCHAIN_OPENAI_AVAILABLE = False
    AzureChatOpenAI = None
    ChatOpenAI = None

logging.basicConfig(
  level=logging.INFO,
  format="%(asctime)s %(levelname)s [llm_factory] %(message)s",
  datefmt="%Y-%m-%d %H:%M:%S",
)

_TRUE = {"1","true","t","yes","y","on"}
_FALSE = {"0","false","f","no","n","off"}

def _as_bool(v: Optional[str], default: bool=False) -> bool:
    if v is None:
        return default
    vv = v.strip().lower()
    if vv in _TRUE:
        return True
    if vv in _FALSE:
        return False
    return default

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

  @classmethod
  def get_supported_providers(cls) -> set[str]:
    """Get the list of supported providers based on available dependencies."""
    providers = set()  # Start with empty set

    if _LANGCHAIN_ANTHROPIC_AVAILABLE:
        providers.add("anthropic-claude")

    if _LANGCHAIN_AWS_AVAILABLE:
        providers.add("aws-bedrock")

    if _LANGCHAIN_OPENAI_AVAILABLE:
        providers.add("azure-openai")
        providers.add("openai")

    if _LANGCHAIN_GOOGLE_GENAI_AVAILABLE:
        providers.add("google-gemini")

    if _LANGCHAIN_GOOGLE_VERTEXAI_AVAILABLE:
        providers.add("gcp-vertexai")

    return providers

  @classmethod
  def is_provider_available(cls, provider: str) -> bool:
    """Check if a specific provider is available."""
    return provider in cls.get_supported_providers()

  @classmethod
  def get_missing_dependencies(cls, provider: str) -> list[str]:
    """Get the missing dependencies for a specific provider."""
    if provider == "anthropic-claude" and not _LANGCHAIN_ANTHROPIC_AVAILABLE:
        return ["langchain-anthropic"]
    elif provider == "aws-bedrock" and not _LANGCHAIN_AWS_AVAILABLE:
        return ["langchain-aws", "boto3"]
    elif provider == "openai" and not _LANGCHAIN_OPENAI_AVAILABLE:
        return ["langchain-openai"]
    elif provider == "azure-openai" and not _LANGCHAIN_OPENAI_AVAILABLE:
        return ["langchain-openai"]
    elif provider == "google-gemini" and not _LANGCHAIN_GOOGLE_GENAI_AVAILABLE:
        return ["langchain-google-genai"]
    elif provider == "gcp-vertexai" and not _LANGCHAIN_GOOGLE_VERTEXAI_AVAILABLE:
        return ["langchain-google-vertexai"]
    else:
        return []

  @property
  def SUPPORTED_PROVIDERS(self) -> set[str]:
    """Get supported providers (property for backward compatibility)."""
    return self.get_supported_providers()

  # ------------------------------------------------------------------ #
  # Construction helpers
  # ------------------------------------------------------------------ #

  def __init__(self, provider: str | None = None) -> None:
    dotenv.load_dotenv()
    if provider is None:
      provider = os.getenv("LLM_PROVIDER")
      if provider is None:
        available_providers = self.get_supported_providers()
        raise ValueError(
          f"Provider must be specified as one of: {available_providers}, "
          "or set the LLM_PROVIDER environment variable"
        )
    if provider not in self.SUPPORTED_PROVIDERS:
      available_providers = self.get_supported_providers()
      raise ValueError(
        f"Unsupported provider: {provider}. "
        f"Available providers are: {available_providers}. "
        f"Install missing dependencies with: pip install 'cnoe-agent-utils[aws]' or 'cnoe-agent-utils[gcp]' etc."
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

  @staticmethod
  def _get_cache_supported_models() -> set[str]:
    """Return set of AWS Bedrock models supporting prompt caching.

    Based on AWS documentation:
    https://docs.aws.amazon.com/bedrock/latest/userguide/prompt-caching.html
    """
    return {
      # Claude 4 models (Generally Available)
      "anthropic.claude-opus-4-1-20250805-v1:0",
      "anthropic.claude-opus-4-20250514-v1:0",
      "anthropic.claude-sonnet-4-5-20250929-v1:0",
      "anthropic.claude-sonnet-4-20250514-v1:0",
      # Claude 3.7 models
      "anthropic.claude-3-7-sonnet-20250219-v1:0",
      # Claude 3.5 models
      "anthropic.claude-3-5-haiku-20241022-v1:0",
      "anthropic.claude-3-5-sonnet-20241022-v2:0",
      # Amazon Nova models
      "amazon.nova-micro-v1:0",
      "amazon.nova-lite-v1:0",
      "amazon.nova-pro-v1:0",
      "us.amazon.nova-premier-v1:0",
    }

  def _normalize_model_id(self, model_id: str) -> str:
    """Normalize model ID by removing regional prefixes and version suffixes.

    Handles regional prefixes like 'us.', 'eu.', 'ap.', etc. and version
    suffixes like ':0' to enable consistent model matching across regions.

    Examples:
      - 'us.anthropic.claude-3-5-sonnet-20241022-v2:0' -> 'anthropic.claude-3-5-sonnet-20241022-v2'
      - 'anthropic.claude-3-7-sonnet-20250219-v1:0' -> 'anthropic.claude-3-7-sonnet-20250219-v1'
    """
    # Remove regional prefixes (us., eu., ap., me., ca., sa., af.)
    regional_prefixes = {"us", "eu", "ap", "me", "ca", "sa", "af"}
    parts = model_id.split(".", 1)
    if len(parts) == 2 and parts[0] in regional_prefixes:
      model_id = parts[1]

    # Remove version suffix (:0, :1, etc.)
    model_id = model_id.split(":", 1)[0]

    return model_id

  def _infer_provider(self, model_id: str) -> str | None:
    """Infer provider from model ID prefix.

    Returns appropriate provider based on model ID, or None if cannot infer.
    """
    normalized = self._normalize_model_id(model_id)

    if normalized.startswith("anthropic."):
      return "anthropic"
    elif normalized.startswith("amazon."):
      return "amazon"

    return None

  def _is_cache_supported(self, model_id: str) -> bool:
    """Check if the given model ID supports prompt caching."""
    normalized = self._normalize_model_id(model_id)
    supported_models = self._get_cache_supported_models()

    # Direct match on normalized IDs
    if normalized in supported_models:
      return True

    # Family prefix match (e.g., "anthropic.claude-3-5-sonnet" matches any version)
    for supported in supported_models:
      # Get family prefix by removing last 2 segments (date and version)
      family = supported.rsplit("-", 2)[0]
      if normalized.startswith(family):
        return True

    return False

  def _build_aws_bedrock_llm(
    self,
    response_format: str | dict | None,
    temperature: float | None,
    **kwargs,
  ):
    if not _LANGCHAIN_AWS_AVAILABLE:
      raise ImportError(
        "AWS Bedrock support requires langchain-aws. "
        "Install with: pip install 'cnoe-agent-utils[aws]'"
      )
    aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    credentials_profile = None
    if not (aws_access_key_id and aws_secret_access_key):
      credentials_profile = os.getenv("AWS_PROFILE") or None
      logging.info("[LLM] Using AWS credentials from profile: %s", credentials_profile)
    else:
      logging.info("[LLM] Using AWS credentials from environment variables")

    model_id = os.getenv("AWS_BEDROCK_MODEL_ID")
    provider = os.getenv("AWS_BEDROCK_PROVIDER")
    region_name = os.getenv("AWS_REGION")

    aws_debug = os.getenv("AWS_CREDENTIALS_DEBUG", "false").lower() == "true"
    if aws_debug:
      import boto3
      try:
        if aws_access_key_id and aws_secret_access_key:
          session = boto3.Session(
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=region_name
          )
        elif credentials_profile:
          session = boto3.Session(profile_name=credentials_profile, region_name=region_name)
        else:
          session = boto3.Session(region_name=region_name)
        sts = session.client("sts", region_name=region_name)
        identity = sts.get_caller_identity()
        arn = identity.get("Arn")
        logging.info(f"[LLM][AWS_DEBUG] STS Arn: {arn}")
      except Exception as e:
        logging.warning(f"[LLM][AWS_DEBUG] Failed to get AWS STS caller identity: {e}")
    missing_vars = []
    if not model_id:
      missing_vars.append("AWS_BEDROCK_MODEL_ID")
    if not region_name:
      missing_vars.append("AWS_REGION")
    if missing_vars:
      raise EnvironmentError(
        f"Missing the following AWS Bedrock environment variable(s): {', '.join(missing_vars)}."
      )
    # Check for prompt caching configuration
    enable_cache = _as_bool(os.getenv("AWS_BEDROCK_ENABLE_PROMPT_CACHE"), False)
    cache_supported = self._is_cache_supported(model_id)

    if enable_cache and cache_supported:
      logging.info(f"[LLM] Prompt caching enabled for Bedrock model={model_id}")
    elif enable_cache and not cache_supported:
      logging.warning(
        f"[LLM] Prompt caching requested but not supported for model={model_id}. "
        f"Supported models: {', '.join(sorted(self._get_cache_supported_models()))}"
      )

    logging.info(f"[LLM] Bedrock model={model_id} profile={credentials_profile} region={region_name}")

    model_kwargs = {"response_format": response_format} if response_format else {}

    # Infer provider from model_id if not explicitly set
    if not provider:
      provider = self._infer_provider(model_id)
      if provider:
        logging.info(f"[LLM] Inferred provider '{provider}' from model_id")

    # Make Converse API usage configurable
    use_converse_api = _as_bool(os.getenv("AWS_BEDROCK_USE_CONVERSE_API", "true"), True)

    bedrock_args = {
      "model_id": model_id,
      "aws_access_key_id": aws_access_key_id,
      "aws_secret_access_key": aws_secret_access_key,
      "region_name": region_name,
      "temperature": temperature if temperature is not None else 0,
      "streaming": _as_bool(os.getenv("AWS_BEDROCK_STREAMING",os.getenv("LLM_STREAMING","true")), True),
      "beta_use_converse_api": use_converse_api,
      "model_kwargs": model_kwargs,
      **kwargs,
    }

    # Only set provider if we have one (inferred or explicit)
    if provider:
      bedrock_args["provider"] = provider
    if credentials_profile:
      bedrock_args["credentials_profile_name"] = credentials_profile
    if region_name:
      bedrock_args["region_name"] = region_name

    llm = ChatBedrock(**bedrock_args)

    # Verify cache support is available if requested
    if enable_cache and cache_supported:
      if not hasattr(llm, 'create_cache_point'):
        logging.warning(
          "[LLM] Prompt caching requested but langchain-aws version doesn't support "
          "create_cache_point(). Please upgrade langchain-aws to enable caching. "
          "See: https://docs.aws.amazon.com/bedrock/latest/userguide/prompt-caching.html"
        )

    return llm

  def _build_anthropic_claude_llm(
    self,
    response_format: str | dict | None,
    temperature: float | None,
    **kwargs,
  ):
    if not _LANGCHAIN_ANTHROPIC_AVAILABLE:
      raise ImportError(
        "Anthropic Claude support requires langchain-anthropic. "
        "Install with: pip install 'cnoe-agent-utils[anthropic]'"
      )
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
    if not _LANGCHAIN_OPENAI_AVAILABLE:
      raise ImportError(
        "Azure OpenAI support requires langchain-openai. "
        "Install with: pip install 'cnoe-agent-utils[azure]'"
      )
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


    # --- GPT-5 support: Responses API + reasoning + streaming ---
    use_responses = _as_bool(os.getenv("AZURE_OPENAI_USE_RESPONSES"),
                            (deployment or "").lower().startswith("gpt-5"))

    reasoning_effort  = os.getenv("AZURE_OPENAI_REASONING_EFFORT")   # low|medium|high
    reasoning_summary = os.getenv("AZURE_OPENAI_REASONING_SUMMARY")  # auto|concise|detailed
    verbosity         = os.getenv("AZURE_OPENAI_VERBOSITY")          # low|medium|high
    os.getenv("AZURE_OPENAI_OUTPUT_VERSION", "responses/v1" if use_responses else "v0")

    streaming = _as_bool(os.getenv("AZURE_OPENAI_STREAMING",
                                  os.getenv("LLM_STREAMING","true")), True)

    model_kwargs: Dict[str, Any] = {"response_format": response_format} if response_format else {}
    if verbosity:
        model_kwargs["verbosity"] = verbosity
    extra_body: Dict[str, Any] = {}
    if use_responses and (reasoning_effort or reasoning_summary):
        extra_body["reasoning"] = {
            **({"effort": reasoning_effort} if reasoning_effort else {}),
            **({"summary": reasoning_summary} if reasoning_summary else {}),
        }

    # For GPT-5 models, don't set temperature if it's 0.0 (use default)
    kwargs_to_pass = {}
    if temperature is not None and temperature != 0.0:
        kwargs_to_pass["temperature"] = temperature
    elif not (deployment or "").lower().startswith("gpt-5"):
        # For non-GPT-5 models, set temperature to 0 if not specified
        kwargs_to_pass["temperature"] = 0

    return AzureChatOpenAI(
        azure_endpoint=endpoint,
        azure_deployment=deployment,
        model=deployment,  # Add model parameter for newer LangChain versions
        api_key=api_key,
        api_version=api_version,
        streaming=streaming,
        **kwargs_to_pass,
        **kwargs,
      )

  def _build_openai_llm(
    self,
    response_format: str | dict | None,
    temperature: float | None,
    **kwargs,
  ):
    if not _LANGCHAIN_OPENAI_AVAILABLE:
      raise ImportError(
        "OpenAI (openai.com) support requires langchain-openai. "
        "Install with: pip install 'cnoe-agent-utils[openai]'"
      )
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_ENDPOINT", "https://api.openai.com/v1")
    model_name = os.getenv("OPENAI_MODEL_NAME")
    user = os.getenv("OPENAI_USER")

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

    # --- GPT-5 support: Responses API + reasoning + streaming ---
    use_responses = _as_bool(os.getenv("OPENAI_USE_RESPONSES"),
                            (model_name or "").lower().startswith("gpt-5"))

    reasoning_effort  = os.getenv("OPENAI_REASONING_EFFORT")   # low|medium|high
    reasoning_summary = os.getenv("OPENAI_REASONING_SUMMARY")  # auto|concise|detailed
    verbosity         = os.getenv("OPENAI_VERBOSITY")          # low|medium|high
    os.getenv("OPENAI_OUTPUT_VERSION", "responses/v1" if use_responses else "v0")

    streaming = _as_bool(os.getenv("OPENAI_STREAMING",
                                  os.getenv("LLM_STREAMING","true")), True)

    model_kwargs: Dict[str, Any] = {"response_format": response_format} if response_format else {}
    if verbosity:
        model_kwargs["verbosity"] = verbosity
    if user:
        model_kwargs["user"] = user

    extra_body: Dict[str, Any] = {}
    if use_responses and (reasoning_effort or reasoning_summary):
        extra_body["reasoning"] = {
            **({"effort": reasoning_effort} if reasoning_effort else {}),
            **({"summary": reasoning_summary} if reasoning_summary else {}),
        }

    # Build kwargs for ChatOpenAI
    openai_kwargs = {
        "model_name": model_name,
        "api_key": api_key,
        "base_url": base_url,
        "use_responses_api": use_responses,
        "streaming": streaming,
    }

    # Only add model_kwargs and extra_body if they have content
    if model_kwargs:
        openai_kwargs["model_kwargs"] = model_kwargs
    if extra_body:
        openai_kwargs["extra_body"] = extra_body

    # Only set temperature when supported (GPT-5 doesn't support temperature=0.0)
    if (model_name or "").lower().startswith("gpt-5"):
        # For GPT-5 models, don't set temperature if it's 0.0 (use default)
        if temperature is not None and temperature != 0.0:
            openai_kwargs["temperature"] = temperature
    else:
        # For non-GPT-5 models, set temperature normally
        openai_kwargs["temperature"] = temperature if temperature is not None else 0

    # Add headers support from the other branch
    openai_headers = os.getenv("OPENAI_DEFAULT_HEADERS")
    if openai_headers:
        try:
            headers = json.loads(openai_headers)
            openai_kwargs["default_headers"] = headers
        except Exception as e:
            logging.warning(f"[LLM] Could not parse OPENAI_HEADERS env var from JSON: {e}")

    # Don't pass output_version to avoid conflicts with underlying OpenAI client
    # LangChain handles this internally

    return ChatOpenAI(
        default_headers=openai_headers,
        **openai_kwargs,
        **kwargs,
    )

  def _build_google_gemini_llm(
    self,
    response_format: str | dict | None,
    temperature: float | None,
    **kwargs,
  ):
    if not _LANGCHAIN_GOOGLE_GENAI_AVAILABLE:
      raise ImportError(
        "Google Gemini support requires langchain-google-genai. "
        "Install with: pip install 'cnoe-agent-utils[gcp]'"
      )

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
    if not _LANGCHAIN_GOOGLE_VERTEXAI_AVAILABLE:
      raise ImportError(
        "Google Vertex AI support requires langchain-google-vertexai. "
        "Install with: pip install 'cnoe-agent-utils[gcp]'"
      )
    import google.auth

    # Check for credentials
    os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    try:
      credentials, _ = google.auth.default()
      logging.info("[LLM] Google VertexAI credentials loaded successfully")
    except Exception as e:
      raise EnvironmentError(
        "Could not load Google Cloud credentials. "
        "Set the GOOGLE_APPLICATION_CREDENTIALS environment variable to the path of your service account JSON file. "
        f"Original error: {e}"
      )

    model_name = os.getenv("VERTEXAI_MODEL_NAME")
    if not model_name:
      raise EnvironmentError("VERTEXAI_MODEL_NAME environment variable is required")

    # Get project and location from environment variables
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
    location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")

    if not project_id:
      raise EnvironmentError("GOOGLE_CLOUD_PROJECT environment variable is required for Vertex AI")

    logging.info(f"[LLM] Google VertexAI model={model_name} project={project_id} location={location}")

    model_kwargs = {"response_format": response_format} if response_format else {}
    return ChatVertexAI(
      model=model_name,
      project=project_id,
      location=location,
      credentials=credentials,
      temperature=temperature if temperature is not None else 0,
      max_tokens=None,
      max_retries=6,
      stop=None,
      model_kwargs=model_kwargs,
      **kwargs,
    )
