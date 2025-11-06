# Copyright 2025 CNOE
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import json
import os
from typing import Any, Iterable, Optional, Dict, Literal, TypedDict
import dotenv


# Conditional imports for optional dependencies
try:
    from langchain_aws import ChatBedrock, ChatBedrockConverse
    _LANGCHAIN_AWS_AVAILABLE = True
except ImportError:
    _LANGCHAIN_AWS_AVAILABLE = False
    ChatBedrock = None
    ChatBedrockConverse = None

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

try:
    from langchain_groq import ChatGroq
    _LANGCHAIN_GROQ_AVAILABLE = True
except ImportError:
    _LANGCHAIN_GROQ_AVAILABLE = False
    ChatGroq = None

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

# Extended thinking configuration constants
THINKING_DEFAULT_BUDGET = 1024
THINKING_MIN_BUDGET = 1024

# TypedDict for extended thinking configuration
class ThinkingConfig(TypedDict):
    """Configuration for extended thinking models."""
    type: Literal["enabled"]
    budget_tokens: int

def _parse_thinking_budget(env_var: str, max_tokens: Optional[int] = None) -> int:
    """Parse and validate thinking budget from environment variable.

    Args:
        env_var: Name of the environment variable to read
        max_tokens: Optional maximum tokens limit to clamp budget to

    Returns:
        Validated thinking budget in tokens
    """
    budget_str = os.getenv(env_var, str(THINKING_DEFAULT_BUDGET))

    # Handle comments like temperature parsing does
    if "#" in budget_str:
        budget_str = budget_str.split("#")[0].strip()

    try:
        thinking_budget = int(budget_str)
    except (ValueError, TypeError):
        logging.warning(f"[LLM] Invalid {env_var}='{budget_str}', using {THINKING_DEFAULT_BUDGET}")
        thinking_budget = THINKING_DEFAULT_BUDGET

    # Validate thinking_budget (minimum tokens)
    if thinking_budget < THINKING_MIN_BUDGET:
        logging.warning(f"[LLM] {env_var}={thinking_budget} is below minimum {THINKING_MIN_BUDGET}, using {THINKING_MIN_BUDGET}")
        thinking_budget = THINKING_MIN_BUDGET

    # Validate upper bound if max_tokens is provided
    if max_tokens and thinking_budget > max_tokens:
        logging.warning(f"[LLM] Thinking budget {thinking_budget} exceeds max_tokens {max_tokens}; clamping to {max_tokens}")
        thinking_budget = max_tokens

    return thinking_budget

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

    if _LANGCHAIN_GROQ_AVAILABLE:
        providers.add("groq")

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
    elif provider == "groq" and not _LANGCHAIN_GROQ_AVAILABLE:
        return ["langchain-groq"]
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

  def _get_default_temperature(self) -> float:
    """Get temperature setting from provider-specific environment variable.

    Checks provider-specific environment variables (e.g., BEDROCK_TEMPERATURE,
    OPENAI_TEMPERATURE) with proper ordering to handle Azure before OpenAI.

    Returns:
        float: Temperature value between 0.0 and 2.0, defaulting to 0.0
               (preserves backward compatibility with original behavior)
    """
    provider = self.provider  # Use instance provider (already normalized with underscores)
    temperature = 0.0  # Default temperature (preserves backward compatibility)

    # Provider to environment variable mapping (order matters for substring matching)
    # Note: Overlapping keys (aws/bedrock, google/gemini) provide fallback coverage
    provider_temp_vars = {
        "azure": "AZURE_TEMPERATURE",  # Check Azure before OpenAI
        "openai": "OPENAI_TEMPERATURE",
        "anthropic": "ANTHROPIC_TEMPERATURE",
        "bedrock": "BEDROCK_TEMPERATURE",  # Matches aws_bedrock
        "aws": "BEDROCK_TEMPERATURE",      # Additional fallback
        "google": "GOOGLE_TEMPERATURE",    # Matches google_gemini
        "gemini": "GOOGLE_TEMPERATURE",    # Additional fallback
        "vertex": "VERTEXAI_TEMPERATURE",
        "groq": "GROQ_TEMPERATURE",
    }

    # Find matching provider-specific environment variable
    env_var_to_check = None
    for provider_keyword, env_var in provider_temp_vars.items():
        if provider_keyword in provider:
            env_var_to_check = env_var
            break

    temp_str = os.getenv(env_var_to_check, "0.0") if env_var_to_check else "0.0"

    # Safe parsing with validation
    # Strip comments if present (handle inline comments like "1  # comment")
    if "#" in temp_str:
        temp_str = temp_str.split("#")[0].strip()

    try:
        temperature = float(temp_str)
        # Validate temperature range (most providers support 0.0 to 2.0)
        if temperature < 0.0:
            logging.warning(f"Temperature {temperature} below 0.0, using 0.0")
            temperature = 0.0
        elif temperature > 2.0:
            logging.warning(f"Temperature {temperature} above 2.0, using 2.0")
            temperature = 2.0
    except (ValueError, TypeError):
        logging.warning(f"Invalid temperature value '{temp_str}', using default 0.0")
        temperature = 0.0

    logging.debug(f"Using temperature {temperature} for provider '{provider}'")
    return temperature

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

    If temperature is not specified, it will be read from provider-specific
    environment variables (e.g., BEDROCK_TEMPERATURE, OPENAI_TEMPERATURE).
    Temperature values are validated and clamped to the range 0.0-2.0.
    """
    # Use environment variable if temperature not explicitly provided
    if temperature is None:
        temperature = self._get_default_temperature()
    else:
        # Validate and clamp explicit temperature values
        try:
            temperature = float(temperature)
            if temperature < 0.0:
                logging.warning(f"Temperature {temperature} below 0.0, clamping to 0.0")
                temperature = 0.0
            elif temperature > 2.0:
                logging.warning(f"Temperature {temperature} above 2.0, clamping to 2.0")
                temperature = 2.0
        except (ValueError, TypeError):
            logging.warning(f"Invalid explicit temperature '{temperature}', using default 0.0")
            temperature = 0.0

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
    # Check for extended thinking configuration
    thinking_enabled = _as_bool(os.getenv("AWS_BEDROCK_THINKING_ENABLED"), False)

    # Check for prompt caching configuration
    enable_cache = _as_bool(os.getenv("AWS_BEDROCK_ENABLE_PROMPT_CACHE"), False)

    if enable_cache:
      logging.info(f"[LLM] Prompt caching enabled for Bedrock model={model_id}. Using ChatBedrockConverse.")
      logging.info("[LLM] If model doesn't support caching, AWS Bedrock API will return an error.")

    logging.info(f"[LLM] Bedrock model={model_id} profile={credentials_profile} region={region_name}")

    # Build common args for both ChatBedrock and ChatBedrockConverse
    common_args = {
      "model_id": model_id,
      "temperature": temperature if temperature is not None else 0,
      **kwargs,
    }

    # Handle extended thinking configuration for AWS Bedrock
    if thinking_enabled:
      logging.info("[LLM] Extended thinking enabled for AWS Bedrock")

      # Create model_kwargs dict if it doesn't exist
      model_kwargs = common_args.get("model_kwargs", {})

      # Remove incompatible parameters when using extended thinking
      incompatible_params = []
      if "temperature" in common_args and common_args["temperature"] != 0:
        del common_args["temperature"]
        incompatible_params.append("temperature")
      if "top_p" in kwargs:
        kwargs.pop("top_p")
        incompatible_params.append("top_p")
      if "top_k" in kwargs:
        kwargs.pop("top_k")
        incompatible_params.append("top_k")

      if incompatible_params:
        logging.warning(
          f"[LLM] Extended thinking is not compatible with: {', '.join(incompatible_params)}. These parameters have been removed."
        )

      max_tokens_limit = kwargs.get("max_tokens")
      thinking_budget = _parse_thinking_budget("AWS_BEDROCK_THINKING_BUDGET", max_tokens_limit)

      thinking_config: ThinkingConfig = {"type": "enabled", "budget_tokens": thinking_budget}
      model_kwargs["thinking"] = thinking_config
      logging.info(f"[LLM] Extended thinking enabled with budget_tokens={thinking_budget}")
      logging.info("[LLM] Note: Extended thinking is not compatible with temperature, top_p, top_k, or forced tool use")

      # Update common_args with the model_kwargs
      common_args["model_kwargs"] = model_kwargs

    # Add optional parameters only if they have values
    if aws_access_key_id:
      common_args["aws_access_key_id"] = aws_access_key_id
    if aws_secret_access_key:
      common_args["aws_secret_access_key"] = aws_secret_access_key
    if credentials_profile:
      common_args["credentials_profile_name"] = credentials_profile
    if region_name:
      common_args["region_name"] = region_name
    if provider:
      common_args["provider"] = provider

    # Merge response_format into existing model_kwargs (preserves thinking config)
    if response_format:
      model_kwargs = common_args.get("model_kwargs", {})
      model_kwargs["response_format"] = response_format
      common_args["model_kwargs"] = model_kwargs

    # Use ChatBedrockConverse when caching is enabled (native prompt caching support)
    # Otherwise use ChatBedrock (legacy)
    if enable_cache:
        # ChatBedrockConverse doesn't support 'streaming' parameter
        # Streaming is enabled by default for Converse API
        llm = ChatBedrockConverse(**common_args)
        logging.info("[LLM] Using ChatBedrockConverse with native prompt caching support")
    else:
        # ChatBedrock supports streaming and needs beta_use_converse_api
        streaming = _as_bool(os.getenv("AWS_BEDROCK_STREAMING", os.getenv("LLM_STREAMING", "true")), True)
        use_converse_api = _as_bool(os.getenv("AWS_BEDROCK_USE_CONVERSE_API", "true"), True)
        llm = ChatBedrock(
          **common_args,
          streaming=streaming,
          beta_use_converse_api=use_converse_api
        )
        logging.info("[LLM] Using ChatBedrock")

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

    # Check for extended thinking configuration (Claude 4+ models)
    thinking_enabled = _as_bool(os.getenv("ANTHROPIC_THINKING_ENABLED"), False)
    if thinking_enabled:
      logging.info("[LLM] Extended thinking enabled for Anthropic")

      max_tokens_limit = kwargs.get("max_tokens")
      thinking_budget = _parse_thinking_budget("ANTHROPIC_THINKING_BUDGET", max_tokens_limit)
      model_kwargs["thinking_budget"] = thinking_budget
      logging.info(f"[LLM] Extended thinking configured with thinking_budget={thinking_budget}")

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

  def _build_groq_llm(
    self,
    response_format: str | dict | None,
    temperature: float | None,
    **kwargs,
  ):
    if not _LANGCHAIN_GROQ_AVAILABLE:
      raise ImportError(
        "Groq support requires langchain-groq. "
        "Install with: pip install 'cnoe-agent-utils[groq]'"
      )
    api_key = os.getenv("GROQ_API_KEY")
    model_name = os.getenv("GROQ_MODEL_NAME")

    # Validate required environment variables
    missing_vars = []
    if not api_key:
      missing_vars.append("GROQ_API_KEY")
    if not model_name:
      missing_vars.append("GROQ_MODEL_NAME")
    if missing_vars:
      raise EnvironmentError(
        f"Missing the following Groq environment variable(s): {', '.join(missing_vars)}."
      )

    missing_vars = []
    if not api_key:
      missing_vars.append("GROQ_API_KEY")
    if not model_name:
      missing_vars.append("GROQ_MODEL_NAME")
    if missing_vars:
      raise EnvironmentError(
        f"Missing the following Groq environment variable(s): {', '.join(missing_vars)}."
      )

    logging.info(f"[LLM] Groq model={model_name}")

    # Configure streaming based on global and provider-specific settings
    streaming = _as_bool(os.getenv("GROQ_STREAMING",
                                os.getenv("LLM_STREAMING", "true")), True)

    model_kwargs = {"response_format": response_format} if response_format else {}

    return ChatGroq(
      model_name=model_name,
      groq_api_key=api_key,
      temperature=temperature if temperature is not None else 0,
      streaming=streaming,
      model_kwargs=model_kwargs,
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

    # Check for extended thinking configuration (Claude 4+ models on Vertex AI)
    thinking_enabled = _as_bool(os.getenv("VERTEXAI_THINKING_ENABLED"), False)
    thinking_budget = None
    if thinking_enabled:
      logging.info("[LLM] Extended thinking enabled for Vertex AI")

      max_tokens_limit = kwargs.get("max_tokens")
      thinking_budget = _parse_thinking_budget("VERTEXAI_THINKING_BUDGET", max_tokens_limit)
      logging.info(f"[LLM] Extended thinking configured with thinking_budget={thinking_budget}")

    # Build ChatVertexAI args - don't pass max_tokens as both explicit param and in kwargs
    vertexai_args = {
      "model": model_name,
      "project": project_id,
      "location": location,
      "credentials": credentials,
      "temperature": temperature if temperature is not None else 0,
      "max_retries": 6,
      "stop": None,
      "model_kwargs": model_kwargs,
    }

    # Add thinking_budget as explicit parameter (not in model_kwargs to avoid warning)
    if thinking_budget is not None:
      vertexai_args["thinking_budget"] = thinking_budget

    # Add kwargs except max_tokens which we set explicitly
    filtered_kwargs = {k: v for k, v in kwargs.items() if k != "max_tokens"}
    max_tokens_value = kwargs.get("max_tokens")
    if max_tokens_value is not None:
      vertexai_args["max_tokens"] = max_tokens_value

    return ChatVertexAI(**vertexai_args, **filtered_kwargs)
