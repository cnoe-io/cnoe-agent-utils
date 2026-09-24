# Copyright 2025 CNOE
# SPDX-License-Identifier: Apache-2.0

"""Global context management configuration for all agent types."""

import logging
import os
from collections.abc import Mapping
from typing import Any

logger = logging.getLogger(__name__)

# Safety margin applied to a model's advertised input window, leaving room
# for tool definitions and response generation.
DEFAULT_CONTEXT_LIMIT_FRACTION = 0.85

# Used only when a model has no resolvable profile (e.g. a Bedrock
# application-inference-profile ARN with no GetInferenceProfile permission,
# or a Strands model that isn't a LangChain BaseChatModel at all).
DEFAULT_CONTEXT_LIMIT_FALLBACK = 200_000


def get_context_limit_for_model(
    model: Any,
    *,
    fraction: float = DEFAULT_CONTEXT_LIMIT_FRACTION,
    fallback: int = DEFAULT_CONTEXT_LIMIT_FALLBACK,
) -> int:
    """
    Get the context token limit for a specific chat model instance.

    Reads `model.profile["max_input_tokens"]`, populated by LangChain/partner
    packages (langchain-aws, langchain-anthropic, langchain-openai, etc.) from
    their own model catalogs, so the limit always matches the model actually
    configured instead of a hand-maintained per-provider guess.

    Args:
        model: A LangChain chat model instance (or `None`/anything without a
            resolvable `.profile`, in which case `fallback` is used).
        fraction: Safety margin applied to `max_input_tokens`.
        fallback: Value to use when no profile is available.

    Returns:
        Context token limit as integer.

    Examples:
        >>> from langchain_anthropic import ChatAnthropic
        >>> get_context_limit_for_model(ChatAnthropic(model="claude-sonnet-4-5"))
        170000

        >>> get_context_limit_for_model(None)
        200000
    """
    profile = getattr(model, "profile", None)
    if isinstance(profile, Mapping):
        max_input_tokens = profile.get("max_input_tokens")
        if isinstance(max_input_tokens, int) and not isinstance(max_input_tokens, bool) and max_input_tokens > 0:
            limit = max(1, int(max_input_tokens * fraction))
            logger.info(
                f"Using model profile context limit: {limit:,} tokens "
                f"({fraction:.0%} of {max_input_tokens:,} max_input_tokens)"
            )
            return limit

    logger.debug(f"No resolvable model profile; using fallback context limit: {fallback:,} tokens")
    return fallback


def get_min_messages_to_keep() -> int:
    """
    Get the minimum number of recent messages to always keep.

    Returns:
        Minimum messages to keep (default: 10)
    """
    try:
        return int(os.getenv("MIN_MESSAGES_TO_KEEP", "10"))
    except ValueError:
        logger.warning(
            f"Invalid value for MIN_MESSAGES_TO_KEEP='{os.getenv('MIN_MESSAGES_TO_KEEP')}', "
            "using default: 10"
        )
        return 10


def is_auto_compression_enabled() -> bool:
    """
    Check if auto-compression is enabled.

    Returns:
        True if enabled (default), False otherwise
    """
    return os.getenv("ENABLE_AUTO_COMPRESSION", "true").lower() == "true"
