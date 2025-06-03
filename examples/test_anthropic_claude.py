# Copyright 2025 CNOE
# SPDX-License-Identifier: Apache-2.0
"""
===============================================================================
 test_anthropic_claude.py - Example usage of Anthropic Claude via cnoe-agent-utils
===============================================================================

This open source example demonstrates how to use the `cnoe-agent-utils` library
to interact with an Anthropic Claude deployment. It loads required Anthropic
configuration from environment variables, which can be set directly or loaded
from a `.env` file in the project directory. The script initializes the
LLMFactory for Anthropic Claude and invokes the model with a sample prompt.

Environment Variables (can be set in a .env file):
  - ANTHROPIC_API_KEY=      Your Anthropic API key.
  - ANTHROPIC_MODEL_NAME=   Name of the Anthropic Claude model (e.g., claude-3-opus-20240229).

Raises:
  EnvironmentError: If any required environment variables are missing.

Example:
  $ echo "ANTHROPIC_API_KEY=your-api-key" >> .env
  $ echo "ANTHROPIC_MODEL_NAME=claude-3-opus-20240229" >> .env
  $ python test_anthropic_claude.py

Dependencies:
  - cnoe-agent-utils
  - python-dotenv
"""

import os
from dotenv import load_dotenv
from cnoe_agent_utils import LLMFactory

def main():
  # Load environment variables from .env file if present
  load_dotenv()

  # List to track missing required Anthropic environment variables
  missing = []

  # Check for required Anthropic environment variables
  required_vars = [
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_MODEL_NAME"
  ]
  for var in required_vars:
    if not os.getenv(var):
      print(f"Missing required environment variable: {var}")
      missing.append(var)

  # If any required variables are missing, raise an error
  if missing:
    raise EnvironmentError(
      f"Missing required environment variable(s): {', '.join(missing)}"
    )

  # Initialize the LLMFactory for Anthropic
  factory = LLMFactory("anthropic-claude")

  # Get the LLM instance
  llm = factory.get_llm()

  # Invoke the model with a prompt and print the response
  response = llm.invoke("Hello, Are you Claude?")

  # Parse and print the response content and token usage
  print(f"Response content: {response.content}")

if __name__ == "__main__":
  main()
