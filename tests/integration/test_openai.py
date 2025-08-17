# Copyright 2025 CNOE
# SPDX-License-Identifier: Apache-2.0
"""
===============================================================================
 test_openai.py - Example usage of OpenAI via cnoe-agent-utils
===============================================================================

This open source example demonstrates how to use the `cnoe-agent-utils` library
to interact with an OpenAI deployment. It loads required OpenAI configuration
from environment variables, which can be set directly or loaded from a `.env`
file in the project directory. The script initializes the LLMFactory for OpenAI
and invokes the model with a sample prompt.

Environment Variables (can be set in a .env file):
  - OPENAI_API_KEY:      Your OpenAI API key.
  - OPENAI_ENDPOINT:     OpenAI endpoint URL (e.g., https://api.openai.com/v1).
  - OPENAI_MODEL_NAME:   Name of the OpenAI model (e.g., gpt-3.5-turbo).

Raises:
  EnvironmentError: If any required environment variables are missing.

Example:
  $ echo "OPENAI_API_KEY=your-api-key" >> .env
  $ echo "OPENAI_ENDPOINT=https://api.openai.com/v1" >> .env
  $ echo "OPENAI_MODEL_NAME=gpt-3.5-turbo" >> .env
  $ python test_openai.py

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

  # List to track missing required OpenAI environment variables
  missing = []

  # Check for required OpenAI environment variables
  required_vars = [
    "OPENAI_API_KEY",
    "OPENAI_ENDPOINT",
    "OPENAI_MODEL_NAME"
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

  # Initialize the LLMFactory for OpenAI
  factory = LLMFactory("openai")

  # Get the LLM instance
  llm = factory.get_llm()

  # Invoke the model with a prompt and print the response
  response = llm.invoke("Hello, Are you OpenAI?")

  # Parse and print the response content and token usage
  print(f"Response content: {response.content}")

if __name__ == "__main__":
  main()
