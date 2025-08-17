# Copyright 2025 CNOE
# SPDX-License-Identifier: Apache-2.0
"""
===============================================================================
 test_azure_openai.py - Example usage of Azure OpenAI via cnoe-agent-utils
===============================================================================

This open source example demonstrates how to use the `cnoe-agent-utils` library
to interact with an Azure OpenAI deployment. It loads required Azure OpenAI
configuration from environment variables, which can be set directly or loaded
from a `.env` file in the project directory. The script initializes the
LLMFactory for Azure OpenAI and invokes the model with a sample prompt.

Environment Variables (can be set in a .env file):
  - AZURE_OPENAI_API_KEY: Azure OpenAI API key.
  - AZURE_OPENAI_API_VERSION: Azure OpenAI API version (e.g., 2023-05-15).
  - AZURE_OPENAI_DEPLOYMENT: Azure OpenAI deployment name.
  - AZURE_OPENAI_ENDPOINT: Azure OpenAI endpoint URL.

Raises:
  EnvironmentError: If any required environment variables are missing.

Example:
  $ echo "AZURE_OPENAI_API_KEY=your-api-key" >> .env
  $ echo "AZURE_OPENAI_API_VERSION=2023-05-15" >> .env
  $ echo "AZURE_OPENAI_DEPLOYMENT=your-deployment" >> .env
  $ echo "AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/" >> .env
  $ python test_azure_openai.py

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

  # Check for required Azure OpenAI environment variables
  required_vars = [
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_API_VERSION",
    "AZURE_OPENAI_DEPLOYMENT",
    "AZURE_OPENAI_ENDPOINT"
  ]
  for var in required_vars:
    if not os.getenv(var):
      missing.append(var)

  # If any required variables are missing, print and raise an error
  if missing:
    for var in missing:
      print(f"Missing required Azure OpenAI environment variable: {var}")
    raise EnvironmentError(
      f"Missing required Azure OpenAI environment variable(s): {', '.join(missing)}"
    )

  # Initialize the LLMFactory for OpenAI
  factory = LLMFactory("azure-openai")

  # Get the LLM instance
  llm = factory.get_llm()

  # Invoke the model with a prompt and print the response
  response = llm.invoke("Hello, Are you Azure OpenAI?")

  # Parse and print the response content and token usage
  print(f"Response content: {response.content}")

if __name__ == "__main__":
  main()
