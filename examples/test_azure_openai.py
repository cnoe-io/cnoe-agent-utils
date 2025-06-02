"""
===============================================================================
 test_openai.py - Example usage of OpenAI via cnoe-agent-utils
===============================================================================

This open source example demonstrates how to use the `cnoe-agent-utils` library
to interact with an OpenAI deployment. It loads required OpenAI
configuration from environment variables, initializes the LLMFactory for
OpenAI, and invokes the model with a sample prompt.

Environment Variables:
  - OPENAI_MODEL_NAME: Name of the OpenAI model (e.g., gpt-3.5-turbo).
  - OPENAI_ENDPOINT: OpenAI endpoint URL (e.g., https://api.openai.com/v1).
  - OPENAI_API_KEY: OpenAI API key.

Raises:
  EnvironmentError: If any required environment variables are missing.

Example:
  $ export OPENAI_MODEL_NAME=gpt-3.5-turbo
  $ export OPENAI_ENDPOINT=https://api.openai.com/v1
  $ export OPENAI_API_KEY=your-api-key
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
    "OPENAI_MODEL_NAME",
    "OPENAI_ENDPOINT",
    "OPENAI_API_KEY"
  ]
  for var in required_vars:
    if not os.getenv(var):
      missing.append(var)

  # If any required variables are missing, print and raise an error
  if missing:
    for var in missing:
      print(f"Missing required OpenAI environment variable: {var}")
    raise EnvironmentError("One or more required OpenAI environment variables are missing.")

  # Initialize the LLMFactory for OpenAI
  factory = LLMFactory("openai")

  # Get the LLM instance
  llm = factory.get_llm()

  # Invoke the model with a prompt and print the response
  response = llm.invoke("Hello, OpenAI!")
  print(response)

if __name__ == "__main__":
  main()
