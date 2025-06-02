# Copyright 2025 CNOE
# SPDX-License-Identifier: Apache-2.0

"""
===============================================================================
 test_google_gemini.py - Example usage of Google Gemini via cnoe-agent-utils
===============================================================================

This open source example demonstrates how to use the `cnoe-agent-utils` library
to interact with the Gemini model via the Google API. It loads required
configuration from environment variables, initializes the LLMFactory
for Google Gemini, and invokes the model with a sample prompt.

Environment Variables:
  - GOOGLE_API_KEY: Google API key for Gemini access.
  - GOOGLE_GEMINI_MODEL_NAME: (Optional) Gemini model name (default: gemini-2.0-flash).

Raises:
  EnvironmentError: If any required environment variables are missing.

Example:
  $ export GOOGLE_API_KEY=your-google-api-key
  $ export GOOGLE_GEMINI_MODEL_NAME=gemini-2.0-flash
  $ python test_google_gemini.py

Dependencies:
  - cnoe-agent-utils
  - python-dotenv
"""

from __future__ import annotations
import os
import dotenv
import logging
from cnoe_agent_utils import LLMFactory

def main():
  # Load environment variables from .env if present
  dotenv.load_dotenv()

  # Set up logging
  logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [test_google_gemini] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
  )

  # Ensure required env vars are set for Google Gemini
  if not os.getenv("GOOGLE_API_KEY"):
    raise EnvironmentError("GOOGLE_API_KEY must be set in your environment or .env file")

  # Optionally set model name via env var
  os.environ.setdefault("GOOGLE_GEMINI_MODEL_NAME", "gemini-2.0-flash")

  # Create the LLMFactory for Google Gemini
  factory = LLMFactory("google-gemini")
  llm = factory.get_llm(temperature=0.2)

  # Run a simple test prompt
  prompt = "What is the capital of France?"
  response = llm.invoke(prompt)
  print("Prompt:", prompt)
  print("Response:", response.content if hasattr(response, "content") else response)

if __name__ == "__main__":
  main()