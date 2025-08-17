# Copyright 2025 CNOE
# SPDX-License-Identifier: Apache-2.0
"""
===============================================================================
 test_gcp_vertex_gemini.py - Example usage of GCP Vertex AI Gemini via cnoe-agent-utils
===============================================================================

This open source example demonstrates how to use the `cnoe-agent-utils` library
to interact with a Gemini model hosted on Google Cloud Vertex AI. It loads required
GCP and model configuration from environment variables, which can be set directly
or loaded from a `.env` file in the project directory. The script initializes the
LLMFactory for Vertex AI and invokes the model with a sample prompt.

Environment Variables (can be set in a .env file):
  - GOOGLE_APPLICATION_CREDENTIALS: Path to your GCP service account JSON key file.
  - VERTEXAI_MODEL_NAME: Name of the Gemini model to use (e.g., "gemini-1.0-pro", "gemini-1.5-pro").

Raises:
  EnvironmentError: If any required environment variables are missing.

Example:
  $ echo "GOOGLE_APPLICATION_CREDENTIALS=/path/to/your/key.json" >> .env
  $ echo "VERTEXAI_MODEL_NAME=gemini-1.0-pro" >> .env
  $ python test_gcp_vertex_gemini.py

Dependencies:
  - cnoe-agent-utils
  - python-dotenv

References:
  - https://cloud.google.com/docs/authentication/application-default-credentials#GAC
  - https://googleapis.dev/python/google-auth/latest/reference/google.auth.html#module-google.auth
"""

import os
import logging
from cnoe_agent_utils import LLMFactory
import dotenv

def main():
  # Load environment variables from .env if present
  dotenv.load_dotenv()

  # Set up logging
  logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [test_gcp_vertex_gemini] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
  )

  # Validate required environment variables and print which are missing
  required_env = ["GOOGLE_APPLICATION_CREDENTIALS", "VERTEXAI_MODEL_NAME"]
  missing = [var for var in required_env if not os.environ.get(var)]
  if missing:
    for var in missing:
      logging.error(f"Environment variable not set: {var}")
    raise EnvironmentError(f"Missing required environment variables: {', '.join(missing)}")

  # Create the LLMFactory for Vertex AI
  factory = LLMFactory("gcp-vertexai")
  llm = factory.get_llm()

  # Run a simple test prompt
  prompt = "What is the capital of France?"
  response = llm.invoke(prompt)
  print("Prompt:", prompt)
  print("Response:", response.content if hasattr(response, "content") else response)

if __name__ == "__main__":
  main()