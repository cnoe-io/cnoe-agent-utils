# Copyright 2025 CNOE
# SPDX-License-Identifier: Apache-2.0

"""
===============================================================================
 test_gcp_vertex_gemini.py - Example usage of GCP Vertex AI Gemini via cnoe-agent-utils
===============================================================================

This open source example demonstrates how to use the `cnoe-agent-utils` library
to interact with a Gemini model hosted on Google Cloud Vertex AI. It loads required
GCP and model configuration from environment variables, initializes the LLMFactory
for Vertex AI, and invokes the model with a sample prompt.

Environment Variables:
  - VERTEXAI_PROJECT: GCP project ID.
  - VERTEXAI_LOCATION: GCP region (e.g., us-central1).
  - VERTEXAI_MODEL_NAME: Vertex AI model name (e.g., gemini-1.5-flash-001).

Raises:
  EnvironmentError: If any required environment variables are missing.

Example:
  $ export VERTEXAI_PROJECT=your-gcp-project-id
  $ export VERTEXAI_LOCATION=us-central1
  $ export VERTEXAI_MODEL_NAME=gemini-1.5-flash-001
  $ python test_gcp_vertex_gemini.py

Dependencies:
  - cnoe-agent-utils
  - python-dotenv
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

  # Validate required environment variables
  required_env = ["VERTEXAI_PROJECT", "VERTEXAI_LOCATION", "VERTEXAI_MODEL_NAME"]
  missing = [var for var in required_env if not os.environ.get(var)]
  if missing:
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