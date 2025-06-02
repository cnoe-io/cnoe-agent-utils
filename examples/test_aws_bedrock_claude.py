# Copyright 2025 CNOE
# SPDX-License-Identifier: Apache-2.0

"""
===============================================================================
 test_aws_bedrock_claude.py - Example usage of AWS Bedrock LLM via cnoe-agent-utils
===============================================================================

This open source example demonstrates how to use the `cnoe-agent-utils` library
to interact with an AWS Bedrock-hosted Claude model. It loads required AWS and
model configuration from environment variables, initializes the LLMFactory for
AWS Bedrock, and invokes the model with a sample prompt.

Environment Variables:
  - AWS_BEDROCK_PROFILE or AWS_PROFILE: AWS credentials profile name.
  - BEDROCK_MODEL_ID: Identifier for the Bedrock model to use.

Raises:
  EnvironmentError: If any required environment variables are missing.

Example:
  $ export AWS_PROFILE=my-aws-profile
  $ export BEDROCK_MODEL_ID=anthropic.claude-v2
  $ python test_aws_bedrock_claude.py

Dependencies:
  - cnoe-agent-utils
  - python-dotenv
"""
import os
from dotenv import load_dotenv
from cnoe_agent_utils import LLMFactory

def main():
  # Load environment variables from a .env file if present
  load_dotenv()

  # List to track missing required environment variables
  missing = []

  # Check for either AWS_BEDROCK_PROFILE or AWS_PROFILE
  profile_set = os.getenv("AWS_BEDROCK_PROFILE") or os.getenv("AWS_PROFILE")
  if not profile_set:
    missing.append("AWS_BEDROCK_PROFILE or AWS_PROFILE")

  # Check for BEDROCK_MODEL_ID
  if not os.getenv("BEDROCK_MODEL_ID"):
    missing.append("BEDROCK_MODEL_ID")

  # If any required variables are missing, print and raise an error
  if missing:
    for var in missing:
      print(f"Missing required AWS Bedrock environment variable: {var}")
    raise EnvironmentError("One or more required AWS Bedrock environment variables are missing.")

  # Retrieve credentials profile and model ID from environment
  credentials_profile = os.getenv("AWS_BEDROCK_PROFILE") or os.getenv("AWS_PROFILE") or None
  model_id = os.getenv("BEDROCK_MODEL_ID")

  # Initialize the LLMFactory for AWS Bedrock
  factory = LLMFactory("aws-bedrock")

  # Get the LLM instance
  llm = factory.get_llm()

  # Invoke the model with a prompt and print the response
  response = llm.invoke("Hello, Claude!")
  print(response)

if __name__ == "__main__":
  main()