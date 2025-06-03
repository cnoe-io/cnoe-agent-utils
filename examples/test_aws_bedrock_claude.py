# Copyright 2025 CNOE
# SPDX-License-Identifier: Apache-2.0

"""
===============================================================================
 test_aws_bedrock_claude.py - Example usage of AWS Bedrock LLM via cnoe-agent-utils
===============================================================================

This open source example demonstrates how to use the `cnoe-agent-utils` library
to interact with an AWS Bedrock-hosted Claude model. It loads required AWS and
model configuration from environment variables, which can be set directly or
loaded from a `.env` file in the project directory. The script initializes the
LLMFactory for AWS Bedrock and invokes the model with a sample prompt.

Environment Variables (can be set in a .env file):
  - AWS_PROFILE: AWS credentials profile name.
  - AWS_BEDROCK_MODEL_ID: Identifier for the Bedrock model to use.
  - AWS_BEDROCK_PROVIDER: Bedrock provider (e.g., 'anthropic').
  - AWS_REGION: AWS region for Bedrock.

Raises:
  EnvironmentError: If any required environment variables are missing.

Example:
  $ echo "AWS_PROFILE=my-aws-profile" >> .env
  $ echo "AWS_BEDROCK_MODEL_ID=anthropic.claude-v2" >> .env
  $ echo "AWS_BEDROCK_PROVIDER=anthropic" >> .env
  $ echo "AWS_REGION=us-east-1" >> .env
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

  print("Testing AWS Bedrock LLM with cnoe-agent-utils")
  # List to track missing required environment variables
  missing = []

  # Check for either AWS_BEDROCK_PROFILE or AWS_PROFILE
  # Check for AWS_PROFILE
  if not os.getenv("AWS_PROFILE"):
    missing.append("AWS_PROFILE")

  # Check for AWS_BEDROCK_MODEL_ID
  if not os.getenv("AWS_BEDROCK_MODEL_ID"):
    missing.append("AWS_BEDROCK_MODEL_ID")

  # Check for AWS_BEDROCK_PROVIDER
  if not os.getenv("AWS_BEDROCK_PROVIDER"):
    missing.append("AWS_BEDROCK_PROVIDER")

  # Check for AWS_REGION
  if not os.getenv("AWS_REGION"):
    missing.append("AWS_REGION")

  # If any required variables are missing, print and raise an error
  if missing:
    for var in missing:
      print(f"Missing required AWS Bedrock environment variable: {var}")
    raise EnvironmentError("One or more required AWS Bedrock environment variables are missing.")

  # Initialize the LLMFactory for AWS Bedrock
  factory = LLMFactory("aws-bedrock")

  # Get the LLM instance
  llm = factory.get_llm()

  # Prepare messages for the LLM in the required format
  messages = [
      (
          "system",
          "You are a helpful assistant that translates English to French. Translate the user sentence.",
      ),
      ("human", "I love programming."),
  ]
  ai_msg = llm.invoke(messages)
  # Parse and print the content text from the response
  if isinstance(ai_msg, dict) and "content" in ai_msg:
    for item in ai_msg["content"]:
      if item.get("type") == "text":
        print(f"AI Response: {item.get('text')}")
  elif hasattr(ai_msg, "content") and isinstance(ai_msg.content, list):
    for item in ai_msg.content:
      if item.get("type") == "text":
        print(f"AI Response: {item.get('text')}")
  else:
    print(f"AI Response: {ai_msg}")

if __name__ == "__main__":
  main()