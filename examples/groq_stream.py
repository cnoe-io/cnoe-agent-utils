#!/usr/bin/env python3
"""
Groq Test Script

Tests the Groq LLM integration with both invoke and streaming modes 
to ensure comprehensive coverage.
"""

import os
import sys
from dotenv import load_dotenv

from cnoe_agent_utils.llm_factory import LLMFactory
from cnoe_agent_utils.utils import invoke_with_spinner, stream_with_spinner

load_dotenv()

def test_groq_invoke(model_name):
    """Test Groq invoke mode."""
    
    try:
        # Set the model for this test
        os.environ["GROQ_MODEL_NAME"] = model_name
        llm = LLMFactory("groq").get_llm()

        prompt = "Write one short sentence about quantum computing."
        print(f"Prompt: {prompt}")

        result = invoke_with_spinner(llm, prompt, f"Processing Groq request")

        # Extract content from the result
        content = getattr(result, "content", None)
        if callable(content):
            content = content()
        text = getattr(result, "text", None)
        if callable(text):
            text = text()

        response = str(content or text or result or "")
        print(f"Response: {response}")
        print("Groq invoke test PASSED")
        return True

    except Exception as e:
        print(f"Groq invoke test FAILED: {e}")
        return False

def test_groq_stream(model_name):
    """Test Groq streaming mode."""

    try:
        # Set the model for this test
        os.environ["GROQ_MODEL_NAME"] = model_name
        llm = LLMFactory("groq").get_llm()

        prompt = "Write one short sentence about quantum computing."
        print(f"Prompt: {prompt}")
        print("Streaming response:")

        # Stream with spinner
        for chunk in stream_with_spinner(llm, prompt, "Waiting for Groq response"):
            text = getattr(chunk, "text", None)
            if callable(text):
                text = text()
            if text:
                sys.stdout.write(str(text))
                sys.stdout.flush()

        print("\n Groq stream test PASSED")
        return True

    except Exception as e:
        print(f"Groq stream test FAILED: {e}")
        return False

def main():
    """Run all Groq tests."""

    # Check required environment variables
    required_vars = [
        "GROQ_API_KEY",
        "GROQ_MODEL_NAME"
    ]

    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        print(" Missing required environment variables:")
        for var in missing_vars:
            print(f"  - {var}")
        return 1

    # Define test models
    models_to_test = [os.getenv("GROQ_MODEL_NAME", "llama2-70b-v2")]

    # Run tests
    all_passed = True
    for model in models_to_test:
        if not test_groq_invoke(model):
            all_passed = False
        if not test_groq_stream(model):
            all_passed = False

    if not all_passed:
        print("\n Some tests failed")
        return 1

    print("\n All tests passed!")
    return 0

if __name__ == "__main__":
    sys.exit(main())