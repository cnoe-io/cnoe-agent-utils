#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import dotenv
from cnoe_agent_utils.llm_factory import LLMFactory
from cnoe_agent_utils.utils import stream_with_spinner

dotenv.load_dotenv()

def check_vertex_credentials():
    """Check if required Vertex AI credentials are available."""
    missing_vars = []

    # Check for Google Cloud credentials
    if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        missing_vars.append("GOOGLE_APPLICATION_CREDENTIALS")

    # Check for Google Cloud project
    if not os.getenv("GOOGLE_CLOUD_PROJECT"):
        missing_vars.append("GOOGLE_CLOUD_PROJECT")

    # Check for model name
    if not os.getenv("VERTEXAI_MODEL_NAME"):
        missing_vars.append("VERTEXAI_MODEL_NAME")

    if missing_vars:
        print("ERROR: Missing required environment variables for Vertex AI:")
        for var in missing_vars:
            print(f"   - {var}")
        print("\nTo fix this:")
        print("1. Set GOOGLE_APPLICATION_CREDENTIALS to your service account JSON file path")
        print("2. Set GOOGLE_CLOUD_PROJECT to your Google Cloud project ID")
        print("3. Set VERTEXAI_MODEL_NAME to your desired model (e.g., gemini-2.0-flash-001)")
        print("4. Or add these to your .env file")
        return False

    return True

def main():
    if not check_vertex_credentials():
        print("\nSKIPPING: Vertex AI example due to missing credentials")
        return

    try:
        llm = LLMFactory("gcp-vertexai").get_llm()
        print("=== Vertex AI (stream) ===")
        # Stream with spinner
        for chunk in stream_with_spinner(llm, "Write long paragraph about solar panels.", "Waiting for Vertex AI response"):
            sys.stdout.write(getattr(chunk, "content", "") or getattr(chunk, "text", "") or "")
            sys.stdout.flush()
        print("\n=== done ===")
    except Exception as e:
        print(f"ERROR: Error running Vertex AI example: {e}")
        print("This example requires proper Google Cloud credentials and configuration.")

if __name__ == "__main__":
    main()
