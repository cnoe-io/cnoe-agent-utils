#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import dotenv
from cnoe_agent_utils.llm_factory import LLMFactory
from cnoe_agent_utils.utils import stream_with_spinner

dotenv.load_dotenv()

def handle_google_application_credentials_env():
    """
    If GOOGLE_APPLICATION_CREDENTIALS_ENV is set, write its contents to a /tmp file
    and set GOOGLE_APPLICATION_CREDENTIALS to that file path.
    """
    creds_env = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_ENV")
    if creds_env:
        creds_path = "/tmp/gcp-creds.json"
        try:
            with open(creds_path, "w") as f:
                f.write(creds_env)
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds_path
        except Exception as e:
            print(f"ERROR: Failed to write GOOGLE_APPLICATION_CREDENTIALS_ENV to file: {e}")
            return False
    return True

def check_vertex_credentials():
    """Check if required Vertex AI credentials are available."""
    missing_vars = []

    # Accept either GOOGLE_APPLICATION_CREDENTIALS or GOOGLE_APPLICATION_CREDENTIALS_ENV
    if not (os.getenv("GOOGLE_APPLICATION_CREDENTIALS") or os.getenv("GOOGLE_APPLICATION_CREDENTIALS_ENV")):
        missing_vars.append("GOOGLE_APPLICATION_CREDENTIALS or GOOGLE_APPLICATION_CREDENTIALS_ENV")

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
        print("   OR set GOOGLE_APPLICATION_CREDENTIALS_ENV to the JSON contents")
        print("2. Set GOOGLE_CLOUD_PROJECT to your Google Cloud project ID")
        print("3. Set VERTEXAI_MODEL_NAME to your desired model (e.g., gemini-2.0-flash-001)")
        print("4. Or add these to your .env file")
        return False

    return True

def main():
    if not handle_google_application_credentials_env():
        print("SKIPPING: Could not process GOOGLE_APPLICATION_CREDENTIALS_ENV")
        return

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
