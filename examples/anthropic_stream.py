
import os
import sys
from cnoe_agent_utils.llm_factory import LLMFactory
from cnoe_agent_utils.utils import stream_with_spinner

def check_anthropic_credentials():
    """Check if required Anthropic credentials are available."""
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("❌ Missing required environment variable for Anthropic:")
        print("   - ANTHROPIC_API_KEY")
        print("\nTo fix this:")
        print("1. Get your API key from https://console.anthropic.com/")
        print("2. Set ANTHROPIC_API_KEY environment variable")
        print("3. Or add it to your .env file")
        return False

    return True

def main():
    if not check_anthropic_credentials():
        print("\n⏭️  Skipping Anthropic example due to missing credentials")
        return

    try:
        llm = LLMFactory("anthropic-claude").get_llm()
        print("=== Anthropic (stream) ===")
        # Stream with spinner
        for chunk in stream_with_spinner(llm, "Write one short sentence about ocean currents.", "Waiting for Anthropic Claude response"):
            sys.stdout.write(getattr(chunk, "content", "") or getattr(chunk, "text", "") or "")
            sys.stdout.flush()
        print("\n=== done ===")
    except Exception as e:
        print(f"❌ Error running Anthropic example: {e}")
        print("This example requires proper Anthropic API key and configuration.")

if __name__ == "__main__":
    main()
