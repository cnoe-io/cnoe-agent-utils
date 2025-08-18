
import os
import sys
from cnoe_agent_utils.llm_factory import LLMFactory
from cnoe_agent_utils.utils import stream_with_spinner

def main():
    llm = LLMFactory("google-gemini").get_llm()
    print("=== Google Gemini (stream) ===")
    # Stream with spinner
    for chunk in stream_with_spinner(llm, "Write one short sentence about wind turbines.", "Waiting for Google Gemini response"):
        sys.stdout.write(getattr(chunk, "content", "") or getattr(chunk, "text", "") or "")
        sys.stdout.flush()
    print("\n=== done ===")

if __name__ == "__main__":
    if not os.getenv("GOOGLE_API_KEY"):
        raise SystemExit("GOOGLE_API_KEY is required")
    main()
