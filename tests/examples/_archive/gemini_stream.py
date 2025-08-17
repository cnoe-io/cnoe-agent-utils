import os, sys
from cnoe_agent_utils.llm_factory import LLMFactory

# Required: GOOGLE_API_KEY
# Optional: GEMINI_MODEL_NAME=gemini-2.0-flash-001

def main():
    llm = LLMFactory("google-gemini").get_llm()
    print("=== Google Gemini (stream) ===")
    for chunk in llm.stream("Write one short sentence about wind turbines."):
        sys.stdout.write(getattr(chunk, "content", "") or getattr(chunk, "text", "") or "")
        sys.stdout.flush()
    print("\n=== done ===")

if __name__ == "__main__":
    if not os.getenv("GOOGLE_API_KEY"):
        raise SystemExit("GOOGLE_API_KEY is required")
    main()
