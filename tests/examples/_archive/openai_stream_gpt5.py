import os, sys
from cnoe_agent_utils.llm_factory import LLMFactory

# Required: OPENAI_API_KEY
# Recommended: OPENAI_MODEL_NAME=gpt-5
# Optional:   OPENAI_USE_RESPONSES=true (auto if name starts with gpt-5)

def main():
    llm = LLMFactory("openai").get_llm()
    prompt = "Write one short sentence about Mars exploration."
    print("=== OpenAI (stream) ===")
    for chunk in llm.stream(prompt):
        sys.stdout.write(getattr(chunk, "content", "") or getattr(chunk, "text", "") or "")
        sys.stdout.flush()
    print("\n=== done ===")

if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is required")
    main()
