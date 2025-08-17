import os, sys
from cnoe_agent_utils.llm_factory import LLMFactory

# Required: ANTHROPIC_API_KEY
# Recommended: ANTHROPIC_MODEL_NAME=claude-3-7-sonnet-20250219

def main():
    llm = LLMFactory("anthropic-claude").get_llm()
    print("=== Anthropic (stream) ===")
    for chunk in llm.stream("Write one short sentence about ocean currents."):
        sys.stdout.write(getattr(chunk, "content", "") or getattr(chunk, "text", "") or "")
        sys.stdout.flush()
    print("\n=== done ===")

if __name__ == "__main__":
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY is required")
    main()
