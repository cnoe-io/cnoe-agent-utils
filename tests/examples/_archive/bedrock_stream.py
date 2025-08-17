import os, sys
from cnoe_agent_utils.llm_factory import LLMFactory

# Required:
#   AWS credentials (env or profile/role)
#   AWS_REGION=...
#   AWS_BEDROCK_MODEL_ID=anthropic.claude-3-7-sonnet-20250219-v1:0   (example)
# Optional:
#   AWS_BEDROCK_PROVIDER=anthropic

def main():
    llm = LLMFactory("aws-bedrock").get_llm()
    print("=== AWS Bedrock (stream) ===")
    for chunk in llm.stream("Write one short sentence about river deltas."):
        sys.stdout.write(getattr(chunk, "content", "") or getattr(chunk, "text", "") or "")
        sys.stdout.flush()
    print("\n=== done ===")

if __name__ == "__main__":
    if not os.getenv("AWS_BEDROCK_MODEL_ID") or not os.getenv("AWS_REGION"):
        raise SystemExit("AWS_BEDROCK_MODEL_ID and AWS_REGION are required")
    main()
