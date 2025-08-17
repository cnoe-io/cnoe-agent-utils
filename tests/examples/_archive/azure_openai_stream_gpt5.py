import os, sys
from cnoe_agent_utils.llm_factory import LLMFactory

# Required:
#   AZURE_OPENAI_ENDPOINT=https://<resource>.openai.azure.com
#   AZURE_OPENAI_API_KEY=...
#   AZURE_OPENAI_API_VERSION=2024-10-21-preview  # or your required version
#   AZURE_OPENAI_DEPLOYMENT=gpt-5                # your deployment name

def main():
    llm = LLMFactory("azure-openai").get_llm()
    prompt = "Write one short sentence about the Moon's surface."
    print("=== Azure OpenAI (stream) ===")
    for chunk in llm.stream(prompt):
        sys.stdout.write(getattr(chunk, "content", "") or getattr(chunk, "text", "") or "")
        sys.stdout.flush()
    print("\n=== done ===")

if __name__ == "__main__":
    need = ["AZURE_OPENAI_ENDPOINT","AZURE_OPENAI_API_KEY","AZURE_OPENAI_API_VERSION","AZURE_OPENAI_DEPLOYMENT"]
    missing = [k for k in need if not os.getenv(k)]
    if missing:
        raise SystemExit(f"Missing env vars: {', '.join(missing)}")
    main()
