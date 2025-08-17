import os, sys
from cnoe_agent_utils.llm_factory import LLMFactory

# Required:
#   GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json (or ADC)
#   VERTEXAI_MODEL_NAME=gemini-2.0-flash-001  (or set via env already)
# Note: Project/location are picked up by Vertex SDK via ADC in your code.

def main():
    llm = LLMFactory("gcp-vertexai").get_llm()
    print("=== Vertex AI (stream) ===")
    for chunk in llm.stream("Write one short sentence about solar panels."):
        sys.stdout.write(getattr(chunk, "content", "") or getattr(chunk, "text", "") or "")
        sys.stdout.flush()
    print("\n=== done ===")

if __name__ == "__main__":
    # Let ADC handle credentials; emit a friendly hint if unset:
    if not os.getenv("VERTEXAI_MODEL_NAME"):
        print("Hint: set VERTEXAI_MODEL_NAME (default used if your factory sets one).")
    main()
