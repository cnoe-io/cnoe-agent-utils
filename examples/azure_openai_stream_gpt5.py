import os
import sys
import dotenv
from cnoe_agent_utils.llm_factory import LLMFactory
from cnoe_agent_utils.utils import stream_with_spinner

dotenv.load_dotenv()

def main():
    llm = LLMFactory("azure-openai").get_llm()
    prompt = "Write one short sentence about the Moon's surface."
    print("=== Azure OpenAI (stream) ===")

    # Stream with spinner
    for chunk in stream_with_spinner(llm, prompt, "Waiting for Azure OpenAI response"):
        content = getattr(chunk, "content", "")
        if callable(content):
            content = content()
        text = getattr(chunk, "text", "")
        if callable(text):
            text = text()
        sys.stdout.write(str(content or text or ""))
        sys.stdout.flush()

    print("\n=== done ===")

if __name__ == "__main__":
    need = ["AZURE_OPENAI_ENDPOINT","AZURE_OPENAI_API_KEY","AZURE_OPENAI_API_VERSION","AZURE_OPENAI_DEPLOYMENT"]
    missing = [k for k in need if not os.getenv(k)]
    if missing:
        raise SystemExit(f"Missing env vars: {', '.join(missing)}")
    main()
