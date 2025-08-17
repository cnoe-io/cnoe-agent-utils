
import os
import dotenv
from cnoe_agent_utils.llm_factory import LLMFactory
from cnoe_agent_utils.utils import invoke_with_spinner

dotenv.load_dotenv()

def main():
    llm = LLMFactory("azure-openai").get_llm()
    prompt = "Write one short sentence about the Moon's surface."
    print("=== Azure OpenAI (invoke) ===")
    result = invoke_with_spinner(llm, prompt, "Processing Azure OpenAI request")
    # Try to extract content or text from the result
    content = getattr(result, "content", None)
    if callable(content):
        content = content()
    text = getattr(result, "text", None)
    if callable(text):
        text = text()
    # If result is a dict (sometimes for responses API), try to get "content" or "text"
    if content is None and isinstance(result, dict):
        content = result.get("content")
        text = result.get("text")
    print(str(content or text or result or ""))  # fallback to printing the result if nothing else

    print("=== done ===")

if __name__ == "__main__":
    need = ["AZURE_OPENAI_ENDPOINT","AZURE_OPENAI_API_KEY","AZURE_OPENAI_API_VERSION","AZURE_OPENAI_DEPLOYMENT"]
    missing = [k for k in need if not os.getenv(k)]
    if missing:
        raise SystemExit(f"Missing env vars: {', '.join(missing)}")
    main()
