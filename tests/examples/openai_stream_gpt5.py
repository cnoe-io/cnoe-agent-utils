
import os, sys
import dotenv
dotenv.load_dotenv()
from cnoe_agent_utils.llm_factory import LLMFactory
from cnoe_agent_utils.utils import stream_with_spinner

def main():
    llm = LLMFactory("openai").get_llm()
    prompt = "Write one short sentence about Mars exploration."
    print("=== OpenAI (stream) ===")
    # Stream with spinner
    for chunk in stream_with_spinner(llm, prompt, "Waiting for OpenAI response"):
        # Only print the "text" field if present (for OpenAI responses API streaming)
        if isinstance(chunk, dict) and "text" in chunk:
            sys.stdout.write(str(chunk["text"]))
            sys.stdout.flush()
        else:
            text = getattr(chunk, "text", None)
            if callable(text):
                text = text()
            if text:
                sys.stdout.write(str(text))
                sys.stdout.flush()
    print("\n=== done ===")

if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is required")
    if not os.getenv("OPENAI_MODEL_NAME"):
        os.environ["OPENAI_MODEL_NAME"] = "gpt-5"
    main()
