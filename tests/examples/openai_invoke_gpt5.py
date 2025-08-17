
import os, sys
import dotenv
dotenv.load_dotenv()
from cnoe_agent_utils.llm_factory import LLMFactory
from cnoe_agent_utils.utils import invoke_with_spinner

def main():
    llm = LLMFactory("openai").get_llm()
    prompt = "Write one short sentence about Mars exploration."
    print("=== OpenAI (invoke) ===")
    response = invoke_with_spinner(llm, prompt, "Processing OpenAI request")
    print(response.content)
    print("=== done ===")

if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is required")
    if not os.getenv("OPENAI_MODEL_NAME"):
        os.environ["OPENAI_MODEL_NAME"] = "gpt-5"
    main()
