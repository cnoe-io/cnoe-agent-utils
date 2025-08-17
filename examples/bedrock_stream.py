import os
import sys
import dotenv
from cnoe_agent_utils.llm_factory import LLMFactory
from cnoe_agent_utils.utils import stream_with_spinner

dotenv.load_dotenv()

def main():
    llm = LLMFactory("aws-bedrock").get_llm()
    print("=== AWS Bedrock (stream) ===")
    # Stream with spinner
    for chunk in stream_with_spinner(llm, "Write a short sentence about river deltas.", "Waiting for AWS Bedrock response"):
        text = getattr(chunk, "text", None)
        if callable(text):
            text = text()
        sys.stdout.write(str(text or ""))
        sys.stdout.flush()
    print("\n=== done ===")

if __name__ == "__main__":
    if not os.getenv("AWS_BEDROCK_MODEL_ID") or not os.getenv("AWS_REGION"):
        raise SystemExit("AWS_BEDROCK_MODEL_ID and AWS_REGION are required")
    main()
