
import os
import itertools
from cnoe_agent_utils.llm_factory import LLMFactory
from .conftest import require

def test_bedrock_invoke_and_stream():
    need = ["AWS_BEDROCK_MODEL_ID","AWS_REGION"]
    have = all(os.getenv(k) for k in need)
    require(have, f"Missing Bedrock envs: {', '.join([k for k in need if not os.getenv(k)])}")

    llm = LLMFactory("aws-bedrock").get_llm()

    msg = llm.invoke("Say 'hello from Bedrock' in five words.")
    text = getattr(msg, "content", "") or getattr(msg, "text", "")
    assert isinstance(text, str) and len(text.strip()) > 0

    chunks = []
    for ch in itertools.islice(llm.stream("Write a short sentence about forests."), 0, 20):
        s = getattr(ch, "content", "") or getattr(ch, "text", "")
        if s:
            chunks.append(s)
            if len("".join(chunks)) > 10:
                break
    assert len("".join(chunks).strip()) > 0
