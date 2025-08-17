
import os
import itertools
from cnoe_agent_utils.llm_factory import LLMFactory
from .conftest import require

def test_openai_invoke_and_stream():
    require(os.getenv("OPENAI_API_KEY"), "OPENAI_API_KEY missing")
    if not os.getenv("OPENAI_MODEL_NAME"):
        os.environ["OPENAI_MODEL_NAME"] = "gpt-5"
    llm = LLMFactory("openai").get_llm()

    msg = llm.invoke("Say 'hello from OpenAI' in five words.")
    text = getattr(msg, "content", "") or getattr(msg, "text", "")
    assert isinstance(text, str) and len(text.strip()) > 0

    chunks = []
    for ch in itertools.islice(llm.stream("Write a short sentence about stars."), 0, 20):
        s = getattr(ch, "content", "") or getattr(ch, "text", "")
        if s:
            chunks.append(s)
            if len("".join(chunks)) > 10:
                break
    assert len("".join(chunks).strip()) > 0
