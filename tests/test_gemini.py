
import os
import itertools
from cnoe_agent_utils.llm_factory import LLMFactory
from .conftest import require

def test_gemini_invoke_and_stream():
    require(os.getenv("GOOGLE_API_KEY"), "GOOGLE_API_KEY missing")

    llm = LLMFactory("google-gemini").get_llm()

    msg = llm.invoke("Say 'hello from Gemini' in five words.")
    text = getattr(msg, "content", "") or getattr(msg, "text", "")
    assert isinstance(text, str) and len(text.strip()) > 0

    chunks = []
    for ch in itertools.islice(llm.stream("Write a short sentence about wind."), 0, 20):
        s = getattr(ch, "content", "") or getattr(ch, "text", "")
        if s:
            chunks.append(s)
            if len("".join(chunks)) > 10:
                break
    assert len("".join(chunks).strip()) > 0
