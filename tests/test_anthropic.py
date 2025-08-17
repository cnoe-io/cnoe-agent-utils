
import os
import itertools
from cnoe_agent_utils.llm_factory import LLMFactory
from .conftest import require

def test_anthropic_invoke_and_stream():
    require(os.getenv("ANTHROPIC_API_KEY"), "ANTHROPIC_API_KEY missing")

    llm = LLMFactory("anthropic-claude").get_llm()

    msg = llm.invoke("Say 'hello from Anthropic' in five words.")
    text = getattr(msg, "content", "") or getattr(msg, "text", "")
    assert isinstance(text, str) and len(text.strip()) > 0

    chunks = []
    for ch in itertools.islice(llm.stream("Write a short sentence about oceans."), 0, 20):
        s = getattr(ch, "content", "") or getattr(ch, "text", "")
        if s:
            chunks.append(s)
            if len("".join(chunks)) > 10:
                break
    assert len("".join(chunks).strip()) > 0
