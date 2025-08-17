
import os
import itertools
from cnoe_agent_utils.llm_factory import LLMFactory
from .conftest import require

def test_vertex_invoke_and_stream():
    require(os.getenv("VERTEXAI_MODEL_NAME"), "VERTEXAI_MODEL_NAME missing")

    llm = LLMFactory("gcp-vertexai").get_llm()

    msg = llm.invoke("Say 'hello from Vertex AI' in five words.")
    text = getattr(msg, "content", "") or getattr(msg, "text", "")
    assert isinstance(text, str) and len(text.strip()) > 0

    chunks = []
    for ch in itertools.islice(llm.stream("Write a short sentence about robotics."), 0, 20):
        s = getattr(ch, "content", "") or getattr(ch, "text", "")
        if s:
            chunks.append(s)
            if len("".join(chunks)) > 10:
                break
    assert len("".join(chunks).strip()) > 0
