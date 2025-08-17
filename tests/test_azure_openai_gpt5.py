
import os
import itertools
from cnoe_agent_utils.llm_factory import LLMFactory
from .conftest import require

def test_azure_invoke_and_stream():
    need = ["AZURE_OPENAI_ENDPOINT","AZURE_OPENAI_API_KEY","AZURE_OPENAI_API_VERSION","AZURE_OPENAI_DEPLOYMENT"]
    have = all(os.getenv(k) for k in need)
    require(have, f"Missing Azure envs: {', '.join([k for k in need if not os.getenv(k)])}")

    llm = LLMFactory("azure-openai").get_llm()

    msg = llm.invoke("Say 'hello from Azure OpenAI' in five words.")
    text = getattr(msg, "content", "") or getattr(msg, "text", "")
    assert isinstance(text, str) and len(text.strip()) > 0

    chunks = []
    for ch in itertools.islice(llm.stream("Write a short sentence about galaxies."), 0, 20):
        s = getattr(ch, "content", "") or getattr(ch, "text", "")
        if s:
            chunks.append(s)
            if len("".join(chunks)) > 10:
                break
    assert len("".join(chunks).strip()) > 0
