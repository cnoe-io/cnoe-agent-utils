
import os
import pytest

RUN = os.getenv("RUN_LLM_INTEGRATION") == "1"

def require(flag: bool, reason: str):
    if not RUN:
        pytest.skip("RUN_LLM_INTEGRATION=1 not set")
    if not flag:
        pytest.skip(reason)
