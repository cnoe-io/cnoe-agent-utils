#!/usr/bin/env python3
"""
Test script for the timing features in the utils module.
"""

import os
from cnoe_agent_utils.utils import time_llm_operation

def test_timing_features():
    """Test the timing features with different environment variable settings."""
    print("Testing Timing Features...")

    # Test 1: Default timing (enabled)
    print("\n=== Test 1: Default timing (enabled) ===")
    with time_llm_operation("Test operation"):
        # Simulate some work
        import time
        time.sleep(1)

    # Test 2: Disabled timing
    print("\n=== Test 2: Disabled timing ===")
    os.environ["LLM_SHOW_TIMING"] = "false"
    with time_llm_operation("Test operation (timing disabled)"):
        # Simulate some work
        import time
        time.sleep(1)

    # Test 3: Re-enable timing
    print("\n=== Test 3: Re-enabled timing ===")
    os.environ["LLM_SHOW_TIMING"] = "true"
    with time_llm_operation("Test operation (timing re-enabled)"):
        # Simulate some work
        import time
        time.sleep(1)

    print("\n=== Timing test completed! ===")

if __name__ == "__main__":
    test_timing_features()
