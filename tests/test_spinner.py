#!/usr/bin/env python3
"""
Test script for the Spinner utility class.
"""

import time
from cnoe_agent_utils.utils import Spinner

def test_spinner():
    """Test the spinner functionality."""
    print("Testing Spinner utility...")

    # Test basic spinner
    spinner = Spinner("Processing")
    spinner.start()
    time.sleep(3)  # Simulate some work
    spinner.stop()
    print("Basic spinner test completed!")

    # Test with custom message
    spinner = Spinner("Loading data")
    spinner.start()
    time.sleep(2)  # Simulate some work
    spinner.stop()
    print("Custom message spinner test completed!")

if __name__ == "__main__":
    test_spinner()
