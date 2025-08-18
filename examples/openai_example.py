#!/usr/bin/env python3
"""
Consolidated OpenAI Test Script

Tests all three OpenAI models (gpt-4o, gpt-4o-mini, gpt-5) with both
invoke and streaming modes to ensure comprehensive coverage.
"""

import os
import sys
import dotenv
from cnoe_agent_utils.llm_factory import LLMFactory
from cnoe_agent_utils.utils import invoke_with_spinner, stream_with_spinner

dotenv.load_dotenv()

def test_openai_invoke(model_name, model_id):
    """Test OpenAI invoke mode for a specific model."""
    print(f"\n{'='*60}")
    print(f" Testing OpenAI INVOKE: {model_name} ({model_id})")
    print(f"{'='*60}")

    try:
        # Set the model for this test
        os.environ["OPENAI_MODEL_NAME"] = model_id
        llm = LLMFactory("openai").get_llm()

        prompt = "Write one short sentence about Mars exploration."
        print(f"Prompt: {prompt}")

        result = invoke_with_spinner(llm, prompt, f"Processing {model_name} request")

        # Extract content from the result
        content = getattr(result, "content", None)
        if callable(content):
            content = content()
        text = getattr(result, "text", None)
        if callable(text):
            text = text()

        # Handle dict results (for responses API)
        if content is None and isinstance(result, dict):
            content = result.get("content")
            text = result.get("text")

        response = str(content or text or result or "")
        print(f"Response: {response}")
        print(f"✅ {model_name} invoke test PASSED")
        return True

    except Exception as e:
        print(f"❌ {model_name} invoke test FAILED: {e}")
        return False

def test_openai_stream(model_name, model_id):
    """Test OpenAI streaming mode for a specific model."""
    print(f"\n{'='*60}")
    print(f" Testing OpenAI STREAM: {model_name} ({model_id})")
    print(f"{'='*60}")

    try:
        # Set the model for this test
        os.environ["OPENAI_MODEL_NAME"] = model_id
        llm = LLMFactory("openai").get_llm()

        prompt = "Write one short sentence about Mars exploration."
        print(f"Prompt: {prompt}")
        print("Streaming response:")

        # Stream with spinner
        for chunk in stream_with_spinner(llm, prompt, f"Waiting for {model_name} response"):
            # Handle different chunk formats
            if isinstance(chunk, dict) and "text" in chunk:
                # OpenAI responses API streaming
                sys.stdout.write(str(chunk["text"]))
                sys.stdout.flush()
            else:
                # Standard OpenAI streaming
                text = getattr(chunk, "text", None)
                if callable(text):
                    text = text()
                if text:
                    sys.stdout.write(str(text))
                    sys.stdout.flush()

        print(f"\n✅ {model_name} stream test PASSED")
        return True

    except Exception as e:
        print(f"❌ {model_name} stream test FAILED: {e}")
        return False

def main():
    """Run all OpenAI tests."""
    print("🚀 Starting Consolidated OpenAI Tests")
    print("=" * 60)

    # Check required environment variables
    required_vars = [
        "OPENAI_API_KEY"
    ]

    missing = [var for var in required_vars if not os.getenv(var)]
    if missing:
        raise SystemExit(f"Missing required environment variables: {', '.join(missing)}")

    # Test configurations: (model_name, model_id)
    test_configs = [
        ("GPT-4o", "gpt-4o"),
        ("GPT-4o Mini", "gpt-4o-mini"),
        ("GPT-4.1", "gpt-4.1"),
        ("GPT-5", "gpt-5"),
        ("GPT-5 Mini", "gpt-5-mini")
    ]

    results = {
        "invoke": {},
        "stream": {}
    }

    # Run invoke tests
    print("\n📝 Testing INVOKE mode for all models...")
    for model_name, model_id in test_configs:
        success = test_openai_invoke(model_name, model_id)
        results["invoke"][model_name] = success

    # Run streaming tests
    print("\n🌊 Testing STREAMING mode for all models...")
    for model_name, model_id in test_configs:
        success = test_openai_stream(model_name, model_id)
        results["stream"][model_name] = success

    # Print summary
    print(f"\n{'='*60}")
    print(" 📊 TEST RESULTS SUMMARY")
    print(f"{'='*60}")

    print("\nINVOKE Tests:")
    for model_name, success in results["invoke"].items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"  {model_name}: {status}")

    print("\nSTREAM Tests:")
    for model_name, success in results["stream"].items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"  {model_name}: {status}")

    # Calculate overall success rate
    total_tests = len(results["invoke"]) + len(results["stream"])
    passed_tests = sum(results["invoke"].values()) + sum(results["stream"].values())
    success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0

    print(f"\nOverall: {passed_tests}/{total_tests} tests passed ({success_rate:.1f}%)")

    if passed_tests == total_tests:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n💥 {total_tests - passed_tests} tests failed!")
        return 1

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⏹️  Tests interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1)
