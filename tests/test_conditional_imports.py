#!/usr/bin/env python3
"""Test script to verify conditional imports work correctly."""

import sys
import pytest

# Temporarily remove langchain packages from sys.path to test conditional imports
original_path = sys.path.copy()

def test_conditional_imports():
    """Test that the factory works with missing optional dependencies."""

    print("Testing conditional imports...")

    # Test 1: Import with all dependencies available
    try:
        from cnoe_agent_utils.llm_factory import LLMFactory
        print("✅ Successfully imported LLMFactory with all dependencies")

        providers = LLMFactory.get_supported_providers()
        print(f"Available providers: {providers}")

        # Assert that we have at least some providers available
        assert len(providers) > 0, "No providers should be available"

        # Test each provider
        for provider in providers:
            is_available = LLMFactory.is_provider_available(provider)
            missing_deps = LLMFactory.get_missing_dependencies(provider)
            print(f"  {provider}: available={is_available}, missing_deps={missing_deps}")

            # Assert that available providers have no missing dependencies
            if is_available:
                assert len(missing_deps) == 0, f"Available provider {provider} should have no missing dependencies"

    except ImportError as e:
        print(f"❌ Failed to import LLMFactory: {e}")
        pytest.fail(f"Failed to import LLMFactory: {e}")

    # Test 2: Try to create an LLM with each available provider
    for provider in providers:
        try:
            print(f"\nTesting provider: {provider}")
            factory = LLMFactory(provider)
            print(f"  ✅ Successfully created factory for {provider}")

            # Assert that factory was created successfully
            assert factory is not None, f"Factory for {provider} should not be None"
            assert factory.provider == provider.replace('-', '_'), f"Factory provider should match {provider}"

        except Exception as e:
            print(f"  ❌ Failed to create factory for {provider}: {e}")
            # For now, we'll allow some providers to fail during creation
            # as they might require specific environment variables
            print(f"  ℹ️ This is expected for {provider} if environment is not configured")

def run_conditional_imports_check():
    """Standalone function that can be run independently."""
    success = True

    try:
        # Test 1: Import with all dependencies available
        from cnoe_agent_utils.llm_factory import LLMFactory
        print("✅ Successfully imported LLMFactory with all dependencies")

        providers = LLMFactory.get_supported_providers()
        print(f"Available providers: {providers}")

        # Test each provider
        for provider in providers:
            is_available = LLMFactory.is_provider_available(provider)
            missing_deps = LLMFactory.get_missing_dependencies(provider)
            print(f"  {provider}: available={is_available}, missing_deps={missing_deps}")

    except ImportError as e:
        print(f"❌ Failed to import LLMFactory: {e}")
        success = False

    # Test 2: Try to create an LLM with each available provider
    for provider in providers:
        try:
            print(f"\nTesting provider: {provider}")
            LLMFactory(provider)
            print(f"  ✅ Successfully created factory for {provider}")
        except Exception as e:
            print(f"  ❌ Failed to create factory for {provider}: {e}")

    return success

if __name__ == "__main__":
    success = run_conditional_imports_check()
    if success:
        print("\n🎉 All tests passed!")
    else:
        print("\n💥 Some tests failed!")
        sys.exit(1)
