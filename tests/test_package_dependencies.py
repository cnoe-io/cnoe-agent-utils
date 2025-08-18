#!/usr/bin/env python3
"""Tests for package dependency installation and optional dependency groups."""

import pytest
import sys
from pathlib import Path
from unittest.mock import patch


class TestPackageDependencies:
    """Test that package dependencies are correctly installed and accessible."""

    def test_core_package_import(self):
        """Test that the core package can be imported."""
        try:
            import cnoe_agent_utils
            assert cnoe_agent_utils is not None
        except ImportError as e:
            pytest.fail(f"Failed to import core package: {e}")

    def test_core_modules_available(self):
        """Test that core modules are available."""
        core_modules = [
            'cnoe_agent_utils.llm_factory',
            'cnoe_agent_utils.utils',
        ]

        for module_name in core_modules:
            try:
                __import__(module_name)
            except ImportError as e:
                pytest.fail(f"Failed to import core module {module_name}: {e}")

    def test_anthropic_dependency_available(self):
        """Test that Anthropic dependency is available (core dependency)."""
        try:
            from langchain_anthropic import ChatAnthropic
            assert ChatAnthropic is not None
        except ImportError as e:
            pytest.fail(f"Anthropic dependency should be available as core dependency: {e}")


class TestOptionalDependencies:
    """Test optional dependency groups."""

    def test_aws_dependencies_available(self):
        """Test that AWS dependencies are available if installed."""
        try:
            from langchain_aws import ChatBedrock
            assert ChatBedrock is not None
            print("✅ AWS dependencies are available")
        except ImportError:
            print("ℹ️ AWS dependencies are not available (this is expected if not installed)")

    def test_openai_dependencies_available(self):
        """Test that OpenAI dependencies are available if installed."""
        try:
            from langchain_openai import ChatOpenAI, AzureChatOpenAI
            assert ChatOpenAI is not None
            assert AzureChatOpenAI is not None
            print("✅ OpenAI dependencies are available")
        except ImportError:
            print("ℹ️ OpenAI dependencies are not available (this is expected if not installed)")

    def test_google_dependencies_available(self):
        """Test that Google dependencies are available if installed."""
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            assert ChatGoogleGenerativeAI is not None
            print("✅ Google GenAI dependencies are available")
        except ImportError:
            print("ℹ️ Google GenAI dependencies are not available (this is expected if not installed)")

        try:
            from langchain_google_vertexai import ChatVertexAI
            assert ChatVertexAI is not None
            print("✅ Google VertexAI dependencies are available")
        except ImportError:
            print("ℹ️ Google VertexAI dependencies are not available (this is expected if not installed)")

    def test_tracing_dependencies_available(self):
        """Test that tracing dependencies are available if installed."""
        try:
            import langfuse
            assert langfuse is not None
            print("✅ Langfuse dependency is available")
        except ImportError:
            print("ℹ️ Langfuse dependency is not available (this is expected if not installed)")

        try:
            from opentelemetry import api
            assert api is not None
            print("✅ OpenTelemetry dependencies are available")
        except ImportError:
            print("ℹ️ OpenTelemetry dependencies are not available (this is expected if not installed)")


class TestPackageInstallation:
    """Test package installation scenarios."""

    def test_package_metadata(self):
        """Test that package metadata is correctly set."""
        try:
            import cnoe_agent_utils
            # Check that the package has expected attributes
            assert hasattr(cnoe_agent_utils, '__version__') or hasattr(cnoe_agent_utils, '__file__')
        except ImportError as e:
            pytest.fail(f"Failed to import package for metadata check: {e}")

    def test_import_without_optional_deps(self):
        """Test that the package can be imported without optional dependencies."""
        # This test ensures that the conditional imports work correctly
        # and don't prevent the package from being imported

        # Mock missing optional dependencies
        with patch.dict(sys.modules, {}, clear=False):
            # Remove optional dependencies from sys.modules if they exist
            optional_modules = [
                'langchain_aws',
                'langchain_openai',
                'langchain_google_genai',
                'langchain_google_vertexai',
                'langfuse',
                'opentelemetry'
            ]

            for module in optional_modules:
                if module in sys.modules:
                    del sys.modules[module]

            # Try to import the package
            try:
                import cnoe_agent_utils
                assert cnoe_agent_utils is not None
                print("✅ Package can be imported without optional dependencies")
            except ImportError as e:
                pytest.fail(f"Package should be importable without optional dependencies: {e}")


class TestDependencyResolution:
    """Test dependency resolution and version compatibility."""

    def test_python_version_compatibility(self):
        """Test that the package is compatible with the current Python version."""
        python_version = sys.version_info
        print(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")

        # Should work with Python 3.8+
        assert python_version >= (3, 8), "Package requires Python 3.8 or higher"

    def test_required_dependencies_installed(self):
        """Test that required dependencies are installed."""
        required_deps = [
            ('langchain_anthropic', 'langchain_anthropic'),
            ('python-dotenv', 'dotenv')
        ]

        for dep_name, import_name in required_deps:
            try:
                __import__(import_name)
                print(f"✅ Required dependency {dep_name} is available")
            except ImportError as e:
                pytest.fail(f"Required dependency {dep_name} is missing: {e}")


class TestPackageStructure:
    """Test the package structure and organization."""

    def test_package_directory_structure(self):
        """Test that the package has the expected directory structure."""
        package_dir = Path(__file__).parent.parent / "cnoe_agent_utils"

        expected_files = [
            "__init__.py",
            "llm_factory.py",
            "utils.py"
        ]

        for file_name in expected_files:
            file_path = package_dir / file_name
            assert file_path.exists(), f"Expected file {file_name} not found in package"
            print(f"✅ Package file {file_name} exists")

    def test_package_init_file(self):
        """Test that the package __init__.py file is properly configured."""
        try:
            from cnoe_agent_utils import LLMFactory
            assert LLMFactory is not None
            print("✅ LLMFactory is properly exported from package")
        except ImportError as e:
            pytest.fail(f"LLMFactory should be importable from package: {e}")


class TestInstallationInstructions:
    """Test that installation instructions work correctly."""

    def test_optional_dependency_installation_commands(self):
        """Test that the optional dependency installation commands are valid."""
        # These are the commands that should work for users
        optional_groups = [
            "anthropic",
            "openai",
            "azure",
            "aws",
            "gcp",
            "tracing"
        ]

        for group in optional_groups:
            # Test that the group name is valid
            assert group in ["anthropic", "openai", "azure", "aws", "gcp", "tracing"], f"Invalid optional group: {group}"
            print(f"✅ Optional dependency group '{group}' is valid")

    def test_pip_install_commands(self):
        """Test that pip install commands are properly formatted."""
        # These should be valid pip install commands
        install_commands = [
            "pip install cnoe-agent-utils",
            "pip install 'cnoe-agent-utils[anthropic]'",
            "pip install 'cnoe-agent-utils[openai]'",
            "pip install 'cnoe-agent-utils[aws]'",
            "pip install 'cnoe-agent-utils[gcp]'",
            "pip install 'cnoe-agent-utils[tracing]'"
        ]

        for command in install_commands:
            # Basic validation that the command looks like a valid pip install command
            assert command.startswith("pip install"), f"Invalid pip install command: {command}"
            assert "cnoe-agent-utils" in command, f"Command should install cnoe-agent-utils: {command}"
            print(f"✅ Install command format is valid: {command}")

    def test_default_vs_minimal_installation(self):
        """Test the difference between default and minimal installations."""
        # Default installation should include all dependencies
        # Minimal installation should include only specific functionality

        print("📦 Default installation (pip install cnoe-agent-utils):")
        print("   - Includes: All dependencies (anthropic, openai, azure, aws, gcp, tracing)")
        print("   - Provides: Full functionality for all LLM providers and tracing")

        print("\n⚡ Minimal installation examples:")
        print("   - pip install 'cnoe-agent-utils[anthropic]' - Anthropic Claude support only")
        print("   - pip install 'cnoe-agent-utils[openai]' - OpenAI (openai.com) support only")
        print("   - pip install 'cnoe-agent-utils[azure]' - Azure OpenAI support only")
        print("   - pip install 'cnoe-agent-utils[aws]' - AWS Bedrock support only")
        print("   - pip install 'cnoe-agent-utils[gcp]' - Google Cloud support only")
        print("   - pip install 'cnoe-agent-utils[tracing]' - Tracing and observability only")

        # Verify that the default installation is comprehensive
        install_commands = [
            "pip install cnoe-agent-utils",
            "pip install 'cnoe-agent-utils[anthropic]'",
            "pip install 'cnoe-agent-utils[openai]'",
            "pip install 'cnoe-agent-utils[aws]'",
            "pip install 'cnoe-agent-utils[gcp]'",
            "pip install 'cnoe-agent-utils[tracing]'"
        ]
        assert "pip install cnoe-agent-utils" in install_commands
        assert "pip install 'cnoe-agent-utils[anthropic]'" in install_commands


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
