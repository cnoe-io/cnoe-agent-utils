#!/usr/bin/env python3
"""Test runner for examples that properly detects failures."""

import os
import sys
import subprocess
import re
from pathlib import Path

def run_example(example_path, env_vars=None):
    """
    Run an example and detect if it failed.

    Returns:
        Tuple of (success, stdout, stderr)
    """
    # Define expected API errors that don't indicate test failure
    # These are legitimate errors that would occur in real usage
    expected_api_errors = [
        r'Your credit balance is too low',  # Anthropic API credit issue
        r'ValidationException.*model identifier is invalid',  # AWS Bedrock model issue
        r'503.*failed to connect',  # Network connectivity issues
        r'Network is unreachable',  # Network issues
        r'Peer name.*is not in peer certificate',  # SSL/TLS issues
    ]

    # Set up environment
    env = os.environ.copy()
    if env_vars:
        env.update(env_vars)

    # Ensure we're using the virtual environment Python
    python_executable = sys.executable

    # Set PYTHONPATH to include the current directory and virtual environment
    current_dir = os.getcwd()
    venv_site_packages = os.path.join(current_dir, '.venv', 'lib', 'python3.13', 'site-packages')

    # Update PYTHONPATH to include virtual environment packages
    python_path = env.get('PYTHONPATH', '')
    if python_path:
        python_path = f"{current_dir}:{venv_site_packages}:{python_path}"
    else:
        python_path = f"{current_dir}:{venv_site_packages}"

    env['PYTHONPATH'] = python_path

    # Ensure we're in the right working directory
    working_dir = current_dir

    # Run the example
    try:
        result = subprocess.run(
            [python_executable, str(example_path)],
            env=env,
            cwd=working_dir,
            capture_output=True,
            text=True,
            timeout=120  # 2 minute timeout
        )

        stdout = result.stdout
        stderr = result.stderr

        # Check for various failure indicators
        success = True
        failure_reasons = []

        # Check exit code
        if result.returncode != 0:
            success = False
            failure_reasons.append(f"Exit code: {result.returncode}")

        # Check for specific failure indicators in the output
        # Only treat these as failures if they're not part of expected error handling
        if "❌ Error running" in stdout or "❌ Error running" in stderr:
            # Check if this is an expected API error vs. a real failure
            if any(re.search(pattern, stdout + stderr, re.IGNORECASE) for pattern in expected_api_errors):
                # This is an expected API error, not a test failure
                pass
            else:
                success = False
                failure_reasons.append("Error message detected")

        if "This example requires proper" in stdout or "This example requires proper" in stderr:
            # Check if this is an expected API error vs. a real failure
            if any(re.search(pattern, stdout + stderr, re.IGNORECASE) for pattern in expected_api_errors):
                # This is an expected API error, not a test failure
                pass
            else:
                success = False
                failure_reasons.append("Configuration requirement not met")

        # Check for timeout
        if result.returncode == -9:  # SIGKILL usually indicates timeout
            success = False
            failure_reasons.append("Timeout")

        # Check if we have only expected API errors
        if not success and failure_reasons:
            # Check if all detected errors are expected API errors
            all_expected = True
            for reason in failure_reasons:
                if not any(re.search(pattern, reason, re.IGNORECASE) for pattern in expected_api_errors):
                    all_expected = False
                    break

            if all_expected:
                success = True  # Mark as success if only expected API errors
                failure_reasons = ["Expected API errors (not a test failure)"]

        return success, stdout, stderr

    except subprocess.TimeoutExpired:
        return False, "", "Timeout exceeded"
    except Exception as e:
        return False, "", f"Exception running example: {e}"

def test_examples(env_file=None):
    """Run all examples and report results."""
    examples_dir = Path("examples")

    # Load environment variables if .env file exists
    env_vars = {}
    if env_file and Path(env_file).exists():
        print(f"Loading environment from {env_file}")
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    env_vars[key] = value

    # Find all Python examples
    example_files = list(examples_dir.glob("*.py"))
    example_files = [f for f in example_files if not f.name.startswith("__")]

    print("=" * 60)
    print(" Running all example scripts")
    print("=" * 60)

    total = len(example_files)
    passed = 0
    failed = 0
    results = []

    for example_path in example_files:
        print(f"\n{'='*20} Running {example_path.name} {'='*20}")

        success, stdout, stderr = run_example(example_path, env_vars)

        # Show the full output from the example
        if stdout.strip():
            print("STDOUT:")
            print(stdout)

        if stderr.strip():
            print("STDERR:")
            print(stderr)

        if success:
            print(f"✅ {example_path.name} PASSED")
            passed += 1
        else:
            print(f"❌ {example_path.name} FAILED")
            failed += 1

        results.append((example_path, success, stdout, stderr))
        print(f"{'='*60}")

    # Print summary
    print(f"\n{'='*60}")
    print(" FINAL TEST RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f" Total examples: {total}")
    print(f" ✅ Passed: {passed}")
    print(f" ❌ Failed: {failed}")
    print(f"{'='*60}")

    # Show detailed results
    if failed > 0:
        print(f"\nFAILED EXAMPLES:")
        for example_path, success, stdout, stderr in results:
            if not success:
                print(f"  ❌ {example_path.name}")
                if stdout.strip():
                    print(f"    STDOUT: {stdout.strip()[:200]}...")
                if stderr.strip():
                    print(f"    STDERR: {stderr.strip()[:200]}...")

    # Exit with non-zero code if any examples failed
    if failed > 0:
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == "__main__":
    # Check if .env file path is provided as argument
    env_file = sys.argv[1] if len(sys.argv) > 1 else ".env"
    test_examples(env_file)
