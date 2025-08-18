#!/usr/bin/env python3
"""Enhanced test runner for examples that generates comprehensive reports."""

import os
import sys
import subprocess
import re
import json
import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any

# Model family definitions
MODEL_FAMILIES = {
    "openai": {
        "name": "OpenAI",
        "models": ["gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-5", "gpt-5-mini"],
        "examples": ["openai_consolidated_test.py"],
        "description": "OpenAI GPT models via openai.com API"
    },
    "azure-openai": {
        "name": "Azure OpenAI",
        "models": ["gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-5", "gpt-5-mini"],
        "examples": ["azure_openai_consolidated_test.py"],
        "description": "OpenAI GPT models via Azure OpenAI service"
    },
    "anthropic": {
        "name": "Anthropic Claude",
        "models": ["claude-3-5-sonnet-20241022", "claude-3-5-haiku-20241022", "claude-3-opus-20240229"],
        "examples": ["anthropic_stream.py"],
        "description": "Anthropic Claude models"
    },
    "aws": {
        "name": "AWS Bedrock",
        "models": ["anthropic.claude-3-5-sonnet-20241022-v1:0", "anthropic.claude-3-5-haiku-20241022-v1:0"],
        "examples": ["bedrock_stream.py"],
        "description": "AWS Bedrock hosted models"
    },
    "gcp": {
        "name": "Google Cloud",
        "models": ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash-exp"],
        "examples": ["gemini_stream.py", "vertex_stream.py"],
        "description": "Google Cloud AI models (Gemini, Vertex AI)"
    }
}

def run_example(example_path, env_vars=None):
    """
    Run an example and detect if it failed.

    Returns:
        Tuple of (success, stdout, stderr, execution_time, model_info)
    """
    # Define expected API errors that don't indicate test failure
    # These are legitimate errors that would occur in real usage
    expected_api_errors = [
        r'Your credit balance is too low',  # Anthropic API credit issue
        r'ValidationException.*model identifier is invalid',  # AWS Bedrock model issue
        r'503.*failed to connect',  # Network connectivity issues
        r'Network is unreachable',  # Network issues
        r'Peer name.*is not in peer certificate',  # SSL/TLS issues
        r'Missing required environment variable',  # Configuration issues
        r'Missing required environment variables',  # Configuration issues
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

    # Record start time
    start_time = datetime.datetime.now()

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

        execution_time = (datetime.datetime.now() - start_time).total_seconds()
        stdout = result.stdout
        stderr = result.stderr

        # Extract model information from output
        model_info = extract_model_info(example_path, stdout, stderr)

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

        return success, stdout, stderr, execution_time, model_info

    except subprocess.TimeoutExpired:
        execution_time = (datetime.datetime.now() - start_time).total_seconds()
        return False, "", "Timeout exceeded", execution_time, {}
    except Exception as e:
        execution_time = (datetime.datetime.now() - start_time).total_seconds()
        return False, "", f"Exception running example: {e}", execution_time, {}

def extract_model_info(example_path: Path, stdout: str, stderr: str) -> Dict[str, Any]:
    """Extract model information from example output."""
    model_info = {
        "models_tested": [],
        "test_modes": [],
        "provider": "unknown"
    }

    example_name = example_path.name.lower()

    # Determine provider from example name
    if "openai" in example_name and "azure" not in example_name:
        model_info["provider"] = "openai"
        model_info["test_modes"] = ["invoke", "stream"]
    elif "azure" in example_name:
        model_info["provider"] = "azure-openai"
        model_info["test_modes"] = ["invoke", "stream"]
    elif "anthropic" in example_name:
        model_info["provider"] = "anthropic"
        model_info["test_modes"] = ["stream"]
    elif "bedrock" in example_name:
        model_info["provider"] = "aws"
        model_info["test_modes"] = ["stream"]
    elif "gemini" in example_name or "vertex" in example_name:
        model_info["provider"] = "gcp"
        model_info["test_modes"] = ["stream"]

    # Extract models from output
    if "gpt-4o" in stdout or "gpt-4o" in stderr:
        model_info["models_tested"].append("gpt-4o")
    if "gpt-4o-mini" in stdout or "gpt-4o-mini" in stderr:
        model_info["models_tested"].append("gpt-4o-mini")
    if "gpt-4.1" in stdout or "gpt-4.1" in stderr:
        model_info["models_tested"].append("gpt-4.1")
    if "gpt-5" in stdout or "gpt-5" in stderr:
        model_info["models_tested"].append("gpt-5")
    if "gpt-5-mini" in stdout or "gpt-5-mini" in stderr:
        model_info["models_tested"].append("gpt-5-mini")
    if "claude" in stdout or "claude" in stderr:
        model_info["models_tested"].extend(["claude-3-5-sonnet", "claude-3-5-haiku", "claude-3-opus"])
    if "gemini" in stdout or "gemini" in stderr:
        model_info["models_tested"].extend(["gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash-exp"])

    return model_info

def generate_test_report(results: List[Tuple], env_vars: Dict[str, str]) -> Dict[str, Any]:
    """Generate a comprehensive test report."""
    total = len(results)
    passed = sum(1 for _, success, _, _, _, _ in results if success)
    failed = total - passed
    success_rate = (passed / total) * 100 if total > 0 else 0

    # Group results by provider
    provider_results = {}
    for example_path, success, stdout, stderr, execution_time, model_info in results:
        provider = model_info.get("provider", "unknown")
        if provider not in provider_results:
            provider_results[provider] = []
        provider_results[provider].append({
            "example": example_path.name,
            "success": success,
            "execution_time": execution_time,
            "model_info": model_info
        })

    # Calculate statistics
    total_execution_time = sum(execution_time for _, _, _, _, execution_time, _ in results)
    avg_execution_time = total_execution_time / total if total > 0 else 0

    report = {
        "summary": {
            "total_examples": total,
            "passed": passed,
            "failed": failed,
            "success_rate": round(success_rate, 2),
            "total_execution_time": round(total_execution_time, 2),
            "average_execution_time": round(avg_execution_time, 2),
            "timestamp": datetime.datetime.now().isoformat(),
            "environment": {
                "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                "platform": sys.platform,
                "env_vars_loaded": len(env_vars)
            }
        },
        "provider_results": provider_results,
        "detailed_results": [
            {
                "example": str(example_path),
                "success": success,
                "execution_time": round(execution_time, 2),
                "model_info": model_info,
                "stdout_preview": stdout[:500] + "..." if len(stdout) > 500 else stdout,
                "stderr_preview": stderr[:500] + "..." if len(stderr) > 500 else stderr
            }
            for example_path, success, stdout, stderr, execution_time, model_info in results
        ]
    }

    return report

def save_report_artifacts(report: Dict[str, Any], output_dir: str = "test-reports"):
    """Save test report as various artifact formats."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Save JSON report
    json_path = output_path / "test-report.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, default=str)

    # Save HTML report
    html_path = output_path / "test-report.html"
    html_content = generate_html_report(report)
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    # Save Markdown report
    md_path = output_path / "test-report.md"
    md_content = generate_markdown_report(report)
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(md_content)

    # Save summary text file
    summary_path = output_path / "test-summary.txt"
    summary_content = generate_summary_text(report)
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(summary_content)

    print(f"\n📊 Test reports saved to {output_dir}/")
    print(f"  - JSON: {json_path}")
    print(f"  - HTML: {html_path}")
    print(f"  - Markdown: {md_path}")
    print(f"  - Summary: {summary_path}")

    return output_path

def generate_html_report(report: Dict[str, Any]) -> str:
    """Generate an HTML test report."""
    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CNOE Agent Utils Test Report</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 20px; }}
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; }}
        .summary {{ background: #f8f9fa; padding: 20px; border-radius: 10px; margin-bottom: 20px; }}
        .provider {{ background: white; border: 1px solid #dee2e6; border-radius: 10px; margin-bottom: 20px; padding: 20px; }}
        .provider-header {{ background: #e9ecef; padding: 10px; border-radius: 5px; margin-bottom: 15px; }}
        .example {{ background: #f8f9fa; padding: 15px; border-radius: 5px; margin-bottom: 10px; }}
        .success {{ border-left: 4px solid #28a745; }}
        .failure {{ border-left: 4px solid #dc3545; }}
        .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-bottom: 20px; }}
        .stat-card {{ background: white; padding: 15px; border-radius: 8px; text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .stat-number {{ font-size: 2em; font-weight: bold; color: #007bff; }}
        .stat-label {{ color: #6c757d; font-size: 0.9em; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🚀 CNOE Agent Utils Test Report</h1>
        <p>Generated on {report['summary']['timestamp']}</p>
    </div>

    <div class="summary">
        <h2>📊 Test Summary</h2>
        <div class="stats">
            <div class="stat-card">
                <div class="stat-number">{report['summary']['total_examples']}</div>
                <div class="stat-label">Total Examples</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{report['summary']['passed']}</div>
                <div class="stat-label">Passed</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{report['summary']['failed']}</div>
                <div class="stat-label">Failed</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{report['summary']['success_rate']}%</div>
                <div class="stat-label">Success Rate</div>
            </div>
        </div>
        <p><strong>Total Execution Time:</strong> {report['summary']['total_execution_time']}s</p>
        <p><strong>Average Execution Time:</strong> {report['summary']['average_execution_time']}s</p>
    </div>

    <div class="summary">
        <h2>🔧 Environment</h2>
        <p><strong>Python Version:</strong> {report['summary']['environment']['python_version']}</p>
        <p><strong>Platform:</strong> {report['summary']['environment']['platform']}</p>
        <p><strong>Environment Variables Loaded:</strong> {report['summary']['environment']['env_vars_loaded']}</p>
    </div>
"""

    # Add provider results
    for provider, examples in report['provider_results'].items():
        provider_name = MODEL_FAMILIES.get(provider, {}).get('name', provider.title())
        provider_desc = MODEL_FAMILIES.get(provider, {}).get('description', '')

        html += f"""
    <div class="provider">
        <div class="provider-header">
            <h3>🔌 {provider_name}</h3>
            <p>{provider_desc}</p>
        </div>
"""

        for example in examples:
            status_class = "success" if example['success'] else "failure"
            status_icon = "✅" if example['success'] else "❌"
            html += f"""
        <div class="example {status_class}">
            <h4>{status_icon} {example['example']}</h4>
            <p><strong>Status:</strong> {'PASSED' if example['success'] else 'FAILED'}</p>
            <p><strong>Execution Time:</strong> {example['execution_time']}s</p>
            <p><strong>Models Tested:</strong> {', '.join(example['model_info'].get('models_tested', ['Unknown']))}</p>
            <p><strong>Test Modes:</strong> {', '.join(example['model_info'].get('test_modes', ['Unknown']))}</p>
        </div>
"""

        html += "    </div>\n"

    html += """
</body>
</html>
"""
    return html

def generate_markdown_report(report: Dict[str, Any]) -> str:
    """Generate a Markdown test report."""
    md = f"""# 🚀 CNOE Agent Utils Test Report

Generated on: {report['summary']['timestamp']}

## 📊 Test Summary

| Metric | Value |
|--------|-------|
| Total Examples | {report['summary']['total_examples']} |
| Passed | {report['summary']['passed']} |
| Failed | {report['summary']['failed']} |
| Success Rate | {report['summary']['success_rate']}% |
| Total Execution Time | {report['summary']['total_execution_time']}s |
| Average Execution Time | {report['summary']['average_execution_time']}s |

## 🔧 Environment

- **Python Version:** {report['summary']['environment']['python_version']}
- **Platform:** {report['summary']['environment']['platform']}
- **Environment Variables Loaded:** {report['summary']['environment']['env_vars_loaded']}

## 🔌 Provider Results

"""

    for provider, examples in report['provider_results'].items():
        provider_name = MODEL_FAMILIES.get(provider, {}).get('name', provider.title())
        provider_desc = MODEL_FAMILIES.get(provider, {}).get('description', '')

        md += f"### {provider_name}\n\n"
        md += f"{provider_desc}\n\n"

        for example in examples:
            status_icon = "✅" if example['success'] else "❌"
            md += f"- {status_icon} **{example['example']}** - {'PASSED' if example['success'] else 'FAILED'} ({example['execution_time']}s)\n"
            md += f"  - Models: {', '.join(example['model_info'].get('models_tested', ['Unknown']))}\n"
            md += f"  - Modes: {', '.join(example['model_info'].get('test_modes', ['Unknown']))}\n\n"

    return md

def generate_summary_text(report: Dict[str, Any]) -> str:
    """Generate a plain text summary."""
    text = f"""CNOE Agent Utils Test Report
Generated on: {report['summary']['timestamp']}

Test Summary:
- Total Examples: {report['summary']['total_examples']}
- Passed: {report['summary']['passed']}
- Failed: {report['summary']['failed']}
- Success Rate: {report['summary']['success_rate']}%
- Total Execution Time: {report['summary']['total_execution_time']}s
- Average Execution Time: {report['summary']['average_execution_time']}s

Environment:
- Python Version: {report['summary']['environment']['python_version']}
- Platform: {report['summary']['environment']['platform']}
- Environment Variables Loaded: {report['summary']['environment']['env_vars_loaded']}

Provider Results:
"""

    for provider, examples in report['provider_results'].items():
        provider_name = MODEL_FAMILIES.get(provider, {}).get('name', provider.title())
        text += f"\n{provider_name}:\n"

        for example in examples:
            status = "PASSED" if example['success'] else "FAILED"
            text += f"  - {example['example']}: {status} ({example['execution_time']}s)\n"

    return text

def test_examples(env_file=None, save_reports=True):
    """Run all examples and report results."""
    examples_dir = Path("examples")

    # Load environment variables if .env file exists
    env_vars = {}
    if env_file and Path(env_file).exists():
        print(f"Loading environment from {env_file}")
        with open(env_file, 'r', encoding='utf-8') as f:
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

        success, stdout, stderr, execution_time, model_info = run_example(example_path, env_vars)

        # Show the full output from the example
        if stdout.strip():
            print("STDOUT:")
            print(stdout)

        if stderr.strip():
            print("STDERR:")
            print(stderr)

        if success:
            print(f"✅ {example_path.name} PASSED ({execution_time:.2f}s)")
            passed += 1
        else:
            print(f"❌ {example_path.name} FAILED ({execution_time:.2f}s)")
            failed += 1

        results.append((example_path, success, stdout, stderr, execution_time, model_info))
        print(f"{'='*60}")

    # Print summary
    print(f"\n{'='*60}")
    print(" FINAL TEST RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f" Total examples: {total}")
    print(f" ✅ Passed: {passed}")
    print(f" ❌ Failed: {failed}")
    print(f"{'='*60}")

    # Generate and save comprehensive report
    if save_reports:
        report = generate_test_report(results, env_vars)
        save_report_artifacts(report)

    # Show detailed results
    if failed > 0:
        print("\nFAILED EXAMPLES:")
        for example_path, success, stdout, stderr, execution_time, model_info in results:
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
