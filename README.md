# 🤖 cnoe-agent-utils

[![PyPI version](https://img.shields.io/pypi/v/cnoe-agent-utils.svg)](https://pypi.org/project/cnoe-agent-utils/)
[![Publish Python Package](https://github.com/cnoe-io/cnoe-agent-utils/actions/workflows/pypi.yml/badge.svg)](https://github.com/cnoe-io/cnoe-agent-utils/actions/workflows/pypi.yml)
[![commit-msg](https://github.com/cnoe-io/cnoe-agent-utils/actions/workflows/conventional_commits.yml/badge.svg)](https://github.com/cnoe-io/cnoe-agent-utils/actions/workflows/conventional_commits.yml)

* **Reusable utilities and abstractions** for building agent-based (LLM-powered) systems.
* **Centralized LLM Factory** supporting major providers (AWS, Azure, GCP, OpenAI, Gemini, Anthropic).
* **Centralized Tracing Utilities** (since v0.2.0) to eliminate duplicated tracing code across CNOE agents.

## Key Features

### **Core Utilities**

* Unified interface (LLM Factory) for seamless LLM instantiation across multiple clouds and vendors.
  - 🏭 **LLM Factory** for easy model instantiation across:
    - ☁️ AWS
    - ☁️ Azure
    - ☁️ GCP Vertex
    - 🤖 Google Gemini
    - 🤖 Anthropic Claude
    - 🤖 OpenAI
* Simple, environment-variable-driven configuration.
* Example scripts for each LLM provider with setup instructions.

### **Agent Tracing (since v0.2.0)**

* **Centralized tracing logic:** Removes 350+ lines of repeated code per agent.
* **Single import/decorator:** No more copy-pasting tracing logic.
* **Environment-based toggling:** Use `ENABLE_TRACING` env var to control all tracing.
* **A2A Tracing Disabling:** Single method to monkey-patch/disable agent-to-agent tracing everywhere.
* **Graceful fallback:** Works with or without Langfuse; tracing is zero-overhead when disabled.

---

**Note:** Checkout this tutorial on [Tracing](TRACING.md)

## 🚀 LLM Factory Getting Started

### 🛡️ Create and Activate a Virtual Environment

It is recommended to use a virtual environment to manage dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### ⚡ Prerequisite: Install `uv`

Before running the examples, install [`uv`](https://github.com/astral-sh/uv):

```bash
pip install uv
```

### 📦 Installation

```bash
pip install cnoe-agent-utils
```

Or, if you are developing locally:

```bash
git clone https://github.com/cnoe-agent-utils/cnoe-agent-utils.git
cd cnoe-agent-utils
poetry build
poetry install
```

---

## 🧑‍💻 Usage

To test integration with different LLM providers, configure the required environment variables for each provider as shown below. Then, run the corresponding example script using `uv`.

---

### 🤖 Anthropic

Set the following environment variables:

```bash
export ANTHROPIC_API_KEY=<your_anthropic_api_key>
export ANTHROPIC_MODEL_NAME=<model_name>
```

Run the example:

```bash
uv run examples/test_anthropic.py
```

---

### ☁️ AWS Bedrock (Anthropic Claude)

Set the following environment variables:

```bash
export AWS_PROFILE=<your_aws_profile>
export AWS_REGION=<your_aws_region>
export AWS_BEDROCK_MODEL_ID="us.anthropic.claude-3-7-sonnet-20250219-v1:0"
export AWS_BEDROCK_PROVIDER="anthropic"
```

Run the example:

```bash
uv run examples/test_aws_bedrock_claude.py
```

---

### ☁️ Azure OpenAI

Set the following environment variables:

```bash
export AZURE_OPENAI_API_KEY=<your_azure_openai_api_key>
export AZURE_OPENAI_API_VERSION=<api_version>
export AZURE_OPENAI_DEPLOYMENT=gpt-4.1
export AZURE_OPENAI_ENDPOINT=<your_azure_openai_endpoint>
```

Run the example:

```bash
uv run examples/test_azure_openai.py
```

---

### 🤖 OpenAI

Set the following environment variables:

```bash
export OPENAI_API_KEY=<your_openai_api_key>
export OPENAI_ENDPOINT=https://api.openai.com/v1
export OPENAI_MODEL_NAME=gpt-4.1
```

Optional configuration:

```bash
export OPENAI_DEFAULT_HEADERS='{"my-header-key":"my-value"}'
export OPENAI_USER=user-identifier
```

Run the example:

```bash
uv run examples/test_openai.py
```

---

### 🤖 Google Gemini

Set the following environment variable:

```bash
export GOOGLE_API_KEY=<your_google_api_key>
```

Run the example:

```bash
uv run examples/test_google_gemini.py
```

---

### ☁️ GCP Vertex AI

Set the following environment variables:

```bash
export GOOGLE_APPLICATION_CREDENTIALS=~/.config/gcp.json
export VERTEXAI_MODEL_NAME="gemini-2.0-flash-001"
```

Run the example:

```bash
uv run examples/test_gcp_vertexai.py
```

This demonstrates how to use the LLM Factory and other utilities provided by the library.

---

## 📜 License

Apache 2.0 (see [LICENSE](./LICENSE))

---

## 👥 Maintainers

See [MAINTAINERS.md](MAINTAINERS.md)

- Contributions welcome via PR or issue!