# cnoe-agent-utils

**cnoe-agent-utils** is an open-source Python library providing utility functions and abstractions for building agent-based systems, including LLM (Large Language Model) factories and integrations.

## Features

- LLM Factory for easy model instantiation across AWS, Azure, GCP Vertex, Google Gemini, Anthropic Claude, and OpenAI
- Utilities for agent orchestration
- Extensible and modular design

## Getting Started

### Create and Activate a Virtual Environment

It is recommended to use a virtual environment to manage dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Installation

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

## Usage

Run the example script to test integration with AWS Bedrock Claude:

```bash
uv run examples/test_aws_bedrock_claude.py
```

This will demonstrate how to use the LLM Factory and other utilities provided by the library.

---

## 📜 License

Apache 2.0 (see [LICENSE](./LICENSE))

---

## 👥 Maintainers

See [MAINTAINERS.md](MAINTAINERS.md)

- Contributions welcome via PR or issue!