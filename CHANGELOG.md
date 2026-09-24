## 0.6.0 (2026-09-24)

### BREAKING CHANGE

- **agents**: removes `get_context_limit_for_provider`, `get_context_config`, `log_context_config`, `DEFAULT_PROVIDER_CONTEXT_LIMITS`, and `PROVIDER_ENV_VARS` from `cnoe_agent_utils.agents.context_config` (and their re-exports). Also drops `MAX_CONTEXT_TOKENS`, `AZURE_OPENAI_MAX_CONTEXT_TOKENS`, `OPENAI_MAX_CONTEXT_TOKENS`, `AWS_BEDROCK_MAX_CONTEXT_TOKENS`, `ANTHROPIC_MAX_CONTEXT_TOKENS`, `GOOGLE_GEMINI_MAX_CONTEXT_TOKENS`, `GCP_VERTEXAI_MAX_CONTEXT_TOKENS`. Use `get_context_limit_for_model(model)` instead.
- **agents**: removes `BaseLangGraphAgentExecutor`, `BaseStrandsAgentExecutor`, and the `a2a-sdk` dependency / `[a2a]` extra. `disable_a2a_tracing`/`is_a2a_disabled` (tracing) are unaffected.

### Feat

- **agents**: derive agent context limits from the constructed model's own `.profile["max_input_tokens"]` instead of a hand-maintained per-provider table

### Fix

- **aws-bedrock**: degrade gracefully (retry with `base_model_id=""`) when `bedrock:GetInferenceProfile` is denied for an application-inference-profile ARN, instead of failing agent construction

### Build

- upgrade langchain-*, boto3/botocore, langgraph, langchain-core, langchain-mcp-adapters, opentelemetry-*, google-auth, pydantic, tiktoken, typing-extensions, and dev tooling to latest compatible releases; `requires-python` moves to `>=3.14`
- hold `langfuse` at 3.15.0 and `google-cloud-aiplatform` at 1.165.1 below their breaking 4.x/2.x rewrites, pending dedicated migration follow-up

## 0.5.1 (2026-09-24)

### Fix

- **llm**: keep max_tokens above manual thinking budgets

## 0.5.0 (2026-09-22)

### Feat

- **llm**: add portable reasoning effort across supported providers (#43)

## 0.4.2 (2026-09-03)

### Fix

- don't send default temperature=0 to Claude Opus/Sonnet 5+ models (#40)

## 0.4.1 (2026-06-05)

### Feat

- **llm**: expose bedrock client resolution helper

### Fix

- **llm**: handle Bedrock client params for Anthropic
- **llm**: prefer Anthropic Bedrock client for Claude models

## 0.4.0 (2026-05-01)

### Feat

- **llm_factory**: lazy provider imports to reduce baseline memory ~200MB

## 0.3.15 (2026-03-23)

### Feat

- **llm**: add configurable Bedrock timeout support

## 0.3.14 (2026-03-18)

### Feat

- **aws-bedrock**: support AWS_BEDROCK_BASE_MODEL_ID to skip GetInferenceProfile (#34)

## 0.3.13 (2026-03-08)

### Feat

- **llm**: add model override parameter to get_llm()

## 0.3.12 (2026-02-26)

### Fix

- **tracing**: forward **kwargs in trace_agent_stream decorator (#32)

## 0.3.11 (2026-02-24)

### Fix

- **tracing**: suppress OTel context-detach ERROR logs in async generators (#31)

## 0.3.10 (2026-02-18)

### Fix

- **tracing**: silence OTel context detach error in async generators (#30)

## 0.3.9 (2025-12-08)

### Feat

- synchronize agent base classes with ai-platform-engineering
- **middleware**: add custom deepagents middleware

### Fix

- **lint**: fix linting

## 0.3.6 (2025-11-10)

### Feat

- agent base classes (#26)

### Fix

- OpenAI default_headers config option

## 0.3.3 (2025-10-07)

### Feat

- add extended thinking support for AWS Bedrock, Anthropic, and Vertex AI Claude 4+ models

## 0.3.2 (2025-10-02)

### Fix

- **Makefile**: use folder name cnoe_agent_utils

## 0.3.1 (2025-09-30)

### Feat

- make streaming option configurable for AWS bedrock models

### Fix

- lint errors
- lint errors
- **conventional-commits**: add bump, release

### Refactor

- **aws-bedrock**: simplify prompt caching implementation

## 0.3.0 (2025-08-17)

### Feat

- Add OpenAI config options
- add gpt-5 and streaming support

### Fix

- update README.md
- uv sync
- updates
- revert pyproject.toml to 0.2.2
- use uv in .cz.toml
- **pyproject.toml**: remove all package and update default dependency
- updates
- updates
- updates
- update new GHA tests
- add coverage as comment
- unit tests gha
- update gcp creds for example run
- unit tests
- update GHA
- update GHA
- updates
- lint
- updates
- 'LLMFactory' object has no attribute 'provider'

### Refactor

- add unit tests and examples

## 0.2.2 (2025-08-07)

### Fix

- release 0.2.2

## 0.2.1 (2025-08-07)

### Fix

- remove the unused is_generic_processing_message
- trace input does not match the user input
- silence unwanted logs llmfactory
- docs
- update docs

## 0.1.5 (2025-07-23)

### Feat

- **tracing**: add package for tracing

## 0.1.4 (2025-06-03)

### Fix

- **aws**: accept IAM AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY

## 0.1.3 (2025-06-03)

### Fix

- bump 0.1.3
- update changelog

## 0.1.2 (2025-06-03)

### Fix

- update README.md
- sign-off
- changelog

## 0.1.1 (2025-06-03)

### Fix

- bump 0.1.1
- **examples**: update examples

## 0.1.0 (2025-06-02)

### Fix

- update pyproject.toml
