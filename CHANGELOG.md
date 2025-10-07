## 0.3.3 (Unreleased)

### Feat

- **llm_factory**: add extended thinking support for AWS Bedrock, Anthropic, and Vertex AI Claude 4+ models
  - New environment variables: `AWS_BEDROCK_THINKING_ENABLED`, `AWS_BEDROCK_THINKING_BUDGET`
  - New environment variables: `ANTHROPIC_THINKING_ENABLED`, `ANTHROPIC_THINKING_BUDGET`
  - New environment variables: `VERTEXAI_THINKING_ENABLED`, `VERTEXAI_THINKING_BUDGET`
  - Default thinking budget: 1024 tokens (minimum required), maximum: max_tokens if provided
  - Automatic validation and clamping to max_tokens when provided

### Refactor

- **llm_factory**: extract duplicate thinking configuration logic into `_parse_thinking_budget()` helper
- **llm_factory**: add `ThinkingConfig` TypedDict for type safety
- **llm_factory**: define constants `THINKING_DEFAULT_BUDGET` and `THINKING_MIN_BUDGET`

### Fix

- **llm_factory**: fix `max_tokens` validation bug in thinking configuration
- **llm_factory**: automatically reset temperature to 0 and remove top_p/top_k parameters when extended thinking is enabled (thinking is incompatible with these parameters)
- **llm_factory**: fix `model_kwargs` clobbering bug in AWS Bedrock builder - merge `response_format` into existing kwargs instead of replacing, preserving extended thinking configuration

### Tests

- **test_extended_thinking**: add comprehensive test suite with 45 tests covering all aspects of extended thinking including parameter compatibility and model_kwargs preservation

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
