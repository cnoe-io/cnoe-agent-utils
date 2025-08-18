# Testing Documentation

This repository uses GitHub Actions to test various LLM provider examples. Each provider has its own dedicated workflow for independent testing and reporting.

## Individual Test Workflows

### 🚀 AWS Bedrock Examples
- **Workflow**: [Test AWS Bedrock Examples](.github/workflows/test-aws-bedrock.yml)
- **Badge**: ![AWS Bedrock Tests](https://github.com/{owner}/{repo}/workflows/Test%20AWS%20Bedrock%20Examples/badge.svg)
- **Triggers**: Push to main, PR to main, manual dispatch
- **Tests**: `examples/aws_bedrock_stream.py`
- **Requirements**: AWS credentials (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION, AWS_BEDROCK_MODEL_ID)

### 🔵 Azure OpenAI Examples
- **Workflow**: [Test Azure OpenAI Examples](.github/workflows/test-azure-openai.yml)
- **Badge**: ![Azure OpenAI Tests](https://github.com/{owner}/{repo}/workflows/Test%20Azure%20OpenAI%20Examples/badge.svg)
- **Triggers**: Push to main, PR to main, manual dispatch
- **Tests**: `examples/azure_openai_example.py`
- **Requirements**: Azure OpenAI credentials (AZURE_OPENAI_API_KEY, AZURE_OPENAI_DEPLOYMENT, AZURE_OPENAI_API_VERSION, AZURE_OPENAI_ENDPOINT)

### 🤖 OpenAI Examples
- **Workflow**: [Test OpenAI Examples](.github/workflows/test-openai.yml)
- **Badge**: ![OpenAI Tests](https://github.com/{owner}/{repo}/workflows/Test%20OpenAI%20Examples/badge.svg)
- **Triggers**: Push to main, PR to main, manual dispatch
- **Tests**: `examples/openai_example.py`
- **Requirements**: OpenAI credentials (OPENAI_API_KEY, OPENAI_MODEL_NAME)

### 🌟 Google Gemini Examples
- **Workflow**: [Test Google Gemini Examples](.github/workflows/test-google-gemini.yml)
- **Badge**: ![Google Gemini Tests](https://github.com/{owner}/{repo}/workflows/Test%20Google%20Gemini%20Examples/badge.svg)
- **Triggers**: Push to main, PR to main, manual dispatch
- **Tests**: `examples/google_gemini_stream.py`
- **Requirements**: Google API key (GOOGLE_API_KEY)

### ☁️ GCP Vertex AI Examples
- **Workflow**: [Test GCP Vertex AI Examples](.github/workflows/test-gcp-vertex.yml)
- **Badge**: ![GCP Vertex AI Tests](https://github.com/{owner}/{repo}/workflows/Test%20GCP%20Vertex%20AI%20Examples/badge.svg)
- **Triggers**: Push to main, PR to main, manual dispatch
- **Tests**: `examples/gcp_vertex_stream.py`
- **Requirements**: Google Cloud credentials (GOOGLE_APPLICATION_CREDENTIALS_ENV, GOOGLE_CLOUD_PROJECT, VERTEXAI_MODEL_NAME)



## Badge Usage

To use these badges in your README.md, replace `{owner}` and `{repo}` with your actual GitHub username and repository name:

```markdown
![AWS Bedrock Tests](https://github.com/{owner}/{repo}/workflows/Test%20AWS%20Bedrock%20Examples/badge.svg)
![Azure OpenAI Tests](https://github.com/{owner}/{repo}/workflows/Test%20Azure%20OpenAI%20Examples/badge.svg)
![OpenAI Tests](https://github.com/{owner}/{repo}/workflows/Test%20OpenAI%20Examples/badge.svg)
![Google Gemini Tests](https://github.com/{owner}/{repo}/workflows/Test%20Google%20Gemini%20Examples/badge.svg)
![GCP Vertex AI Tests](https://github.com/{owner}/{repo}/workflows/Test%20GCP%20Vertex%20AI%20Examples/badge.svg)

```

## PR Reporting

Each individual workflow automatically comments on pull requests with:
- ✅ **PASS**: Test completed successfully
- ❌ **FAIL**: Test failed
- ⚠️ **SKIPPED**: Test skipped due to missing credentials

Each individual workflow provides detailed test results for its specific provider.

## Manual Testing

You can manually trigger any workflow using the GitHub Actions UI:
1. Go to the Actions tab in your repository
2. Select the workflow you want to run
3. Click "Run workflow"
4. Choose the branch and click "Run workflow"

## Credentials

Each workflow only loads the environment variables it needs, making it easier to:
- Debug credential issues
- Run tests for specific providers
- Maintain security by limiting access to secrets

## Benefits of Split Workflows

1. **Independent Badges**: Each provider has its own status badge
2. **Focused Testing**: Run only the tests you need
3. **Better Reporting**: Clear pass/fail status per provider
4. **Easier Debugging**: Isolate issues to specific providers
5. **Parallel Execution**: Multiple workflows can run simultaneously
6. **Selective Triggering**: Run specific tests without running all
