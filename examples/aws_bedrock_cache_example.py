#!/usr/bin/env python3
"""
AWS Bedrock Prompt Caching Example

Demonstrates how to use prompt caching with AWS Bedrock to reduce latency
and costs for repeated context. Shows cache hit performance improvements.

Requirements:
- AWS credentials configured (AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY or AWS_PROFILE)
- AWS_REGION set
- AWS_BEDROCK_MODEL_ID set to a cache-supported model
  (see https://docs.aws.amazon.com/bedrock/latest/userguide/prompt-caching.html)
- AWS_BEDROCK_ENABLE_PROMPT_CACHE=true

Cache Benefits:
- Up to 85% reduction in latency for cached content
- Up to 90% reduction in costs for cached tokens
- 5-minute cache TTL
- Automatic cache management by AWS Bedrock
"""

import os
import sys
import time
import dotenv
from cnoe_agent_utils.llm_factory import LLMFactory

dotenv.load_dotenv()


def main():
    # Verify required environment variables
    required_vars = ["AWS_BEDROCK_MODEL_ID", "AWS_REGION"]
    missing = [var for var in required_vars if not os.getenv(var)]
    if missing:
        print(f"Error: Missing required environment variables: {', '.join(missing)}")
        sys.exit(1)

    # Enable prompt caching
    os.environ["AWS_BEDROCK_ENABLE_PROMPT_CACHE"] = "true"

    print("=" * 70)
    print("AWS Bedrock Prompt Caching Example")
    print("=" * 70)

    # Initialize LLM with caching enabled
    llm = LLMFactory("aws-bedrock").get_llm()

    # Verify caching is supported
    if not hasattr(llm, 'create_cache_point'):
        print("\n" + "=" * 70)
        print("ERROR: Prompt caching not available")
        print("=" * 70)
        print("Ensure the following:")
        print("  1. AWS_BEDROCK_ENABLE_PROMPT_CACHE=true")
        print("  2. Model supports caching (check AWS docs)")
        print(f"\nCurrent model: {os.getenv('AWS_BEDROCK_MODEL_ID')}")
        print("=" * 70)
        sys.exit(1)

    # Create a long system prompt (>1024 tokens recommended for caching benefits)
    system_context = """
You are an expert software architect with deep knowledge of distributed systems,
microservices, cloud-native architectures, and modern development practices.

Your expertise includes:
- System design and architecture patterns (microservices, event-driven, serverless)
- Cloud platforms (AWS, GCP, Azure) and their services
- Container orchestration (Kubernetes, Docker, ECS)
- CI/CD pipelines and DevOps best practices
- Database design (SQL, NoSQL, time-series, graph databases)
- API design (REST, GraphQL, gRPC)
- Security best practices and compliance
- Performance optimization and scalability
- Observability and monitoring strategies
- Infrastructure as Code (Terraform, CloudFormation, Pulumi)

When answering questions:
1. Provide architectural context and trade-offs
2. Consider scalability, reliability, and maintainability
3. Suggest industry best practices
4. Be concise but comprehensive
5. Use examples when helpful
"""

    # To enable caching, we need to use ChatPromptTemplate with cache points
    # For this example, we'll demonstrate the concept using direct invocation
    from langchain_core.messages import HumanMessage, SystemMessage

    # Create cache point for system message
    cache_point = llm.create_cache_point()

    messages_with_cache = [
        SystemMessage(content=[
            {"text": system_context},
            cache_point  # This marks the cache checkpoint
        ]),
    ]

    print("\n" + "=" * 70)
    print("Test 1: First request (COLD - Cache Miss)")
    print("=" * 70)
    print("Sending first request to establish cache...")

    start_time = time.time()
    messages = messages_with_cache + [
        HumanMessage(content="What are the key considerations for designing a microservices architecture?")
    ]
    response1 = llm.invoke(messages)
    elapsed1 = time.time() - start_time

    print(f"\nResponse (truncated): {response1.content[:200]}...")
    print(f"\nElapsed time: {elapsed1:.2f} seconds")

    # Check for usage metadata (cache statistics)
    if hasattr(response1, 'response_metadata') and 'usage' in response1.response_metadata:
        usage = response1.response_metadata['usage']
        print(f"\nToken Usage:")
        print(f"  Input tokens: {usage.get('inputTokens', 'N/A')}")
        print(f"  Output tokens: {usage.get('outputTokens', 'N/A')}")
        if 'cacheReadInputTokens' in usage:
            print(f"  Cache read tokens: {usage.get('cacheReadInputTokens', 0)}")
        if 'cacheCreationInputTokens' in usage:
            print(f"  Cache creation tokens: {usage.get('cacheCreationInputTokens', 0)}")

    # Wait a moment to ensure the cache is established
    print("\nWaiting 2 seconds for cache to propagate...")
    time.sleep(2)

    print("\n" + "=" * 70)
    print("Test 2: Second request (WARM - Cache Hit)")
    print("=" * 70)
    print("Sending second request with same system context...")

    start_time = time.time()
    messages = messages_with_cache + [
        HumanMessage(content="How do you implement effective monitoring in a distributed system?")
    ]
    response2 = llm.invoke(messages)
    elapsed2 = time.time() - start_time

    print(f"\nResponse (truncated): {response2.content[:200]}...")
    print(f"\nElapsed time: {elapsed2:.2f} seconds")

    # Check for cache hit in usage metadata
    if hasattr(response2, 'response_metadata') and 'usage' in response2.response_metadata:
        usage = response2.response_metadata['usage']
        print(f"\nToken Usage:")
        print(f"  Input tokens: {usage.get('inputTokens', 'N/A')}")
        print(f"  Output tokens: {usage.get('outputTokens', 'N/A')}")
        if 'cacheReadInputTokens' in usage:
            cache_read = usage.get('cacheReadInputTokens', 0)
            print(f"  Cache read tokens: {cache_read}")
            if cache_read > 0:
                print("  ✓ CACHE HIT! System prompt was read from cache")
        if 'cacheCreationInputTokens' in usage:
            print(f"  Cache creation tokens: {usage.get('cacheCreationInputTokens', 0)}")

    # Calculate performance improvement
    print("\n" + "=" * 70)
    print("Performance Comparison")
    print("=" * 70)
    if elapsed2 < elapsed1:
        speedup = elapsed1 / elapsed2
        time_saved = elapsed1 - elapsed2
        print(f"First request (cold):  {elapsed1:.2f}s")
        print(f"Second request (warm): {elapsed2:.2f}s")
        print(f"Time saved: {time_saved:.2f}s ({((time_saved/elapsed1)*100):.1f}% faster)")
        print(f"Speedup: {speedup:.2f}x")
    else:
        print(f"First request:  {elapsed1:.2f}s")
        print(f"Second request: {elapsed2:.2f}s")
        print("\nNote: Cache benefits are most pronounced with longer prompts (>1024 tokens)")
        print("and may not be visible in network latency-dominated scenarios.")

    print("\n" + "=" * 70)
    print("Cache Configuration")
    print("=" * 70)
    print(f"Caching enabled: {os.getenv('AWS_BEDROCK_ENABLE_PROMPT_CACHE', 'false')}")
    print(f"Model: {os.getenv('AWS_BEDROCK_MODEL_ID')}")
    print(f"Region: {os.getenv('AWS_REGION')}")
    print("\nFor more details on cache hits/misses, check CloudWatch metrics:")
    print("- CacheReadInputTokens")
    print("- CacheCreationInputTokens")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()