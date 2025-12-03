"""
Low-level LLM client for OpenRouter API.

This module provides a client for interacting with LLM providers
through OpenRouter, with retry logic, rate limiting, and cost tracking.
"""

import asyncio
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, AsyncGenerator
import httpx

from palet8_agents.core.config import ModelProfile, get_config
from palet8_agents.core.exceptions import (
    LLMClientError,
    LLMTimeoutError,
    LLMResponseError,
    RateLimitError,
)


@dataclass
class LLMResponse:
    """
    Response from an LLM API call.

    Contains the generated content along with usage statistics
    and cost information.
    """
    content: str
    model: str
    finish_reason: str = "stop"

    # Token usage
    tokens_input: int = 0
    tokens_output: int = 0
    tokens_total: int = 0

    # Cost (calculated based on model pricing)
    cost_usd: float = 0.0

    # Timing
    latency_ms: int = 0

    # Raw response for debugging
    raw_response: Dict[str, Any] = field(default_factory=dict)

    # Tool calls if any
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "content": self.content,
            "model": self.model,
            "finish_reason": self.finish_reason,
            "tokens_input": self.tokens_input,
            "tokens_output": self.tokens_output,
            "tokens_total": self.tokens_total,
            "cost_usd": self.cost_usd,
            "latency_ms": self.latency_ms,
            "tool_calls": self.tool_calls,
        }


class LLMClient:
    """
    Low-level client for LLM API interactions via OpenRouter.

    Features:
    - Automatic retry with exponential backoff
    - Rate limiting awareness
    - Cost tracking
    - Streaming support
    - Tool/function calling support
    """

    # OpenRouter API endpoint
    BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(
        self,
        api_key: Optional[str] = None,
        timeout: float = 120.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        """
        Initialize the LLM client.

        Args:
            api_key: OpenRouter API key. If None, reads from environment.
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            retry_delay: Initial delay between retries (doubles each retry)
        """
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # HTTP client
        self._client: Optional[httpx.AsyncClient] = None

        # Rate limiting tracking
        self._last_request_time: float = 0
        self._min_request_interval: float = 0.1  # 100ms between requests

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout),
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://palet8.com",
                    "X-Title": "Palet8 Agent System",
                },
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def _wait_for_rate_limit(self) -> None:
        """Ensure minimum interval between requests."""
        now = time.time()
        elapsed = now - self._last_request_time
        if elapsed < self._min_request_interval:
            await asyncio.sleep(self._min_request_interval - elapsed)
        self._last_request_time = time.time()

    async def chat(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        profile: Optional[ModelProfile] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> LLMResponse:
        """
        Send a chat completion request.

        Args:
            messages: List of messages in OpenAI format
            model: Model ID to use (overrides profile)
            profile: ModelProfile to use for configuration
            tools: List of tool definitions for function calling
            temperature: Sampling temperature (overrides profile)
            max_tokens: Maximum tokens to generate (overrides profile)
            **kwargs: Additional parameters for the API

        Returns:
            LLMResponse with generated content and usage stats

        Raises:
            LLMClientError: For API errors
            RateLimitError: When rate limited
            LLMTimeoutError: On timeout
        """
        # Determine model and parameters
        if profile:
            model = model or profile.primary_model
            temperature = temperature if temperature is not None else profile.temperature
            max_tokens = max_tokens or profile.max_tokens
        else:
            model = model or ""
            temperature = temperature if temperature is not None else 0.7
            max_tokens = max_tokens or 1000

        if not model:
            raise LLMClientError("No model specified")

        # Build request payload
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = kwargs.get("tool_choice", "auto")

        # Add any additional kwargs
        for key, value in kwargs.items():
            if key not in payload and value is not None:
                payload[key] = value

        # Execute with retry
        return await self._execute_with_retry(payload, profile)

    async def _execute_with_retry(
        self,
        payload: Dict[str, Any],
        profile: Optional[ModelProfile] = None,
    ) -> LLMResponse:
        """Execute request with retry logic."""
        client = await self._get_client()
        last_error: Optional[Exception] = None
        model = payload.get("model", "")

        max_retries = profile.max_retries if profile else self.max_retries
        retry_delay = profile.retry_delay_seconds if profile else self.retry_delay

        for attempt in range(max_retries + 1):
            try:
                await self._wait_for_rate_limit()

                start_time = time.time()
                response = await client.post(
                    f"{self.BASE_URL}/chat/completions",
                    json=payload,
                )
                latency_ms = int((time.time() - start_time) * 1000)

                # Handle rate limiting
                if response.status_code == 429:
                    retry_after = int(response.headers.get("Retry-After", retry_delay * (2 ** attempt)))
                    if attempt < max_retries:
                        await asyncio.sleep(retry_after)
                        continue
                    raise RateLimitError(
                        f"Rate limit exceeded for {model}",
                        retry_after=retry_after,
                    )

                # Handle other errors
                if response.status_code != 200:
                    error_body = response.text
                    try:
                        error_data = response.json()
                        error_msg = error_data.get("error", {}).get("message", error_body)
                    except Exception:
                        error_msg = error_body

                    if response.status_code >= 500 and attempt < max_retries:
                        await asyncio.sleep(retry_delay * (2 ** attempt))
                        continue

                    raise LLMClientError(
                        f"API error ({response.status_code}): {error_msg}",
                        details={"status_code": response.status_code},
                    )

                # Parse successful response
                data = response.json()
                return self._parse_response(data, model, latency_ms, profile)

            except httpx.TimeoutException as e:
                last_error = e
                if attempt < max_retries:
                    await asyncio.sleep(retry_delay * (2 ** attempt))
                    continue
                raise LLMTimeoutError(f"Request timed out after {self.timeout}s")

            except (RateLimitError, LLMClientError):
                raise

            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    await asyncio.sleep(retry_delay * (2 ** attempt))
                    continue
                raise LLMClientError(f"Unexpected error: {str(e)}")

        raise LLMClientError(f"Max retries exceeded: {str(last_error)}")

    def _parse_response(
        self,
        data: Dict[str, Any],
        model: str,
        latency_ms: int,
        profile: Optional[ModelProfile] = None,
    ) -> LLMResponse:
        """Parse API response into LLMResponse."""
        try:
            choice = data.get("choices", [{}])[0]
            message = choice.get("message", {})
            usage = data.get("usage", {})

            content = message.get("content", "")
            finish_reason = choice.get("finish_reason", "stop")

            tokens_input = usage.get("prompt_tokens", 0)
            tokens_output = usage.get("completion_tokens", 0)
            tokens_total = usage.get("total_tokens", tokens_input + tokens_output)

            # Calculate cost
            cost_usd = 0.0
            if profile:
                cost_usd = (
                    (tokens_input / 1000) * profile.cost_per_1k_input +
                    (tokens_output / 1000) * profile.cost_per_1k_output
                )

            # Extract tool calls if present
            tool_calls = []
            if "tool_calls" in message:
                for tc in message["tool_calls"]:
                    tool_calls.append({
                        "id": tc.get("id", ""),
                        "type": tc.get("type", "function"),
                        "function": {
                            "name": tc.get("function", {}).get("name", ""),
                            "arguments": tc.get("function", {}).get("arguments", "{}"),
                        },
                    })

            return LLMResponse(
                content=content,
                model=data.get("model", model),
                finish_reason=finish_reason,
                tokens_input=tokens_input,
                tokens_output=tokens_output,
                tokens_total=tokens_total,
                cost_usd=cost_usd,
                latency_ms=latency_ms,
                raw_response=data,
                tool_calls=tool_calls,
            )

        except Exception as e:
            raise LLMResponseError(f"Failed to parse response: {str(e)}")

    async def chat_stream(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        profile: Optional[ModelProfile] = None,
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        """
        Send a streaming chat completion request.

        Args:
            messages: List of messages in OpenAI format
            model: Model ID to use
            profile: ModelProfile for configuration
            **kwargs: Additional parameters

        Yields:
            String chunks of the generated content
        """
        if profile:
            model = model or profile.primary_model
            kwargs.setdefault("temperature", profile.temperature)
            kwargs.setdefault("max_tokens", profile.max_tokens)

        if not model:
            raise LLMClientError("No model specified")

        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
            **kwargs,
        }

        client = await self._get_client()
        await self._wait_for_rate_limit()

        async with client.stream(
            "POST",
            f"{self.BASE_URL}/chat/completions",
            json=payload,
        ) as response:
            if response.status_code != 200:
                error_text = await response.aread()
                raise LLMClientError(f"Streaming error: {error_text.decode()}")

            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str.strip() == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                        delta = data.get("choices", [{}])[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            yield content
                    except json.JSONDecodeError:
                        continue

    async def __aenter__(self) -> "LLMClient":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()


# Convenience function for one-off requests
async def chat_completion(
    messages: List[Dict[str, Any]],
    model: str,
    **kwargs,
) -> LLMResponse:
    """
    One-off chat completion request.

    Args:
        messages: List of messages
        model: Model ID
        **kwargs: Additional parameters

    Returns:
        LLMResponse with result
    """
    async with LLMClient() as client:
        return await client.chat(messages, model=model, **kwargs)
