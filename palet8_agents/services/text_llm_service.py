"""
Text LLM Service

Pure model tooling service for text generation using LLM models via OpenRouter.
Provides generic text generation with automatic failover support.

This service has NO domain-specific logic or system prompts.
Domain-specific prompts belong in:
- PromptComposerService: Prompt composition
- ReasoningService: Quality assessment, classification, etc.

Documentation Reference: Section 4.1
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, AsyncGenerator
import logging

from palet8_agents.core.llm_client import LLMClient, LLMResponse
from palet8_agents.core.config import get_config, ModelProfile
from palet8_agents.core.exceptions import (
    LLMClientError,
    LLMTimeoutError,
    RateLimitError,
)

logger = logging.getLogger(__name__)


class TextLLMServiceError(Exception):
    """Base exception for TextLLMService errors."""
    pass


class ModelNotAvailableError(TextLLMServiceError):
    """Raised when no models are available (primary and fallback failed)."""
    pass


@dataclass
class TextGenerationResult:
    """Result from text generation."""
    content: str
    model_used: str
    tokens_input: int = 0
    tokens_output: int = 0
    cost_usd: float = 0.0
    latency_ms: int = 0
    used_fallback: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "content": self.content,
            "model_used": self.model_used,
            "tokens_input": self.tokens_input,
            "tokens_output": self.tokens_output,
            "cost_usd": self.cost_usd,
            "latency_ms": self.latency_ms,
            "used_fallback": self.used_fallback,
            "metadata": self.metadata,
        }


class TextLLMService:
    """
    High-level service for text generation.

    Features:
    - Automatic failover from primary to fallback model
    - Profile-based configuration
    - Cost tracking
    - Specialized methods for common tasks
    """

    # Failover triggers: HTTP status codes that trigger fallback
    FAILOVER_STATUS_CODES = {429, 500, 502, 503, 504}

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        default_profile_name: str = "pali",
    ):
        """
        Initialize TextLLMService.

        Args:
            llm_client: Optional LLMClient instance. Creates new one if None.
            default_profile_name: Default model profile to use.
        """
        self._client = llm_client
        self._owns_client = llm_client is None
        self.default_profile_name = default_profile_name
        self._config = get_config()

    async def _get_client(self) -> LLMClient:
        """Get or create LLM client."""
        if self._client is None:
            self._client = LLMClient()
        return self._client

    async def close(self) -> None:
        """Close the service and release resources."""
        if self._client and self._owns_client:
            await self._client.close()
            self._client = None

    def _get_profile(self, profile_name: Optional[str] = None) -> Optional[ModelProfile]:
        """Get model profile by name."""
        name = profile_name or self.default_profile_name
        return self._config.get_model_profile(name)

    async def generate_text(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        profile_name: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> TextGenerationResult:
        """
        Generate text from a prompt.

        Args:
            prompt: User prompt text
            system_prompt: Optional system instructions
            profile_name: Model profile to use (defaults to default_profile_name)
            temperature: Override profile temperature
            max_tokens: Override profile max_tokens
            **kwargs: Additional parameters for the API

        Returns:
            TextGenerationResult with generated content and metadata

        Raises:
            ModelNotAvailableError: If both primary and fallback fail
            TextLLMServiceError: For other generation errors
        """
        profile = self._get_profile(profile_name)
        client = await self._get_client()

        # Build messages
        messages: List[Dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Try primary model
        primary_model = profile.primary_model if profile else ""
        fallback_model = profile.fallback_model if profile else ""

        if not primary_model:
            raise TextLLMServiceError("No primary model configured")

        used_fallback = False
        model_to_use = primary_model

        try:
            response = await client.chat(
                messages=messages,
                model=model_to_use,
                profile=profile,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )
            return self._to_result(response, used_fallback)

        except (RateLimitError, LLMTimeoutError, LLMClientError) as e:
            logger.warning(f"Primary model {primary_model} failed: {e}")

            # Try fallback if available
            if fallback_model and fallback_model != primary_model:
                logger.info(f"Attempting fallback model: {fallback_model}")
                used_fallback = True
                model_to_use = fallback_model

                try:
                    response = await client.chat(
                        messages=messages,
                        model=model_to_use,
                        profile=profile,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        **kwargs,
                    )
                    return self._to_result(response, used_fallback)

                except Exception as fallback_error:
                    logger.error(f"Fallback model {fallback_model} also failed: {fallback_error}")
                    raise ModelNotAvailableError(
                        f"Both primary ({primary_model}) and fallback ({fallback_model}) models failed"
                    ) from fallback_error
            else:
                raise ModelNotAvailableError(
                    f"Primary model ({primary_model}) failed and no fallback available"
                ) from e

    async def generate_text_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        profile_name: Optional[str] = None,
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        """
        Stream text generation.

        Args:
            prompt: User prompt text
            system_prompt: Optional system instructions
            profile_name: Model profile to use
            **kwargs: Additional parameters

        Yields:
            String chunks of generated content
        """
        profile = self._get_profile(profile_name)
        client = await self._get_client()

        messages: List[Dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        async for chunk in client.chat_stream(
            messages=messages,
            profile=profile,
            **kwargs,
        ):
            yield chunk

    def _to_result(self, response: LLMResponse, used_fallback: bool) -> TextGenerationResult:
        """Convert LLMResponse to TextGenerationResult."""
        return TextGenerationResult(
            content=response.content,
            model_used=response.model,
            tokens_input=response.tokens_input,
            tokens_output=response.tokens_output,
            cost_usd=response.cost_usd,
            latency_ms=response.latency_ms,
            used_fallback=used_fallback,
            metadata={
                "finish_reason": response.finish_reason,
                "tool_calls": response.tool_calls,
            },
        )

    async def __aenter__(self) -> "TextLLMService":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()
