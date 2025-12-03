"""
ChatGPT reasoning model connector.
"""
from openai import AsyncOpenAI, APIError, RateLimitError, APIConnectionError, Timeout
from typing import AsyncIterator, Optional
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import httpx
from config import settings
from src.connectors.engine_interface import ReasoningEngineInterface, ModelMetadata
from src.connectors.gemini_reasoning import LLMResponse
from src.utils import get_logger

logger = get_logger(__name__)


class ChatGPTReasoningEngine(ReasoningEngineInterface):
    """
    OpenAI ChatGPT reasoning model connector.

    Supports chain-of-thought reasoning patterns.
    """

    def __init__(self) -> None:
        """Initialize ChatGPT reasoning engine."""
        # Create httpx client without proxies to avoid compatibility issues
        http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(60.0, connect=10.0),
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10)
        )

        self.client = AsyncOpenAI(
            api_key=settings.openai_api_key,
            organization=settings.openai_org_id if settings.openai_org_id else None,
            http_client=http_client,
        )
        self.model_name = settings.openai_model

        logger.info("ChatGPT reasoning engine initialized", model=self.model_name)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((
            RateLimitError,  # Rate limit exceeded
            APIConnectionError,  # Network issues
            Timeout,  # Request timeout
        )),
        reraise=True,
    )
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False,
    ) -> LLMResponse | AsyncIterator[str]:
        """
        Generate a response using ChatGPT.

        Args:
            prompt: User prompt
            system_prompt: System instructions
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            stream: Whether to stream the response

        Returns:
            Generated text or async iterator for streaming

        Raises:
            ValueError: Invalid request parameters
            PermissionError: API key invalid
            RateLimitError: Rate limit exceeded (retried)
        """
        try:
            logger.info("Generating with ChatGPT", prompt_length=len(prompt), stream=stream)

            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            if stream:
                # Streaming generation
                async def _stream() -> AsyncIterator[str]:
                    response = await self.client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        stream=True,
                    )
                    async for chunk in response:
                        if chunk.choices[0].delta.content:
                            yield chunk.choices[0].delta.content

                return _stream()
            else:
                # Non-streaming generation
                response = await self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )

                # Validate response
                if not response.choices:
                    raise ValueError("No choices returned from ChatGPT")

                content = response.choices[0].message.content
                if not content:
                    # Check if content was filtered
                    if hasattr(response.choices[0], 'finish_reason'):
                        if response.choices[0].finish_reason == 'content_filter':
                            logger.warning("ChatGPT response filtered")
                            raise ValueError("Content blocked by OpenAI content filters")
                    raise ValueError("Empty response from ChatGPT")

                # Extract token usage from response
                input_tokens = 0
                output_tokens = 0
                total_tokens = 0
                reasoning_text = None

                if hasattr(response, 'usage'):
                    usage = response.usage
                    input_tokens = getattr(usage, 'prompt_tokens', 0)
                    output_tokens = getattr(usage, 'completion_tokens', 0)
                    total_tokens = getattr(usage, 'total_tokens', 0)

                    # For o1/o3 reasoning models, check for reasoning tokens
                    if hasattr(usage, 'completion_tokens_details'):
                        details = usage.completion_tokens_details
                        if hasattr(details, 'reasoning_tokens'):
                            reasoning_tokens = details.reasoning_tokens
                            logger.debug(f"Reasoning tokens: {reasoning_tokens}")

                logger.info(
                    "ChatGPT generation complete",
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    total_tokens=total_tokens
                )

                # Return enhanced response with metadata
                return LLMResponse(
                    text=content,
                    reasoning=reasoning_text,
                    tokens_used=total_tokens,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    model_name=self.model_name
                )

        except RateLimitError as e:
            logger.warning("ChatGPT rate limit hit, retrying", error=str(e))
            raise  # Will be retried by @retry decorator
        except APIConnectionError as e:
            logger.warning("ChatGPT connection error, retrying", error=str(e))
            raise  # Will be retried by @retry decorator
        except Timeout as e:
            logger.warning("ChatGPT timeout, retrying", error=str(e))
            raise  # Will be retried by @retry decorator
        except APIError as e:
            # Check for specific error codes
            if hasattr(e, 'status_code'):
                if e.status_code == 401:
                    logger.error("ChatGPT API key invalid", error=str(e))
                    raise PermissionError(f"OpenAI API authentication failed: {str(e)}")
                elif e.status_code == 400:
                    logger.error("Invalid ChatGPT request", error=str(e))
                    raise ValueError(f"Invalid request to OpenAI: {str(e)}")
            logger.error("ChatGPT API error", error=str(e), exc_info=True)
            raise
        except Exception as e:
            logger.error("ChatGPT generation error", error=str(e), exc_info=True)
            raise

    async def estimate_cost(self, prompt: str, max_tokens: Optional[int] = None) -> float:
        """
        Estimate the cost of a generation request.

        Args:
            prompt: Input prompt
            max_tokens: Expected output tokens

        Returns:
            Estimated cost in USD
        """
        # Estimate token count (rough: 1 token ≈ 4 characters)
        input_tokens = len(prompt) // 4
        output_tokens = max_tokens or 1024

        # GPT-4o pricing (January 2025)
        # Input: $2.50 per 1M tokens, Output: $10.00 per 1M tokens
        input_cost = (input_tokens / 1_000_000) * 2.50
        output_cost = (output_tokens / 1_000_000) * 10.00

        return input_cost + output_cost

    def get_metadata(self) -> ModelMetadata:
        """
        Get model metadata.

        Returns:
            Model metadata
        """
        return ModelMetadata(
            model_name=self.model_name,
            provider="openai",
            cost_per_request=settings.chatgpt_reasoning_cost,
            average_latency=2.0,  # GPT-4o is faster
            supports_streaming=True,
            max_tokens=128_000,  # GPT-4o supports 128K context
            supports_images=False,
        )
