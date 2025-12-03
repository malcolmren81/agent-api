"""
Gemini reasoning model connector.
"""
import google.generativeai as genai
from typing import AsyncIterator, Optional
from dataclasses import dataclass
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from google.api_core import exceptions as google_exceptions
from config import settings
from src.connectors.engine_interface import ReasoningEngineInterface, ModelMetadata
from src.utils import get_logger

logger = get_logger(__name__)


@dataclass
class LLMResponse:
    """
    Enhanced LLM response with metadata.

    Contains the main response text plus reasoning/thinking outputs
    and token usage information from the model.
    """
    text: str                       # Main response text
    reasoning: Optional[str] = None # Chain-of-thought/thinking output
    tokens_used: int = 0            # Total tokens consumed
    input_tokens: int = 0           # Prompt tokens
    output_tokens: int = 0          # Completion tokens
    model_name: str = ""            # Actual model used


class GeminiReasoningEngine(ReasoningEngineInterface):
    """
    Google Gemini reasoning model connector.

    Supports thinking models with controllable reasoning depth.
    """

    def __init__(self) -> None:
        """Initialize Gemini reasoning engine."""
        genai.configure(api_key=settings.gemini_api_key)
        self.model_name = settings.gemini_model_reasoning
        self.thinking_budget = settings.gemini_thinking_budget

        # Initialize the model
        # TODO: Configure thinking budget and other model parameters
        self.model = genai.GenerativeModel(self.model_name)

        logger.info(
            "Gemini reasoning engine initialized",
            model=self.model_name,
            thinking_budget=self.thinking_budget,
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((
            google_exceptions.ResourceExhausted,  # Rate limit
            google_exceptions.ServiceUnavailable,  # Temporary service issues
            google_exceptions.DeadlineExceeded,  # Timeout
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
        Generate a response using Gemini.

        Args:
            prompt: User prompt
            system_prompt: System instructions
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            stream: Whether to stream the response

        Returns:
            Generated text or async iterator for streaming

        Raises:
            google_exceptions.InvalidArgument: Invalid request parameters
            google_exceptions.PermissionDenied: API key invalid
            google_exceptions.ResourceExhausted: Rate limit exceeded (retried)
        """
        try:
            # Validate prompt is not empty
            if not prompt or not prompt.strip():
                raise ValueError("Empty prompt provided")

            logger.info("Generating with Gemini", prompt_length=len(prompt), stream=stream)

            # Combine system prompt and user prompt
            full_prompt = prompt
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"

            # Configure generation parameters
            generation_config = genai.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
                candidate_count=1,
            )

            if stream:
                # Streaming generation
                async def _stream() -> AsyncIterator[str]:
                    response = await self.model.generate_content_async(
                        full_prompt,
                        generation_config=generation_config,
                        stream=True,
                    )
                    async for chunk in response:
                        if chunk.text:
                            yield chunk.text

                return _stream()
            else:
                # Non-streaming generation
                response = await self.model.generate_content_async(
                    full_prompt,
                    generation_config=generation_config,
                )

                # Check for safety filtering and blocked responses
                # finish_reason: 1=STOP (normal), 2=SAFETY, 3=RECITATION, 4=OTHER
                if response.candidates:
                    finish_reason = response.candidates[0].finish_reason
                    if finish_reason == 2:  # SAFETY filter
                        logger.warning(
                            "Gemini response blocked by safety filters",
                            finish_reason=finish_reason,
                            safety_ratings=str(response.candidates[0].safety_ratings) if hasattr(response.candidates[0], 'safety_ratings') else None
                        )
                        raise ValueError(f"Content blocked by safety filters (finish_reason={finish_reason})")
                    elif finish_reason not in [1]:  # Not a normal stop
                        logger.warning(
                            "Gemini response stopped unexpectedly",
                            finish_reason=finish_reason
                        )

                # Now safely access text
                try:
                    if not response.text:
                        if hasattr(response, 'prompt_feedback'):
                            logger.warning(
                                "Gemini response blocked",
                                feedback=str(response.prompt_feedback)
                            )
                            raise ValueError(f"Content blocked: {response.prompt_feedback}")
                        else:
                            raise ValueError("No text returned from Gemini")

                    # Extract token usage metadata
                    input_tokens = 0
                    output_tokens = 0
                    total_tokens = 0
                    reasoning_text = None

                    if hasattr(response, 'usage_metadata'):
                        usage = response.usage_metadata
                        input_tokens = getattr(usage, 'prompt_token_count', 0)
                        output_tokens = getattr(usage, 'candidates_token_count', 0)
                        total_tokens = getattr(usage, 'total_token_count', 0)

                        # For Gemini 2.0 thinking models, check for thinking tokens
                        # Note: API may expose thinking output in future versions
                        if hasattr(usage, 'cached_content_token_count'):
                            logger.debug(f"Cached tokens: {usage.cached_content_token_count}")

                    logger.info(
                        "Gemini generation complete",
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        total_tokens=total_tokens
                    )

                    # Return enhanced response with metadata
                    return LLMResponse(
                        text=response.text,
                        reasoning=reasoning_text,
                        tokens_used=total_tokens,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        model_name=self.model_name
                    )
                except ValueError as e:
                    # Re-raise if it's our own ValueError
                    if "Content blocked" in str(e) or "No text returned" in str(e):
                        raise
                    # Handle finish_reason errors when accessing .text
                    logger.error("Error accessing response text", error=str(e))
                    raise ValueError(f"Invalid response from Gemini: {str(e)}")

        except google_exceptions.InvalidArgument as e:
            logger.error("Invalid Gemini request", error=str(e), prompt_length=len(prompt))
            raise ValueError(f"Invalid request to Gemini: {str(e)}")
        except google_exceptions.PermissionDenied as e:
            logger.error("Gemini API key invalid", error=str(e))
            raise PermissionError(f"Gemini API authentication failed: {str(e)}")
        except google_exceptions.ResourceExhausted as e:
            logger.warning("Gemini rate limit hit, retrying", error=str(e))
            raise  # Will be retried by @retry decorator
        except Exception as e:
            logger.error("Gemini generation error", error=str(e), exc_info=True)
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((
            google_exceptions.ResourceExhausted,
            google_exceptions.ServiceUnavailable,
            google_exceptions.DeadlineExceeded,
        )),
        reraise=True,
    )
    async def generate_with_image(
        self,
        prompt: str,
        image_base64: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        """
        Generate a response using Gemini Vision (multimodal).

        Args:
            prompt: User prompt
            image_base64: Base64-encoded image
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            Generated text

        Raises:
            ValueError: Invalid request
            PermissionError: API key invalid
            google_exceptions.ResourceExhausted: Rate limit exceeded
        """
        try:
            import base64
            from PIL import Image
            from io import BytesIO

            logger.info("Generating with Gemini Vision", prompt_length=len(prompt))

            # Decode base64 image
            image_bytes = base64.b64decode(image_base64)
            image = Image.open(BytesIO(image_bytes))

            # Configure generation parameters
            generation_config = genai.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
                candidate_count=1,
            )

            # Generate with image and text
            response = await self.model.generate_content_async(
                [prompt, image],
                generation_config=generation_config,
            )

            # Check for safety filtering
            if response.candidates:
                finish_reason = response.candidates[0].finish_reason
                if finish_reason == 2:  # SAFETY filter
                    logger.warning("Gemini Vision response blocked by safety filters")
                    raise ValueError(f"Content blocked by safety filters")

            # Extract text and token usage
            try:
                if not response.text:
                    raise ValueError("No text returned from Gemini Vision")

                # Extract token usage metadata
                input_tokens = 0
                output_tokens = 0
                total_tokens = 0

                if hasattr(response, 'usage_metadata'):
                    usage = response.usage_metadata
                    input_tokens = getattr(usage, 'prompt_token_count', 0)
                    output_tokens = getattr(usage, 'candidates_token_count', 0)
                    total_tokens = getattr(usage, 'total_token_count', 0)

                logger.info(
                    "Gemini Vision generation complete",
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    total_tokens=total_tokens
                )

                # Return enhanced response with metadata
                return LLMResponse(
                    text=response.text,
                    reasoning=None,  # Vision models don't expose thinking
                    tokens_used=total_tokens,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    model_name=self.model_name
                )
            except ValueError as e:
                if "Content blocked" in str(e) or "No text returned" in str(e):
                    raise
                logger.error("Error accessing vision response text", error=str(e))
                raise ValueError(f"Invalid response from Gemini Vision: {str(e)}")

        except google_exceptions.InvalidArgument as e:
            logger.error("Invalid Gemini Vision request", error=str(e))
            raise ValueError(f"Invalid request to Gemini Vision: {str(e)}")
        except google_exceptions.PermissionDenied as e:
            logger.error("Gemini API key invalid", error=str(e))
            raise PermissionError(f"Gemini API authentication failed: {str(e)}")
        except google_exceptions.ResourceExhausted as e:
            logger.warning("Gemini Vision rate limit hit, retrying", error=str(e))
            raise
        except Exception as e:
            logger.error("Gemini Vision generation error", error=str(e), exc_info=True)
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
        # Count input tokens (rough estimate: 1 token ≈ 4 characters)
        input_tokens = len(prompt) // 4
        output_tokens = max_tokens or 1024

        # Gemini 2.0 Flash pricing (January 2025)
        # Free tier: up to 15 RPM, 1M TPM, 1500 RPD
        # Paid: Input: $0.075 per 1M tokens, Output: $0.30 per 1M tokens
        input_cost = (input_tokens / 1_000_000) * 0.075
        output_cost = (output_tokens / 1_000_000) * 0.30

        return input_cost + output_cost

    def get_metadata(self) -> ModelMetadata:
        """
        Get model metadata.

        Returns:
            Model metadata
        """
        return ModelMetadata(
            model_name=self.model_name,
            provider="google",
            cost_per_request=settings.gemini_reasoning_cost,
            average_latency=1.0,  # Gemini 2.0 Flash is very fast
            supports_streaming=True,
            max_tokens=1_000_000,  # Gemini 2.0 Flash supports up to 1M tokens
            supports_images=False,
        )
