"""
Text LLM Service

High-level service for text generation using LLM models via OpenRouter.
Provides methods for text generation, prompt rewriting, clarifying questions,
and conversation summarization with automatic failover support.

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

    async def rewrite_prompt(
        self,
        original_prompt: str,
        constraints: Optional[Dict[str, Any]] = None,
        profile_name: Optional[str] = None,
    ) -> str:
        """
        Rewrite a prompt to improve quality.

        Args:
            original_prompt: Original user prompt
            constraints: Optional constraints (style, format, etc.)
            profile_name: Model profile to use

        Returns:
            Rewritten prompt string
        """
        constraint_text = ""
        if constraints:
            constraint_items = [f"- {k}: {v}" for k, v in constraints.items()]
            constraint_text = f"\n\nConstraints:\n" + "\n".join(constraint_items)

        system_prompt = """You are an expert prompt engineer for image generation.
Your task is to rewrite the user's prompt to be more effective for AI image generation.

Guidelines:
- Be specific and descriptive
- Include style, lighting, and composition details
- Maintain the original intent
- Use clear, unambiguous language
- Optimize for the target model's capabilities

Return ONLY the rewritten prompt, no explanations."""

        user_prompt = f"""Original prompt: {original_prompt}{constraint_text}

Rewrite this prompt for optimal image generation:"""

        result = await self.generate_text(
            prompt=user_prompt,
            system_prompt=system_prompt,
            profile_name=profile_name or "planner",
            temperature=0.3,
        )

        return result.content.strip()

    async def generate_clarifying_questions(
        self,
        context: str,
        current_requirements: Optional[Dict[str, Any]] = None,
        max_questions: int = 3,
        profile_name: Optional[str] = None,
    ) -> List[str]:
        """
        Generate clarifying questions to gather more requirements.

        Args:
            context: Current conversation context
            current_requirements: Requirements gathered so far
            max_questions: Maximum number of questions to generate
            profile_name: Model profile to use

        Returns:
            List of clarifying question strings
        """
        requirements_text = ""
        if current_requirements:
            req_items = [f"- {k}: {v}" for k, v in current_requirements.items()]
            requirements_text = f"\n\nCurrent requirements:\n" + "\n".join(req_items)

        system_prompt = f"""You are a helpful assistant gathering requirements for image generation.
Based on the conversation, generate up to {max_questions} clarifying questions to better understand the user's needs.

Guidelines:
- Ask about missing but important details (style, colors, composition, mood)
- Be concise and specific
- Focus on what would most improve the generation result
- Don't repeat information already provided

Return each question on a new line, numbered (1., 2., etc.)"""

        user_prompt = f"""Conversation context:
{context}{requirements_text}

Generate clarifying questions:"""

        result = await self.generate_text(
            prompt=user_prompt,
            system_prompt=system_prompt,
            profile_name=profile_name or "pali",
            temperature=0.5,
        )

        # Parse questions from response
        questions = []
        for line in result.content.strip().split("\n"):
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith("-")):
                # Remove numbering/bullets
                question = line.lstrip("0123456789.-) ").strip()
                if question and question.endswith("?"):
                    questions.append(question)

        return questions[:max_questions]

    async def summarize_conversation(
        self,
        messages: List[Dict[str, str]],
        max_length: int = 200,
        profile_name: Optional[str] = None,
    ) -> str:
        """
        Summarize a conversation history.

        Args:
            messages: List of message dicts with 'role' and 'content'
            max_length: Maximum summary length in words
            profile_name: Model profile to use

        Returns:
            Summary string
        """
        # Format conversation
        conversation_text = "\n".join([
            f"{msg['role'].capitalize()}: {msg['content']}"
            for msg in messages
        ])

        system_prompt = f"""Summarize the following conversation concisely.
Focus on key decisions, requirements, and outcomes.
Maximum {max_length} words."""

        result = await self.generate_text(
            prompt=f"Conversation:\n{conversation_text}\n\nSummary:",
            system_prompt=system_prompt,
            profile_name=profile_name or "planner",
            temperature=0.3,
            max_tokens=max_length * 2,  # Approximate tokens
        )

        return result.content.strip()

    async def classify_intent(
        self,
        text: str,
        categories: List[str],
        profile_name: Optional[str] = None,
    ) -> str:
        """
        Classify text into one of the given categories.

        Args:
            text: Text to classify
            categories: List of possible categories
            profile_name: Model profile to use

        Returns:
            Selected category string
        """
        categories_text = "\n".join([f"- {cat}" for cat in categories])

        system_prompt = f"""Classify the following text into exactly one of these categories:
{categories_text}

Respond with ONLY the category name, nothing else."""

        result = await self.generate_text(
            prompt=text,
            system_prompt=system_prompt,
            profile_name=profile_name or "safety",
            temperature=0.0,  # Deterministic
            max_tokens=50,
        )

        # Clean and validate response
        response = result.content.strip().lower()
        for category in categories:
            if category.lower() in response or response in category.lower():
                return category

        # Return first category as fallback
        logger.warning(f"Could not match response '{response}' to categories, using first")
        return categories[0]

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
