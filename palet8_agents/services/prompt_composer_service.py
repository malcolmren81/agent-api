"""
Prompt Composer Service - Composes prompts from Planner decisions.

This service WRITES prompts based on:
- Mode decision (from Planner)
- Selected dimensions (from Planner)
- Context (from Planner via RAG)
- Rules/templates (from PromptTemplateService)

It uses LLM to compose natural, high-quality prompts from structured dimensions.

Flow:
  Planner Agent → decides mode, selects dimensions, builds context
  Planner Agent → calls PromptComposerService.compose_prompt(mode, dimensions, context)
  PromptComposerService → uses LLM to write final prompt

Documentation Reference: Section 5.3
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import logging
import json

from palet8_agents.services.prompt_template_service import (
    PromptTemplateService,
    PromptMode,
)
from palet8_agents.services.text_llm_service import TextLLMService, TextGenerationResult

logger = logging.getLogger(__name__)


class PromptComposerServiceError(Exception):
    """Base exception for Prompt Composer Service errors."""
    pass


class CompositionError(PromptComposerServiceError):
    """Error during prompt composition."""
    pass


@dataclass
class PromptDimensions:
    """Structured prompt dimensions from Planner."""
    subject: str
    aesthetic: Optional[str] = None
    color: Optional[str] = None
    composition: Optional[str] = None
    background: Optional[str] = None
    lighting: Optional[str] = None
    texture: Optional[str] = None
    mood: Optional[str] = None
    detail: Optional[str] = None
    expression: Optional[str] = None
    pose: Optional[str] = None
    art_movement: Optional[str] = None
    reference_style: Optional[str] = None
    technical: Optional[Dict[str, str]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding None values."""
        result = {}
        for key, value in self.__dict__.items():
            if value is not None:
                result[key] = value
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PromptDimensions":
        """Create from dictionary."""
        return cls(
            subject=data.get("subject", ""),
            aesthetic=data.get("aesthetic"),
            color=data.get("color"),
            composition=data.get("composition"),
            background=data.get("background"),
            lighting=data.get("lighting"),
            texture=data.get("texture"),
            mood=data.get("mood"),
            detail=data.get("detail"),
            expression=data.get("expression"),
            pose=data.get("pose"),
            art_movement=data.get("art_movement"),
            reference_style=data.get("reference_style"),
            technical=data.get("technical"),
        )


@dataclass
class ComposedPrompt:
    """Result from prompt composition."""
    positive_prompt: str
    negative_prompt: str
    mode: str
    dimensions_used: Dict[str, Any]
    token_count: int
    model_used: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "positive_prompt": self.positive_prompt,
            "negative_prompt": self.negative_prompt,
            "mode": self.mode,
            "dimensions_used": self.dimensions_used,
            "token_count": self.token_count,
            "model_used": self.model_used,
            "metadata": self.metadata,
        }


class PromptComposerService:
    """
    Composes prompts from structured dimensions using LLM.

    Input (from Planner Agent):
        - mode: RELAX, STANDARD, or COMPLEX
        - dimensions: PromptDimensions with filled values
        - context: RAG context (similar designs, user history)
        - print_method: Optional print method for technical specs

    Output:
        - ComposedPrompt with positive/negative prompts

    Usage:
        composer = PromptComposerService()
        result = await composer.compose_prompt(
            mode="STANDARD",
            dimensions=PromptDimensions(subject="tiger", aesthetic="streetwear"),
            context={"similar_designs": [...], "user_history": [...]},
            print_method="screen_print",
        )
    """

    COMPOSER_SYSTEM_PROMPT = """You are an image prompt composer.

Task: Convert structured dimensions into image generation prompts.

IMPORTANT: All info, context, and materials are provided by the Planner Agent. Follow the planner's requirements exactly.

Rules:
1. Use structural language.
2. One point per sentence.
3. No complex sentences.
4. Prioritize: subject, aesthetic, color.
5. Match technical print constraints.

Scenario Rules:
- Layout design: Specify each component position (top, center, bottom, left, right).
- Text in design: Specify text position, size, font type.
- Image insert: Include reference image description.
- Character selected: Include character details and placement.
- Aesthetic method: Specify the aesthetic style clearly.
- Photorealistic: Specify camera, angle, lens, position, lighting.

Mode Token Limits:
- RELAX: 20-50 tokens. Subject and style only.
- STANDARD: 50-150 tokens. Add composition and mood.
- COMPLEX: 150-400 tokens. Full specification.

Output JSON:
{
  "positive_prompt": "composed prompt here",
  "negative_prompt": "elements to avoid"
}"""

    def __init__(
        self,
        template_service: Optional[PromptTemplateService] = None,
        text_llm_service: Optional[TextLLMService] = None,
        profile_name: str = "composer",
    ):
        """
        Initialize the Prompt Composer Service.

        Args:
            template_service: PromptTemplateService for rules (creates if None)
            text_llm_service: TextLLMService for LLM (creates if None)
            profile_name: Model profile for composition
        """
        self._template_service = template_service
        self._text_llm_service = text_llm_service
        self._owns_template = template_service is None
        self._owns_llm = text_llm_service is None
        self._profile_name = profile_name

    async def _get_template_service(self) -> PromptTemplateService:
        """Get or create template service."""
        if self._template_service is None:
            self._template_service = PromptTemplateService()
        return self._template_service

    async def _get_llm_service(self) -> TextLLMService:
        """Get or create LLM service."""
        if self._text_llm_service is None:
            self._text_llm_service = TextLLMService()
        return self._text_llm_service

    async def compose_prompt(
        self,
        mode: str,
        dimensions: PromptDimensions,
        context: Optional[Dict[str, Any]] = None,
        print_method: Optional[str] = None,
        product: Optional[str] = None,
    ) -> ComposedPrompt:
        """
        Compose a prompt from Planner's decisions.

        Args:
            mode: Selected mode (RELAX, STANDARD, COMPLEX)
            dimensions: Filled dimensions from Planner
            context: RAG context (similar designs, user history, etc.)
            print_method: Optional print method for technical constraints
            product: Optional product type

        Returns:
            ComposedPrompt with positive and negative prompts

        Raises:
            CompositionError: If composition fails
        """
        try:
            template_service = await self._get_template_service()
            llm_service = await self._get_llm_service()

            # Get composition context from template service
            composition_context = template_service.get_composition_context(
                mode=mode,
                print_method=print_method,
            )

            # Build the composition request
            user_prompt = self._build_composition_request(
                mode=mode,
                dimensions=dimensions,
                context=context,
                composition_context=composition_context,
                print_method=print_method,
            )

            # Call LLM to compose
            result = await llm_service.generate_text(
                prompt=user_prompt,
                system_prompt=self.COMPOSER_SYSTEM_PROMPT,
                profile_name=self._profile_name,
                temperature=0.4,  # Slightly creative but controlled
            )

            # Parse the response
            composed = self._parse_composition_response(result.content)

            # Count tokens (approximate)
            token_count = len(composed["positive_prompt"].split())

            return ComposedPrompt(
                positive_prompt=composed["positive_prompt"],
                negative_prompt=composed["negative_prompt"],
                mode=mode,
                dimensions_used=dimensions.to_dict(),
                token_count=token_count,
                model_used=result.model_used,
                metadata={
                    "context_used": bool(context),
                    "print_method": print_method,
                    "product": product,
                    "llm_tokens": result.tokens_output,
                },
            )

        except Exception as e:
            logger.error(f"Prompt composition failed: {e}", exc_info=True)
            raise CompositionError(f"Failed to compose prompt: {e}") from e

    def _build_composition_request(
        self,
        mode: str,
        dimensions: PromptDimensions,
        context: Optional[Dict[str, Any]],
        composition_context: Dict[str, Any],
        print_method: Optional[str],
    ) -> str:
        """Build the user prompt for composition."""
        parts = []

        # Mode info
        mode_rule = composition_context.get("mode_rule", {})
        token_range = mode_rule.get("token_range", [50, 150])
        parts.append(f"Mode: {mode}")
        parts.append(f"Target token range: {token_range[0]}-{token_range[1]} tokens")
        parts.append("")

        # Dimensions
        parts.append("Dimensions to compose:")
        dims = dimensions.to_dict()
        for key, value in dims.items():
            if key != "technical" and value:
                parts.append(f"  {key}: {value}")

        # Technical specs (for COMPLEX mode)
        if dimensions.technical:
            parts.append("")
            parts.append("Technical specifications:")
            for key, value in dimensions.technical.items():
                parts.append(f"  {key}: {value}")

        # Print constraints
        if print_method:
            constraints = composition_context.get("print_constraints", {})
            if constraints:
                parts.append("")
                parts.append(f"Print method: {print_method}")
                if not constraints.get("supports_gradients", True):
                    parts.append("  Note: Use halftone dots instead of smooth gradients")
                if constraints.get("max_colors"):
                    parts.append(f"  Max colors: {constraints['max_colors']}")

        # Context from RAG
        if context:
            if context.get("similar_designs"):
                parts.append("")
                parts.append("Reference from similar designs:")
                for design in context["similar_designs"][:2]:
                    if isinstance(design, dict):
                        parts.append(f"  - {design.get('description', design)}")
                    else:
                        parts.append(f"  - {design}")

            if context.get("user_preferences"):
                parts.append("")
                parts.append("User preferences:")
                prefs = context["user_preferences"]
                if isinstance(prefs, dict):
                    for key, value in prefs.items():
                        parts.append(f"  {key}: {value}")

        parts.append("")
        parts.append("Compose the prompt now. Return JSON with positive_prompt and negative_prompt.")

        return "\n".join(parts)

    def _parse_composition_response(self, response: str) -> Dict[str, str]:
        """Parse the LLM response to extract prompts."""
        # Try to parse as JSON
        try:
            # Find JSON in response
            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                json_str = response[start:end]
                parsed = json.loads(json_str)
                return {
                    "positive_prompt": parsed.get("positive_prompt", ""),
                    "negative_prompt": parsed.get("negative_prompt", ""),
                }
        except json.JSONDecodeError:
            pass

        # Fallback: treat entire response as positive prompt
        logger.warning("Could not parse JSON from composition response, using raw text")
        return {
            "positive_prompt": response.strip(),
            "negative_prompt": "blurry, low quality, distorted, ugly, bad anatomy",
        }

    async def compose_simple(
        self,
        subject: str,
        aesthetic: Optional[str] = None,
        color: Optional[str] = None,
        mode: str = "RELAX",
    ) -> ComposedPrompt:
        """
        Quick composition for simple requests.

        Args:
            subject: Main subject
            aesthetic: Optional style
            color: Optional color scheme
            mode: Mode (default RELAX)

        Returns:
            ComposedPrompt
        """
        dimensions = PromptDimensions(
            subject=subject,
            aesthetic=aesthetic,
            color=color,
        )
        return await self.compose_prompt(mode=mode, dimensions=dimensions)

    async def close(self) -> None:
        """Clean up resources."""
        if self._template_service and self._owns_template:
            await self._template_service.close()
            self._template_service = None
        if self._text_llm_service and self._owns_llm:
            await self._text_llm_service.close()
            self._text_llm_service = None

    async def __aenter__(self) -> "PromptComposerService":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()
