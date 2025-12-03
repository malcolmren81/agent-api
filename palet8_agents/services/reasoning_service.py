"""
Reasoning Service

High-level service for reasoning tasks including prompt quality assessment,
design alignment evaluation, intent classification, and prompt revision.

Documentation Reference: Section 4.3
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
import logging
import json

from palet8_agents.services.text_llm_service import TextLLMService, TextLLMServiceError

logger = logging.getLogger(__name__)


class ReasoningServiceError(Exception):
    """Base exception for ReasoningService errors."""
    pass


class QualityLevel(Enum):
    """Quality assessment levels."""
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    UNACCEPTABLE = "unacceptable"


@dataclass
class QualityScore:
    """Result from prompt quality assessment."""
    overall_score: float  # 0.0 to 1.0
    level: QualityLevel
    clarity_score: float = 0.0
    specificity_score: float = 0.0
    feasibility_score: float = 0.0
    issues: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "overall_score": self.overall_score,
            "level": self.level.value,
            "clarity_score": self.clarity_score,
            "specificity_score": self.specificity_score,
            "feasibility_score": self.feasibility_score,
            "issues": self.issues,
            "suggestions": self.suggestions,
            "metadata": self.metadata,
        }

    @property
    def is_acceptable(self) -> bool:
        """Check if quality is acceptable for generation."""
        return self.level in [QualityLevel.EXCELLENT, QualityLevel.GOOD, QualityLevel.ACCEPTABLE]


@dataclass
class AlignmentScore:
    """Result from design alignment assessment."""
    overall_score: float  # 0.0 to 1.0
    prompt_adherence: float = 0.0
    style_match: float = 0.0
    composition_score: float = 0.0
    technical_quality: float = 0.0
    issues: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "overall_score": self.overall_score,
            "prompt_adherence": self.prompt_adherence,
            "style_match": self.style_match,
            "composition_score": self.composition_score,
            "technical_quality": self.technical_quality,
            "issues": self.issues,
            "metadata": self.metadata,
        }

    @property
    def passes_threshold(self, threshold: float = 0.45) -> bool:
        """Check if score passes the quality threshold."""
        return self.overall_score >= threshold


class ReasoningService:
    """
    Service for reasoning and analysis tasks.

    Features:
    - Prompt quality assessment
    - Design alignment evaluation
    - Intent classification
    - Prompt revision suggestions
    """

    def __init__(
        self,
        text_llm_service: Optional[TextLLMService] = None,
    ):
        """
        Initialize ReasoningService.

        Args:
            text_llm_service: Optional TextLLMService instance.
        """
        self._text_service = text_llm_service
        self._owns_service = text_llm_service is None

    async def _get_text_service(self) -> TextLLMService:
        """Get or create text LLM service."""
        if self._text_service is None:
            self._text_service = TextLLMService(default_profile_name="planner")
        return self._text_service

    async def close(self) -> None:
        """Close the service and release resources."""
        if self._text_service and self._owns_service:
            await self._text_service.close()
            self._text_service = None

    async def assess_prompt_quality(
        self,
        prompt: str,
        constraints: Optional[Dict[str, Any]] = None,
    ) -> QualityScore:
        """
        Assess the quality of an image generation prompt.

        Args:
            prompt: The prompt to assess
            constraints: Optional constraints (product type, style, etc.)

        Returns:
            QualityScore with detailed assessment
        """
        service = await self._get_text_service()

        constraint_text = ""
        if constraints:
            constraint_items = [f"- {k}: {v}" for k, v in constraints.items()]
            constraint_text = f"\n\nConstraints:\n" + "\n".join(constraint_items)

        system_prompt = """You are an expert prompt quality assessor for AI image generation.
Evaluate the given prompt and provide a JSON response with the following structure:

{
    "overall_score": 0.0-1.0,
    "clarity_score": 0.0-1.0,
    "specificity_score": 0.0-1.0,
    "feasibility_score": 0.0-1.0,
    "issues": ["list of problems"],
    "suggestions": ["list of improvements"]
}

Scoring criteria:
- Clarity (0-1): How clear and unambiguous is the prompt?
- Specificity (0-1): How detailed and specific is the description?
- Feasibility (0-1): How achievable is this with current AI models?

Return ONLY valid JSON, no other text."""

        user_prompt = f"""Prompt to assess: "{prompt}"{constraint_text}

Provide your assessment as JSON:"""

        try:
            result = await service.generate_text(
                prompt=user_prompt,
                system_prompt=system_prompt,
                profile_name="planner",
                temperature=0.2,
            )

            # Parse JSON response
            try:
                data = json.loads(result.content.strip())
            except json.JSONDecodeError:
                # Try to extract JSON from response
                import re
                json_match = re.search(r'\{[^{}]*\}', result.content, re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group())
                else:
                    raise ReasoningServiceError("Failed to parse quality assessment response")

            overall_score = float(data.get("overall_score", 0.5))

            # Determine quality level
            if overall_score >= 0.9:
                level = QualityLevel.EXCELLENT
            elif overall_score >= 0.7:
                level = QualityLevel.GOOD
            elif overall_score >= 0.5:
                level = QualityLevel.ACCEPTABLE
            elif overall_score >= 0.3:
                level = QualityLevel.POOR
            else:
                level = QualityLevel.UNACCEPTABLE

            return QualityScore(
                overall_score=overall_score,
                level=level,
                clarity_score=float(data.get("clarity_score", 0.5)),
                specificity_score=float(data.get("specificity_score", 0.5)),
                feasibility_score=float(data.get("feasibility_score", 0.5)),
                issues=data.get("issues", []),
                suggestions=data.get("suggestions", []),
                metadata={"raw_response": result.content},
            )

        except TextLLMServiceError as e:
            raise ReasoningServiceError(f"Quality assessment failed: {e}")

    async def propose_prompt_revision(
        self,
        prompt: str,
        feedback: str,
        constraints: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Propose a revised prompt based on feedback.

        Args:
            prompt: Original prompt
            feedback: Feedback on what to improve
            constraints: Optional constraints

        Returns:
            Revised prompt string
        """
        service = await self._get_text_service()

        constraint_text = ""
        if constraints:
            constraint_items = [f"- {k}: {v}" for k, v in constraints.items()]
            constraint_text = f"\n\nConstraints to maintain:\n" + "\n".join(constraint_items)

        system_prompt = """You are an expert prompt engineer for AI image generation.
Given an original prompt and feedback, create an improved version.

Guidelines:
- Address all feedback points
- Maintain the original intent
- Be specific and descriptive
- Optimize for the target model

Return ONLY the revised prompt, no explanations."""

        user_prompt = f"""Original prompt: "{prompt}"

Feedback: {feedback}{constraint_text}

Revised prompt:"""

        try:
            result = await service.generate_text(
                prompt=user_prompt,
                system_prompt=system_prompt,
                profile_name="planner",
                temperature=0.3,
            )

            return result.content.strip().strip('"')

        except TextLLMServiceError as e:
            raise ReasoningServiceError(f"Prompt revision failed: {e}")

    async def assess_design_alignment(
        self,
        prompt: str,
        description: str,
        product_type: Optional[str] = None,
    ) -> AlignmentScore:
        """
        Assess how well a design description aligns with the original prompt.

        This is used by the Evaluator Agent to check generated image quality
        based on a description of the image.

        Args:
            prompt: Original generation prompt
            description: Description of the generated design
            product_type: Optional product type for context

        Returns:
            AlignmentScore with detailed assessment
        """
        service = await self._get_text_service()

        product_context = f"\nProduct type: {product_type}" if product_type else ""

        system_prompt = """You are an expert design evaluator.
Compare a generated image description against its original prompt and provide a JSON assessment:

{
    "overall_score": 0.0-1.0,
    "prompt_adherence": 0.0-1.0,
    "style_match": 0.0-1.0,
    "composition_score": 0.0-1.0,
    "technical_quality": 0.0-1.0,
    "issues": ["list of problems"]
}

Scoring criteria:
- Prompt adherence: How well does the result match the prompt intent?
- Style match: Does the style match what was requested?
- Composition: Is the composition appropriate for the use case?
- Technical quality: Are there any technical issues described?

Return ONLY valid JSON."""

        user_prompt = f"""Original prompt: "{prompt}"

Generated design description: "{description}"{product_context}

Provide your alignment assessment as JSON:"""

        try:
            result = await service.generate_text(
                prompt=user_prompt,
                system_prompt=system_prompt,
                profile_name="evaluator",
                temperature=0.2,
            )

            # Parse JSON response
            try:
                data = json.loads(result.content.strip())
            except json.JSONDecodeError:
                import re
                json_match = re.search(r'\{[^{}]*\}', result.content, re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group())
                else:
                    raise ReasoningServiceError("Failed to parse alignment assessment response")

            return AlignmentScore(
                overall_score=float(data.get("overall_score", 0.5)),
                prompt_adherence=float(data.get("prompt_adherence", 0.5)),
                style_match=float(data.get("style_match", 0.5)),
                composition_score=float(data.get("composition_score", 0.5)),
                technical_quality=float(data.get("technical_quality", 0.5)),
                issues=data.get("issues", []),
                metadata={"raw_response": result.content},
            )

        except TextLLMServiceError as e:
            raise ReasoningServiceError(f"Design alignment assessment failed: {e}")

    async def classify_intent(
        self,
        text: str,
        categories: List[str],
    ) -> str:
        """
        Classify text into one of the given categories.

        Args:
            text: Text to classify
            categories: List of possible categories

        Returns:
            Selected category string
        """
        service = await self._get_text_service()

        try:
            return await service.classify_intent(text, categories, profile_name="safety")
        except TextLLMServiceError as e:
            raise ReasoningServiceError(f"Intent classification failed: {e}")

    async def extract_requirements(
        self,
        conversation: List[Dict[str, str]],
    ) -> Dict[str, Any]:
        """
        Extract structured requirements from a conversation.

        Args:
            conversation: List of message dicts with 'role' and 'content'

        Returns:
            Dictionary of extracted requirements
        """
        service = await self._get_text_service()

        conversation_text = "\n".join([
            f"{msg['role'].capitalize()}: {msg['content']}"
            for msg in conversation
        ])

        system_prompt = """Extract structured requirements from this conversation for image generation.

Return a JSON object with these fields (leave null if not mentioned):
{
    "subject": "main subject of the image",
    "style": "artistic style or aesthetic",
    "mood": "emotional tone or atmosphere",
    "colors": ["list", "of", "colors"],
    "composition": "composition preferences",
    "background": "background description",
    "additional_elements": ["other", "elements"],
    "avoid": ["things", "to", "avoid"],
    "reference_images": ["urls if provided"],
    "product_specific": {"any product-specific requirements"}
}

Return ONLY valid JSON."""

        user_prompt = f"""Conversation:
{conversation_text}

Extract requirements as JSON:"""

        try:
            result = await service.generate_text(
                prompt=user_prompt,
                system_prompt=system_prompt,
                profile_name="planner",
                temperature=0.2,
            )

            try:
                return json.loads(result.content.strip())
            except json.JSONDecodeError:
                import re
                json_match = re.search(r'\{[^{}]*\}', result.content, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
                else:
                    return {"raw_response": result.content}

        except TextLLMServiceError as e:
            raise ReasoningServiceError(f"Requirement extraction failed: {e}")

    async def __aenter__(self) -> "ReasoningService":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()
