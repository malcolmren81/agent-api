"""
Requirements Analysis Service

Service for analyzing conversations to extract design requirements and calculate
completeness scores. This extracts the requirements gathering logic from PaliAgent
into a reusable service.

Documentation Reference: Section 5.2.1 (Pali Agent)
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import logging
import yaml

from palet8_agents.models import RequirementsStatus
from palet8_agents.core.message import Conversation
from palet8_agents.services.text_llm_service import TextLLMService, TextLLMServiceError

logger = logging.getLogger(__name__)


class RequirementsAnalysisError(Exception):
    """Base exception for RequirementsAnalysisService errors."""
    pass


class ExtractionError(RequirementsAnalysisError):
    """Raised when LLM-based extraction fails."""
    pass


@dataclass
class RequirementsConfig:
    """Configuration for requirements analysis."""
    completeness_threshold: float = 0.5
    completeness_weights: Dict[str, float] = field(default_factory=lambda: {
        "subject": 0.50,
        "style": 0.20,
        "colors": 0.15,
        "mood": 0.15,
    })
    required_fields: List[str] = field(default_factory=lambda: ["subject"])
    recommended_fields: List[str] = field(default_factory=lambda: ["style", "colors", "mood"])
    extraction_temperature: float = 0.2
    min_input_length: int = 3
    max_input_length: int = 10000


class RequirementsAnalysisService:
    """
    Service for analyzing conversations to extract design requirements.

    This service:
    - Uses LLM to extract structured requirements from conversation
    - Calculates weighted completeness scores
    - Determines if requirements meet minimum threshold
    - Identifies missing required and recommended fields
    """

    # Default extraction prompt
    DEFAULT_EXTRACTION_PROMPT = """Extract design requirements from this conversation.
Return a JSON object with these fields (null if not mentioned):
{
    "subject": "main subject/concept",
    "style": "visual style",
    "colors": ["list", "of", "colors"],
    "mood": "emotional tone",
    "composition": "composition notes",
    "include_elements": ["elements to include"],
    "avoid_elements": ["elements to avoid"]
}
Return ONLY valid JSON."""

    def __init__(
        self,
        text_service: Optional[TextLLMService] = None,
        config: Optional[RequirementsConfig] = None,
        config_path: Optional[Path] = None,
    ):
        """
        Initialize RequirementsAnalysisService.

        Args:
            text_service: TextLLMService for LLM calls. Creates one if not provided.
            config: Optional RequirementsConfig. Loaded from file if not provided.
            config_path: Path to config YAML file. Uses default if not provided.
        """
        self._text_service = text_service
        self._owns_service = text_service is None
        self._config = config or self._load_config(config_path)

    def _load_config(self, config_path: Optional[Path] = None) -> RequirementsConfig:
        """Load configuration from YAML file."""
        if config_path is None:
            # Try default config location
            default_paths = [
                Path("config/pali_config.yaml"),
                Path(__file__).parent.parent.parent / "config" / "pali_config.yaml",
            ]
            for path in default_paths:
                if path.exists():
                    config_path = path
                    break

        if config_path and config_path.exists():
            try:
                with open(config_path, "r") as f:
                    data = yaml.safe_load(f)

                completeness = data.get("completeness", {})
                llm = data.get("llm", {})
                validation = data.get("input_validation", {})

                return RequirementsConfig(
                    completeness_threshold=completeness.get("threshold", 0.5),
                    completeness_weights=completeness.get("weights", {}),
                    required_fields=data.get("required_fields", ["subject"]),
                    recommended_fields=data.get("recommended_fields", []),
                    extraction_temperature=llm.get("extraction_temperature", 0.2),
                    min_input_length=validation.get("min_length", 3),
                    max_input_length=validation.get("max_length", 10000),
                )
            except Exception as e:
                logger.warning(f"Failed to load config from {config_path}: {e}")

        return RequirementsConfig()

    async def _get_text_service(self) -> TextLLMService:
        """Get or create text LLM service."""
        if self._text_service is None:
            self._text_service = TextLLMService(default_profile_name="pali")
        return self._text_service

    async def close(self) -> None:
        """Close the service and release resources."""
        if self._text_service and self._owns_service:
            await self._text_service.close()
            self._text_service = None

    async def analyze_conversation(
        self,
        conversation: Conversation,
        existing_requirements: Optional[Dict[str, Any]] = None,
        system_prompt: Optional[str] = None,
    ) -> RequirementsStatus:
        """
        Extract requirements from a conversation using LLM.

        Args:
            conversation: Conversation history to analyze
            existing_requirements: Previously extracted requirements to merge
            system_prompt: Optional override for extraction prompt

        Returns:
            RequirementsStatus with extracted fields and completeness score

        Raises:
            ExtractionError: If LLM extraction fails after retries
        """
        # Start with existing requirements or empty status
        status = RequirementsStatus()

        if existing_requirements:
            status.subject = existing_requirements.get("subject")
            status.style = existing_requirements.get("style")
            status.colors = existing_requirements.get("colors", [])
            status.mood = existing_requirements.get("mood")
            status.composition = existing_requirements.get("composition")
            status.include_elements = existing_requirements.get("include_elements", [])
            status.avoid_elements = existing_requirements.get("avoid_elements", [])

        # If no messages, return current status
        if not conversation.messages:
            return status

        # Build conversation text for LLM
        conversation_text = "\n".join([
            f"{msg.role.value}: {msg.content}"
            for msg in conversation.messages
        ])

        # Use LLM to extract requirements
        text_service = await self._get_text_service()
        extraction_prompt = system_prompt or self.DEFAULT_EXTRACTION_PROMPT

        try:
            result = await text_service.generate_text(
                prompt=f"Conversation:\n{conversation_text}\n\nExtract requirements:",
                system_prompt=extraction_prompt,
                temperature=self._config.extraction_temperature,
            )

            # Parse JSON response
            try:
                extracted = json.loads(result.content.strip())
                status.subject = extracted.get("subject") or status.subject
                status.style = extracted.get("style") or status.style
                status.colors = extracted.get("colors") or status.colors
                status.mood = extracted.get("mood") or status.mood
                status.composition = extracted.get("composition") or status.composition
                status.include_elements = extracted.get("include_elements", []) or status.include_elements
                status.avoid_elements = extracted.get("avoid_elements", []) or status.avoid_elements
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse requirements extraction response: {e}")
                # Don't raise - return partial results

        except TextLLMServiceError as e:
            logger.warning(f"Requirements extraction LLM call failed: {e}")
            # Don't raise - return partial results

        return status

    def calculate_completeness(
        self,
        requirements: RequirementsStatus,
        weights: Optional[Dict[str, float]] = None,
    ) -> float:
        """
        Calculate weighted completeness score for requirements.

        Args:
            requirements: RequirementsStatus to score
            weights: Optional weight overrides (uses config if not provided)

        Returns:
            Completeness score from 0.0 to 1.0
        """
        weights = weights or self._config.completeness_weights

        score = 0.0
        if requirements.subject:
            score += weights.get("subject", 0.5)
        if requirements.style:
            score += weights.get("style", 0.2)
        if requirements.colors:
            score += weights.get("colors", 0.15)
        if requirements.mood:
            score += weights.get("mood", 0.15)

        return min(1.0, score)

    def get_missing_fields(
        self,
        requirements: RequirementsStatus,
        include_recommended: bool = True,
    ) -> List[str]:
        """
        Get list of missing required and optionally recommended fields.

        Args:
            requirements: RequirementsStatus to check
            include_recommended: Whether to include recommended fields

        Returns:
            List of missing field names
        """
        missing = []

        # Check required fields
        for field in self._config.required_fields:
            value = getattr(requirements, field, None)
            if value is None or (isinstance(value, list) and not value):
                missing.append(field)

        # Check recommended fields
        if include_recommended:
            for field in self._config.recommended_fields:
                value = getattr(requirements, field, None)
                if value is None or (isinstance(value, list) and not value):
                    missing.append(field)

        return missing

    def is_complete(
        self,
        requirements: RequirementsStatus,
        threshold: Optional[float] = None,
    ) -> bool:
        """
        Check if requirements meet minimum completeness threshold.

        The primary check is that subject is present. Completeness score
        is used for UI feedback but not for determining "completeness".

        Args:
            requirements: RequirementsStatus to check
            threshold: Optional threshold override

        Returns:
            True if requirements are complete
        """
        # Subject is always required
        if not requirements.subject:
            return False

        return True

    def validate_input(self, user_input: str) -> Dict[str, Any]:
        """
        Validate user input before processing.

        Args:
            user_input: Raw user input string

        Returns:
            Dict with is_valid flag and any issues
        """
        issues = []

        # Check for empty input
        if not user_input or not user_input.strip():
            issues.append("Input cannot be empty")
            return {"is_valid": False, "issues": issues}

        # Check minimum length
        if len(user_input.strip()) < self._config.min_input_length:
            issues.append("Input is too short")

        # Check maximum length
        if len(user_input) > self._config.max_input_length:
            issues.append(f"Input is too long (max {self._config.max_input_length} characters)")

        return {"is_valid": len(issues) == 0, "issues": issues}

    async def __aenter__(self) -> "RequirementsAnalysisService":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()
