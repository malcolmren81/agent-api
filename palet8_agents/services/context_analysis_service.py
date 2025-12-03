"""
Context Analysis Service

Service for evaluating context completeness for planner decision-making.
This extracts the context analysis logic from PlannerAgent into a reusable service.

Documentation Reference: Section 5.2.2 (Planner Agent)
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import logging
import yaml

from palet8_agents.models import ContextCompleteness

logger = logging.getLogger(__name__)


class ContextAnalysisError(Exception):
    """Base exception for ContextAnalysisService errors."""
    pass


@dataclass
class ContextConfig:
    """Configuration for context analysis."""
    required_fields: List[str] = field(default_factory=lambda: ["subject"])
    important_fields: List[str] = field(default_factory=lambda: [
        "style", "aesthetic", "colors", "product_type"
    ])
    optional_fields: List[str] = field(default_factory=lambda: [
        "mood", "composition", "background", "lighting"
    ])
    completeness_weights: Dict[str, float] = field(default_factory=lambda: {
        "subject": 0.40,
        "style": 0.15,
        "aesthetic": 0.15,
        "colors": 0.10,
        "product_type": 0.10,
        "mood": 0.05,
        "composition": 0.05,
    })
    min_completeness: float = 0.5
    clarifying_questions: Dict[str, str] = field(default_factory=lambda: {
        "subject": "What would you like the image to show? Please describe the main subject.",
        "style": "What style are you looking for? (e.g., realistic, cartoon, minimalist)",
        "aesthetic": "What aesthetic or visual style do you prefer?",
        "colors": "Do you have any color preferences for this design?",
        "product_type": "What product is this design for? (e.g., t-shirt, poster, phone case)",
        "mood": "What mood or feeling should the image convey?",
        "composition": "How would you like the elements arranged in the image?",
        "background": "What kind of background would you prefer?",
    })


class ContextAnalysisService:
    """
    Service for evaluating context completeness.

    This service:
    - Evaluates if gathered requirements provide sufficient context for planning
    - Calculates weighted completeness scores
    - Generates clarifying questions for missing fields
    - Prioritizes fields by importance (required > important > optional)
    """

    def __init__(
        self,
        config: Optional[ContextConfig] = None,
        config_path: Optional[Path] = None,
    ):
        """
        Initialize ContextAnalysisService.

        Args:
            config: Optional ContextConfig. Loaded from file if not provided.
            config_path: Path to config YAML file. Uses default if not provided.
        """
        self._config = config or self._load_config(config_path)

    def _load_config(self, config_path: Optional[Path] = None) -> ContextConfig:
        """Load configuration from YAML file."""
        if config_path is None:
            # Try default config location
            default_paths = [
                Path("config/context_analysis.yaml"),
                Path(__file__).parent.parent.parent / "config" / "context_analysis.yaml",
            ]
            for path in default_paths:
                if path.exists():
                    config_path = path
                    break

        if config_path and config_path.exists():
            try:
                with open(config_path, "r") as f:
                    data = yaml.safe_load(f)

                fields = data.get("fields", {})
                thresholds = data.get("thresholds", {})

                return ContextConfig(
                    required_fields=fields.get("required", ["subject"]),
                    important_fields=fields.get("important", []),
                    optional_fields=fields.get("optional", []),
                    completeness_weights=data.get("completeness_weights", {}),
                    min_completeness=thresholds.get("min_completeness", 0.5),
                    clarifying_questions=data.get("clarifying_questions", {}),
                )
            except Exception as e:
                logger.warning(f"Failed to load config from {config_path}: {e}")

        return ContextConfig()

    def evaluate_completeness(
        self,
        requirements: Dict[str, Any],
        threshold: Optional[float] = None,
    ) -> ContextCompleteness:
        """
        Evaluate if requirements provide sufficient context for planning.

        Args:
            requirements: Dict of gathered requirements
            threshold: Optional threshold override

        Returns:
            ContextCompleteness with score, missing fields, clarifying questions
        """
        threshold = threshold or self._config.min_completeness
        weights = self._config.completeness_weights

        score = 0.0
        missing = []
        questions = []
        metadata = {"required_missing": [], "important_missing": [], "optional_missing": []}

        # Evaluate required fields
        for field_name in self._config.required_fields:
            value = requirements.get(field_name)
            if self._has_value(value):
                score += weights.get(field_name, 0.1)
            else:
                missing.append(field_name)
                metadata["required_missing"].append(field_name)
                question = self._generate_question(field_name)
                if question:
                    questions.append(question)

        # Evaluate important fields
        for field_name in self._config.important_fields:
            value = requirements.get(field_name)
            if self._has_value(value):
                score += weights.get(field_name, 0.1)
            else:
                missing.append(field_name)
                metadata["important_missing"].append(field_name)
                question = self._generate_question(field_name)
                if question:
                    questions.append(question)

        # Evaluate optional fields
        for field_name in self._config.optional_fields:
            value = requirements.get(field_name)
            if self._has_value(value):
                score += weights.get(field_name, 0.05)
            else:
                metadata["optional_missing"].append(field_name)

        # Check if sufficient
        is_sufficient = score >= threshold and len(metadata["required_missing"]) == 0

        return ContextCompleteness(
            score=min(1.0, score),
            is_sufficient=is_sufficient,
            missing_fields=missing,
            clarifying_questions=questions,
            metadata=metadata,
        )

    def _has_value(self, value: Any) -> bool:
        """Check if a value is present and non-empty."""
        if value is None:
            return False
        if isinstance(value, str):
            return bool(value.strip())
        if isinstance(value, (list, dict)):
            return bool(value)
        return True

    def _generate_question(self, field_name: str) -> Optional[str]:
        """
        Generate clarifying question for a missing field.

        Args:
            field_name: Name of the missing field

        Returns:
            Question string or None if no template found
        """
        return self._config.clarifying_questions.get(field_name)

    def get_priority_missing_fields(
        self,
        requirements: Dict[str, Any],
        max_fields: int = 3,
    ) -> List[str]:
        """
        Get the most important missing fields to ask about.

        Returns missing fields in priority order: required > important > optional.

        Args:
            requirements: Dict of gathered requirements
            max_fields: Maximum number of fields to return

        Returns:
            List of missing field names in priority order
        """
        priority_missing = []

        # Check required first
        for field_name in self._config.required_fields:
            if not self._has_value(requirements.get(field_name)):
                priority_missing.append(field_name)
                if len(priority_missing) >= max_fields:
                    return priority_missing

        # Then important
        for field_name in self._config.important_fields:
            if not self._has_value(requirements.get(field_name)):
                priority_missing.append(field_name)
                if len(priority_missing) >= max_fields:
                    return priority_missing

        # Then optional
        for field_name in self._config.optional_fields:
            if not self._has_value(requirements.get(field_name)):
                priority_missing.append(field_name)
                if len(priority_missing) >= max_fields:
                    return priority_missing

        return priority_missing

    def get_priority_questions(
        self,
        requirements: Dict[str, Any],
        max_questions: int = 3,
    ) -> List[str]:
        """
        Get clarifying questions for the most important missing fields.

        Args:
            requirements: Dict of gathered requirements
            max_questions: Maximum number of questions to return

        Returns:
            List of clarifying question strings
        """
        missing_fields = self.get_priority_missing_fields(requirements, max_questions)
        questions = []

        for field_name in missing_fields:
            question = self._generate_question(field_name)
            if question:
                questions.append(question)

        return questions

    def calculate_field_score(
        self,
        field_name: str,
        value: Any,
    ) -> float:
        """
        Calculate score contribution for a single field.

        Args:
            field_name: Name of the field
            value: Field value

        Returns:
            Score contribution (0.0 if empty, weight value if present)
        """
        if not self._has_value(value):
            return 0.0
        return self._config.completeness_weights.get(field_name, 0.0)

    def get_all_fields(self) -> Dict[str, List[str]]:
        """
        Get all field categories.

        Returns:
            Dict with 'required', 'important', 'optional' keys
        """
        return {
            "required": self._config.required_fields.copy(),
            "important": self._config.important_fields.copy(),
            "optional": self._config.optional_fields.copy(),
        }

    def merge_requirements(
        self,
        base: Dict[str, Any],
        updates: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Merge requirement updates into base requirements.

        Updates take precedence. Empty values in updates don't override.

        Args:
            base: Base requirements dict
            updates: Updates to merge

        Returns:
            Merged requirements dict
        """
        result = base.copy()

        for key, value in updates.items():
            if self._has_value(value):
                result[key] = value

        return result
