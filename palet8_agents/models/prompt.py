"""
Prompt dimension and quality models.

Used by PlannerAgent and EvaluatorAgent for prompt composition and assessment.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class PromptQualityDimension(Enum):
    """Dimensions for prompt quality assessment."""
    COVERAGE = "coverage"                    # Required elements present
    CLARITY = "clarity"                      # Clear, unambiguous language
    PRODUCT_CONSTRAINTS = "product_constraints"  # Print/product requirements met
    STYLE_ALIGNMENT = "style_alignment"      # Consistent style direction
    CONTROL_SURFACE = "control_surface"      # Effective negative prompts


@dataclass
class PromptDimensions:
    """Selected dimensions for prompt assembly."""
    subject: Optional[str] = None
    aesthetic: Optional[str] = None
    color: Optional[str] = None
    composition: Optional[str] = None
    background: Optional[str] = None
    lighting: Optional[str] = None
    texture: Optional[str] = None
    detail_level: Optional[str] = None
    mood: Optional[str] = None
    expression: Optional[str] = None
    pose: Optional[str] = None
    art_movement: Optional[str] = None
    reference_style: Optional[str] = None
    technical: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding None values."""
        result = {}
        for key, value in self.__dict__.items():
            if value is not None:
                if isinstance(value, dict) and not value:
                    continue
                result[key] = value
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PromptDimensions":
        """Create from dictionary."""
        return cls(
            subject=data.get("subject"),
            aesthetic=data.get("aesthetic"),
            color=data.get("color"),
            composition=data.get("composition"),
            background=data.get("background"),
            lighting=data.get("lighting"),
            texture=data.get("texture"),
            detail_level=data.get("detail_level"),
            mood=data.get("mood"),
            expression=data.get("expression"),
            pose=data.get("pose"),
            art_movement=data.get("art_movement"),
            reference_style=data.get("reference_style"),
            technical=data.get("technical", {}),
        )


@dataclass
class PromptQualityResult:
    """Result of prompt quality assessment."""
    overall: float  # 0.0 to 1.0
    dimensions: Dict[str, float] = field(default_factory=dict)
    mode: str = "STANDARD"  # RELAX, STANDARD, COMPLEX
    threshold: float = 0.70
    decision: str = "PASS"  # PASS, FIX_REQUIRED, POLICY_FAIL
    feedback: List[str] = field(default_factory=list)
    failed_dimensions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_acceptable(self) -> bool:
        """Check if prompt quality meets threshold."""
        return self.overall >= self.threshold and self.decision == "PASS"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "overall": self.overall,
            "dimensions": self.dimensions,
            "mode": self.mode,
            "threshold": self.threshold,
            "decision": self.decision,
            "feedback": self.feedback,
            "failed_dimensions": self.failed_dimensions,
            "metadata": self.metadata,
            "is_acceptable": self.is_acceptable,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PromptQualityResult":
        """Create from dictionary."""
        return cls(
            overall=data.get("overall", 0.0),
            dimensions=data.get("dimensions", {}),
            mode=data.get("mode", "STANDARD"),
            threshold=data.get("threshold", 0.70),
            decision=data.get("decision", "PASS"),
            feedback=data.get("feedback", []),
            failed_dimensions=data.get("failed_dimensions", []),
            metadata=data.get("metadata", {}),
        )
