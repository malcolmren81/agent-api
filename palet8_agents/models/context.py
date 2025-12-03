"""
Context completeness models.

Used by PlannerAgent for context evaluation.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class ContextCompleteness:
    """Result of context completeness evaluation."""
    score: float  # 0.0 to 1.0
    is_sufficient: bool
    missing_fields: List[str]
    clarifying_questions: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "score": self.score,
            "is_sufficient": self.is_sufficient,
            "missing_fields": self.missing_fields,
            "clarifying_questions": self.clarifying_questions,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ContextCompleteness":
        """Create from dictionary."""
        return cls(
            score=data.get("score", 0.0),
            is_sufficient=data.get("is_sufficient", False),
            missing_fields=data.get("missing_fields", []),
            clarifying_questions=data.get("clarifying_questions", []),
            metadata=data.get("metadata", {}),
        )
