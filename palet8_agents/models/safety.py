"""
Safety classification models.

Used by SafetyAgent and PlannerAgent for content safety checks.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class SafetyCategory(Enum):
    """Categories of safety concerns."""
    NSFW = "nsfw"
    VIOLENCE = "violence"
    HATE = "hate"
    IP_TRADEMARK = "ip_trademark"
    ILLEGAL = "illegal"


class SafetySeverity(Enum):
    """Severity levels for safety flags."""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SafetyClassification:
    """Safety classification result from PlannerAgent quick check."""
    is_safe: bool
    requires_review: bool
    risk_level: str  # "low", "medium", "high"
    categories: List[str]  # Detected risk categories
    flags: Dict[str, bool] = field(default_factory=dict)
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "is_safe": self.is_safe,
            "requires_review": self.requires_review,
            "risk_level": self.risk_level,
            "categories": self.categories,
            "flags": self.flags,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SafetyClassification":
        """Create from dictionary."""
        return cls(
            is_safe=data.get("is_safe", True),
            requires_review=data.get("requires_review", False),
            risk_level=data.get("risk_level", "low"),
            categories=data.get("categories", []),
            flags=data.get("flags", {}),
            reason=data.get("reason", ""),
        )


@dataclass
class SafetyFlag:
    """Individual safety flag from SafetyAgent analysis."""
    category: SafetyCategory
    severity: SafetySeverity
    score: float  # 0.0 to 1.0
    description: str
    source: str  # "input", "prompt", "image"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "category": self.category.value,
            "severity": self.severity.value,
            "score": self.score,
            "description": self.description,
            "source": self.source,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SafetyFlag":
        """Create from dictionary."""
        return cls(
            category=SafetyCategory(data.get("category", "nsfw")),
            severity=SafetySeverity(data.get("severity", "none")),
            score=data.get("score", 0.0),
            description=data.get("description", ""),
            source=data.get("source", "input"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class SafetyResult:
    """Complete safety analysis result from SafetyAgent."""
    job_id: str
    is_safe: bool
    overall_score: float  # 0.0 to 1.0 (higher = safer)
    flags: List[SafetyFlag] = field(default_factory=list)
    blocked_categories: List[str] = field(default_factory=list)
    user_message: Optional[str] = None
    alternatives: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "job_id": self.job_id,
            "is_safe": self.is_safe,
            "overall_score": self.overall_score,
            "flags": [f.to_dict() for f in self.flags],
            "blocked_categories": self.blocked_categories,
            "user_message": self.user_message,
            "alternatives": self.alternatives,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SafetyResult":
        """Create from dictionary."""
        return cls(
            job_id=data.get("job_id", ""),
            is_safe=data.get("is_safe", True),
            overall_score=data.get("overall_score", 1.0),
            flags=[SafetyFlag.from_dict(f) for f in data.get("flags", [])],
            blocked_categories=data.get("blocked_categories", []),
            user_message=data.get("user_message"),
            alternatives=data.get("alternatives", []),
            metadata=data.get("metadata", {}),
        )
