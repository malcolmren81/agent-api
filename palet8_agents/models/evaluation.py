"""
Evaluation models for quality assessment.

Used by EvaluatorAgent for prompt and result quality evaluation.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from .prompt import PromptQualityResult


class EvaluationDecision(Enum):
    """Decision outcomes from evaluation."""
    PASS = "PASS"                  # Prompt quality acceptable
    FIX_REQUIRED = "FIX_REQUIRED"  # Needs revision
    APPROVE = "APPROVE"            # Result quality acceptable
    REJECT = "REJECT"              # Result quality unacceptable
    POLICY_FAIL = "POLICY_FAIL"    # Safety/policy violation


class ResultQualityDimension(Enum):
    """Dimensions for result quality assessment."""
    PROMPT_FIDELITY = "prompt_fidelity"           # Matches prompt description
    PRODUCT_READINESS = "product_readiness"       # Ready for print/product
    TECHNICAL_QUALITY = "technical_quality"       # Resolution, sharpness, etc.
    BACKGROUND_COMPOSITION = "background_composition"  # Background handling
    AESTHETIC = "aesthetic"                       # Visual appeal
    TEXT_LEGIBILITY = "text_legibility"           # Text clarity if present
    SET_CONSISTENCY = "set_consistency"           # Multi-image consistency


@dataclass
class RetrySuggestion:
    """Suggestion for retrying generation."""
    dimension: str
    suggested_changes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "dimension": self.dimension,
            "suggested_changes": self.suggested_changes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RetrySuggestion":
        """Create from dictionary."""
        return cls(
            dimension=data.get("dimension", ""),
            suggested_changes=data.get("suggested_changes", []),
        )


@dataclass
class ResultQualityResult:
    """Result of image quality assessment."""
    overall: float  # 0.0 to 1.0
    dimensions: Dict[str, float] = field(default_factory=dict)
    mode: str = "STANDARD"  # RELAX, STANDARD, COMPLEX
    threshold: float = 0.80
    decision: str = "APPROVE"  # APPROVE, REJECT
    feedback: List[str] = field(default_factory=list)
    failed_dimensions: List[str] = field(default_factory=list)
    retry_suggestions: List[RetrySuggestion] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_acceptable(self) -> bool:
        """Check if result quality meets threshold."""
        return self.overall >= self.threshold and self.decision == "APPROVE"

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
            "retry_suggestions": [s.to_dict() for s in self.retry_suggestions],
            "metadata": self.metadata,
            "is_acceptable": self.is_acceptable,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ResultQualityResult":
        """Create from dictionary."""
        return cls(
            overall=data.get("overall", 0.0),
            dimensions=data.get("dimensions", {}),
            mode=data.get("mode", "STANDARD"),
            threshold=data.get("threshold", 0.80),
            decision=data.get("decision", "APPROVE"),
            feedback=data.get("feedback", []),
            failed_dimensions=data.get("failed_dimensions", []),
            retry_suggestions=[
                RetrySuggestion.from_dict(s) for s in data.get("retry_suggestions", [])
            ],
            metadata=data.get("metadata", {}),
        )


@dataclass
class EvaluationPlan:
    """Plan for evaluating a generation job."""
    job_id: str
    prompt: str
    negative_prompt: str = ""
    mode: str = "STANDARD"
    product_type: Optional[str] = None
    print_method: Optional[str] = None
    dimensions_requested: Dict[str, Any] = field(default_factory=dict)
    prompt_quality: Optional[PromptQualityResult] = None
    result_weights: Dict[str, float] = field(default_factory=dict)
    result_thresholds: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "job_id": self.job_id,
            "prompt": self.prompt,
            "negative_prompt": self.negative_prompt,
            "mode": self.mode,
            "product_type": self.product_type,
            "print_method": self.print_method,
            "dimensions_requested": self.dimensions_requested,
            "prompt_quality": self.prompt_quality.to_dict() if self.prompt_quality else None,
            "result_weights": self.result_weights,
            "result_thresholds": self.result_thresholds,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvaluationPlan":
        """Create from dictionary."""
        prompt_quality_data = data.get("prompt_quality")
        prompt_quality = None
        if prompt_quality_data:
            prompt_quality = PromptQualityResult.from_dict(prompt_quality_data)

        return cls(
            job_id=data.get("job_id", ""),
            prompt=data.get("prompt", ""),
            negative_prompt=data.get("negative_prompt", ""),
            mode=data.get("mode", "STANDARD"),
            product_type=data.get("product_type"),
            print_method=data.get("print_method"),
            dimensions_requested=data.get("dimensions_requested", {}),
            prompt_quality=prompt_quality,
            result_weights=data.get("result_weights", {}),
            result_thresholds=data.get("result_thresholds", {}),
        )


@dataclass
class EvaluationFeedback:
    """Feedback from Evaluator Agent for Fix Plan loop."""
    passed: bool
    overall_score: float
    issues: List[str]
    retry_suggestions: List[str]
    dimension_scores: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "passed": self.passed,
            "overall_score": self.overall_score,
            "issues": self.issues,
            "retry_suggestions": self.retry_suggestions,
            "dimension_scores": self.dimension_scores,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvaluationFeedback":
        """Create from dictionary."""
        return cls(
            passed=data.get("passed", False),
            overall_score=data.get("overall_score", 0.0),
            issues=data.get("issues", []),
            retry_suggestions=data.get("retry_suggestions", []),
            dimension_scores=data.get("dimension_scores", {}),
            metadata=data.get("metadata", {}),
        )
