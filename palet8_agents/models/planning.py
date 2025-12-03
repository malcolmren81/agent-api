"""
Planning models for agent coordination.

Defines data contracts for communication between PlannerAgent and ReactPromptAgent.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional


@dataclass
class PlanningTask:
    """
    Task passed from Planner to ReactPromptAgent.

    Contains all context needed for prompt building, including phase
    information to guide the ReAct agent's behavior.

    Phases:
    - initial: Full context + prompt build from scratch
    - fix_plan: Seeded with previous prompt + EvaluationFeedback, minimal fixes
    - edit: Seeded with existing plan + user edit instructions, preserve as much as possible
    """
    job_id: str
    user_id: str
    phase: Literal["initial", "fix_plan", "edit"]
    requirements: Dict[str, Any]
    complexity: Literal["simple", "standard", "complex"]
    product_type: str
    print_method: Optional[str] = None

    # For fix_plan and edit phases - previous prompt plan
    previous_plan: Optional[Dict[str, Any]] = None

    # For fix_plan phase - feedback from Evaluator
    evaluation_feedback: Optional[Dict[str, Any]] = None

    # For edit phase - user's edit instructions
    edit_instructions: Optional[str] = None

    # Additional context
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "job_id": self.job_id,
            "user_id": self.user_id,
            "phase": self.phase,
            "requirements": self.requirements,
            "complexity": self.complexity,
            "product_type": self.product_type,
            "print_method": self.print_method,
            "previous_plan": self.previous_plan,
            "evaluation_feedback": self.evaluation_feedback,
            "edit_instructions": self.edit_instructions,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PlanningTask":
        """Create from dictionary."""
        return cls(
            job_id=data.get("job_id", ""),
            user_id=data.get("user_id", ""),
            phase=data.get("phase", "initial"),
            requirements=data.get("requirements", {}),
            complexity=data.get("complexity", "standard"),
            product_type=data.get("product_type", "general"),
            print_method=data.get("print_method"),
            previous_plan=data.get("previous_plan"),
            evaluation_feedback=data.get("evaluation_feedback"),
            edit_instructions=data.get("edit_instructions"),
            metadata=data.get("metadata", {}),
        )

    @property
    def is_initial(self) -> bool:
        """Check if this is an initial generation task."""
        return self.phase == "initial"

    @property
    def is_fix(self) -> bool:
        """Check if this is a fix/revision task."""
        return self.phase == "fix_plan"

    @property
    def is_edit(self) -> bool:
        """Check if this is an edit task."""
        return self.phase == "edit"


@dataclass
class ContextSummary:
    """Summary of context sources used during prompt building."""
    user_history_count: int = 0
    art_references_count: int = 0
    web_search_count: int = 0
    rag_sources: List[str] = field(default_factory=list)
    reference_images: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "user_history_count": self.user_history_count,
            "art_references_count": self.art_references_count,
            "web_search_count": self.web_search_count,
            "rag_sources": self.rag_sources,
            "reference_images": self.reference_images,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ContextSummary":
        """Create from dictionary."""
        return cls(
            user_history_count=data.get("user_history_count", 0),
            art_references_count=data.get("art_references_count", 0),
            web_search_count=data.get("web_search_count", 0),
            rag_sources=data.get("rag_sources", []),
            reference_images=data.get("reference_images", []),
            metadata=data.get("metadata", {}),
        )


@dataclass
class PromptPlan:
    """
    Result from ReactPromptAgent back to Planner.

    Contains the built prompt along with quality assessment and
    context usage summary.
    """
    prompt: str
    negative_prompt: str = ""

    # Dimensions used to build the prompt
    dimensions: Dict[str, Any] = field(default_factory=dict)

    # Provider-specific generation parameters
    # These are model/provider-specific settings passed to the final generation
    # Examples: steps, guidance_scale, sampler, style_preset, seed, etc.
    provider_params: Dict[str, Any] = field(default_factory=dict)

    # Quality assessment
    quality_score: float = 0.0
    quality_acceptable: bool = False
    quality_feedback: List[str] = field(default_factory=list)
    failed_dimensions: List[str] = field(default_factory=list)

    # Revision tracking
    revision_count: int = 0
    revision_history: List[str] = field(default_factory=list)

    # Context usage summary
    context_summary: ContextSummary = field(default_factory=ContextSummary)

    # Mode used for generation
    mode: str = "STANDARD"  # RELAX, STANDARD, COMPLEX

    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_acceptable(self) -> bool:
        """Check if the prompt plan meets quality requirements."""
        return self.quality_acceptable and self.quality_score >= 0.7

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "prompt": self.prompt,
            "negative_prompt": self.negative_prompt,
            "dimensions": self.dimensions,
            "provider_params": self.provider_params,
            "quality_score": self.quality_score,
            "quality_acceptable": self.quality_acceptable,
            "quality_feedback": self.quality_feedback,
            "failed_dimensions": self.failed_dimensions,
            "revision_count": self.revision_count,
            "revision_history": self.revision_history,
            "context_summary": (
                self.context_summary.to_dict()
                if isinstance(self.context_summary, ContextSummary)
                else self.context_summary
            ),
            "mode": self.mode,
            "metadata": self.metadata,
            "is_acceptable": self.is_acceptable,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PromptPlan":
        """Create from dictionary."""
        context_summary_data = data.get("context_summary", {})
        if isinstance(context_summary_data, dict):
            context_summary = ContextSummary.from_dict(context_summary_data)
        else:
            context_summary = context_summary_data

        return cls(
            prompt=data.get("prompt", ""),
            negative_prompt=data.get("negative_prompt", ""),
            dimensions=data.get("dimensions", {}),
            provider_params=data.get("provider_params", {}),
            quality_score=data.get("quality_score", 0.0),
            quality_acceptable=data.get("quality_acceptable", False),
            quality_feedback=data.get("quality_feedback", []),
            failed_dimensions=data.get("failed_dimensions", []),
            revision_count=data.get("revision_count", 0),
            revision_history=data.get("revision_history", []),
            context_summary=context_summary,
            mode=data.get("mode", "STANDARD"),
            metadata=data.get("metadata", {}),
        )
