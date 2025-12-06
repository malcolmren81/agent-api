"""
GenPlan models for generation planning.

Defines data contracts for the GenPlan Agent which handles:
- Complexity analysis
- User info parsing
- Genflow (generation method) selection
- Model selection and parameter extraction
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

from palet8_agents.models import PipelineConfig


# =============================================================================
# User Info Models
# =============================================================================


@dataclass
class UserParseResult:
    """
    Parsed user info extracted from requirements.

    Contains structured information about what the user wants to generate,
    extracted from the raw requirements dictionary.
    """
    # Required
    subject: str

    # Visual style
    style: Optional[str] = None
    mood: Optional[str] = None
    colors: List[str] = field(default_factory=list)

    # Product context
    product_type: str = "general"
    print_method: Optional[str] = None

    # Reference handling
    has_reference: bool = False
    reference_image_url: Optional[str] = None

    # Text content for overlays
    text_content: Optional[str] = None

    # Extracted intents (e.g., ["poster", "text_overlay", "vintage"])
    extracted_intents: List[str] = field(default_factory=list)

    # Additional parsed fields
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "subject": self.subject,
            "style": self.style,
            "mood": self.mood,
            "colors": self.colors,
            "product_type": self.product_type,
            "print_method": self.print_method,
            "has_reference": self.has_reference,
            "reference_image_url": self.reference_image_url,
            "text_content": self.text_content,
            "extracted_intents": self.extracted_intents,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UserParseResult":
        """Create from dictionary."""
        return cls(
            subject=data.get("subject", ""),
            style=data.get("style"),
            mood=data.get("mood"),
            colors=data.get("colors", []),
            product_type=data.get("product_type", "general"),
            print_method=data.get("print_method"),
            has_reference=data.get("has_reference", False),
            reference_image_url=data.get("reference_image_url"),
            text_content=data.get("text_content"),
            extracted_intents=data.get("extracted_intents", []),
            metadata=data.get("metadata", {}),
        )


# =============================================================================
# Genflow Models
# =============================================================================


@dataclass
class GenflowConfig:
    """
    Generation flow configuration.

    Determines whether to use single or dual pipeline generation
    and which specific pipeline configuration to use.
    """
    # Flow type: "single" or "dual"
    flow_type: Literal["single", "dual"]

    # Flow name (e.g., "single_standard", "creative_art", "photorealistic", "layout_poster")
    flow_name: str

    # Human-readable description
    description: str = ""

    # Rationale for this selection
    rationale: str = ""

    # Trigger that activated this flow (if any)
    triggered_by: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "flow_type": self.flow_type,
            "flow_name": self.flow_name,
            "description": self.description,
            "rationale": self.rationale,
            "triggered_by": self.triggered_by,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GenflowConfig":
        """Create from dictionary."""
        return cls(
            flow_type=data.get("flow_type", "single"),
            flow_name=data.get("flow_name", "single_standard"),
            description=data.get("description", ""),
            rationale=data.get("rationale", ""),
            triggered_by=data.get("triggered_by"),
        )

    @property
    def is_dual(self) -> bool:
        """Check if this is a dual pipeline flow."""
        return self.flow_type == "dual"

    @property
    def is_single(self) -> bool:
        """Check if this is a single pipeline flow."""
        return self.flow_type == "single"


# =============================================================================
# Generation Plan Models
# =============================================================================


@dataclass
class GenerationPlan:
    """
    Complete generation plan output from GenPlan Agent.

    Contains all decisions made during generation planning:
    - Complexity assessment
    - Parsed user info
    - Generation flow configuration
    - Model selection with alternatives
    - Input and provider parameters
    - Pipeline configuration
    """
    # Job context
    job_id: str
    user_id: str

    # Complexity assessment
    complexity: Literal["simple", "standard", "complex"]
    complexity_rationale: str

    # Parsed user info
    user_info: UserParseResult

    # Generation flow
    genflow: GenflowConfig

    # Model selection
    model_id: str
    model_rationale: str
    model_alternatives: List[str] = field(default_factory=list)
    model_specs: Dict[str, Any] = field(default_factory=dict)

    # Parameters (flat structure for downstream compatibility)
    # For dual pipelines, these contain stage_1 params for backward compatibility
    model_input_params: Dict[str, Any] = field(default_factory=dict)
    provider_params: Dict[str, Any] = field(default_factory=dict)

    # Stage-specific parameters (for dual pipelines)
    # Structure: {"stage_1": {...}, "stage_2": {...}}
    # AssemblyService uses these for dual pipeline execution
    model_input_params_by_stage: Optional[Dict[str, Dict[str, Any]]] = None
    provider_params_by_stage: Optional[Dict[str, Dict[str, Any]]] = None

    # Pipeline configuration
    pipeline: Optional[PipelineConfig] = None

    # Cost and timing estimates
    estimated_cost: Optional[float] = None
    estimated_latency_ms: Optional[int] = None

    # Validation
    is_valid: bool = True
    validation_errors: List[str] = field(default_factory=list)

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "job_id": self.job_id,
            "user_id": self.user_id,
            "complexity": self.complexity,
            "complexity_rationale": self.complexity_rationale,
            "user_info": self.user_info.to_dict() if self.user_info else None,
            "genflow": self.genflow.to_dict() if self.genflow else None,
            "model_id": self.model_id,
            "model_rationale": self.model_rationale,
            "model_alternatives": self.model_alternatives,
            "model_specs": self.model_specs,
            "model_input_params": self.model_input_params,
            "provider_params": self.provider_params,
            "pipeline": self.pipeline.to_dict() if self.pipeline else None,
            "estimated_cost": self.estimated_cost,
            "estimated_latency_ms": self.estimated_latency_ms,
            "is_valid": self.is_valid,
            "validation_errors": self.validation_errors,
            "metadata": self.metadata,
        }
        # Include stage-specific params if present (dual pipelines)
        if self.model_input_params_by_stage:
            result["model_input_params_by_stage"] = self.model_input_params_by_stage
        if self.provider_params_by_stage:
            result["provider_params_by_stage"] = self.provider_params_by_stage
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GenerationPlan":
        """Create from dictionary."""
        user_info_data = data.get("user_info")
        user_info = UserParseResult.from_dict(user_info_data) if user_info_data else None

        genflow_data = data.get("genflow")
        genflow = GenflowConfig.from_dict(genflow_data) if genflow_data else None

        pipeline_data = data.get("pipeline")
        pipeline = PipelineConfig.from_dict(pipeline_data) if pipeline_data else None

        return cls(
            job_id=data.get("job_id", ""),
            user_id=data.get("user_id", ""),
            complexity=data.get("complexity", "standard"),
            complexity_rationale=data.get("complexity_rationale", ""),
            user_info=user_info,
            genflow=genflow,
            model_id=data.get("model_id", ""),
            model_rationale=data.get("model_rationale", ""),
            model_alternatives=data.get("model_alternatives", []),
            model_specs=data.get("model_specs", {}),
            model_input_params=data.get("model_input_params", {}),
            provider_params=data.get("provider_params", {}),
            model_input_params_by_stage=data.get("model_input_params_by_stage"),
            provider_params_by_stage=data.get("provider_params_by_stage"),
            pipeline=pipeline,
            estimated_cost=data.get("estimated_cost"),
            estimated_latency_ms=data.get("estimated_latency_ms"),
            is_valid=data.get("is_valid", True),
            validation_errors=data.get("validation_errors", []),
            metadata=data.get("metadata", {}),
        )

    @property
    def is_dual_pipeline(self) -> bool:
        """Check if this plan uses dual pipeline."""
        return self.genflow.is_dual if self.genflow else False

    @property
    def is_complex(self) -> bool:
        """Check if this is a complex generation."""
        return self.complexity == "complex"

    @property
    def is_simple(self) -> bool:
        """Check if this is a simple generation."""
        return self.complexity == "simple"


# =============================================================================
# GenPlan State (for ReAct loop tracking)
# =============================================================================


@dataclass
class GenPlanState:
    """
    Tracks accumulated state through GenPlan ReAct loop.

    Each step of the ReAct loop populates different fields,
    building up to a complete GenerationPlan.
    """
    # Job context (input)
    job_id: str = ""
    user_id: str = ""

    # Step 1: ANALYZE_COMPLEXITY output
    complexity: Optional[str] = None
    complexity_rationale: Optional[str] = None

    # Step 2: PARSE_USER_INFO output
    user_info: Optional[UserParseResult] = None

    # Step 3: DETERMINE_GENFLOW output
    genflow: Optional[GenflowConfig] = None

    # Step 4: SELECT_MODEL output
    model_id: Optional[str] = None
    model_rationale: Optional[str] = None
    model_alternatives: List[str] = field(default_factory=list)
    model_specs: Dict[str, Any] = field(default_factory=dict)

    # Step 5: EXTRACT_PARAMETERS output
    # Flat params for downstream compatibility (ReactPrompt, Planner)
    model_input_params: Dict[str, Any] = field(default_factory=dict)
    provider_params: Dict[str, Any] = field(default_factory=dict)
    # Stage-specific params for dual pipelines (AssemblyService)
    model_input_params_by_stage: Optional[Dict[str, Dict[str, Any]]] = None
    provider_params_by_stage: Optional[Dict[str, Dict[str, Any]]] = None
    parameters_extracted: bool = False

    # Step 6: VALIDATE_PLAN output
    validated: bool = False
    validation_errors: List[str] = field(default_factory=list)

    # Pipeline config (built from genflow + model)
    pipeline: Optional[PipelineConfig] = None

    # Cost/timing estimates
    estimated_cost: Optional[float] = None
    estimated_latency_ms: Optional[int] = None

    # Step tracking
    current_step: int = 0
    steps_completed: List[str] = field(default_factory=list)

    def to_generation_plan(self) -> GenerationPlan:
        """Convert state to final GenerationPlan."""
        return GenerationPlan(
            job_id=self.job_id,
            user_id=self.user_id,
            complexity=self.complexity or "standard",
            complexity_rationale=self.complexity_rationale or "",
            user_info=self.user_info or UserParseResult(subject=""),
            genflow=self.genflow or GenflowConfig(flow_type="single", flow_name="single_standard"),
            model_id=self.model_id or "",
            model_rationale=self.model_rationale or "",
            model_alternatives=self.model_alternatives,
            model_specs=self.model_specs,
            model_input_params=self.model_input_params,
            provider_params=self.provider_params,
            model_input_params_by_stage=self.model_input_params_by_stage,
            provider_params_by_stage=self.provider_params_by_stage,
            pipeline=self.pipeline,
            estimated_cost=self.estimated_cost,
            estimated_latency_ms=self.estimated_latency_ms,
            is_valid=self.validated and len(self.validation_errors) == 0,
            validation_errors=self.validation_errors,
        )

    @property
    def is_complete(self) -> bool:
        """Check if all steps are complete."""
        return (
            self.complexity is not None
            and self.user_info is not None
            and self.genflow is not None
            and self.model_id is not None
            and self.parameters_extracted
            and self.validated
        )
