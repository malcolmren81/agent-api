"""
Image generation models.

Used by PlannerAgent for building generation requests
and AssemblyService for execution results.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum

from .safety import SafetyClassification


class ExecutionStatus(Enum):
    """Status of an execution."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class GenerationParameters:
    """Image generation parameters.

    Note: steps and guidance_scale are Optional because not all models support them.
    Provider-hosted models (Midjourney, Ideogram, Imagen) don't support steps.
    GenPlan determines which parameters are supported based on model config.
    """
    width: int = 1024
    height: int = 1024
    steps: Optional[int] = None  # Only for models that support it (e.g., FLUX, SD)
    guidance_scale: Optional[float] = None  # Only for models that support cfg_scale
    seed: Optional[int] = None
    num_images: int = 1
    # Provider-specific settings (varies by model)
    provider_settings: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, omitting None values for optional params."""
        result = {
            "width": self.width,
            "height": self.height,
            "num_images": self.num_images,
        }
        # Only include optional params if they have values
        if self.steps is not None:
            result["steps"] = self.steps
        if self.guidance_scale is not None:
            result["guidance_scale"] = self.guidance_scale
        if self.seed is not None:
            result["seed"] = self.seed
        if self.provider_settings:
            result["provider_settings"] = self.provider_settings
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GenerationParameters":
        """Create from dictionary, preserving None for unspecified optional params."""
        return cls(
            width=data.get("width", 1024),
            height=data.get("height", 1024),
            steps=data.get("steps"),  # No default - respect what GenPlan decided
            guidance_scale=data.get("guidance_scale"),  # No default
            seed=data.get("seed"),
            num_images=data.get("num_images", 1),
            provider_settings=data.get("provider_settings", {}),
        )


@dataclass
class PipelineConfig:
    """Configuration for generation pipeline (single or dual model)."""
    pipeline_type: str = "single"  # "single" or "dual"
    pipeline_name: Optional[str] = None  # e.g., "creative_art", "photorealistic", "layout_poster"

    # Stage 1 (generator) - always used
    stage_1_model: str = ""
    stage_1_purpose: str = ""

    # Stage 2 (editor) - only for dual pipeline
    stage_2_model: Optional[str] = None
    stage_2_purpose: Optional[str] = None

    # Decision rationale
    decision_rationale: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "pipeline_type": self.pipeline_type,
            "pipeline_name": self.pipeline_name,
            "stage_1_model": self.stage_1_model,
            "stage_1_purpose": self.stage_1_purpose,
            "stage_2_model": self.stage_2_model,
            "stage_2_purpose": self.stage_2_purpose,
            "decision_rationale": self.decision_rationale,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PipelineConfig":
        """Create from dictionary."""
        return cls(
            pipeline_type=data.get("pipeline_type", "single"),
            pipeline_name=data.get("pipeline_name"),
            stage_1_model=data.get("stage_1_model", ""),
            stage_1_purpose=data.get("stage_1_purpose", ""),
            stage_2_model=data.get("stage_2_model"),
            stage_2_purpose=data.get("stage_2_purpose"),
            decision_rationale=data.get("decision_rationale", ""),
        )


@dataclass
class AssemblyRequest:
    """
    Structured request for downstream services (Generation, Writer, Evaluation).

    This is the OUTPUT of Planner Agent - contains everything needed for generation.
    """
    # Core prompt data
    prompt: str = ""
    negative_prompt: str = ""
    mode: str = "STANDARD"
    dimensions: Dict[str, Any] = field(default_factory=dict)

    # Pipeline configuration (single vs dual model)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)

    # Model selection (primary model for single, stage_1 for dual)
    model_id: str = ""
    model_rationale: str = ""
    model_alternatives: List[str] = field(default_factory=list)

    # Generation parameters
    parameters: GenerationParameters = field(default_factory=GenerationParameters)

    # Reference assets
    reference_image_url: Optional[str] = None
    reference_strength: float = 0.75

    # Quality assessment
    prompt_quality_score: float = 0.0
    quality_acceptable: bool = False

    # Safety
    safety: SafetyClassification = field(default_factory=lambda: SafetyClassification(
        is_safe=True, requires_review=False, risk_level="low", categories=[]
    ))

    # Cost estimation
    estimated_cost: float = 0.0
    estimated_time_ms: int = 0

    # Context used
    context_used: Optional[Dict[str, Any]] = None

    # Job metadata
    job_id: str = ""
    user_id: str = ""
    product_type: str = ""
    print_method: Optional[str] = None
    revision_count: int = 0

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "prompt": self.prompt,
            "negative_prompt": self.negative_prompt,
            "mode": self.mode,
            "dimensions": self.dimensions,
            "pipeline": self.pipeline.to_dict() if isinstance(self.pipeline, PipelineConfig) else self.pipeline,
            "model_id": self.model_id,
            "model_rationale": self.model_rationale,
            "model_alternatives": self.model_alternatives,
            "parameters": self.parameters.to_dict() if isinstance(self.parameters, GenerationParameters) else self.parameters,
            "reference_image_url": self.reference_image_url,
            "reference_strength": self.reference_strength,
            "prompt_quality_score": self.prompt_quality_score,
            "quality_acceptable": self.quality_acceptable,
            "safety": self.safety.to_dict() if isinstance(self.safety, SafetyClassification) else self.safety,
            "estimated_cost": self.estimated_cost,
            "estimated_time_ms": self.estimated_time_ms,
            "context_used": self.context_used,
            "job_id": self.job_id,
            "user_id": self.user_id,
            "product_type": self.product_type,
            "print_method": self.print_method,
            "revision_count": self.revision_count,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AssemblyRequest":
        """Create from dictionary."""
        pipeline_data = data.get("pipeline", {})
        if isinstance(pipeline_data, dict):
            pipeline = PipelineConfig.from_dict(pipeline_data)
        else:
            pipeline = pipeline_data

        params_data = data.get("parameters", {})
        if isinstance(params_data, dict):
            parameters = GenerationParameters.from_dict(params_data)
        else:
            parameters = params_data

        safety_data = data.get("safety", {})
        if isinstance(safety_data, dict):
            safety = SafetyClassification.from_dict(safety_data)
        else:
            safety = safety_data

        return cls(
            prompt=data.get("prompt", ""),
            negative_prompt=data.get("negative_prompt", ""),
            mode=data.get("mode", "STANDARD"),
            dimensions=data.get("dimensions", {}),
            pipeline=pipeline,
            model_id=data.get("model_id", ""),
            model_rationale=data.get("model_rationale", ""),
            model_alternatives=data.get("model_alternatives", []),
            parameters=parameters,
            reference_image_url=data.get("reference_image_url"),
            reference_strength=data.get("reference_strength", 0.75),
            prompt_quality_score=data.get("prompt_quality_score", 0.0),
            quality_acceptable=data.get("quality_acceptable", False),
            safety=safety,
            estimated_cost=data.get("estimated_cost", 0.0),
            estimated_time_ms=data.get("estimated_time_ms", 0),
            context_used=data.get("context_used"),
            job_id=data.get("job_id", ""),
            user_id=data.get("user_id", ""),
            product_type=data.get("product_type", ""),
            print_method=data.get("print_method"),
            revision_count=data.get("revision_count", 0),
            metadata=data.get("metadata", {}),
        )


@dataclass
class GeneratedImageData:
    """Data for a single generated image."""
    url: Optional[str] = None
    base64_data: Optional[str] = None
    seed: Optional[int] = None
    revised_prompt: Optional[str] = None
    provider: str = ""
    model_used: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "url": self.url,
            "has_base64": self.base64_data is not None,
            "seed": self.seed,
            "revised_prompt": self.revised_prompt,
            "provider": self.provider,
            "model_used": self.model_used,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GeneratedImageData":
        """Create from dictionary."""
        return cls(
            url=data.get("url"),
            base64_data=data.get("base64_data"),
            seed=data.get("seed"),
            revised_prompt=data.get("revised_prompt"),
            provider=data.get("provider", ""),
            model_used=data.get("model_used", ""),
            metadata=data.get("metadata", {}),
        )


@dataclass
class ExecutionResult:
    """
    Result from the AssemblyService after generation execution.

    Contains generated images, cost, timing, and status information.
    """
    # Status
    status: ExecutionStatus = ExecutionStatus.PENDING
    success: bool = False
    error: Optional[str] = None
    error_code: Optional[str] = None

    # Generated images
    images: List[GeneratedImageData] = field(default_factory=list)

    # Model and provider info
    model_used: str = ""
    provider: str = ""
    pipeline_type: str = "single"

    # Cost and timing
    actual_cost: float = 0.0
    duration_ms: int = 0

    # Stage tracking (for dual pipeline)
    stage_1_result: Optional[Dict[str, Any]] = None
    stage_2_result: Optional[Dict[str, Any]] = None

    # Job reference
    job_id: str = ""
    task_id: str = ""

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def first_image_url(self) -> Optional[str]:
        """Get URL of first image."""
        return self.images[0].url if self.images else None

    @property
    def first_image_base64(self) -> Optional[str]:
        """Get base64 data of first image."""
        return self.images[0].base64_data if self.images else None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "status": self.status.value,
            "success": self.success,
            "error": self.error,
            "error_code": self.error_code,
            "images": [img.to_dict() for img in self.images],
            "model_used": self.model_used,
            "provider": self.provider,
            "pipeline_type": self.pipeline_type,
            "actual_cost": self.actual_cost,
            "duration_ms": self.duration_ms,
            "stage_1_result": self.stage_1_result,
            "stage_2_result": self.stage_2_result,
            "job_id": self.job_id,
            "task_id": self.task_id,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExecutionResult":
        """Create from dictionary."""
        status_str = data.get("status", "pending")
        status = ExecutionStatus(status_str) if isinstance(status_str, str) else status_str

        images_data = data.get("images", [])
        images = [
            GeneratedImageData.from_dict(img) if isinstance(img, dict) else img
            for img in images_data
        ]

        return cls(
            status=status,
            success=data.get("success", False),
            error=data.get("error"),
            error_code=data.get("error_code"),
            images=images,
            model_used=data.get("model_used", ""),
            provider=data.get("provider", ""),
            pipeline_type=data.get("pipeline_type", "single"),
            actual_cost=data.get("actual_cost", 0.0),
            duration_ms=data.get("duration_ms", 0),
            stage_1_result=data.get("stage_1_result"),
            stage_2_result=data.get("stage_2_result"),
            job_id=data.get("job_id", ""),
            task_id=data.get("task_id", ""),
            metadata=data.get("metadata", {}),
        )
