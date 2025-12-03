"""
Request and response schemas for API endpoints.
"""
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field


class ReasoningModel(str, Enum):
    """Supported reasoning models."""
    GEMINI = "gemini"
    CHATGPT = "chatgpt"


class ImageModel(str, Enum):
    """Supported image generation models."""
    FLUX = "flux"
    GEMINI = "gemini"


class ModelType(str, Enum):
    """Model type categorization."""
    REASONING = "reasoning"
    IMAGE = "image"


# Interactive Agent Schemas
class InteractiveRequest(BaseModel):
    """Request for interactive agent."""
    prompt: str = Field(..., description="User prompt in English")
    reasoning_model: Optional[ReasoningModel] = Field(
        default=None, description="Preferred reasoning model"
    )
    image_model: Optional[ImageModel] = Field(
        default=None, description="Preferred image model"
    )
    user_id: str = Field(..., description="User identifier")
    session_id: Optional[str] = Field(default=None, description="Session identifier")
    product_types: Optional[list[str]] = Field(default=None, description="Product types to generate")
    num_images: Optional[int] = Field(default=2, ge=1, le=4, description="Number of images to generate")

    # Phase 5: Credit system integration
    customer_id: str = Field(..., description="Shopify customer ID for credit operations")
    email: str = Field(..., description="User email address")
    shop_domain: Optional[str] = Field(default=None, description="Multi-tenant shop domain (e.g., store.myshopify.com)")

    # Phase 7.1.2: Product template integration
    template_id: Optional[str] = Field(default=None, description="Selected product template UUID (optional)")
    template_category: Optional[str] = Field(default=None, description="Product category filter (apparel, drinkware, wall-art, accessories)")

    # Phase 7.3: Aesthetic reference integration
    aesthetic_id: Optional[str] = Field(default=None, description="Selected aesthetic reference UUID (optional)")

    # Phase 7.4: Character reference integration
    character_id: Optional[str] = Field(default=None, description="Selected character reference UUID (optional)")


class InteractiveResponse(BaseModel):
    """Response from interactive agent."""
    task_id: str = Field(..., description="Unique task identifier")
    status: str = Field(..., description="Task status")
    message: str = Field(..., description="Response message")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# Planner Agent Schemas
class PlanStep(BaseModel):
    """Individual step in execution plan."""
    step_number: int
    action: str
    description: str
    estimated_cost: float
    estimated_time: float


class PlannerResponse(BaseModel):
    """Response from planner agent."""
    task_id: str
    steps: list[PlanStep]
    total_estimated_cost: float
    total_estimated_time: float
    selected_reasoning_model: ReasoningModel
    selected_image_model: ImageModel


# Generation Agent Schemas
class GenerationRequest(BaseModel):
    """Request for image generation."""
    prompt: str = Field(..., description="Generation prompt in English")
    image_model: ImageModel = Field(..., description="Image model to use")
    style: Optional[str] = Field(default=None, description="Style guidance")
    reference_image: Optional[str] = Field(
        default=None, description="Base64 encoded reference image"
    )
    num_images: int = Field(default=1, ge=1, le=4, description="Number of images")
    task_id: str = Field(..., description="Associated task ID")


class GeneratedImage(BaseModel):
    """Generated image metadata."""
    image_id: str
    url: Optional[str] = None
    base64_data: Optional[str] = None
    model_used: ImageModel
    cost: float
    generation_time: float
    has_watermark: bool


class GenerationResponse(BaseModel):
    """Response from generation agent."""
    task_id: str
    images: list[GeneratedImage]
    total_cost: float
    total_time: float


# Evaluation Agent Schemas
class EvaluationCriteria(BaseModel):
    """Evaluation criteria and weights."""
    prompt_adherence: float = Field(default=0.3, ge=0.0, le=1.0)
    aesthetics: float = Field(default=0.25, ge=0.0, le=1.0)
    product_suitability: float = Field(default=0.25, ge=0.0, le=1.0)
    safety: float = Field(default=0.2, ge=0.0, le=1.0)


class EvaluationRequest(BaseModel):
    """Request for image evaluation."""
    image_id: str = Field(..., description="Image to evaluate")
    original_prompt: str = Field(..., description="Original generation prompt")
    criteria: EvaluationCriteria = Field(default_factory=EvaluationCriteria)
    reasoning_model: ReasoningModel = Field(..., description="Model for evaluation")


class EvaluationScore(BaseModel):
    """Individual evaluation scores."""
    prompt_adherence: float = Field(..., ge=0.0, le=1.0)
    aesthetics: float = Field(..., ge=0.0, le=1.0)
    product_suitability: float = Field(..., ge=0.0, le=1.0)
    safety: float = Field(..., ge=0.0, le=1.0)
    overall: float = Field(..., ge=0.0, le=1.0)


class EvaluationResponse(BaseModel):
    """Response from evaluation agent."""
    image_id: str
    scores: EvaluationScore
    feedback: str
    approved: bool
    reasoning_model_used: ReasoningModel


# Product Generator Schemas
class ProductType(str, Enum):
    """Supported product types."""
    TSHIRT = "tshirt"
    MUG = "mug"
    POSTER = "poster"
    PHONE_CASE = "phone_case"


class ProductRequest(BaseModel):
    """Request for product compositing."""
    image_id: str = Field(..., description="Approved image ID")
    product_types: list[ProductType] = Field(..., description="Product types to generate")
    task_id: str = Field(..., description="Associated task ID")


class ProductImage(BaseModel):
    """Product composite image."""
    product_id: str
    product_type: ProductType
    url: Optional[str] = None
    base64_data: Optional[str] = None
    processing_time: float


class ProductResponse(BaseModel):
    """Response from product generator."""
    task_id: str
    products: list[ProductImage]
    total_time: float


# Model Selection Schemas
class ModelSelectionCriteria(BaseModel):
    """Criteria for model selection."""
    max_cost: Optional[float] = None
    max_latency: Optional[float] = None
    required_features: list[str] = Field(default_factory=list)
    priority: str = Field(default="balanced")  # balanced, cost, speed, quality


class SelectedModels(BaseModel):
    """Selected reasoning and image models."""
    reasoning_model: ReasoningModel
    image_model: ImageModel
    reasoning_cost: float
    image_cost: float
    estimated_latency: float
    rationale: str


# Health & Status Schemas
class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    version: str = "0.1.0"
    services: dict[str, bool]


# Error Schemas
class ErrorDetail(BaseModel):
    """Error detail."""
    code: str
    message: str
    details: Optional[dict[str, Any]] = None


class ErrorResponse(BaseModel):
    """Error response."""
    error: ErrorDetail
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    request_id: Optional[str] = None


# Full Pipeline Response
class FullPipelineResponse(BaseModel):
    """Complete pipeline execution result."""
    success: bool
    products: list[dict[str, Any]] = Field(default_factory=list)
    best_image: dict[str, Any] = Field(default_factory=dict)
    all_evaluations: list[dict[str, Any]] = Field(default_factory=list)
    selected_models: dict[str, Any] = Field(default_factory=dict)
    estimated_cost: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)
    pipeline_stages: Optional[dict[str, Any]] = None
    error: Optional[str] = None
    stage: Optional[str] = None


# Template Management Schemas
class TemplateCategory(str, Enum):
    """Template categories."""
    PRODUCT = "product"
    LIFESTYLE = "lifestyle"
    CREATIVE = "creative"


class TemplateSource(str, Enum):
    """Template source type."""
    MANUAL = "manual"
    LLM_PROMOTED = "llm-promoted"


class CreateTemplateRequest(BaseModel):
    """Request to create a new template."""
    name: str = Field(..., description="Unique template name", min_length=1, max_length=255)
    category: TemplateCategory = Field(..., description="Template category")
    promptText: str = Field(..., description="Template prompt text", min_length=1)
    style: str = Field(default="realistic", description="Default style for template")
    tags: list[str] = Field(default_factory=list, description="Search tags")
    language: str = Field(default="en", description="Template language code")
    createdBy: Optional[str] = Field(default=None, description="Creator user ID")
    source: TemplateSource = Field(default=TemplateSource.MANUAL, description="Template source")


class UpdateTemplateRequest(BaseModel):
    """Request to update an existing template."""
    name: Optional[str] = Field(None, description="Template name", min_length=1, max_length=255)
    category: Optional[TemplateCategory] = Field(None, description="Template category")
    promptText: Optional[str] = Field(None, description="Template prompt text", min_length=1)
    style: Optional[str] = Field(None, description="Default style for template")
    tags: Optional[list[str]] = Field(None, description="Search tags")
    language: Optional[str] = Field(None, description="Template language code")
    isActive: Optional[bool] = Field(None, description="Active status")


class TemplateResponse(BaseModel):
    """Template response model."""
    id: str
    name: str
    category: str
    promptText: str
    style: str
    tags: list[str]
    language: str
    isActive: bool
    createdBy: Optional[str]
    usageCount: int
    acceptRate: Optional[float]
    avgScore: Optional[float]
    lastUsed: Optional[datetime]
    source: str
    createdAt: datetime
    updatedAt: datetime


class TemplateListResponse(BaseModel):
    """Paginated list of templates."""
    templates: list[TemplateResponse]
    total: int
    page: int
    pageSize: int
    hasMore: bool


class TemplateStatsResponse(BaseModel):
    """Template usage statistics."""
    totalTemplates: int
    activeTemplates: int
    totalUsages: int
    averageAcceptRate: float
    averageScore: float
    mostUsedTemplate: Optional[TemplateResponse]
    categoryBreakdown: dict[str, int]


# Task Management Schemas (Comprehensive Pipeline Logs)
class StageResult(BaseModel):
    """Result from a single pipeline stage."""
    stage: str = Field(..., description="Stage name (agent name)")
    keyInput: dict[str, Any] = Field(..., description="Key input fields for the stage")
    keyOutput: dict[str, Any] = Field(..., description="Key output fields from the stage")
    duration: int = Field(..., description="Execution duration in milliseconds")
    creditsUsed: int = Field(..., description="Credits consumed by this stage")
    status: str = Field(..., description="Status: success, failed, skipped")


class PromptTransformation(BaseModel):
    """Prompt transformation at a specific stage."""
    stage: str = Field(..., description="Stage that transformed the prompt")
    prompt: str = Field(..., description="Prompt text at this stage")
    timestamp: datetime = Field(..., description="When this transformation occurred")


class PerformanceData(BaseModel):
    """Performance metrics for a task."""
    byStage: list[dict[str, Any]] = Field(..., description="Per-stage performance breakdown")
    bottlenecks: list[str] = Field(default_factory=list, description="Identified bottlenecks")
    totalDuration: int = Field(..., description="Total duration in milliseconds")


class EvaluationData(BaseModel):
    """Evaluation results for a task."""
    overallScore: Optional[float] = Field(None, description="Overall quality score (0-1)")
    objectiveScores: Optional[dict[str, float]] = Field(None, description="Objective metrics")
    subjectiveScores: Optional[dict[str, float]] = Field(None, description="Subjective ratings")
    reasoning: Optional[str] = Field(None, description="Evaluation reasoning")
    recommendations: list[str] = Field(default_factory=list, description="Improvement recommendations")


class LogCheckpoint(BaseModel):
    """Structured log checkpoint from pipeline execution."""
    event: str = Field(..., description="Log event name (e.g., pali.session.start)")
    level: str = Field(..., description="Log level: INFO, WARNING, ERROR, DEBUG")
    timestamp: datetime = Field(..., description="When the event occurred")
    fields: dict[str, Any] = Field(default_factory=dict, description="Event-specific fields")
    component: str = Field(..., description="Component name (e.g., pali, planner_v2)")


class TaskResponse(BaseModel):
    """Comprehensive task execution response."""
    id: str = Field(..., description="Task database ID")
    taskId: str = Field(..., description="External task identifier")
    shop: str = Field(..., description="Shop domain")
    originalPrompt: str = Field(..., description="User's original prompt")
    userRequest: Optional[dict[str, Any]] = Field(None, description="Complete user input")
    stages: list[dict[str, Any]] = Field(..., description="All 10 pipeline stages with I/O")
    promptJourney: list[dict[str, Any]] = Field(..., description="Prompt transformations")
    totalDuration: int = Field(..., description="Total execution time in ms")
    creditsCost: int = Field(..., description="Total credits used")
    performanceBreakdown: dict[str, Any] = Field(..., description="Performance metrics")
    evaluationResults: Optional[dict[str, Any]] = Field(None, description="Evaluation data")
    logCheckpoints: Optional[list[dict[str, Any]]] = Field(None, description="Structured log checkpoints")
    generatedImageUrl: Optional[str] = Field(None, description="Final generated image URL")
    mockupUrls: Optional[list[str]] = Field(None, description="Product mockup URLs")
    finalPrompt: Optional[str] = Field(None, description="Final prompt sent to Flux")
    status: str = Field(..., description="Task status: completed, failed, processing")
    errorMessage: Optional[str] = Field(None, description="Error details if failed")
    createdAt: datetime = Field(..., description="Task creation timestamp")
    completedAt: Optional[datetime] = Field(None, description="Task completion timestamp")
    updatedAt: datetime = Field(..., description="Last update timestamp")


class TaskListResponse(BaseModel):
    """Paginated list of tasks."""
    tasks: list[TaskResponse]
    total: int
    page: int
    pageSize: int
    hasMore: bool


class CreateTaskRequest(BaseModel):
    """Request to create a task (used internally by post-processing)."""
    taskId: str
    shop: str
    originalPrompt: str
    userRequest: Optional[dict[str, Any]] = None
    stages: list[dict[str, Any]]
    promptJourney: list[dict[str, Any]]
    totalDuration: int
    creditsCost: int
    performanceBreakdown: dict[str, Any]
    evaluationResults: Optional[dict[str, Any]] = None
    generatedImageUrl: Optional[str] = None
    mockupUrls: Optional[list[str]] = None
    finalPrompt: Optional[str] = None
    status: str = "completed"
    errorMessage: Optional[str] = None
    completedAt: Optional[datetime] = None
