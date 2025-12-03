"""
Planner Agent - Central orchestrator for AI image generation planning.

This agent is the DECISION MAKER that orchestrates the entire planning flow:

Key Responsibilities (from Swimlane Diagram):
1. Evaluate Task - Check if we have enough context
2. Knowledge Acquire - RAG (user history, art library, similar designs)
3. Decide prompt MODE from initial user input
4. Select dimensions using reasoning model
5. Call PromptComposerService to write the final prompt
6. Evaluate Prompt Quality with thresholds
7. Select image model (decision based on Model Info Service data)
8. Create structured AssemblyRequest for downstream services
9. Handle evaluation feedback for Fix Plan loop
10. Safety classification and review flags

Architecture:
- PromptTemplateService: Rule keeper (templates, dimensions, constraints)
- ModelInfoService: Data provider (model info, capabilities, constraints)
- Planner Agent: Decision maker (mode, dimensions, model, context)
- PromptComposerService: Writer (composes final prompt using LLM)

Documentation Reference: Section 5.2.2
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import logging
import asyncio
import re

from palet8_agents.core.agent import BaseAgent, AgentContext, AgentResult
from palet8_agents.core.config import get_config
from palet8_agents.tools.base import BaseTool

from palet8_agents.services.text_llm_service import TextLLMService
from palet8_agents.services.reasoning_service import ReasoningService, QualityScore
from palet8_agents.services.context_service import ContextService, Context
from palet8_agents.services.web_search_service import WebSearchService, WebSearchResponse
from palet8_agents.services.model_info_service import ModelInfoService, ModelSelection, ModelCapability
from palet8_agents.services.prompt_template_service import PromptTemplateService, PromptMode, Scenario
from palet8_agents.services.prompt_composer_service import (
    PromptComposerService,
    PromptDimensions as ComposerDimensions,
    ComposedPrompt,
)

logger = logging.getLogger(__name__)


# =============================================================================
# EXECUTION PHASES
# =============================================================================

class PlannerPhase(Enum):
    """Planner Agent execution phases."""
    INITIAL = "initial"           # First run - evaluate context, create plan
    FIX_PLAN = "fix_plan"         # After evaluation rejection - fix the plan
    CLARIFY = "clarify"           # After receiving clarification from Pali


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ContextCompleteness:
    """Result of context completeness evaluation."""
    score: float  # 0.0 to 1.0
    is_sufficient: bool
    missing_fields: List[str]
    clarifying_questions: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": self.score,
            "is_sufficient": self.is_sufficient,
            "missing_fields": self.missing_fields,
            "clarifying_questions": self.clarifying_questions,
            "metadata": self.metadata,
        }


@dataclass
class SafetyClassification:
    """Safety classification result."""
    is_safe: bool
    requires_review: bool
    risk_level: str  # "low", "medium", "high"
    categories: List[str]  # Detected risk categories
    flags: Dict[str, bool] = field(default_factory=dict)
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_safe": self.is_safe,
            "requires_review": self.requires_review,
            "risk_level": self.risk_level,
            "categories": self.categories,
            "flags": self.flags,
            "reason": self.reason,
        }


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


@dataclass
class GenerationParameters:
    """Image generation parameters."""
    width: int = 1024
    height: int = 1024
    steps: int = 30
    guidance_scale: float = 7.5
    seed: Optional[int] = None
    num_images: int = 1
    # Provider-specific settings (varies by model)
    provider_settings: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "width": self.width,
            "height": self.height,
            "steps": self.steps,
            "guidance_scale": self.guidance_scale,
            "seed": self.seed,
            "num_images": self.num_images,
        }
        if self.provider_settings:
            result["provider_settings"] = self.provider_settings
        return result


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
        return {
            "pipeline_type": self.pipeline_type,
            "pipeline_name": self.pipeline_name,
            "stage_1_model": self.stage_1_model,
            "stage_1_purpose": self.stage_1_purpose,
            "stage_2_model": self.stage_2_model,
            "stage_2_purpose": self.stage_2_purpose,
            "decision_rationale": self.decision_rationale,
        }


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


@dataclass
class EvaluationFeedback:
    """Feedback from Evaluator Agent for Fix Plan loop."""
    passed: bool
    overall_score: float
    issues: List[str]
    retry_suggestions: List[str]
    dimension_scores: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvaluationFeedback":
        return cls(
            passed=data.get("passed", False),
            overall_score=data.get("overall_score", 0.0),
            issues=data.get("issues", []),
            retry_suggestions=data.get("retry_suggestions", []),
            dimension_scores=data.get("dimension_scores", {}),
            metadata=data.get("metadata", {}),
        )


# =============================================================================
# SYSTEM PROMPT
# =============================================================================

PLANNER_SYSTEM_PROMPT = """You are the Planner for Palet8's image generation system.

YOUR ROLE
Transform creative briefs into complete, safe, high-quality generation plans. Handle refinements after users see results. You don't talk to users - you work with structured data from other agents.

CORE DECISIONS

1. CONTEXT CHECK
   Assess if you have enough to proceed:
   - If critical info is missing, return a small set of clarification points/questions for the user-facing agent (don't guess or invent user intent)
   - If sufficient, continue planning

2. COMPLEXITY LEVEL
   Decide how detailed the plan needs to be:
   - Simple: Quick concepts, minimal styling needed
   - Standard: Clear style direction, composition matters
   - Advanced: Technical specs, print constraints, production requirements

   Let the request's nature guide this.

3. PROMPT BUILDING
   Construct prompts that are:
   - Specific and unambiguous
   - Free of contradictions
   - Aligned with product constraints
   - Appropriately detailed for the complexity level

4. MODEL SELECTION
   Choose generation model and parameters based on:
   - Quality requirements
   - Any speed or latency needs signaled by the system
   - Technical constraints
   - Cost or efficiency considerations when available

EXECUTION PHASES

INITIAL
Full planning flow from creative brief.

FIX PLAN
Evaluator rejected the result. Apply feedback surgically - don't restart from scratch.
- Prefer minimal, targeted changes over completely rewriting a working plan.

EDIT REQUEST
User saw result and wants changes ("make it darker", "remove text", "blue background").
- Preserve what worked, modify only what's requested
- Prefer minimal changes when possible; use full regeneration only when necessary
- Treat major conceptual shifts as a new prompt with updated assumptions
- Avoid bouncing between similar plans; incorporate prior feedback so the same mistake is not repeated

CLARIFY
Received additional context from Pali. Incorporate and continue.

COLLABORATION
- Use feedback from the Evaluator to improve prompts and constraints
- Respect signals from the Safety agent; when content is problematic, propose safer alternatives

OUTPUT
Complete plan with prompts, parameters, model choice, and rationale."""


# =============================================================================
# PLANNER AGENT
# =============================================================================

class PlannerAgent(BaseAgent):
    """
    Central orchestrator for AI image generation planning.

    Responsibilities (from Swimlane Diagram):
    1. Evaluate Task - "Enough Context?" decision
    2. Knowledge Acquire - RAG (user history, art library, similar designs)
    3. Decide prompt MODE from initial user input
    4. Select dimensions using reasoning model
    5. Call PromptComposerService to write the final prompt
    6. Evaluate Prompt Quality with thresholds
    7. Select image model (decision based on Model Info Service data)
    8. Create structured AssemblyRequest
    9. Handle evaluation feedback for Fix Plan loop
    10. Safety classification and review flags

    Execution Flow:
    - Phase INITIAL: Full planning flow
    - Phase FIX_PLAN: Revise plan based on evaluation feedback
    - Phase CLARIFY: Continue after receiving clarification
    """

    # Context completeness thresholds
    REQUIRED_FIELDS = ["subject"]
    IMPORTANT_FIELDS = ["style", "aesthetic", "colors", "product_type"]
    OPTIONAL_FIELDS = ["mood", "composition", "background", "lighting"]

    # Field weights for completeness scoring
    COMPLETENESS_WEIGHTS = {
        "subject": 0.40,
        "style": 0.15,
        "aesthetic": 0.15,
        "colors": 0.10,
        "product_type": 0.10,
        "mood": 0.05,
        "composition": 0.05,
    }

    # Safety risk keywords (quick check - full safety via SafetyAgent)
    # High-risk keywords from safety_config.yaml: nsfw.critical_keywords + extreme_block_keywords
    SAFETY_RISK_KEYWORDS = {
        "critical": [  # Always block - from safety_config.yaml
            "nude", "naked", "xxx", "porn", "pornograph", "nsfw", "sexual", "erotic", "hentai",
            "child abuse", "terror attack", "mass shooting", "torture of children",
            "nazi propaganda", "genocide", "ethnic cleansing", "white supremacy",
            "child exploitation", "human trafficking", "csam",
        ],
        "high": ["explicit", "violence", "gore", "hate"],
        "medium": ["sexy", "provocative", "weapon", "blood", "scary"],
        "low": ["adult", "mature", "dark", "aggressive"],
    }

    def __init__(
        self,
        tools: Optional[List[BaseTool]] = None,
        text_service: Optional[TextLLMService] = None,
        reasoning_service: Optional[ReasoningService] = None,
        context_service: Optional[ContextService] = None,
        model_info_service: Optional[ModelInfoService] = None,
        prompt_template_service: Optional[PromptTemplateService] = None,
        prompt_composer_service: Optional[PromptComposerService] = None,
        web_search_service: Optional[WebSearchService] = None,
    ):
        """Initialize the Planner Agent."""
        super().__init__(
            name="planner",
            description="Central orchestrator for planning, RAG, and model selection",
            tools=tools,
        )

        self._text_service = text_service
        self._reasoning_service = reasoning_service
        self._context_service = context_service
        self._model_info_service = model_info_service
        self._prompt_template_service = prompt_template_service
        self._prompt_composer_service = prompt_composer_service
        self._web_search_service = web_search_service

        self._owns_services = {
            "text": text_service is None,
            "reasoning": reasoning_service is None,
            "context": context_service is None,
            "model_info": model_info_service is None,
            "prompt_template": prompt_template_service is None,
            "prompt_composer": prompt_composer_service is None,
            "web_search": web_search_service is None,
        }

        # System prompt
        self.system_prompt = PLANNER_SYSTEM_PROMPT

        # Model profile (from agent_routing_policy.yaml)
        self.model_profile = "planner"

        # Configuration thresholds
        self.config = get_config()
        self.min_context_completeness = 0.5  # Minimum completeness to proceed
        self.min_prompt_quality = 0.45       # From evaluation_config.yaml (acceptance_threshold)
        self.max_prompt_revisions = 3        # Max prompt revision attempts
        self.max_fix_iterations = 3          # From evaluation_config.yaml (max_retries)

    # =========================================================================
    # SERVICE GETTERS
    # =========================================================================

    async def _get_text_service(self) -> TextLLMService:
        if self._text_service is None:
            self._text_service = TextLLMService(default_profile_name=self.model_profile)
        return self._text_service

    async def _get_reasoning_service(self) -> ReasoningService:
        if self._reasoning_service is None:
            self._reasoning_service = ReasoningService()
        return self._reasoning_service

    async def _get_context_service(self) -> ContextService:
        if self._context_service is None:
            self._context_service = ContextService()
        return self._context_service

    async def _get_model_info_service(self) -> ModelInfoService:
        if self._model_info_service is None:
            self._model_info_service = ModelInfoService()
        return self._model_info_service

    async def _get_prompt_template_service(self) -> PromptTemplateService:
        if self._prompt_template_service is None:
            self._prompt_template_service = PromptTemplateService()
        return self._prompt_template_service

    async def _get_prompt_composer_service(self) -> PromptComposerService:
        if self._prompt_composer_service is None:
            self._prompt_composer_service = PromptComposerService(
                template_service=await self._get_prompt_template_service(),
                text_llm_service=await self._get_text_service(),
            )
        return self._prompt_composer_service

    async def _get_web_search_service(self) -> WebSearchService:
        if self._web_search_service is None:
            self._web_search_service = WebSearchService()
        return self._web_search_service

    async def close(self) -> None:
        """Close resources."""
        if self._text_service and self._owns_services["text"]:
            await self._text_service.close()
        if self._reasoning_service and self._owns_services["reasoning"]:
            await self._reasoning_service.close()
        if self._context_service and self._owns_services["context"]:
            await self._context_service.close()
        if self._prompt_template_service and self._owns_services["prompt_template"]:
            await self._prompt_template_service.close()
        if self._prompt_composer_service and self._owns_services["prompt_composer"]:
            await self._prompt_composer_service.close()
        if self._web_search_service and self._owns_services["web_search"]:
            await self._web_search_service.close()

    # =========================================================================
    # MAIN EXECUTION
    # =========================================================================

    async def run(
        self,
        context: AgentContext,
        user_input: Optional[str] = None,
        phase: str = "initial",
        evaluation_feedback: Optional[Dict[str, Any]] = None,
    ) -> AgentResult:
        """
        Execute the Planner Agent's task.

        Args:
            context: Shared execution context
            user_input: Optional additional user input
            phase: Execution phase ("initial", "fix_plan", "clarify")
            evaluation_feedback: Feedback from Evaluator for fix_plan phase

        Returns:
            AgentResult with:
            - AssemblyRequest for successful planning
            - Clarification request if context insufficient
            - Error details if planning failed
        """
        self._start_execution()

        try:
            requirements = context.requirements or {}

            # Route to appropriate phase handler
            if phase == "fix_plan" and evaluation_feedback:
                return await self._handle_fix_plan(context, evaluation_feedback)
            elif phase == "clarify":
                return await self._handle_clarify(context, user_input)
            else:
                return await self._handle_initial(context)

        except Exception as e:
            logger.error(f"Planner Agent error: {e}", exc_info=True)
            return self._create_result(
                success=False,
                data=None,
                error=f"Planning failed: {e}",
                error_code="PLANNING_ERROR",
            )

    async def _handle_initial(self, context: AgentContext) -> AgentResult:
        """Handle initial planning phase."""
        requirements = context.requirements or {}

        # =====================================================================
        # STEP 1: Evaluate Context Completeness ("Enough Context?" decision)
        # =====================================================================
        completeness = await self._evaluate_context_completeness(requirements)

        if not completeness.is_sufficient:
            logger.info(f"Context incomplete (score={completeness.score:.2f}), requesting clarification")
            return self._create_result(
                success=True,
                data={
                    "action": "needs_clarification",
                    "completeness": completeness.to_dict(),
                    "questions": completeness.clarifying_questions,
                    "missing_fields": completeness.missing_fields,
                },
                next_agent="pali",  # Return to Pali for clarification
            )

        # =====================================================================
        # STEP 2: Safety Classification (early check)
        # =====================================================================
        safety = await self._classify_safety(requirements)

        if not safety.is_safe:
            logger.warning(f"Safety check failed: {safety.reason}")
            return self._create_result(
                success=False,
                data={
                    "action": "blocked",
                    "safety": safety.to_dict(),
                },
                error=f"Content blocked: {safety.reason}",
                error_code="SAFETY_BLOCKED",
            )

        # =====================================================================
        # STEP 3: Decide Prompt MODE
        # =====================================================================
        mode = await self._decide_mode(requirements)
        logger.info(f"Selected prompt mode: {mode.value}")

        # =====================================================================
        # STEP 4: Get Planning Context from Template Service
        # =====================================================================
        prompt_template = await self._get_prompt_template_service()
        product = requirements.get("product_type", "mens_tshirt")
        print_method = requirements.get("print_method")

        planning_context = prompt_template.get_planning_context(
            product=product,
            print_method=print_method,
        )

        # =====================================================================
        # STEP 5: CONCURRENT - Knowledge Acquire (RAG) + Model Info
        # =====================================================================
        rag_task = self._acquire_knowledge(context, requirements)
        model_info_task = self._get_model_selection_context(requirements)

        rag_context, model_info_context = await asyncio.gather(
            rag_task,
            model_info_task,
        )

        # =====================================================================
        # STEP 6: Select Dimensions
        # =====================================================================
        dimensions = await self._select_dimensions(
            mode=mode,
            requirements=requirements,
            planning_context=planning_context,
            rag_context=rag_context,
        )

        # =====================================================================
        # STEP 7: Compose Prompt via PromptComposerService
        # =====================================================================
        composed = await self._compose_prompt(
            mode=mode,
            dimensions=dimensions,
            rag_context=rag_context,
            planning_context=planning_context,
            print_method=print_method,
            product=product,
        )

        # =====================================================================
        # STEP 8: Evaluate Prompt Quality
        # =====================================================================
        prompt = composed.positive_prompt
        negative_prompt = composed.negative_prompt

        quality_score, revision_count = await self._evaluate_and_revise_prompt(
            prompt=prompt,
            requirements=requirements,
        )

        if revision_count > 0:
            prompt = quality_score.revised_prompt if hasattr(quality_score, 'revised_prompt') else prompt

        # =====================================================================
        # STEP 9: Decide Pipeline (Single vs Dual Model)
        # =====================================================================
        pipeline_config = await self._decide_pipeline(
            requirements=requirements,
            mode=mode,
        )

        # =====================================================================
        # STEP 10: Select Image Model (Planner makes decision)
        # =====================================================================
        # For dual pipeline, stage_1 model is preset; for single, select here
        if pipeline_config.pipeline_type == "dual":
            model_id = pipeline_config.stage_1_model
            model_rationale = f"Dual pipeline stage 1: {pipeline_config.stage_1_purpose}"
            model_alternatives = []
            # Get model specs from model_info_context
            model_specs = model_info_context.get("model_registry", {}).get(model_id, {})
        else:
            model_id, model_rationale, model_alternatives, model_specs = await self._select_model(
                requirements=requirements,
                mode=mode,
                model_info_context=model_info_context,
            )
            # Update pipeline config with selected model
            pipeline_config.stage_1_model = model_id

        # =====================================================================
        # STEP 11: Build AssemblyRequest
        # =====================================================================
        assembly_request = await self._build_assembly_request(
            context=context,
            prompt=prompt,
            negative_prompt=negative_prompt,
            mode=mode,
            dimensions=dimensions,
            pipeline=pipeline_config,
            model_id=model_id,
            model_rationale=model_rationale,
            model_alternatives=model_alternatives,
            model_specs=model_specs,
            quality_score=quality_score,
            safety=safety,
            rag_context=rag_context,
            requirements=requirements,
            revision_count=revision_count,
        )

        # Update context
        context.plan = assembly_request.to_dict()
        context.image_model = model_id

        # Return result
        return self._create_result(
            success=True,
            data={
                "action": "proceed",
                "assembly_request": assembly_request.to_dict(),
            },
            next_agent="evaluator",  # Proceed to create eval plan
        )

    async def _handle_fix_plan(
        self,
        context: AgentContext,
        evaluation_feedback: Dict[str, Any],
    ) -> AgentResult:
        """Handle fix plan phase after evaluation rejection."""
        feedback = EvaluationFeedback.from_dict(evaluation_feedback)
        requirements = context.requirements or {}

        # Get current plan
        current_plan = context.plan or {}
        revision_count = current_plan.get("revision_count", 0) + 1

        if revision_count > self.max_fix_iterations:
            logger.warning(f"Max fix iterations ({self.max_fix_iterations}) reached")
            return self._create_result(
                success=False,
                data={
                    "action": "max_retries_exceeded",
                    "feedback": feedback.__dict__,
                    "revision_count": revision_count,
                },
                error="Maximum revision attempts exceeded",
                error_code="MAX_RETRIES_EXCEEDED",
            )

        logger.info(f"Fix plan iteration {revision_count}, issues: {feedback.issues}")

        # Apply retry suggestions to requirements
        revised_requirements = await self._apply_feedback_to_requirements(
            requirements=requirements,
            feedback=feedback,
        )

        # Re-run planning with revised requirements
        context.requirements = revised_requirements
        result = await self._handle_initial(context)

        # Update revision count in result
        if result.success and result.data and "assembly_request" in result.data:
            result.data["assembly_request"]["revision_count"] = revision_count
            result.data["assembly_request"]["metadata"]["fix_feedback"] = feedback.issues

        return result

    async def _handle_clarify(
        self,
        context: AgentContext,
        user_input: Optional[str],
    ) -> AgentResult:
        """Handle clarification phase after Pali asks questions."""
        # User input should be incorporated into requirements by Pali
        # Just re-run initial planning
        return await self._handle_initial(context)

    # =========================================================================
    # CONTEXT COMPLETENESS (Swimlane: "Enough Context?")
    # =========================================================================
    #
    # IMPORTANT: Planner is the SOLE decision maker for context completeness.
    # Pali only checks if user has a subject (minimal gate), then passes here.
    # Planner does the thorough evaluation with weighted fields.
    #
    # If context is insufficient, Planner returns to Pali with specific
    # clarifying questions. Pali asks user, then passes back to Planner.
    # =========================================================================

    async def _evaluate_context_completeness(
        self,
        requirements: Dict[str, Any],
    ) -> ContextCompleteness:
        """
        Evaluate if we have enough context to proceed.

        Returns ContextCompleteness with:
        - score: 0.0 to 1.0
        - is_sufficient: True if we can proceed
        - missing_fields: What's missing
        - clarifying_questions: Questions to ask user
        """
        score = 0.0
        missing_required = []
        missing_important = []
        questions = []

        # Check required fields
        for field in self.REQUIRED_FIELDS:
            if requirements.get(field):
                score += self.COMPLETENESS_WEIGHTS.get(field, 0.1)
            else:
                missing_required.append(field)
                questions.append(self._generate_question_for_field(field))

        # Check important fields
        for field in self.IMPORTANT_FIELDS:
            if requirements.get(field):
                score += self.COMPLETENESS_WEIGHTS.get(field, 0.05)
            else:
                missing_important.append(field)

        # Check optional fields (don't add to missing, just boost score)
        for field in self.OPTIONAL_FIELDS:
            if requirements.get(field):
                score += self.COMPLETENESS_WEIGHTS.get(field, 0.02)

        # Determine if sufficient
        is_sufficient = (
            len(missing_required) == 0 and
            score >= self.min_context_completeness
        )

        # If missing important fields, add questions
        if not is_sufficient and len(questions) < 3:
            for field in missing_important[:3 - len(questions)]:
                questions.append(self._generate_question_for_field(field))

        return ContextCompleteness(
            score=min(1.0, score),
            is_sufficient=is_sufficient,
            missing_fields=missing_required + missing_important,
            clarifying_questions=questions[:3],
            metadata={
                "required_missing": missing_required,
                "important_missing": missing_important,
            },
        )

    def _generate_question_for_field(self, field: str) -> str:
        """Generate a clarifying question for a missing field."""
        questions = {
            "subject": "What would you like the image to show? Please describe the main subject.",
            "style": "What style are you looking for? (e.g., realistic, cartoon, minimalist)",
            "aesthetic": "What aesthetic or visual style do you prefer?",
            "colors": "Do you have any color preferences for this design?",
            "product_type": "What product is this design for? (e.g., t-shirt, poster, phone case)",
            "mood": "What mood or feeling should the image convey?",
            "composition": "How would you like the elements arranged in the image?",
            "background": "What kind of background would you prefer?",
        }
        return questions.get(field, f"Could you provide more details about the {field}?")

    # =========================================================================
    # SAFETY CLASSIFICATION
    # =========================================================================

    async def _classify_safety(
        self,
        requirements: Dict[str, Any],
    ) -> SafetyClassification:
        """
        Classify content safety and determine if review is needed.

        Returns SafetyClassification with risk level and flags.
        """
        # Combine all text content
        text_content = " ".join([
            str(requirements.get("subject", "")),
            str(requirements.get("style", "")),
            str(requirements.get("description", "")),
        ]).lower()

        detected_categories = []
        risk_level = "low"

        # Check for risk keywords
        for level, keywords in self.SAFETY_RISK_KEYWORDS.items():
            for keyword in keywords:
                if keyword in text_content:
                    detected_categories.append(keyword)
                    if level == "high":
                        risk_level = "high"
                    elif level == "medium" and risk_level != "high":
                        risk_level = "medium"

        # Determine safety and review flags
        is_safe = risk_level != "high"
        requires_review = risk_level in ["medium", "high"]

        reason = ""
        if not is_safe:
            reason = f"Content contains prohibited elements: {', '.join(detected_categories)}"
        elif requires_review:
            reason = f"Content requires review: {', '.join(detected_categories)}"

        return SafetyClassification(
            is_safe=is_safe,
            requires_review=requires_review,
            risk_level=risk_level,
            categories=detected_categories,
            flags={
                "nsfw_check_required": "nsfw" in text_content or "nude" in text_content,
                "violence_check_required": any(w in text_content for w in ["violence", "gore", "blood"]),
                "ip_check_required": any(w in text_content for w in ["logo", "brand", "trademark", "disney", "marvel"]),
            },
            reason=reason,
        )

    # =========================================================================
    # MODE DECISION
    # =========================================================================

    async def _decide_mode(self, requirements: Dict[str, Any]) -> PromptMode:
        """
        Decide prompt mode from initial user input.

        Priority:
        1. Technical fields present → COMPLEX
        2. Style/composition fields present → STANDARD
        3. Simple concept only → RELAX
        """
        # Check for COMPLEX indicators
        complex_indicators = [
            "print_method", "color_separation", "bleed", "safe_zone",
            "dpi", "thread_colors", "stitch_types", "color_count", "max_colors",
        ]

        for indicator in complex_indicators:
            if requirements.get(indicator):
                logger.info(f"COMPLEX mode: found technical field '{indicator}'")
                return PromptMode.COMPLEX

        # Print methods that require technical specs
        if requirements.get("print_method") in ["screen_print", "embroidery"]:
            logger.info("COMPLEX mode: print_method requires technical specs")
            return PromptMode.COMPLEX

        # Check for STANDARD indicators
        standard_indicators = [
            "composition", "background", "lighting", "mood", "style",
            "reference_image", "colors", "color_palette",
        ]

        for indicator in standard_indicators:
            if requirements.get(indicator):
                logger.info(f"STANDARD mode: found style field '{indicator}'")
                return PromptMode.STANDARD

        # Default to RELAX
        logger.info("RELAX mode: simple request with minimal constraints")
        return PromptMode.RELAX

    # =========================================================================
    # KNOWLEDGE ACQUIRE (RAG + Conditional Web Search)
    # =========================================================================

    # Threshold for determining if RAG context is sufficient
    MIN_CONTEXT_ITEMS = 2  # Minimum combined references/prompts before triggering search

    async def _acquire_knowledge(
        self,
        context: AgentContext,
        requirements: Dict[str, Any],
    ) -> Context:
        """
        Acquire knowledge via RAG, with conditional web search if insufficient.

        Flow:
        1. First get context from RAG (user history, art library, similar designs)
        2. Check if context is sufficient for prompt composition
        3. If NOT sufficient → supplement with web search
        4. Return combined context

        Note: Web search is only triggered when RAG provides insufficient context,
        not when info is already sufficient.
        """
        # Step 1: Get RAG context
        rag_context = await self._get_rag_context(context, requirements)

        # Step 2: Check if RAG context is sufficient
        if self._is_context_sufficient(rag_context, requirements):
            logger.info("RAG context is sufficient, skipping web search")
            return rag_context

        # Step 3: RAG context is insufficient - supplement with web search
        logger.info("RAG context insufficient, supplementing with web search")
        search_context = await self._get_search_context(requirements)

        # Step 4: Merge web search results into RAG context
        return self._merge_contexts(rag_context, search_context)

    async def _get_rag_context(
        self,
        context: AgentContext,
        requirements: Dict[str, Any],
    ) -> Context:
        """Get context from RAG (ContextService)."""
        try:
            context_service = await self._get_context_service()
            return await context_service.build_context(
                user_id=context.user_id,
                requirements=requirements,
                query=requirements.get("subject", ""),
            )
        except Exception as e:
            logger.warning(f"RAG context retrieval failed: {e}")
            return Context(
                user_history=[],
                art_references=[],
                similar_prompts=[],
            )

    def _is_context_sufficient(
        self,
        rag_context: Context,
        requirements: Dict[str, Any],
    ) -> bool:
        """
        Determine if RAG context provides enough info for prompt composition.

        Context is considered sufficient when:
        - We have art references OR similar prompts to draw from
        - OR the request is a simple/well-specified one
        """
        # Count available context items
        art_refs_count = len(rag_context.art_references or [])
        similar_prompts_count = len(rag_context.similar_prompts or [])
        total_refs = art_refs_count + similar_prompts_count

        # If we have enough references, context is sufficient
        if total_refs >= self.MIN_CONTEXT_ITEMS:
            return True

        # If user provided detailed requirements, may not need more context
        detail_fields = ["style", "aesthetic", "composition", "background", "mood"]
        provided_details = sum(1 for f in detail_fields if requirements.get(f))

        # Well-specified requests (3+ detail fields) don't need extra search
        if provided_details >= 3:
            return True

        # Otherwise, we need more context
        return False

    async def _get_search_context(
        self,
        requirements: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Get additional context from web search.

        Searches for:
        - Style references
        - Trend information
        - Design inspiration
        """
        try:
            search_service = await self._get_web_search_service()

            # Build search query from requirements
            subject = requirements.get("subject", "")
            style = requirements.get("style", "")
            product_type = requirements.get("product_type", "")

            query_parts = [subject]
            if style:
                query_parts.append(style)
            if product_type:
                query_parts.append(f"{product_type} design")

            query = " ".join(query_parts)
            if not query.strip():
                return {"search_results": [], "search_answer": None}

            # Add design/art context to query
            search_query = f"{query} design style reference"

            response = await search_service.search(
                query=search_query,
                max_results=3,
                include_answer=True,
            )

            return {
                "search_results": [r.to_dict() for r in response.results],
                "search_answer": response.answer,
                "search_provider": response.provider,
            }

        except Exception as e:
            logger.warning(f"Web search failed: {e}")
            return {"search_results": [], "search_answer": None}

    def _merge_contexts(
        self,
        rag_context: Context,
        search_context: Dict[str, Any],
    ) -> Context:
        """
        Merge web search results into RAG context.

        Adds search results as additional references while preserving
        the original RAG context structure.
        """
        # Create web-sourced references from search results
        web_references = []
        for result in search_context.get("search_results", []):
            web_references.append({
                "source": "web_search",
                "title": result.get("title", ""),
                "url": result.get("url", ""),
                "snippet": result.get("snippet", ""),
            })

        # Add search answer as a reference if available
        if search_context.get("search_answer"):
            web_references.append({
                "source": "web_search_answer",
                "content": search_context["search_answer"],
            })

        # Merge into existing art_references
        merged_art_refs = list(rag_context.art_references or [])
        merged_art_refs.extend(web_references)

        # Return updated context
        return Context(
            user_history=rag_context.user_history,
            art_references=merged_art_refs,
            similar_prompts=rag_context.similar_prompts,
            metadata={
                **(getattr(rag_context, 'metadata', None) or {}),
                "web_search_used": len(web_references) > 0,
                "search_provider": search_context.get("search_provider"),
            },
        )

    # =========================================================================
    # MODEL SELECTION (Planner decides based on Model Info data)
    # =========================================================================

    async def _get_model_selection_context(
        self,
        requirements: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Get model info context for selection decision."""
        try:
            model_info_service = await self._get_model_info_service()
            return model_info_service.get_selection_context(
                product_type=requirements.get("product_type"),
                requirements=requirements,
            )
        except Exception as e:
            logger.warning(f"Failed to get model info context: {e}")
            return {"available_models": [], "model_count": 0}

    async def _select_model(
        self,
        requirements: Dict[str, Any],
        mode: PromptMode,
        model_info_context: Dict[str, Any],
    ) -> Tuple[str, str, List[str], Dict[str, Any]]:
        """
        Select optimal image model (Planner makes the decision).

        Returns:
            Tuple of (model_id, rationale, alternatives, model_specs)
        """
        model_info_service = await self._get_model_info_service()

        # Get requirements
        needs_speed = requirements.get("priority") == "fast"
        needs_quality = requirements.get("quality_level") == "premium"
        needs_reference = requirements.get("reference_image") is not None
        max_cost = requirements.get("max_cost_per_image")

        # Score models
        compatible_models = []

        for model_data in model_info_context.get("available_models", []):
            model_id = model_data.get("model_id")
            compat = model_info_context.get("compatibility_results", {}).get(model_id, {})

            if not compat.get("compatible", True):
                continue

            score = compat.get("score", 0.5)

            # Adjust score based on requirements
            if needs_speed and "fast" in str(model_data.get("capabilities", [])):
                score += 0.1
            if needs_quality:
                score += model_data.get("quality_score", 0.5) * 0.2
            if mode == PromptMode.COMPLEX:
                # Prefer higher quality for complex modes
                score += model_data.get("quality_score", 0.5) * 0.1

            compatible_models.append((model_id, score, model_data))

        # Sort by score
        compatible_models.sort(key=lambda x: x[1], reverse=True)

        if not compatible_models:
            # Fallback to default
            default_model = self.config.image_models.primary or "flux-1-kontext-pro"
            return default_model, "Default model (no compatible models found)", [], {}

        best_model_id, best_score, best_model_data = compatible_models[0]
        alternatives = [m[0] for m in compatible_models[1:4]]

        # Generate rationale
        rationale_parts = [f"Selected {best_model_data.get('display_name', best_model_id)}"]
        if needs_quality:
            rationale_parts.append("for premium quality")
        if needs_speed:
            rationale_parts.append("for fast generation")
        if mode == PromptMode.COMPLEX:
            rationale_parts.append("suitable for complex prompts")

        rationale = " ".join(rationale_parts)

        # Extract model specs for provider settings
        model_specs = {
            "air_id": best_model_data.get("air_id"),
            "provider": best_model_data.get("provider"),
            "specs": best_model_data.get("specs", {}),
            "provider_params": best_model_data.get("provider_params", []),
            "workflows": best_model_data.get("workflows", []),
            "cost": best_model_data.get("cost", {}),
        }

        return best_model_id, rationale, alternatives, model_specs

    # =========================================================================
    # PIPELINE DECISION (Single vs Dual Model)
    # =========================================================================
    #
    # Planner decides based on TASK SCOPE, not mode:
    # - Text accuracy needed? → Dual (generator + text editor)
    # - Character refinement? → Dual (generator + editor)
    # - Complex multi-element? → Dual
    # - Simple/quick request? → Single
    #
    # User credit will factor in later (cost consideration)
    # =========================================================================

    # Dual pipeline triggers from image_models_config.yaml
    DUAL_PIPELINE_TRIGGERS = {
        "text_in_image": ["text", "typography", "lettering", "font", "words", "title", "headline", "quote"],
        "character_refinement": ["character edit", "face fix", "expression change", "pose adjust"],
        "multi_element": ["multiple subjects", "complex composition", "layered design"],
        "production_quality": ["print-ready", "production", "high accuracy", "4k", "poster"],
    }

    # Pipeline mapping from config
    PIPELINE_MAPPING = {
        "creative_with_text": "creative_art",       # Midjourney → Nano Banana 2 Pro
        "concept_art_refined": "creative_art",
        "product_photo_edited": "photorealistic",   # Imagen 4 Ultra → Nano Banana 2 Pro
        "portrait_with_adjustments": "photorealistic",
        "marketing_poster": "layout_poster",        # FLUX.2 Flex → Qwen Image Edit
        "infographic": "layout_poster",
        "banner_design": "layout_poster",
    }

    # Dual pipeline configurations
    DUAL_PIPELINES = {
        "creative_art": {
            "name": "High-Creative Art Pipeline",
            "stage_1_model": "midjourney-v7",
            "stage_1_purpose": "Generate creative, non-realistic composition",
            "stage_2_model": "nano-banana-2-pro",
            "stage_2_purpose": "Refine characters, add/correct text, adjust elements",
        },
        "photorealistic": {
            "name": "Photorealistic Pipeline",
            "stage_1_model": "imagen-4-ultra",
            "stage_1_purpose": "Generate photorealistic base with exceptional detail",
            "stage_2_model": "nano-banana-2-pro",
            "stage_2_purpose": "Character edits, text overlays, stylistic adjustments",
        },
        "layout_poster": {
            "name": "Layout & Poster Design Pipeline",
            "stage_1_model": "flux-2-flex",
            "stage_1_purpose": "Generate layout with accurate text placement",
            "stage_2_model": "qwen-image-edit",
            "stage_2_purpose": "Targeted edits, text correction, color adjustments",
        },
    }

    async def _decide_pipeline(
        self,
        requirements: Dict[str, Any],
        mode: PromptMode,
    ) -> PipelineConfig:
        """
        Decide whether to use single or dual pipeline based on task scope.

        Decision factors:
        - Text accuracy needed → Dual
        - Character refinement → Dual
        - Multi-element composition → Dual
        - Simple request → Single

        Args:
            requirements: User requirements
            mode: Prompt mode (for context, not decision)

        Returns:
            PipelineConfig with pipeline type and model info
        """
        # Combine all text content for analysis
        text_content = " ".join([
            str(requirements.get("subject", "")),
            str(requirements.get("prompt", "")),
            str(requirements.get("description", "")),
            str(requirements.get("style", "")),
        ]).lower()

        # Check triggers for dual pipeline
        triggers_found = []
        pipeline_intent = None

        for trigger_type, keywords in self.DUAL_PIPELINE_TRIGGERS.items():
            for keyword in keywords:
                if keyword in text_content:
                    triggers_found.append((trigger_type, keyword))
                    break

        # Determine pipeline intent
        has_text_need = any(t[0] == "text_in_image" for t in triggers_found)
        has_character_need = any(t[0] == "character_refinement" for t in triggers_found)
        has_multi_element = any(t[0] == "multi_element" for t in triggers_found)
        has_production_need = any(t[0] == "production_quality" for t in triggers_found)

        # Determine intent and pipeline
        style = requirements.get("style", "").lower()
        is_photorealistic = any(w in style for w in ["photo", "realistic", "product"])
        is_layout = any(w in text_content for w in ["poster", "banner", "infographic", "layout"])

        if has_text_need or has_production_need:
            if is_layout:
                pipeline_intent = "layout_poster"
            elif is_photorealistic:
                pipeline_intent = "photorealistic"
            else:
                pipeline_intent = "creative_art"
        elif has_character_need or has_multi_element:
            if is_photorealistic:
                pipeline_intent = "photorealistic"
            else:
                pipeline_intent = "creative_art"

        # Build pipeline config
        if pipeline_intent and pipeline_intent in self.DUAL_PIPELINES:
            pipeline_def = self.DUAL_PIPELINES[pipeline_intent]
            rationale = f"Dual pipeline selected: {', '.join([f'{t[0]}:{t[1]}' for t in triggers_found])}"
            logger.info(f"Pipeline decision: DUAL ({pipeline_intent}) - {rationale}")

            return PipelineConfig(
                pipeline_type="dual",
                pipeline_name=pipeline_intent,
                stage_1_model=pipeline_def["stage_1_model"],
                stage_1_purpose=pipeline_def["stage_1_purpose"],
                stage_2_model=pipeline_def["stage_2_model"],
                stage_2_purpose=pipeline_def["stage_2_purpose"],
                decision_rationale=rationale,
            )
        else:
            rationale = "Single pipeline: no dual triggers detected"
            logger.info(f"Pipeline decision: SINGLE - {rationale}")

            return PipelineConfig(
                pipeline_type="single",
                pipeline_name=None,
                stage_1_model="",  # Will be filled by _select_model
                stage_1_purpose="Generate final image",
                decision_rationale=rationale,
            )

    # =========================================================================
    # DIMENSION SELECTION
    # =========================================================================

    async def _select_dimensions(
        self,
        mode: PromptMode,
        requirements: Dict[str, Any],
        planning_context: Dict[str, Any],
        rag_context: Context,
    ) -> PromptDimensions:
        """Select and fill dimensions using reasoning model."""
        dimensions = PromptDimensions()

        # Get mode rules
        mode_rules = planning_context.get("mode_rules", {}).get(mode.value, {})
        required_dims = set(mode_rules.get("required", []))

        # Map requirements to dimensions
        dimension_mapping = {
            "subject": "subject", "style": "aesthetic", "aesthetic": "aesthetic",
            "colors": "color", "color_palette": "color", "color": "color",
            "composition": "composition", "layout": "composition",
            "background": "background", "lighting": "lighting", "texture": "texture",
            "detail": "detail_level", "mood": "mood", "emotion": "mood",
            "expression": "expression", "pose": "pose",
            "art_style": "art_movement", "reference_style": "reference_style",
        }

        # Fill from requirements
        for req_key, dim_key in dimension_mapping.items():
            if req_key in requirements and requirements[req_key]:
                value = requirements[req_key]
                if isinstance(value, list):
                    value = ", ".join(str(v) for v in value)
                setattr(dimensions, dim_key, str(value))

        # Fill missing required dimensions using LLM
        missing_required = [
            dim for dim in required_dims
            if dim != "technical" and getattr(dimensions, dim, None) is None
        ]

        if missing_required:
            await self._fill_missing_dimensions(
                dimensions=dimensions,
                missing=missing_required,
                planning_context=planning_context,
            )

        # Build technical specs for COMPLEX mode
        if mode == PromptMode.COMPLEX:
            dimensions.technical = await self._build_technical_specs(
                requirements=requirements,
                planning_context=planning_context,
            )

        return dimensions

    async def _fill_missing_dimensions(
        self,
        dimensions: PromptDimensions,
        missing: List[str],
        planning_context: Dict[str, Any],
    ) -> None:
        """Fill missing dimensions using LLM reasoning."""
        text_service = await self._get_text_service()
        dim_definitions = planning_context.get("dimensions", {})

        system_prompt = """You are an expert at describing images for AI generation.
Given a subject and context, suggest appropriate values for missing dimensions.
Return values in format: dimension: value (one per line)."""

        user_prompt = f"""Subject: {dimensions.subject or 'unknown'}
Current dimensions: {dimensions.to_dict()}
Missing dimensions to fill: {missing}
Dimension definitions: {dim_definitions}

Suggest values for the missing dimensions:"""

        try:
            result = await text_service.generate_text(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=0.5,
            )

            for line in result.content.strip().split("\n"):
                if ":" in line:
                    key, value = line.split(":", 1)
                    key = key.strip().lower()
                    value = value.strip()
                    if hasattr(dimensions, key) and value:
                        setattr(dimensions, key, value)

        except Exception as e:
            logger.warning(f"Failed to fill missing dimensions: {e}")

    async def _build_technical_specs(
        self,
        requirements: Dict[str, Any],
        planning_context: Dict[str, Any],
    ) -> Dict[str, str]:
        """Build technical specifications for COMPLEX mode."""
        technical = {}
        template = planning_context.get("technical_template", {})
        product_spec = planning_context.get("product_spec", {})

        # Apply template
        if template:
            for key, value in template.items():
                if isinstance(value, str):
                    technical[key] = value

        # Add product specs
        print_area = product_spec.get("print_area", {})
        if print_area.get("width") and print_area.get("height"):
            technical["size"] = f"{print_area['width']}x{print_area['height']} inch"
            technical["dpi"] = f"{print_area.get('dpi', 300)} DPI"

        # Override with explicit requirements
        for key in ["dpi", "color_separation", "bleed", "safe_zone"]:
            if requirements.get(key):
                technical[key] = str(requirements[key])

        return technical

    # =========================================================================
    # PROMPT COMPOSITION
    # =========================================================================

    async def _compose_prompt(
        self,
        mode: PromptMode,
        dimensions: PromptDimensions,
        rag_context: Context,
        planning_context: Dict[str, Any],
        print_method: Optional[str],
        product: str,
    ) -> ComposedPrompt:
        """Compose prompt via PromptComposerService."""
        composer_context = {
            "similar_designs": [
                p.to_dict() if hasattr(p, 'to_dict') else str(p)
                for p in (rag_context.similar_prompts or [])[:3]
            ],
            "user_preferences": getattr(rag_context, 'user_preferences', {}),
        }

        composer_dimensions = ComposerDimensions(
            subject=dimensions.subject or "",
            aesthetic=dimensions.aesthetic,
            color=dimensions.color,
            composition=dimensions.composition,
            background=dimensions.background,
            lighting=dimensions.lighting,
            texture=dimensions.texture,
            mood=dimensions.mood,
            detail=dimensions.detail_level,
            expression=dimensions.expression,
            pose=dimensions.pose,
            art_movement=dimensions.art_movement,
            reference_style=dimensions.reference_style,
            technical=dimensions.technical if dimensions.technical else None,
        )

        prompt_composer = await self._get_prompt_composer_service()
        return await prompt_composer.compose_prompt(
            mode=mode.value,
            dimensions=composer_dimensions,
            context=composer_context,
            print_method=print_method or planning_context.get("product_spec", {}).get("default_method"),
            product=product,
        )

    # =========================================================================
    # PROVIDER SETTINGS BUILDER
    # =========================================================================

    def _build_provider_settings(
        self,
        model_id: str,
        model_specs: Dict[str, Any],
        requirements: Dict[str, Any],
        mode: PromptMode,
    ) -> Dict[str, Any]:
        """
        Build provider-specific settings from model specs and requirements.

        Different providers require different parameters. This method maps
        user requirements to the specific format each provider expects.

        Args:
            model_id: Selected model ID
            model_specs: Model specifications from config
            requirements: User requirements
            mode: Prompt mode (RELAX/STANDARD/COMPLEX)

        Returns:
            Provider-specific settings dict
        """
        provider = model_specs.get("provider", "")
        specs = model_specs.get("specs", {})
        provider_params = model_specs.get("provider_params", [])
        settings = {}

        # Set AIR ID (required for Runware)
        if model_specs.get("air_id"):
            settings["air_id"] = model_specs["air_id"]

        # Build provider settings dynamically from config specs
        provider_params = specs.get("provider_params", {})
        provider_settings = {}

        for param_name, param_spec in provider_params.items():
            if isinstance(param_spec, dict):
                # Get value from requirements with provider prefix or direct key
                req_key = f"{provider}_{param_name}"  # e.g., "midjourney_quality"
                alt_key = param_name  # e.g., "quality"

                value = requirements.get(req_key) or requirements.get(alt_key)

                param_type = param_spec.get("type", "string")
                default = param_spec.get("default")

                # Apply mode-based defaults if specified
                if value is None and mode == PromptMode.COMPLEX:
                    value = param_spec.get("complex_default", default)
                elif value is None and requirements.get("priority") == "fast":
                    value = param_spec.get("fast_default", default)
                elif value is None:
                    value = default

                if value is not None:
                    # Type conversion and validation
                    if param_type == "integer":
                        value = int(value)
                        if "min" in param_spec:
                            value = max(param_spec["min"], value)
                        if "max" in param_spec:
                            value = min(param_spec["max"], value)
                    elif param_type == "string":
                        value = str(value)
                        # Validate against allowed values if specified
                        allowed = param_spec.get("values", [])
                        if allowed and value not in allowed:
                            value = default
                    elif param_type == "array":
                        max_items = param_spec.get("max_items")
                        if max_items and isinstance(value, list):
                            value = value[:max_items]

                    provider_settings[param_name] = value

        # Handle special array fields from requirements
        if specs.get("style_reference_images"):
            style_refs = requirements.get("style_reference_images", [])
            max_refs = specs.get("style_reference_images", 4)
            if style_refs:
                provider_settings["styleReferenceImages"] = style_refs[:max_refs]

        # Handle colorPalette object (special case)
        if requirements.get("color_palette"):
            palette = requirements["color_palette"]
            if isinstance(palette, str):
                provider_settings["colorPalette"] = {"name": palette}
            elif isinstance(palette, dict):
                provider_settings["colorPalette"] = palette

        # Handle styleCode exclusivity
        if provider_settings.get("styleCode"):
            provider_settings.pop("styleType", None)
            provider_settings.pop("styleReferenceImages", None)

        # Wrap in providerSettings object with provider key
        if provider_settings:
            settings["providerSettings"] = {provider: provider_settings}

        # Handle provider-specific output constraints
        if specs.get("number_of_results"):
            constraint = specs["number_of_results"]
            num_images = requirements.get("num_images", 1)
            if "multiple of 4" in str(constraint):
                num_images = max(4, (num_images // 4) * 4)
            settings["numberOfResults"] = num_images

        # Image size tier for Google models
        if provider == "google" and specs.get("dimensions"):
            width = requirements.get("width", 1024)
            height = requirements.get("height", 1024)
            if width > 2048 or height > 2048:
                settings["imageSize"] = "4K"
            elif width > 1024 or height > 1024:
                settings["imageSize"] = "2K"
            else:
                settings["imageSize"] = "1K"

        # Common settings derived from specs
        if specs.get("prompt_length"):
            prompt_limit = specs["prompt_length"]
            if isinstance(prompt_limit, str):
                match = re.search(r"(\d+)", prompt_limit)
                if match:
                    settings["max_prompt_length"] = int(match.group(1))

        # Reference image limits
        max_refs = specs.get("reference_images", 0)
        if max_refs > 0:
            settings["max_reference_images"] = max_refs

        return settings

    # =========================================================================
    # PROMPT QUALITY EVALUATION
    # =========================================================================

    async def _evaluate_and_revise_prompt(
        self,
        prompt: str,
        requirements: Dict[str, Any],
    ) -> Tuple[QualityScore, int]:
        """Evaluate prompt quality and revise if needed."""
        reasoning_service = await self._get_reasoning_service()

        quality_score = await reasoning_service.assess_prompt_quality(
            prompt,
            constraints=requirements,
        )

        revision_count = 0

        while not quality_score.is_acceptable and revision_count < self.max_prompt_revisions:
            logger.info(f"Prompt quality {quality_score.overall_score} below threshold, revising...")

            feedback = "; ".join(quality_score.suggestions[:3])
            prompt = await reasoning_service.propose_prompt_revision(
                prompt,
                feedback,
                constraints=requirements,
            )

            quality_score = await reasoning_service.assess_prompt_quality(
                prompt,
                constraints=requirements,
            )
            revision_count += 1

        return quality_score, revision_count

    # =========================================================================
    # ASSEMBLY REQUEST BUILDING
    # =========================================================================

    async def _build_assembly_request(
        self,
        context: AgentContext,
        prompt: str,
        negative_prompt: str,
        mode: PromptMode,
        dimensions: PromptDimensions,
        pipeline: PipelineConfig,
        model_id: str,
        model_rationale: str,
        model_alternatives: List[str],
        model_specs: Dict[str, Any],
        quality_score: QualityScore,
        safety: SafetyClassification,
        rag_context: Context,
        requirements: Dict[str, Any],
        revision_count: int,
    ) -> AssemblyRequest:
        """Build structured AssemblyRequest for downstream services."""
        # Get model info for cost/time estimation
        model_info_service = await self._get_model_info_service()
        model_info = model_info_service.get_model(model_id)

        # Build provider-specific settings from model specs
        provider_settings = self._build_provider_settings(
            model_id=model_id,
            model_specs=model_specs,
            requirements=requirements,
            mode=mode,
        )

        # Build parameters with provider settings
        parameters = GenerationParameters(
            width=requirements.get("width", 1024),
            height=requirements.get("height", 1024),
            steps=requirements.get("steps", 30),
            guidance_scale=requirements.get("guidance_scale", 7.5),
            seed=requirements.get("seed"),
            num_images=requirements.get("num_images", 1),
            provider_settings=provider_settings,
        )

        # Estimate cost from model specs or service
        estimated_cost = 0.04  # Default
        estimated_time_ms = 15000  # Default

        # Try to get cost from model_specs first
        cost_info = model_specs.get("cost", {})
        if cost_info.get("per_image"):
            estimated_cost = cost_info["per_image"]
        elif cost_info.get("per_mp"):
            # Calculate based on megapixels
            mp = (parameters.width * parameters.height) / 1_000_000
            estimated_cost = cost_info["per_mp"] * mp

        # Override with model_info service if available
        if model_info:
            service_cost = model_info_service.estimate_cost(
                model_id=model_id,
                num_images=parameters.num_images,
                steps=parameters.steps,
            )
            if service_cost:
                estimated_cost = service_cost
            if hasattr(model_info, 'performance'):
                estimated_time_ms = model_info.performance.average_latency_ms

        # For dual pipeline, calculate combined cost
        if pipeline.pipeline_type == "dual" and pipeline.stage_2_model:
            stage_2_info = model_info_service.get_model(pipeline.stage_2_model)
            if stage_2_info:
                stage_2_cost = model_info_service.estimate_cost(
                    model_id=pipeline.stage_2_model,
                    num_images=1,
                    steps=parameters.steps,
                )
                if stage_2_cost:
                    estimated_cost += stage_2_cost
            # Dual pipeline has ~2.5x latency
            estimated_time_ms = int(estimated_time_ms * 2.5)

        return AssemblyRequest(
            prompt=prompt,
            negative_prompt=negative_prompt,
            mode=mode.value,
            dimensions=dimensions.to_dict(),
            pipeline=pipeline,
            model_id=model_id,
            model_rationale=model_rationale,
            model_alternatives=model_alternatives,
            parameters=parameters,
            reference_image_url=requirements.get("reference_image"),
            reference_strength=requirements.get("reference_strength", 0.75),
            prompt_quality_score=quality_score.overall_score,
            quality_acceptable=quality_score.is_acceptable,
            safety=safety,
            estimated_cost=estimated_cost,
            estimated_time_ms=estimated_time_ms,
            context_used=rag_context.to_dict() if hasattr(rag_context, 'to_dict') else None,
            job_id=context.job_id,
            user_id=context.user_id,
            product_type=requirements.get("product_type", "general"),
            print_method=requirements.get("print_method"),
            revision_count=revision_count,
            metadata={
                "quality_suggestions": quality_score.suggestions if hasattr(quality_score, 'suggestions') else [],
            },
        )

    # =========================================================================
    # FEEDBACK HANDLING
    # =========================================================================

    async def _apply_feedback_to_requirements(
        self,
        requirements: Dict[str, Any],
        feedback: EvaluationFeedback,
    ) -> Dict[str, Any]:
        """Apply evaluation feedback to improve requirements."""
        revised = requirements.copy()

        # Apply retry suggestions
        for suggestion in feedback.retry_suggestions:
            suggestion_lower = suggestion.lower()

            if "specific" in suggestion_lower or "detailed" in suggestion_lower:
                # Need more detail in subject
                if revised.get("subject"):
                    revised["subject"] = f"{revised['subject']}, highly detailed"

            if "composition" in suggestion_lower or "layout" in suggestion_lower:
                if not revised.get("composition"):
                    revised["composition"] = "well-composed, balanced layout"

            if "style" in suggestion_lower:
                if not revised.get("style"):
                    revised["style"] = "high quality, professional"

        # Check dimension scores for weak areas
        for dim, score in feedback.dimension_scores.items():
            if score < 0.4:
                logger.info(f"Low score for {dim}: {score}, adding emphasis")
                if dim in revised:
                    revised[dim] = f"{revised[dim]}, emphasized"

        return revised

    # =========================================================================
    # LIFECYCLE
    # =========================================================================

    async def __aenter__(self) -> "PlannerAgent":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()
