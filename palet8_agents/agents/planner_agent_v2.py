"""
Planner Agent v2 - Thin Coordinator for AI image generation planning.

Refactored to be a Plan-and-Do coordinator that delegates prompt building
to ReactPromptAgent while keeping high-level decision-making.

Key Responsibilities:
1. Context completeness check - DECIDE "sufficient?"
2. Safety check - DECIDE "safe?"
3. Complexity classification - DECIDE "simple/standard/complex"
4. Delegate prompt building to ReactPromptAgent
5. Model/pipeline selection - DECIDE which
6. Build final AssemblyRequest
7. Route between agents: Pali <-> ReactPromptAgent <-> Evaluator

Documentation Reference: Section 5.2.2
"""

from typing import Any, Dict, List, Optional

from src.utils.logger import get_logger, set_correlation_context

from palet8_agents.core.agent import BaseAgent, AgentContext, AgentResult
from palet8_agents.tools.base import BaseTool

# Import from models package
from palet8_agents.models import (
    ContextCompleteness,
    SafetyClassification,
    PromptDimensions,
    GenerationParameters,
    PipelineConfig,
    AssemblyRequest,
)
from palet8_agents.models.planning import PlanningTask, PromptPlan

# Services
from palet8_agents.services.text_llm_service import TextLLMService
from palet8_agents.services.context_analysis_service import ContextAnalysisService
from palet8_agents.services.model_selection_service import ModelSelectionService
from palet8_agents.services.safety_classification_service import SafetyClassificationService

logger = get_logger(__name__)


PLANNER_SYSTEM_PROMPT = """You are the **Planner** for Palet8's image generation system.

## YOUR ROLE
Transform creative briefs into complete, safe, high-quality generation plans.
You coordinate planning phases, decide routing, and assemble the final execution plan.
You do **not** talk to users directly — you operate only on structured data from other agents.

You delegate detailed prompt construction and iterative refinement to a dedicated planning agent.

---

## CORE DECISIONS

### 1. CONTEXT CHECK
Assess whether the information provided is sufficient to continue:
- If critical information is missing, return a small set of clarification points/questions for the user-facing agent.
- Do not guess or invent user intent.
- If sufficient, proceed with planning.

### 2. COMPLEXITY LEVEL
Determine how detailed the overall plan must be:
- **Simple**: Quick concepts, minimal direction needed.
- **Standard**: Clear stylistic guidance, composition matters.
- **Advanced**: Technical or production-level constraints present.

Let the nature of the request guide the required planning depth.

### 3. PROMPT PLANNING (Delegated)
You do **not** craft or refine prompts directly.
- Instead, delegate prompt construction, context enrichment, and prompt improvement to the dedicated ReAct-style planning agent.
- Ensure that the resulting prompt plan follows product or system constraints and is fit for downstream execution.

### 4. MODEL & PIPELINE SELECTION
Choose appropriate generation settings based on:
- Quality needs,
- Speed or latency preferences,
- Technical or system constraints,
- Cost or efficiency considerations.

Use service-provided metadata to make these decisions; do not guess.

---

## EXECUTION PHASES

### INITIAL
Perform the full planning flow:
- Check context completeness and safety.
- Determine complexity.
- Delegate prompt planning.
- Evaluate the resulting plan.
- Assemble the final execution request.

### FIX PLAN
Evaluator rejected the output:
- Apply evaluator feedback **surgically**.
- Prefer minimal, targeted adjustments.
- Delegate focused refinement to the planning agent.
- Avoid restarting from scratch unless absolutely necessary.
- Ensure past issues are not repeated.

### EDIT REQUEST
User has seen the result and wants changes (e.g., "make it darker", "remove text", "change the background").
- Preserve what already works.
- Modify only what the user requested.
- Prefer minimal adjustments; regenerate fully only when required.
- Treat major conceptual shifts as a revised brief with updated assumptions.

Delegate all prompt-level modifications to the planning agent.

### CLARIFY
The user supplied additional info through the assistant agent.
- Incorporate new details.
- Continue the appropriate planning phase (usually INITIAL).

---

## COLLABORATION
- Use evaluator feedback to improve constraints and planning decisions.
- Respect safety signals; when content is risky, propose safer alternatives or require user clarification.
- Coordinate with other agents and services without duplicating their logic.

---

## OUTPUT
Produce a **complete, high-level generation plan** that includes:
- The selected prompt plan returned by the planning agent,
- Parameters needed for execution,
- The chosen model or model configuration,
- Any pipeline decisions,
- A concise rationale for your decisions.

The plan should be ready for evaluation and eventual execution by downstream services."""


class PlannerAgentV2(BaseAgent):
    """
    Thin coordinator for AI image generation planning.

    Delegates prompt building to ReactPromptAgent while maintaining
    high-level decision-making and routing responsibilities.
    """

    def __init__(
        self,
        tools: Optional[List[BaseTool]] = None,
        text_service: Optional[TextLLMService] = None,
        context_analysis_service: Optional[ContextAnalysisService] = None,
        model_selection_service: Optional[ModelSelectionService] = None,
        safety_classification_service: Optional[SafetyClassificationService] = None,
    ):
        """
        Initialize the Planner Agent.

        Args:
            tools: Optional list of tools
            text_service: Optional TextLLMService for LLM calls
            context_analysis_service: Service for context completeness evaluation
            model_selection_service: Service for model and pipeline selection
            safety_classification_service: Service for content safety checks
        """
        super().__init__(
            name="planner",
            description="Thin coordinator for planning, routing, and model selection",
            tools=tools,
        )

        self._text_service = text_service
        self._context_analysis_service = context_analysis_service
        self._model_selection_service = model_selection_service
        self._safety_classification_service = safety_classification_service

        self._owns_services = text_service is None

        self.system_prompt = PLANNER_SYSTEM_PROMPT
        self.model_profile = "planner"

        # Thresholds
        self.min_context_completeness = 0.5

    # =========================================================================
    # Service Getters
    # =========================================================================

    async def _get_text_service(self) -> TextLLMService:
        """Get or create text LLM service."""
        if self._text_service is None:
            self._text_service = TextLLMService(default_profile_name=self.model_profile)
        return self._text_service

    async def _get_context_analysis_service(self) -> ContextAnalysisService:
        """Get or create context analysis service."""
        if self._context_analysis_service is None:
            self._context_analysis_service = ContextAnalysisService()
        return self._context_analysis_service

    async def _get_model_selection_service(self) -> ModelSelectionService:
        """Get or create model selection service."""
        if self._model_selection_service is None:
            self._model_selection_service = ModelSelectionService()
        return self._model_selection_service

    async def _get_safety_classification_service(self) -> SafetyClassificationService:
        """Get or create safety classification service."""
        if self._safety_classification_service is None:
            text_service = await self._get_text_service()
            self._safety_classification_service = SafetyClassificationService(
                text_service=text_service
            )
        return self._safety_classification_service

    async def close(self) -> None:
        """Close resources."""
        if self._owns_services:
            if self._text_service:
                await self._text_service.close()
                self._text_service = None
            if self._context_analysis_service:
                if hasattr(self._context_analysis_service, 'close'):
                    await self._context_analysis_service.close()
                self._context_analysis_service = None
            if self._model_selection_service:
                if hasattr(self._model_selection_service, 'close'):
                    await self._model_selection_service.close()
                self._model_selection_service = None
            if self._safety_classification_service:
                if hasattr(self._safety_classification_service, 'close'):
                    await self._safety_classification_service.close()
                self._safety_classification_service = None

    # =========================================================================
    # Main Entry Point
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
            phase: Execution phase ("initial", "post_prompt", "fix_plan", "edit")
            evaluation_feedback: Feedback from Evaluator for fix_plan phase

        Returns:
            AgentResult with routing information
        """
        self._start_execution()

        # Set correlation context for all downstream logs
        set_correlation_context(
            job_id=context.job_id,
            user_id=context.user_id,
        )

        logger.info(
            "planner_v2.run.start",
            phase=phase,
            has_feedback=evaluation_feedback is not None,
            requirements_count=len(context.requirements or {}),
        )

        try:

            if phase == "initial":
                return await self._handle_initial(context)
            elif phase == "post_prompt":
                return await self._handle_post_prompt(context)
            elif phase == "fix_plan":
                return await self._handle_fix_plan(context, evaluation_feedback)
            elif phase == "edit":
                return await self._handle_edit(context, user_input)
            else:
                return self._create_result(
                    success=False,
                    data=None,
                    error=f"Unknown phase: {phase}",
                    error_code="UNKNOWN_PHASE",
                )

        except Exception as e:
            logger.error(
                "planner_v2.run.error",
                phase=phase,
                error=str(e),
                error_type=type(e).__name__,
                exc_info=True,
            )
            return self._create_result(
                success=False,
                data=None,
                error=f"Planning failed: {e}",
                error_code="PLANNING_ERROR",
            )

    # =========================================================================
    # Phase Handlers
    # =========================================================================

    async def _handle_initial(self, context: AgentContext) -> AgentResult:
        """
        Handle initial planning phase.

        1. Check context completeness - DECIDE
        2. Check safety - DECIDE
        3. Classify complexity - DECIDE
        4. Delegate to ReactPromptAgent
        """
        requirements = context.requirements or {}

        # STEP 1: Context Completeness Check - PLANNER DECIDES
        completeness = await self._evaluate_context_completeness(requirements)

        logger.info(
            "planner_v2.context.evaluated",
            score=completeness.score,
            is_sufficient=self._is_context_sufficient(completeness),
            missing_count=len(completeness.missing_fields),
            missing_fields=completeness.missing_fields[:5],
        )

        if not self._is_context_sufficient(completeness):
            logger.info(
                "planner_v2.context.insufficient",
                score=completeness.score,
                missing_fields=completeness.missing_fields,
                questions_count=len(completeness.clarifying_questions),
            )
            return self._create_result(
                success=True,
                data={
                    "action": "needs_clarification",
                    "completeness": completeness.to_dict(),
                    "questions": completeness.clarifying_questions,
                    "missing_fields": completeness.missing_fields,
                },
                next_agent="pali",
            )

        # STEP 2: Safety Check - PLANNER DECIDES
        safety = await self._classify_safety(requirements)

        logger.info(
            "planner_v2.safety.classified",
            is_safe=self._is_safe_to_proceed(safety),
            risk_level=safety.risk_level,
            requires_review=safety.requires_review,
            categories=safety.categories,
        )

        if not self._is_safe_to_proceed(safety):
            logger.warning(
                "planner_v2.safety.blocked",
                reason=safety.reason,
                categories=safety.categories,
            )
            return self._create_result(
                success=False,
                data={
                    "action": "blocked",
                    "safety": safety.to_dict(),
                },
                error=f"Content blocked: {safety.reason}",
                error_code="SAFETY_BLOCKED",
            )

        # STEP 3: Classify Complexity - PLANNER DECIDES
        complexity = self._classify_complexity(requirements)
        logger.info(
            "planner_v2.complexity.classified",
            complexity=complexity,
            product_type=requirements.get("product_type"),
        )

        # STEP 4: Create PlanningTask and delegate to ReactPromptAgent
        planning_task = PlanningTask(
            job_id=context.job_id,
            user_id=context.user_id,
            phase="initial",
            requirements=requirements,
            complexity=complexity,
            product_type=requirements.get("product_type", "general"),
            print_method=requirements.get("print_method"),
        )

        # Store in context for ReactPromptAgent
        context.metadata["planning_task"] = planning_task.to_dict()
        context.metadata["safety"] = safety.to_dict()

        return self._create_result(
            success=True,
            data={
                "action": "build_prompt",
                "planning_task": planning_task.to_dict(),
            },
            next_agent="react_prompt",
        )

    async def _handle_post_prompt(self, context: AgentContext) -> AgentResult:
        """
        Handle post-prompt phase after ReactPromptAgent returns.

        1. Get PromptPlan from context
        2. Select model - DECIDE
        3. Select pipeline - DECIDE
        4. Build AssemblyRequest
        5. Route to Evaluator
        """
        prompt_plan_data = context.metadata.get("prompt_plan")
        if not prompt_plan_data:
            return self._create_result(
                success=False,
                data=None,
                error="No prompt_plan found in context",
                error_code="MISSING_PROMPT_PLAN",
            )

        prompt_plan = PromptPlan.from_dict(prompt_plan_data)
        requirements = context.requirements or {}
        safety_data = context.metadata.get("safety", {})

        logger.info(
            "planner_v2.post_prompt.start",
            quality_score=prompt_plan.quality_score,
            mode=prompt_plan.mode,
            prompt_length=len(prompt_plan.prompt),
        )

        # STEP 1: Select Model - PLANNER DECIDES
        model_id, rationale, alternatives = await self._select_model(
            mode=prompt_plan.mode,
            requirements=requirements,
        )

        logger.info(
            "planner_v2.model.selected",
            model_id=model_id,
            rationale=rationale[:100] if rationale else None,
            alternatives_count=len(alternatives),
        )

        # STEP 2: Select Pipeline - PLANNER DECIDES
        pipeline = await self._select_pipeline(
            requirements=requirements,
            prompt=prompt_plan.prompt,
        )

        logger.info(
            "planner_v2.pipeline.selected",
            pipeline_type=pipeline.pipeline_type,
            stage_1_model=pipeline.stage_1_model,
            stage_2_model=pipeline.stage_2_model,
        )

        # STEP 3: Build AssemblyRequest
        assembly_request = self._build_assembly_request(
            context=context,
            prompt_plan=prompt_plan,
            model_id=model_id,
            model_rationale=rationale,
            model_alternatives=alternatives,
            pipeline=pipeline,
            safety=SafetyClassification.from_dict(safety_data) if safety_data else None,
        )

        logger.info(
            "planner_v2.assembly_request.created",
            model_id=model_id,
            pipeline_type=pipeline.pipeline_type,
            prompt_length=len(prompt_plan.prompt),
        )

        # Store in context
        context.metadata["assembly_request"] = assembly_request.to_dict()

        return self._create_result(
            success=True,
            data={
                "action": "evaluate_plan",
                "assembly_request": assembly_request.to_dict(),
            },
            next_agent="evaluator",
        )

    async def _handle_fix_plan(
        self,
        context: AgentContext,
        evaluation_feedback: Optional[Dict[str, Any]],
    ) -> AgentResult:
        """
        Handle fix_plan phase after Evaluator rejects.

        Delegate to ReactPromptAgent with feedback.
        """
        requirements = context.requirements or {}
        previous_plan = context.metadata.get("prompt_plan", {})

        logger.info(
            "planner_v2.fix_plan.start",
            has_feedback=evaluation_feedback is not None,
            issues_count=len(evaluation_feedback.get("issues", [])) if evaluation_feedback else 0,
        )

        planning_task = PlanningTask(
            job_id=context.job_id,
            user_id=context.user_id,
            phase="fix_plan",
            requirements=requirements,
            complexity=context.metadata.get("complexity", "standard"),
            product_type=requirements.get("product_type", "general"),
            print_method=requirements.get("print_method"),
            previous_plan=previous_plan,
            evaluation_feedback=evaluation_feedback,
        )

        context.metadata["planning_task"] = planning_task.to_dict()

        return self._create_result(
            success=True,
            data={
                "action": "fix_prompt",
                "planning_task": planning_task.to_dict(),
            },
            next_agent="react_prompt",
        )

    async def _handle_edit(
        self,
        context: AgentContext,
        edit_instructions: Optional[str],
    ) -> AgentResult:
        """
        Handle edit phase when user requests changes.

        Delegate to ReactPromptAgent with edit instructions.
        """
        requirements = context.requirements or {}
        previous_plan = context.metadata.get("prompt_plan", {})

        planning_task = PlanningTask(
            job_id=context.job_id,
            user_id=context.user_id,
            phase="edit",
            requirements=requirements,
            complexity=context.metadata.get("complexity", "standard"),
            product_type=requirements.get("product_type", "general"),
            print_method=requirements.get("print_method"),
            previous_plan=previous_plan,
            edit_instructions=edit_instructions,
        )

        context.metadata["planning_task"] = planning_task.to_dict()

        return self._create_result(
            success=True,
            data={
                "action": "edit_prompt",
                "planning_task": planning_task.to_dict(),
            },
            next_agent="react_prompt",
        )

    # =========================================================================
    # Decision Methods
    # =========================================================================

    async def _evaluate_context_completeness(
        self,
        requirements: Dict[str, Any],
    ) -> ContextCompleteness:
        """Evaluate if requirements provide sufficient context."""
        service = await self._get_context_analysis_service()
        return service.evaluate_completeness(requirements)

    def _is_context_sufficient(self, completeness: ContextCompleteness) -> bool:
        """DECIDE: Is context sufficient to proceed?"""
        return completeness.is_sufficient and completeness.score >= self.min_context_completeness

    async def _classify_safety(
        self,
        requirements: Dict[str, Any],
    ) -> SafetyClassification:
        """Classify content safety."""
        service = await self._get_safety_classification_service()

        # Combine requirements text for safety check
        text_to_check = " ".join([
            str(requirements.get("subject", "")),
            str(requirements.get("style", "")),
            str(requirements.get("mood", "")),
            " ".join(requirements.get("include_elements", [])),
        ])

        flag = await service.classify_content(text_to_check, source="requirements")

        if flag:
            return SafetyClassification(
                is_safe=False,
                requires_review=flag.severity.value in ["medium", "low"],
                risk_level=flag.severity.value,
                categories=[flag.category.value],
                reason=flag.description,
            )

        return SafetyClassification(
            is_safe=True,
            requires_review=False,
            risk_level="low",
            categories=[],
        )

    def _is_safe_to_proceed(self, safety: SafetyClassification) -> bool:
        """DECIDE: Is it safe to proceed?"""
        return safety.is_safe

    def _classify_complexity(self, requirements: Dict[str, Any]) -> str:
        """
        DECIDE: What complexity level?

        - Simple: Quick concepts, minimal direction needed.
        - Standard: Clear stylistic guidance, composition matters.
        - Advanced: Technical or production-level constraints present.
        """
        # Score based on requirements present
        has_technical = bool(requirements.get("print_method"))
        has_composition = bool(requirements.get("composition"))
        has_multiple_elements = len(requirements.get("include_elements", [])) > 2
        has_reference = bool(requirements.get("reference_image"))
        has_style = bool(requirements.get("style") or requirements.get("aesthetic"))
        has_colors = bool(requirements.get("colors"))

        # Technical constraints indicate advanced
        if has_technical:
            return "advanced"

        # Count stylistic/compositional factors
        complexity_score = sum([
            has_composition,
            has_multiple_elements,
            has_reference,
            has_style,
            has_colors,
        ])

        if complexity_score >= 3:
            return "advanced"
        elif complexity_score >= 1:
            return "standard"
        else:
            return "simple"

    async def _select_model(
        self,
        mode: str,
        requirements: Dict[str, Any],
    ) -> tuple:
        """DECIDE: Which model to use?"""
        service = await self._get_model_selection_service()

        try:
            model_id, rationale, alternatives = await service.select_model(
                mode=mode,
                requirements=requirements,
            )
            return model_id, rationale, alternatives
        except Exception as e:
            logger.warning(
                "planner_v2.model.selection_failed",
                error=str(e),
                fallback_model="default-model",
            )
            return "default-model", "Fallback selection", []

    async def _select_pipeline(
        self,
        requirements: Dict[str, Any],
        prompt: str,
    ) -> PipelineConfig:
        """DECIDE: Single or dual pipeline?"""
        service = await self._get_model_selection_service()

        try:
            return await service.select_pipeline(
                requirements=requirements,
                prompt=prompt,
            )
        except Exception as e:
            logger.warning(
                "planner_v2.pipeline.selection_failed",
                error=str(e),
                fallback_pipeline="single",
            )
            return PipelineConfig(pipeline_type="single")

    # =========================================================================
    # Assembly Request Builder
    # =========================================================================

    def _build_assembly_request(
        self,
        context: AgentContext,
        prompt_plan: PromptPlan,
        model_id: str,
        model_rationale: str,
        model_alternatives: List[str],
        pipeline: PipelineConfig,
        safety: Optional[SafetyClassification] = None,
    ) -> AssemblyRequest:
        """Build the final AssemblyRequest for downstream services."""
        requirements = context.requirements or {}

        # Build GenerationParameters from provider_params
        provider_params = prompt_plan.provider_params or {}

        # Extract standard params, put rest in provider_settings
        gen_params = GenerationParameters(
            width=requirements.get("width", 1024),
            height=requirements.get("height", 1024),
            steps=provider_params.get("steps", 30),
            guidance_scale=provider_params.get("guidance_scale", 7.5),
            seed=provider_params.get("seed"),
            num_images=requirements.get("num_images", 1),
            provider_settings={
                k: v for k, v in provider_params.items()
                if k not in ("steps", "guidance_scale", "seed")
            },
        )

        return AssemblyRequest(
            prompt=prompt_plan.prompt,
            negative_prompt=prompt_plan.negative_prompt,
            mode=prompt_plan.mode,
            dimensions=prompt_plan.dimensions,
            pipeline=pipeline,
            model_id=model_id,
            model_rationale=model_rationale,
            model_alternatives=model_alternatives,
            parameters=gen_params,
            prompt_quality_score=prompt_plan.quality_score,
            quality_acceptable=prompt_plan.quality_acceptable,
            safety=safety or SafetyClassification(
                is_safe=True,
                requires_review=False,
                risk_level="low",
                categories=[],
            ),
            context_used=prompt_plan.context_summary.to_dict(),
            job_id=context.job_id,
            user_id=context.user_id,
            product_type=requirements.get("product_type", "general"),
            print_method=requirements.get("print_method"),
            revision_count=prompt_plan.revision_count,
        )

    # =========================================================================
    # Context Manager
    # =========================================================================

    async def __aenter__(self) -> "PlannerAgentV2":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()
