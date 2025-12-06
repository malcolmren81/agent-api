"""
Planner Agent v2 - Pure Orchestrator for AI image generation.

Refactored to be a thin orchestrator that follows pipeline_methods.yaml
checkpoints and delegates specialized tasks to dedicated agents:

- GenPlanAgent: Complexity, genflow, model selection, parameters
- ReactPromptAgent: Context enrichment, prompt composition
- EvaluatorAgent: Pre/post generation quality gates

Key Responsibilities:
1. Context completeness check - Quick gate using ContextAnalysisService
2. Safety check - Gate using SafetyClassificationService
3. Delegate generation planning to GenPlanAgent
4. Delegate prompt building to ReactPromptAgent
5. Delegate evaluation to EvaluatorAgent
6. Coordinate execution via AssemblyService
7. Handle retries and failure routing

Documentation Reference: Section 5.2.2
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
import yaml

from src.utils.logger import get_logger, set_correlation_context


class TodoStatus(Enum):
    """Status of a todo item."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class TodoItem:
    """Internal todo tracking item."""
    id: str
    description: str
    status: TodoStatus = TodoStatus.PENDING
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "status": self.status.value,
            "result": self.result,
            "error": self.error,
        }


@dataclass
class PlannerTodoList:
    """Internal todo list for tracking checkpoint progress."""
    items: List[TodoItem] = field(default_factory=list)
    current_idx: int = 0

    def init_from_checkpoints(self, checkpoints: List[Dict[str, Any]]) -> None:
        """Initialize todo list from pipeline checkpoints."""
        self.items = []
        for cp in checkpoints:
            cp_id = cp.get("id", "unknown")
            desc = self._get_checkpoint_description(cp_id)
            self.items.append(TodoItem(id=cp_id, description=desc))
        self.current_idx = 0

    def _get_checkpoint_description(self, checkpoint_id: str) -> str:
        """Get human-readable description for checkpoint."""
        descriptions = {
            "context_check": "Verify context completeness",
            "safety_check": "Perform safety classification",
            "generation_plan": "Create generation plan (GenPlan agent)",
            "prompt_build": "Build prompt (ReactPrompt agent)",
            "pre_evaluation": "Evaluate prompt quality",
            "execute_generation": "Execute image generation",
            "post_evaluation": "Evaluate generation result",
        }
        return descriptions.get(checkpoint_id, f"Execute {checkpoint_id}")

    def start_item(self, checkpoint_id: str) -> None:
        """Mark a todo item as in progress."""
        for item in self.items:
            if item.id == checkpoint_id:
                item.status = TodoStatus.IN_PROGRESS
                break

    def complete_item(self, checkpoint_id: str, result: Optional[Dict] = None) -> None:
        """Mark a todo item as completed."""
        for item in self.items:
            if item.id == checkpoint_id:
                item.status = TodoStatus.COMPLETED
                item.result = result
                break

    def fail_item(self, checkpoint_id: str, error: str) -> None:
        """Mark a todo item as failed."""
        for item in self.items:
            if item.id == checkpoint_id:
                item.status = TodoStatus.FAILED
                item.error = error
                break

    def get_progress(self) -> Dict[str, Any]:
        """Get current progress summary."""
        completed = sum(1 for i in self.items if i.status == TodoStatus.COMPLETED)
        failed = sum(1 for i in self.items if i.status == TodoStatus.FAILED)
        return {
            "total": len(self.items),
            "completed": completed,
            "failed": failed,
            "pending": len(self.items) - completed - failed,
            "progress_pct": completed / len(self.items) if self.items else 0,
        }

    def to_list(self) -> List[Dict[str, Any]]:
        """Convert to list of dicts for logging."""
        return [item.to_dict() for item in self.items]

from palet8_agents.core.agent import BaseAgent, AgentContext, AgentResult
from palet8_agents.tools.base import BaseTool

# Import from models package
from palet8_agents.models import (
    ContextCompleteness,
    SafetyClassification,
    GenerationParameters,
    PipelineConfig,
    AssemblyRequest,
)
from palet8_agents.models.planning import PlanningTask, PromptPlan
from palet8_agents.models.genplan import GenerationPlan

# Services
from palet8_agents.services.context_analysis_service import ContextAnalysisService
from palet8_agents.services.safety_classification_service import SafetyClassificationService
from palet8_agents.services.text_llm_service import TextLLMService

logger = get_logger(__name__)


def _load_pipeline_methods() -> Dict[str, Any]:
    """Load pipeline methods configuration."""
    config_paths = [
        Path("config/pipeline_methods.yaml"),
        Path(__file__).parent.parent.parent / "config" / "pipeline_methods.yaml",
    ]

    for path in config_paths:
        if path.exists():
            try:
                with open(path, "r") as f:
                    return yaml.safe_load(f)
            except Exception as e:
                logger.warning(f"Failed to load pipeline methods: {e}")

    # Return minimal default if no config found
    return {
        "pipeline_methods": {
            "standard_generation": {
                "checkpoints": [
                    {"id": "context_check", "handler": "internal"},
                    {"id": "safety_check", "handler": "internal"},
                    {"id": "generation_plan", "handler": "agent", "agent": "genplan"},
                    {"id": "prompt_build", "handler": "agent", "agent": "react_prompt"},
                    {"id": "pre_evaluation", "handler": "agent", "agent": "evaluator"},
                    {"id": "execute_generation", "handler": "service", "service": "assembly"},
                    {"id": "post_evaluation", "handler": "agent", "agent": "evaluator"},
                ]
            }
        }
    }


PIPELINE_METHODS = _load_pipeline_methods()


PLANNER_SYSTEM_PROMPT = """You are the **Planner** for Palet8's image generation system.

## YOUR ROLE
Pure orchestrator that coordinates the generation pipeline by:
1. Running checkpoints defined in pipeline_methods.yaml
2. Maintaining an internal todo list to track checkpoint progress
3. Delegating specialized tasks to dedicated agents
4. Handling failures and retry logic
5. Routing clarification requests through Pali

You do NOT make generation planning decisions directly. Those are delegated to:
- GenPlanAgent: Complexity determination, genflow selection, model selection, parameter extraction
- ReactPromptAgent: Context evaluation, clarification questions, prompt composition
- EvaluatorAgent: Pre/post generation quality assessment

## INTERNAL TODO LIST
You track progress through the pipeline with an internal todo list:
- Each checkpoint becomes a todo item with status tracking
- Statuses: PENDING → IN_PROGRESS → COMPLETED/FAILED
- Progress is logged at each status change for visibility
- Todo list enables progress tracking and debugging

## CHECKPOINT FLOW
1. context_check: Verify minimum context is available
2. safety_check: Gate for content safety
3. generation_plan: Delegate to GenPlanAgent (complexity, genflow, model)
4. prompt_build: Delegate to ReactPromptAgent (context, questions, prompt)
5. pre_evaluation: Quality gate before generation
6. execute_generation: Run image generation via AssemblyService
7. post_evaluation: Quality gate on results

## FAILURE HANDLING
- request_clarification: Route questions through Pali to user
- retry: Re-run checkpoint with adjustments (max retries from config)
- block: Stop and report safety concern
"""


class PlannerAgentV2(BaseAgent):
    """
    Pure orchestrator for AI image generation pipeline.

    Follows pipeline_methods.yaml checkpoints and delegates
    specialized tasks to dedicated agents.
    """

    def __init__(
        self,
        tools: Optional[List[BaseTool]] = None,
        text_service: Optional[TextLLMService] = None,
        context_analysis_service: Optional[ContextAnalysisService] = None,
        safety_classification_service: Optional[SafetyClassificationService] = None,
    ):
        """
        Initialize the Planner Agent.

        Args:
            tools: Optional list of tools
            text_service: Optional TextLLMService for LLM calls
            context_analysis_service: Service for context completeness evaluation
            safety_classification_service: Service for content safety checks
        """
        super().__init__(
            name="planner",
            description="Pure orchestrator for generation pipeline",
            tools=tools,
        )

        self._text_service = text_service
        self._context_analysis_service = context_analysis_service
        self._safety_classification_service = safety_classification_service
        self._owns_services = text_service is None
        self._pali_callback: Optional[Callable] = None

        self.system_prompt = PLANNER_SYSTEM_PROMPT
        self.model_profile = "planner"

        # Load pipeline methods config
        self._pipeline_methods = PIPELINE_METHODS

        # Thresholds
        self.min_context_completeness = 0.5
        self.max_retries = 3
        self.max_clarification_rounds = 3  # Max rounds of user clarification before proceeding

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
            if self._safety_classification_service:
                if hasattr(self._safety_classification_service, 'close'):
                    await self._safety_classification_service.close()
                self._safety_classification_service = None

    # =========================================================================
    # Main Orchestration Loop
    # =========================================================================

    async def orchestrate_generation(
        self,
        context: AgentContext,
        pali_callback: Optional[Callable] = None,
    ) -> AgentResult:
        """
        Main orchestration loop following pipeline_methods.yaml checkpoints.

        Args:
            context: Shared execution context
            pali_callback: Optional callback to communicate via Pali

        Returns:
            AgentResult with final generation results
        """
        self._start_execution()
        self._pali_callback = pali_callback
        retry_count = 0

        # Track clarification rounds (persisted across orchestration calls)
        clarification_round = context.metadata.get("clarification_round", 0)

        # Set correlation context
        set_correlation_context(
            job_id=context.job_id,
            user_id=context.user_id,
        )

        logger.info(
            "planner_v2.orchestration.start",
            job_id=context.job_id,
            has_pali_callback=pali_callback is not None,
            clarification_round=clarification_round,
        )

        try:
            requirements = context.requirements or {}

            # Get pipeline method to use
            pipeline_name = self._select_pipeline_method(requirements)
            pipeline_config = self._pipeline_methods.get("pipeline_methods", {}).get(
                pipeline_name, {}
            )
            checkpoints = pipeline_config.get("checkpoints", [])

            # Initialize internal todo list from checkpoints
            todo_list = PlannerTodoList()
            todo_list.init_from_checkpoints(checkpoints)

            logger.info(
                "planner_v2.pipeline.selected",
                pipeline_name=pipeline_name,
                checkpoint_count=len(checkpoints),
            )

            # Log initial todo list
            logger.info(
                "planner_v2.todo.initialized",
                job_id=context.job_id,
                todo_count=len(todo_list.items),
                todos=todo_list.to_list(),
            )

            # Execute checkpoints in order
            checkpoint_idx = 0
            while checkpoint_idx < len(checkpoints) and retry_count < self.max_retries:
                checkpoint = checkpoints[checkpoint_idx]
                checkpoint_id = checkpoint.get("id")

                await self._emit_progress(checkpoint_id, checkpoint_idx / len(checkpoints))

                # Mark todo as in_progress
                todo_list.start_item(checkpoint_id)
                logger.info(
                    "planner_v2.todo.in_progress",
                    checkpoint_id=checkpoint_id,
                    checkpoint_idx=checkpoint_idx,
                    progress=todo_list.get_progress(),
                )

                logger.info(
                    "planner_v2.checkpoint.start",
                    checkpoint_id=checkpoint_id,
                    checkpoint_idx=checkpoint_idx,
                )

                # Execute checkpoint
                result = await self._execute_checkpoint(context, checkpoint)

                if not result.get("success"):
                    # Mark todo as failed
                    todo_list.fail_item(checkpoint_id, result.get("error", "Unknown error"))
                    logger.info(
                        "planner_v2.todo.failed",
                        checkpoint_id=checkpoint_id,
                        error=result.get("error"),
                        progress=todo_list.get_progress(),
                    )

                    # Handle failure based on on_fail action
                    on_fail = checkpoint.get("on_fail", "retry")
                    failure_result = await self._handle_checkpoint_failure(
                        context, checkpoint, result, on_fail, retry_count
                    )

                    if failure_result.get("action") == "return":
                        return failure_result.get("result")
                    elif failure_result.get("action") == "retry":
                        retry_count += 1
                        # Jump to specified checkpoint or stay at current
                        jump_to = failure_result.get("jump_to")
                        if jump_to:
                            checkpoint_idx = self._find_checkpoint_index(checkpoints, jump_to)
                        continue
                    elif failure_result.get("action") == "continue":
                        # Continue with warning
                        pass
                else:
                    # Mark todo as completed
                    todo_list.complete_item(checkpoint_id, result)
                    logger.info(
                        "planner_v2.todo.completed",
                        checkpoint_id=checkpoint_id,
                        progress=todo_list.get_progress(),
                    )

                # Move to next checkpoint
                checkpoint_idx += 1

            # All checkpoints complete
            await self._emit_progress("complete", 1.0)

            # Log final todo list
            logger.info(
                "planner_v2.todo.all_complete",
                job_id=context.job_id,
                final_progress=todo_list.get_progress(),
                todos=todo_list.to_list(),
            )

            # Get final results from context
            images = context.metadata.get("generated_images", [])
            generation_plan = context.metadata.get("generation_plan", {})
            prompt_plan = context.metadata.get("prompt_plan", {})
            post_eval = context.metadata.get("post_evaluation", {})

            logger.info(
                "planner_v2.orchestration.complete",
                job_id=context.job_id,
                images_count=len(images),
                retry_count=retry_count,
            )

            return self._create_result(
                success=True,
                data={
                    "action": "generation_complete",
                    "images": images,
                    "generation_plan": generation_plan,
                    "prompt_plan": prompt_plan,
                    "evaluation": post_eval,
                    "model_id": generation_plan.get("model_id"),
                    "pipeline_type": generation_plan.get("genflow", {}).get("flow_type", "single"),
                },
            )

        except Exception as e:
            logger.error(
                "planner_v2.orchestration.error",
                error_detail=str(e),
                error_type=type(e).__name__,
                exc_info=True,
            )
            return self._create_result(
                success=False,
                data=None,
                error_detail=str(e),
                error_code="ORCHESTRATION_ERROR",
            )

    def _select_pipeline_method(self, requirements: Dict[str, Any]) -> str:
        """Select which pipeline method to use based on requirements."""
        # Check pipeline selection rules
        rules = self._pipeline_methods.get("pipeline_selection_rules", [])
        complexity = requirements.get("complexity", "").lower()

        for rule in rules:
            condition = rule.get("condition", "")
            if condition == "default":
                return rule.get("pipeline", "standard_generation")
            if "simple" in condition and complexity == "simple":
                return rule.get("pipeline", "quick_generation")
            if "RELAX" in condition and requirements.get("mode") == "RELAX":
                return rule.get("pipeline", "quick_generation")

        return "standard_generation"

    def _find_checkpoint_index(self, checkpoints: List[Dict], checkpoint_id: str) -> int:
        """Find index of checkpoint by ID."""
        for i, cp in enumerate(checkpoints):
            if cp.get("id") == checkpoint_id:
                return i
        return 0

    # =========================================================================
    # Checkpoint Execution
    # =========================================================================

    async def _execute_checkpoint(
        self,
        context: AgentContext,
        checkpoint: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute a single checkpoint."""
        checkpoint_id = checkpoint.get("id")
        handler = checkpoint.get("handler")

        if handler == "internal":
            return await self._execute_internal_checkpoint(context, checkpoint)
        elif handler == "agent":
            return await self._execute_agent_checkpoint(context, checkpoint)
        elif handler == "service":
            return await self._execute_service_checkpoint(context, checkpoint)
        else:
            logger.warning(f"Unknown checkpoint handler: {handler}")
            return {"success": False, "error": f"Unknown handler: {handler}"}

    async def _execute_internal_checkpoint(
        self,
        context: AgentContext,
        checkpoint: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute internal checkpoint (context_check, safety_check)."""
        checkpoint_id = checkpoint.get("id")
        requirements = context.requirements or {}

        if checkpoint_id == "context_check":
            completeness = await self._evaluate_context_completeness(requirements)
            context.metadata["context_completeness"] = completeness.to_dict()

            if not self._is_context_sufficient(completeness):
                return {
                    "success": False,
                    "error": "Insufficient context",
                    "data": {
                        "completeness": completeness.to_dict(),
                        "questions": completeness.clarifying_questions,
                        "missing_fields": completeness.missing_fields,
                    },
                }
            return {"success": True}

        elif checkpoint_id == "safety_check":
            safety = await self._classify_safety(requirements)
            context.metadata["safety"] = safety.to_dict()

            if not safety.is_safe:
                return {
                    "success": False,
                    "error": f"Content blocked: {safety.reason}",
                    "data": {"safety": safety.to_dict()},
                }
            return {"success": True}

        return {"success": True}

    async def _execute_agent_checkpoint(
        self,
        context: AgentContext,
        checkpoint: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute agent checkpoint (genplan, react_prompt, evaluator)."""
        agent_name = checkpoint.get("agent")

        if agent_name == "genplan":
            return await self._delegate_to_genplan(context)
        elif agent_name == "react_prompt":
            return await self._delegate_to_react_prompt(context)
        elif agent_name == "evaluator":
            action = checkpoint.get("action", "pre_evaluation")
            return await self._delegate_to_evaluator(context, action)

        return {"success": False, "error": f"Unknown agent: {agent_name}"}

    async def _execute_service_checkpoint(
        self,
        context: AgentContext,
        checkpoint: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute service checkpoint (assembly)."""
        service_name = checkpoint.get("service")

        if service_name == "assembly":
            return await self._execute_generation(context)

        return {"success": False, "error": f"Unknown service: {service_name}"}

    # =========================================================================
    # Agent Delegation
    # =========================================================================

    async def _delegate_to_genplan(self, context: AgentContext) -> Dict[str, Any]:
        """Delegate generation planning to GenPlanAgent."""
        from palet8_agents.agents.genplan_agent import GenPlanAgent

        logger.info("planner_v2.delegate.genplan", job_id=context.job_id)

        try:
            async with GenPlanAgent() as agent:
                result = await agent.run(context=context)

            if not result.success:
                logger.error("planner_v2.genplan.failed", err_msg=result.error)
                return {"success": False, "error": result.error}

            # Check if clarification is needed
            if result.requires_user_input:
                return {
                    "success": False,
                    "needs_clarification": True,
                    "data": result.data,
                }

            # Generation plan stored in context.metadata["generation_plan"]
            return {"success": True, "data": result.data}

        except Exception as e:
            logger.error("planner_v2.genplan.error", error_detail=str(e), exc_info=True)
            return {"success": False, "error": str(e)}

    async def _delegate_to_react_prompt(self, context: AgentContext) -> Dict[str, Any]:
        """Delegate prompt building to ReactPromptAgent."""
        from palet8_agents.agents.react_prompt_agent import ReactPromptAgent

        requirements = context.requirements or {}
        generation_plan_data = context.metadata.get("generation_plan", {})
        retry_feedback = context.metadata.get("evaluation_feedback")

        phase = "fix_plan" if retry_feedback else "initial"

        logger.info(
            "planner_v2.delegate.react_prompt",
            phase=phase,
            has_generation_plan=bool(generation_plan_data),
        )

        # Build PlanningTask from GenerationPlan
        planning_task = PlanningTask(
            job_id=context.job_id,
            user_id=context.user_id,
            phase=phase,
            requirements=requirements,
            complexity=generation_plan_data.get("complexity", "standard"),
            product_type=requirements.get("product_type", "general"),
            print_method=requirements.get("print_method"),
            previous_plan=context.metadata.get("prompt_plan"),
            evaluation_feedback=retry_feedback,
        )

        context.metadata["planning_task"] = planning_task.to_dict()

        try:
            async with ReactPromptAgent() as agent:
                result = await agent.run(context=context)

            if not result.success:
                logger.error("planner_v2.react_prompt.failed", err_msg=result.error)
                return {"success": False, "error": result.error}

            # Check if clarification is needed (similar to GenPlan at lines 624-630)
            # ReactPrompt returns success=True but with needs_clarification in data
            if result.data and result.data.get("needs_clarification"):
                logger.info(
                    "planner_v2.react_prompt.clarification_needed",
                    missing_fields=result.data.get("missing_fields", []),
                    question_count=len(result.data.get("questions", [])),
                )
                return {
                    "success": False,
                    "needs_clarification": True,
                    "data": result.data,
                }

            # Prompt plan stored in context.metadata["prompt_plan"]
            return {"success": True, "data": result.data}

        except Exception as e:
            logger.error("planner_v2.react_prompt.error", error_detail=str(e), exc_info=True)
            return {"success": False, "error": str(e)}

    async def _delegate_to_evaluator(
        self,
        context: AgentContext,
        action: str,
    ) -> Dict[str, Any]:
        """Delegate evaluation to EvaluatorAgentV2."""
        from palet8_agents.agents.evaluator_agent_v2 import EvaluatorAgentV2

        phase = "create_plan" if action == "pre_evaluation" else "execute"
        image_data = context.metadata.get("generated_images", [None])[0] if action == "post_evaluation" else None

        logger.info(
            "planner_v2.delegate.evaluator",
            phase=phase,
            has_image=image_data is not None,
        )

        try:
            async with EvaluatorAgentV2() as evaluator:
                result = await evaluator.run(
                    context=context,
                    phase=phase,
                    image_data=image_data,
                )

            if result.success and result.data:
                decision = result.data.get("decision", "PASS")
                context.metadata[f"{action}_result"] = result.data

                if decision == "FIX_REQUIRED":
                    context.metadata["evaluation_feedback"] = result.data.get("feedback", {})
                    return {"success": False, "data": result.data, "decision": "FIX_REQUIRED"}

                if decision == "POLICY_FAIL":
                    return {"success": False, "data": result.data, "decision": "POLICY_FAIL"}

                if decision == "REJECT":
                    return {
                        "success": False,
                        "data": result.data,
                        "decision": "REJECT",
                        "should_retry": result.data.get("should_retry", False),
                    }

            return {"success": True, "data": result.data if result.success else None}

        except Exception as e:
            logger.error("planner_v2.evaluator.error", error_detail=str(e), exc_info=True)
            return {"success": False, "error": str(e)}

    async def _execute_generation(self, context: AgentContext) -> Dict[str, Any]:
        """Execute generation via AssemblyService."""
        from palet8_agents.services.assembly_service import AssemblyService

        generation_plan_data = context.metadata.get("generation_plan", {})
        prompt_plan_data = context.metadata.get("prompt_plan", {})

        if not prompt_plan_data:
            return {"success": False, "error": "No prompt_plan available"}

        # Build AssemblyRequest from GenerationPlan + PromptPlan
        assembly_request = self._build_assembly_request(context)

        logger.info(
            "planner_v2.generation.start",
            job_id=context.job_id,
            model_id=assembly_request.model_id,
            pipeline_type=assembly_request.pipeline.pipeline_type,
        )

        try:
            async def generation_progress(stage: str, progress: float, message: str = None):
                await self._emit_progress(f"generation_{stage}", 0.4 + (progress * 0.4))

            async with AssemblyService() as service:
                result = await service.execute(
                    request=assembly_request,
                    progress_callback=generation_progress,
                )

            if not result.success:
                logger.error("planner_v2.generation.failed", err_msg=result.error)
                return {"success": False, "error": result.error}

            # Store images in context
            images = []
            for img in result.images:
                img_dict = {
                    "url": img.url,
                    "base64_data": img.base64_data,
                    "width": img.width,
                    "height": img.height,
                }
                images.append(img_dict)

            context.metadata["generated_images"] = images

            logger.info("planner_v2.generation.complete", images_count=len(images))
            return {"success": True, "data": {"images": images}}

        except Exception as e:
            logger.error("planner_v2.generation.error", error_detail=str(e), exc_info=True)
            return {"success": False, "error": str(e)}

    # =========================================================================
    # Failure Handling
    # =========================================================================

    async def _handle_checkpoint_failure(
        self,
        context: AgentContext,
        checkpoint: Dict[str, Any],
        result: Dict[str, Any],
        on_fail: str,
        retry_count: int,
    ) -> Dict[str, Any]:
        """Handle checkpoint failure based on on_fail action."""
        checkpoint_id = checkpoint.get("id")

        logger.warning(
            "planner_v2.checkpoint.failed",
            checkpoint_id=checkpoint_id,
            on_fail=on_fail,
            retry_count=retry_count,
        )

        if on_fail == "request_clarification" or result.get("needs_clarification"):
            # Track clarification rounds
            clarification_round = context.metadata.get("clarification_round", 0) + 1
            context.metadata["clarification_round"] = clarification_round

            logger.info(
                "planner_v2.clarification.round",
                checkpoint_id=checkpoint_id,
                round=clarification_round,
                max_rounds=self.max_clarification_rounds,
            )

            # Check if max rounds exceeded
            if clarification_round > self.max_clarification_rounds:
                logger.warning(
                    "planner_v2.clarification.max_rounds_exceeded",
                    checkpoint_id=checkpoint_id,
                    rounds=clarification_round,
                )
                # Proceed with best effort using defaults
                return await self._apply_clarification_defaults(
                    context, checkpoint, result
                )

            # Route clarification through Pali
            questions = result.get("data", {}).get("questions", [])
            missing_fields = result.get("data", {}).get("missing_fields", [])
            clarification_request = result.get("data", {}).get("clarification_request")

            # Fallback: extract field from clarification_request if missing_fields is empty
            if not missing_fields and clarification_request:
                field = clarification_request.get("field")
                if field:
                    missing_fields = [field]
                    logger.info(
                        "planner_v2.clarification.missing_fields_fallback",
                        field=field,
                        source="clarification_request",
                    )

            # Phase 5 Debug: Log clarification return details
            logger.info(
                "planner_v2.clarification.returning_to_pali",
                checkpoint_id=checkpoint_id,
                round=clarification_round,
                question_count=len(questions),
                missing_fields=missing_fields,
                has_clarification_request=clarification_request is not None,
            )

            return {
                "action": "return",
                "result": self._create_result(
                    success=False,
                    data={
                        "action": "needs_clarification",
                        "round": clarification_round,
                        "max_rounds": self.max_clarification_rounds,
                        "questions": questions,
                        "missing_fields": missing_fields,
                        "clarification_request": clarification_request,
                    },
                    error="Clarification needed",
                    error_code="CLARIFICATION_NEEDED",
                ),
            }

        elif on_fail == "block":
            return {
                "action": "return",
                "result": self._create_result(
                    success=False,
                    data=result.get("data"),
                    error=result.get("error", "Blocked"),
                    error_code="BLOCKED",
                ),
            }

        elif on_fail == "retry" or on_fail == "retry_prompt_build":
            max_retries = checkpoint.get("max_retries", 3)
            if retry_count < max_retries:
                jump_to = "prompt_build" if on_fail == "retry_prompt_build" else None
                return {"action": "retry", "jump_to": jump_to}
            else:
                return {
                    "action": "return",
                    "result": self._create_result(
                        success=False,
                        data=None,
                        error=f"Max retries ({max_retries}) exceeded",
                        error_code="MAX_RETRIES_EXCEEDED",
                    ),
                }

        elif on_fail == "retry_generation":
            max_retries = checkpoint.get("max_retries", 3)
            if retry_count < max_retries and result.get("should_retry"):
                return {"action": "retry", "jump_to": "execute_generation"}
            else:
                return {"action": "continue"}  # Accept with what we have

        elif on_fail == "accept_with_warning":
            logger.warning("planner_v2.accepting_with_warning", checkpoint_id=checkpoint_id)
            return {"action": "continue"}

        # Default: return error
        return {
            "action": "return",
            "result": self._create_result(
                success=False,
                data=result.get("data"),
                error=result.get("error", "Checkpoint failed"),
                error_code="CHECKPOINT_FAILED",
            ),
        }

    async def _apply_clarification_defaults(
        self,
        context: AgentContext,
        checkpoint: Dict[str, Any],
        result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Apply defaults after max clarification rounds exceeded.

        When user has not provided sufficient context after multiple rounds,
        proceed with best effort using reasonable defaults.
        """
        checkpoint_id = checkpoint.get("id")
        missing_fields = result.get("data", {}).get("missing_fields", [])

        logger.info(
            "planner_v2.clarification.applying_defaults",
            checkpoint_id=checkpoint_id,
            missing_fields=missing_fields,
        )

        requirements = context.requirements or {}

        # Define default values for missing fields
        field_defaults = {
            "style": "photorealistic",
            "mood": "neutral",
            "colors": [],
            "product_type": "general",
            "dimensions": {"width": 1024, "height": 1024},
            "character": None,
            "reference_image": None,
            "text_content": None,
        }

        # Apply defaults for missing fields
        defaults_applied = []
        for field in missing_fields:
            if field in field_defaults and field not in requirements:
                requirements[field] = field_defaults[field]
                defaults_applied.append(field)

        context.requirements = requirements

        logger.info(
            "planner_v2.clarification.defaults_applied",
            checkpoint_id=checkpoint_id,
            defaults_applied=defaults_applied,
        )

        # Mark that we proceeded with defaults (for quality warnings)
        context.metadata["proceeded_with_defaults"] = True
        context.metadata["defaults_applied"] = defaults_applied

        # Continue to next checkpoint (retry current one with defaults)
        return {"action": "continue"}

    # =========================================================================
    # Internal Checks
    # =========================================================================

    async def _evaluate_context_completeness(
        self,
        requirements: Dict[str, Any],
    ) -> ContextCompleteness:
        """Evaluate if requirements provide sufficient context."""
        service = await self._get_context_analysis_service()
        return service.evaluate_completeness(requirements)

    def _is_context_sufficient(self, completeness: ContextCompleteness) -> bool:
        """Check if context is sufficient to proceed."""
        return completeness.is_sufficient and completeness.score >= self.min_context_completeness

    async def _classify_safety(
        self,
        requirements: Dict[str, Any],
    ) -> SafetyClassification:
        """Classify content safety."""
        service = await self._get_safety_classification_service()

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

    # =========================================================================
    # Assembly Request Builder
    # =========================================================================

    def _build_assembly_request(self, context: AgentContext) -> AssemblyRequest:
        """Build AssemblyRequest from GenerationPlan + PromptPlan."""
        requirements = context.requirements or {}
        generation_plan_data = context.metadata.get("generation_plan", {})
        prompt_plan_data = context.metadata.get("prompt_plan", {})
        safety_data = context.metadata.get("safety", {})

        prompt_plan = PromptPlan.from_dict(prompt_plan_data)

        # Get values from GenerationPlan (from GenPlanAgent)
        # Model should come from GenPlan - no hardcoded fallback
        model_id = generation_plan_data.get("model_id")
        if not model_id:
            logger.warning("planner.assembly.no_model_id - GenPlan should have provided model_id")
        model_rationale = generation_plan_data.get("model_rationale", "")
        model_alternatives = generation_plan_data.get("model_alternatives", [])
        model_input_params = generation_plan_data.get("model_input_params", {})
        provider_params = generation_plan_data.get("provider_params", {})
        pipeline_data = generation_plan_data.get("pipeline", {})

        # Build PipelineConfig
        pipeline = PipelineConfig(
            pipeline_type=pipeline_data.get("pipeline_type", "single"),
            pipeline_name=pipeline_data.get("pipeline_name", "single_standard"),
            stage_1_model=pipeline_data.get("stage_1_model", model_id),
            stage_1_purpose=pipeline_data.get("stage_1_purpose", "Generate final image"),
            stage_2_model=pipeline_data.get("stage_2_model"),
            stage_2_purpose=pipeline_data.get("stage_2_purpose"),
        )

        # Build GenerationParameters
        # DYNAMIC PARAMETER HANDLING:
        # - Extract core dimensions (width, height, num_images) explicitly
        # - Pass ALL other params from model_input_params through provider_settings
        # - GenPlan determines which params are supported based on model config
        # - This allows flexibility for new parameters without code changes

        # Core params that are always present
        width = model_input_params.get("width", requirements.get("width", 1024))
        height = model_input_params.get("height", requirements.get("height", 1024))
        num_images = model_input_params.get("num_images", 1)
        seed = model_input_params.get("seed")

        # Dynamic params - extract steps and cfg_scale (GenPlan uses cfg_scale, not guidance_scale)
        steps = model_input_params.get("steps")
        guidance_scale = model_input_params.get("cfg_scale") or model_input_params.get("guidance_scale")

        # Merge all remaining model_input_params with provider_params for flexibility
        # This ensures any new params from model config flow through
        all_provider_settings = {**provider_params}

        # Add any extra model_input_params that aren't core fields
        core_fields = {"width", "height", "num_images", "seed", "steps", "cfg_scale", "guidance_scale"}
        for key, value in model_input_params.items():
            if key not in core_fields and value is not None:
                all_provider_settings[key] = value

        gen_params = GenerationParameters(
            width=width,
            height=height,
            steps=steps,
            guidance_scale=guidance_scale,
            seed=seed,
            num_images=num_images,
            provider_settings=all_provider_settings,
        )

        # Phase 5 Debug: Log parameter flow from GenPlan -> Planner
        logger.info(
            "planner_v2.build_assembly.params_debug",
            job_id=context.job_id,
            model_id=model_id,
            genplan_model_input_params=model_input_params,
            genplan_provider_params=provider_params,
            final_steps=steps,
            final_guidance_scale=guidance_scale,
            final_num_images=num_images,
            final_width=width,
            final_height=height,
            provider_settings_keys=list(all_provider_settings.keys()) if all_provider_settings else [],
        )

        # Build safety
        safety = SafetyClassification.from_dict(safety_data) if safety_data else SafetyClassification(
            is_safe=True,
            requires_review=False,
            risk_level="low",
            categories=[],
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
            safety=safety,
            context_used=prompt_plan.context_summary.to_dict() if prompt_plan.context_summary else {},
            job_id=context.job_id,
            user_id=context.user_id,
            product_type=requirements.get("product_type", "general"),
            print_method=requirements.get("print_method"),
            revision_count=prompt_plan.revision_count,
        )

    # =========================================================================
    # Progress & Utilities
    # =========================================================================

    async def _emit_progress(self, stage: str, progress: float):
        """Emit progress update via Pali callback."""
        logger.info("planner_v2.progress", stage=stage, progress=progress)
        if self._pali_callback:
            try:
                await self._pali_callback("progress", {
                    "stage": stage,
                    "progress": progress,
                })
            except Exception as e:
                logger.debug(f"Progress callback failed: {e}")

    # =========================================================================
    # Legacy run() method for backwards compatibility
    # =========================================================================

    async def run(
        self,
        context: AgentContext,
        user_input: Optional[str] = None,
        phase: str = "initial",
        evaluation_feedback: Optional[Dict[str, Any]] = None,
    ) -> AgentResult:
        """
        Legacy run method - redirects to orchestrate_generation.

        For new code, use orchestrate_generation() directly.
        """
        if phase == "initial":
            return await self.orchestrate_generation(context)
        else:
            # Handle legacy phase routing
            logger.warning(
                "planner_v2.legacy_run",
                phase=phase,
                message="Use orchestrate_generation() for new code",
            )
            return await self.orchestrate_generation(context)

    # =========================================================================
    # Context Manager
    # =========================================================================

    async def __aenter__(self) -> "PlannerAgentV2":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()
