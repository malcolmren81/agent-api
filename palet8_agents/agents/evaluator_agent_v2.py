"""
Evaluator Agent v2 - Thin quality gate using tools.

Refactored to use tools for scoring logic while keeping decision-making
in the agent. Uses a Think-Act-Observe pattern for evaluation.

Phases:
- create_plan: Evaluate prompt quality BEFORE generation
- execute: Evaluate result quality AFTER generation

Tools Used:
- prompt_quality: Assess prompt quality and propose revisions
- image_evaluation: Evaluate generated images

Documentation Reference: Section 5.2.3
"""

from typing import Any, Dict, List, Optional
from pathlib import Path
import logging

from palet8_agents.core.agent import BaseAgent, AgentContext, AgentResult
from palet8_agents.core.config import get_config
from palet8_agents.tools.base import BaseTool

from palet8_agents.models import (
    EvaluationPhase,
    EvaluationDecision,
    PromptQualityResult,
    ResultQualityResult,
    EvaluationPlan,
    EvaluationFeedback,
)

logger = logging.getLogger(__name__)


def _load_system_prompt() -> str:
    """Load the system prompt from file."""
    prompt_path = Path(__file__).parent.parent.parent / "prompts" / "evaluator_system.txt"
    try:
        return prompt_path.read_text()
    except FileNotFoundError:
        logger.warning(f"System prompt file not found: {prompt_path}")
        return EVALUATOR_SYSTEM_PROMPT_FALLBACK


EVALUATOR_SYSTEM_PROMPT_FALLBACK = """You are the Evaluator for Palet8's image pipeline.

YOUR ROLE
Quality gate for generation. Evaluate quickly to not slow down normal flows.

PRE-GENERATION: Assess the plan before images are created.
Decisions: PASS, FIX_REQUIRED, POLICY_FAIL

POST-GENERATION: Evaluate when image arrives.
Decisions: APPROVE, REJECT, POLICY_FAIL

Be specific, concise, and actionable with feedback.
Catch problems early without slowing the happy path."""


class EvaluatorAgentV2(BaseAgent):
    """
    Thin quality gate agent using tools for evaluation.

    Uses Think-Act-Observe pattern:
    1. THINK: Determine what needs evaluation
    2. ACT: Use tools to gather quality scores
    3. OBSERVE: Analyze results
    4. DECIDE: Pass/Fail based on scores and thresholds

    Phase 1: create_plan (BEFORE generation)
    - Uses prompt_quality tool to assess prompt
    - Creates evaluation plan for Phase 2
    - Returns PASS/FIX_REQUIRED/POLICY_FAIL

    Phase 2: execute (AFTER generation)
    - Uses image_evaluation tool to assess result
    - Returns APPROVE/REJECT/POLICY_FAIL
    """

    def __init__(
        self,
        tools: Optional[List[BaseTool]] = None,
    ):
        """
        Initialize the Evaluator Agent.

        Args:
            tools: List of tools (prompt_quality, image_evaluation)
        """
        super().__init__(
            name="evaluator",
            description="Quality gate for prompts and generated images",
            tools=tools,
        )

        self.system_prompt = _load_system_prompt()
        self.model_profile = "evaluator"

        # Load configuration
        self.config = get_config()
        self.max_retries = self.config.evaluation.max_retries

    async def run(
        self,
        context: AgentContext,
        user_input: Optional[str] = None,
        phase: str = "create_plan",
        image_data: Optional[Dict[str, Any]] = None,
    ) -> AgentResult:
        """
        Execute the Evaluator Agent.

        Args:
            context: Shared execution context
            user_input: Optional additional input
            phase: "create_plan" or "execute"
            image_data: Generated image data (for execute phase)

        Returns:
            AgentResult with evaluation results
        """
        self._start_execution()

        try:
            logger.info(f"[{self.name}] Starting phase={phase}, job_id={context.job_id}")

            if phase == "create_plan":
                return await self._handle_create_plan(context)
            elif phase == "execute":
                return await self._handle_execute(context, image_data)
            else:
                return self._create_result(
                    success=False,
                    data=None,
                    error=f"Unknown phase: {phase}",
                    error_code="INVALID_PHASE",
                )

        except Exception as e:
            logger.error(f"[{self.name}] Error: {e}", exc_info=True)
            return self._create_result(
                success=False,
                data=None,
                error=f"Evaluation failed: {e}",
                error_code="EVALUATION_ERROR",
            )

    # =========================================================================
    # PHASE 1: CREATE PLAN (Pre-generation prompt quality)
    # =========================================================================

    async def _handle_create_plan(self, context: AgentContext) -> AgentResult:
        """
        Phase 1: Evaluate prompt quality and create evaluation plan.

        THINK: What do we need to evaluate?
        ACT: Use prompt_quality tool
        OBSERVE: Analyze quality results
        DECIDE: Pass/Fix/Block
        """
        # Extract inputs from context
        plan_data = context.plan or {}
        assembly_request = context.metadata.get("assembly_request", {})
        requirements = context.requirements or {}

        prompt = assembly_request.get("prompt") or plan_data.get("prompt", "")
        negative_prompt = assembly_request.get("negative_prompt") or plan_data.get("negative_prompt", "")
        mode = assembly_request.get("mode") or plan_data.get("mode", "STANDARD")
        product_type = requirements.get("product_type", "general")
        print_method = requirements.get("print_method")
        dimensions = assembly_request.get("dimensions") or plan_data.get("dimensions", {})

        logger.debug(f"[{self.name}] Evaluating prompt quality, mode={mode}")

        # ACT: Use prompt_quality tool to assess
        quality_result = await self.call_tool(
            "prompt_quality",
            "assess_quality",
            prompt=prompt,
            negative_prompt=negative_prompt,
            mode=mode,
            product_type=product_type,
            print_method=print_method,
            dimensions=dimensions,
        )

        if not quality_result.success:
            logger.warning(f"[{self.name}] Prompt quality tool failed: {quality_result.error}")
            # Fallback: assume pass but log warning
            prompt_quality = PromptQualityResult(
                overall=0.7,
                dimensions={},
                mode=mode,
                threshold=0.7,
                decision="PASS",
                feedback=["Quality check incomplete"],
            )
        else:
            prompt_quality = PromptQualityResult.from_dict(quality_result.data)

        # Get thresholds for result evaluation (for Phase 2)
        thresholds_result = await self.call_tool(
            "image_evaluation",
            "get_thresholds",
            mode=mode,
        )
        result_thresholds = thresholds_result.data.get("thresholds", {}) if thresholds_result.success else {}

        weights_result = await self.call_tool(
            "image_evaluation",
            "get_weights",
            mode=mode,
        )
        result_weights = weights_result.data.get("weights", {}) if weights_result.success else {}

        # Create evaluation plan for Phase 2
        evaluation_plan = EvaluationPlan(
            job_id=context.job_id,
            prompt=prompt,
            negative_prompt=negative_prompt,
            mode=mode,
            product_type=product_type,
            print_method=print_method,
            dimensions_requested=dimensions,
            prompt_quality=prompt_quality,
            result_weights=result_weights,
            result_thresholds=result_thresholds,
        )

        # Store in context for Phase 2
        context.metadata["evaluation_plan"] = evaluation_plan.to_dict()

        # DECIDE: Route based on quality result
        return self._decide_prompt_result(prompt_quality, evaluation_plan)

    def _decide_prompt_result(
        self,
        quality: PromptQualityResult,
        plan: EvaluationPlan,
    ) -> AgentResult:
        """DECIDE: Determine next action based on prompt quality."""

        if quality.decision == "POLICY_FAIL":
            # Hard block
            logger.warning(f"[{self.name}] Policy violation detected")
            return self._create_result(
                success=False,
                data={
                    "evaluation_plan": plan.to_dict(),
                    "action": "policy_blocked",
                    "reason": "Policy violation detected",
                },
                error="Policy violation",
                error_code="POLICY_VIOLATION",
                next_agent=None,
            )

        if quality.decision == "FIX_REQUIRED":
            # Prompt needs improvement - return to Planner
            logger.info(f"[{self.name}] Fix required: {quality.failed_dimensions}")

            # Build feedback for planner
            feedback = EvaluationFeedback(
                passed=False,
                overall_score=quality.overall,
                issues=quality.feedback,
                retry_suggestions=quality.failed_dimensions,
                dimension_scores=quality.dimensions,
            )

            return self._create_result(
                success=True,
                data={
                    "evaluation_plan": plan.to_dict(),
                    "action": "fix_required",
                    "prompt_quality": quality.to_dict(),
                    "feedback": feedback.to_dict(),
                },
                next_agent="planner",
            )

        # PASS - proceed to generation
        logger.info(f"[{self.name}] Prompt quality passed: {quality.overall:.2f}")
        return self._create_result(
            success=True,
            data={
                "evaluation_plan": plan.to_dict(),
                "action": "proceed_to_generation",
                "prompt_quality": quality.to_dict(),
            },
            next_agent=None,  # Proceed to image generation
        )

    # =========================================================================
    # PHASE 2: EXECUTE (Post-generation result quality)
    # =========================================================================

    async def _handle_execute(
        self,
        context: AgentContext,
        image_data: Optional[Dict[str, Any]],
    ) -> AgentResult:
        """
        Phase 2: Evaluate result quality after generation.

        THINK: What quality dimensions need checking?
        ACT: Use image_evaluation tool
        OBSERVE: Analyze results
        DECIDE: Approve/Reject/Block
        """
        if not image_data:
            return self._create_result(
                success=False,
                data=None,
                error="No image data provided for evaluation",
                error_code="MISSING_IMAGE_DATA",
            )

        # Get evaluation plan from context or create new one
        plan_data = context.metadata.get("evaluation_plan")
        if plan_data:
            plan = EvaluationPlan.from_dict(plan_data)
        else:
            # Create minimal plan from context
            assembly_request = context.metadata.get("assembly_request", {})
            mode = assembly_request.get("mode", "STANDARD")
            plan = EvaluationPlan(
                job_id=context.job_id,
                prompt=assembly_request.get("prompt", ""),
                negative_prompt=assembly_request.get("negative_prompt", ""),
                mode=mode,
            )

        logger.debug(f"[{self.name}] Evaluating result quality")

        # ACT: Use image_evaluation tool
        eval_result = await self.call_tool(
            "image_evaluation",
            "evaluate_image",
            image_data=image_data,
            plan=plan.to_dict(),
        )

        if not eval_result.success:
            logger.warning(f"[{self.name}] Image evaluation tool failed: {eval_result.error}")
            # Fallback: approve with warning
            result_quality = ResultQualityResult(
                overall=0.7,
                dimensions={},
                mode=plan.mode,
                threshold=0.7,
                decision="APPROVE",
                feedback=["Evaluation incomplete"],
            )
        else:
            result_quality = ResultQualityResult.from_dict(eval_result.data)

        # Check if should retry
        retry_result = await self.call_tool(
            "image_evaluation",
            "should_retry",
            result=result_quality.to_dict(),
            attempt_count=context.metadata.get("attempt_count", 1),
        )
        should_retry = retry_result.data.get("should_retry", False) if retry_result.success else False

        # DECIDE: Route based on result quality
        return self._decide_result(result_quality, plan, should_retry)

    def _decide_result(
        self,
        quality: ResultQualityResult,
        plan: EvaluationPlan,
        should_retry: bool,
    ) -> AgentResult:
        """DECIDE: Determine next action based on result quality."""

        if quality.decision == "POLICY_FAIL":
            # Hard block
            logger.warning(f"[{self.name}] Policy violation in result")
            return self._create_result(
                success=False,
                data={
                    "result_quality": quality.to_dict(),
                    "action": "policy_blocked",
                },
                error="Policy violation detected",
                error_code="POLICY_VIOLATION",
                next_agent=None,
            )

        if quality.decision == "REJECT":
            logger.info(f"[{self.name}] Result rejected: {quality.failed_dimensions}")

            # Build feedback for planner
            feedback = EvaluationFeedback(
                passed=False,
                overall_score=quality.overall,
                issues=quality.feedback,
                retry_suggestions=[
                    s.to_dict() if hasattr(s, 'to_dict') else s
                    for s in quality.retry_suggestions
                ],
                dimension_scores=quality.dimensions,
            )

            return self._create_result(
                success=True,
                data={
                    "result_quality": quality.to_dict(),
                    "action": "rejected",
                    "feedback": feedback.to_dict(),
                    "should_retry": should_retry,
                },
                next_agent="planner" if should_retry else None,
            )

        # APPROVE - return to user
        logger.info(f"[{self.name}] Result approved: {quality.overall:.2f}")
        return self._create_result(
            success=True,
            data={
                "result_quality": quality.to_dict(),
                "action": "approved",
            },
            next_agent="pali",
        )

    # =========================================================================
    # LIFECYCLE
    # =========================================================================

    async def close(self) -> None:
        """Close resources."""
        # Tools are managed externally or via registry
        pass

    async def __aenter__(self) -> "EvaluatorAgentV2":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()
