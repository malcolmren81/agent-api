"""
ReactPromptAgent - ReAct-style agent for prompt building.

This agent uses a Think-Act-Observe loop to build high-quality
image generation prompts by gathering context, selecting dimensions,
and iteratively refining until quality thresholds are met.

Phases:
- initial: Full context + prompt build from scratch
- fix_plan: Seeded with previous prompt + EvaluationFeedback, minimal fixes
- edit: Seeded with existing plan + user edit instructions, preserve as much as possible

Documentation Reference: Section 5.2.2
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import logging

from palet8_agents.core.agent import BaseAgent, AgentContext, AgentResult
from palet8_agents.tools.base import BaseTool, ToolResult
from palet8_agents.models.planning import PlanningTask, PromptPlan, ContextSummary
from palet8_agents.models.prompt import PromptDimensions, PromptQualityResult
from palet8_agents.services.text_llm_service import TextLLMService, TextLLMServiceError

logger = logging.getLogger(__name__)


class ReactAction(Enum):
    """Actions available in the ReAct loop."""
    BUILD_CONTEXT = "build_context"
    SELECT_DIMENSIONS = "select_dimensions"
    COMPOSE_PROMPT = "compose_prompt"
    EVALUATE_PROMPT = "evaluate_prompt"
    REFINE_PROMPT = "refine_prompt"
    DONE = "done"


@dataclass
class PromptState:
    """Internal state for the ReAct loop."""
    # Context information
    user_history: List[Dict[str, Any]] = field(default_factory=list)
    art_references: List[Dict[str, Any]] = field(default_factory=list)
    web_results: List[Dict[str, Any]] = field(default_factory=list)
    web_answer: Optional[str] = None  # AI-generated answer from web search
    rag_sources: List[str] = field(default_factory=list)

    # Prompt building state
    dimensions: Optional[PromptDimensions] = None
    prompt: str = ""
    negative_prompt: str = ""

    # Provider-specific generation parameters
    # Passed through to the final generation (e.g., steps, guidance_scale, sampler)
    provider_params: Dict[str, Any] = field(default_factory=dict)

    # Quality state
    quality: Optional[PromptQualityResult] = None
    revision_count: int = 0
    revision_history: List[str] = field(default_factory=list)

    # Control flags
    goal_satisfied: bool = False
    mode: str = "STANDARD"

    @property
    def has_context(self) -> bool:
        """Check if context has been gathered."""
        return bool(self.user_history or self.art_references or self.rag_sources)

    @property
    def has_dimensions(self) -> bool:
        """Check if dimensions have been selected."""
        return self.dimensions is not None

    @property
    def has_prompt(self) -> bool:
        """Check if prompt has been composed."""
        return bool(self.prompt)

    @property
    def has_quality(self) -> bool:
        """Check if quality has been evaluated."""
        return self.quality is not None

    def get_context_summary(self) -> ContextSummary:
        """Build context summary from state."""
        return ContextSummary(
            user_history_count=len(self.user_history),
            art_references_count=len(self.art_references),
            web_search_count=len(self.web_results),
            rag_sources=self.rag_sources,
            reference_images=[
                ref.get("image_url", "")
                for ref in self.art_references
                if ref.get("image_url")
            ],
            metadata={
                "web_search_used": len(self.web_results) > 0,
                "has_web_answer": self.web_answer is not None,
            },
        )


# Load system prompt from file
def _load_system_prompt() -> str:
    """Load the system prompt from file."""
    prompt_path = Path(__file__).parent.parent.parent / "prompts" / "react_prompt_system.txt"
    try:
        return prompt_path.read_text()
    except FileNotFoundError:
        logger.warning(f"System prompt file not found: {prompt_path}")
        return "You are a prompt building agent. Build high-quality image generation prompts."


REACT_PROMPT_SYSTEM = _load_system_prompt()


class ReactPromptAgent(BaseAgent):
    """
    ReAct agent for building and refining prompts.

    Uses a Think-Act-Observe loop to:
    1. Gather context (user history, art references, RAG)
    2. Select dimensions based on mode and requirements
    3. Compose prompt using templates and LLM
    4. Evaluate quality and refine if needed
    5. Return PromptPlan when quality is acceptable
    """

    def __init__(
        self,
        tools: Optional[List[BaseTool]] = None,
        text_service: Optional[TextLLMService] = None,
        prompt_composer_service: Optional[Any] = None,
        prompt_template_service: Optional[Any] = None,
    ):
        """
        Initialize the ReactPromptAgent.

        Args:
            tools: List of tools (context, dimension, prompt_quality)
            text_service: Optional TextLLMService for LLM calls
            prompt_composer_service: Optional PromptComposerService instance
            prompt_template_service: Optional PromptTemplateService instance
        """
        super().__init__(
            name="react_prompt",
            description="Builds optimal prompts using context, dimensions, and iterative refinement",
            tools=tools,
        )

        self._text_service = text_service
        self._prompt_composer = prompt_composer_service
        self._prompt_template = prompt_template_service
        self._owns_services = text_service is None

        self.system_prompt = REACT_PROMPT_SYSTEM
        # Uses react_prompt profile (same config as planner for consistent quality)
        self.model_profile = "react_prompt"
        self.max_steps = 10
        self.max_revisions = 3

    async def _get_text_service(self) -> TextLLMService:
        """Get or create text LLM service."""
        if self._text_service is None:
            self._text_service = TextLLMService(default_profile_name=self.model_profile)
        return self._text_service

    async def _get_prompt_composer(self):
        """Get or create prompt composer service."""
        if self._prompt_composer is None:
            from palet8_agents.services.prompt_composer_service import PromptComposerService
            # Share text service with prompt composer for efficiency
            text_service = await self._get_text_service()
            self._prompt_composer = PromptComposerService(text_service=text_service)
        return self._prompt_composer

    async def _get_prompt_template(self):
        """Get or create prompt template service."""
        if self._prompt_template is None:
            from palet8_agents.services.prompt_template_service import PromptTemplateService
            self._prompt_template = PromptTemplateService()
        return self._prompt_template

    async def close(self) -> None:
        """Close resources."""
        if self._prompt_composer and self._owns_services:
            if hasattr(self._prompt_composer, 'close'):
                await self._prompt_composer.close()
            self._prompt_composer = None
        if self._prompt_template and self._owns_services:
            if hasattr(self._prompt_template, 'close'):
                await self._prompt_template.close()
            self._prompt_template = None
        if self._text_service and self._owns_services:
            await self._text_service.close()
            self._text_service = None

    async def run(
        self,
        context: AgentContext,
        user_input: Optional[str] = None,
    ) -> AgentResult:
        """
        Execute the ReAct loop to build a PromptPlan.

        Args:
            context: Shared execution context containing planning_task
            user_input: Optional user input (not typically used)

        Returns:
            AgentResult with PromptPlan data
        """
        self._start_execution()

        try:
            # Extract PlanningTask from context
            planning_task_data = context.metadata.get("planning_task")
            if not planning_task_data:
                return self._create_result(
                    success=False,
                    data=None,
                    error="No planning_task found in context",
                    error_code="MISSING_TASK",
                )

            task = PlanningTask.from_dict(planning_task_data)
            logger.info(f"[{self.name}] Starting ReAct loop for job={task.job_id}, phase={task.phase}")

            # Initialize state
            state = self._init_state(task)
            steps = 0

            # ReAct loop
            while not state.goal_satisfied and steps < self.max_steps:
                # THINK: Decide next action
                next_action = await self._think(state, task)
                logger.debug(f"[{self.name}] Step {steps + 1}: Action={next_action.value}")

                # ACT: Execute the action
                if next_action == ReactAction.BUILD_CONTEXT:
                    state = await self._build_context(task, state)
                elif next_action == ReactAction.SELECT_DIMENSIONS:
                    state = await self._select_dimensions(task, state)
                elif next_action == ReactAction.COMPOSE_PROMPT:
                    state = await self._compose_prompt(task, state)
                elif next_action == ReactAction.EVALUATE_PROMPT:
                    state = await self._evaluate_prompt(task, state)
                elif next_action == ReactAction.REFINE_PROMPT:
                    state = await self._refine_prompt(task, state)
                elif next_action == ReactAction.DONE:
                    state.goal_satisfied = True

                # OBSERVE: Update goal status
                steps += 1
                if not state.goal_satisfied:
                    state.goal_satisfied = self._check_goal(state)

            # Build result
            prompt_plan = PromptPlan(
                prompt=state.prompt,
                negative_prompt=state.negative_prompt,
                dimensions=state.dimensions.to_dict() if state.dimensions else {},
                provider_params=state.provider_params,
                quality_score=state.quality.overall if state.quality else 0.0,
                quality_acceptable=(
                    state.quality.decision == "PASS" if state.quality else False
                ),
                quality_feedback=state.quality.feedback if state.quality else [],
                failed_dimensions=state.quality.failed_dimensions if state.quality else [],
                revision_count=state.revision_count,
                revision_history=state.revision_history,
                context_summary=state.get_context_summary(),
                mode=state.mode,
            )

            # Store in context for next agent
            context.metadata["prompt_plan"] = prompt_plan.to_dict()

            logger.info(
                f"[{self.name}] Completed: quality={prompt_plan.quality_score:.2f}, "
                f"acceptable={prompt_plan.quality_acceptable}, steps={steps}"
            )

            return self._create_result(
                success=True,
                data={"prompt_plan": prompt_plan.to_dict()},
                next_agent="planner",
            )

        except Exception as e:
            logger.error(f"[{self.name}] Error in ReAct loop: {e}", exc_info=True)
            return self._create_result(
                success=False,
                data=None,
                error=str(e),
                error_code="REACT_ERROR",
            )

    def _init_state(self, task: PlanningTask) -> PromptState:
        """Initialize state based on task phase."""
        state = PromptState()
        state.mode = task.complexity.upper() if task.complexity else "STANDARD"

        # Extract provider_params from requirements if specified
        state.provider_params = task.requirements.get("provider_params", {})

        # For fix_plan or edit, seed with previous data
        if task.previous_plan:
            state.prompt = task.previous_plan.get("prompt", "")
            state.negative_prompt = task.previous_plan.get("negative_prompt", "")
            dims_data = task.previous_plan.get("dimensions", {})
            if dims_data:
                state.dimensions = PromptDimensions.from_dict(dims_data)
            # Preserve provider_params from previous plan if not overridden
            if not state.provider_params and task.previous_plan.get("provider_params"):
                state.provider_params = task.previous_plan.get("provider_params", {})

        return state

    async def _think(self, state: PromptState, task: PlanningTask) -> ReactAction:
        """
        Decide next action based on current state and task phase.

        The thinking logic varies by phase:
        - initial: Full pipeline from context to quality
        - fix_plan: Focus on evaluation and refinement
        - edit: Minimal changes to preserve existing prompt
        """
        # For fix_plan phase, prioritize evaluation and refinement
        if task.is_fix:
            if not state.has_quality:
                return ReactAction.EVALUATE_PROMPT
            if state.quality and state.quality.decision != "PASS":
                if state.revision_count < self.max_revisions:
                    return ReactAction.REFINE_PROMPT
            return ReactAction.DONE

        # For edit phase, minimal changes
        if task.is_edit:
            if not state.has_dimensions:
                return ReactAction.SELECT_DIMENSIONS
            if not state.has_prompt:
                return ReactAction.COMPOSE_PROMPT
            if not state.has_quality:
                return ReactAction.EVALUATE_PROMPT
            return ReactAction.DONE

        # For initial phase, full pipeline
        if not state.has_context:
            return ReactAction.BUILD_CONTEXT
        if not state.has_dimensions:
            return ReactAction.SELECT_DIMENSIONS
        if not state.has_prompt:
            return ReactAction.COMPOSE_PROMPT
        if not state.has_quality:
            return ReactAction.EVALUATE_PROMPT
        if state.quality and state.quality.decision != "PASS":
            if state.revision_count < self.max_revisions:
                return ReactAction.REFINE_PROMPT
        return ReactAction.DONE

    # Threshold for determining if RAG context is sufficient
    MIN_CONTEXT_ITEMS = 2  # Minimum combined references/prompts before triggering web search

    async def _build_context(self, task: PlanningTask, state: PromptState) -> PromptState:
        """
        Build context using RAG, memory, and conditional web search.

        Flow:
        1. First get context from memory (user history, art references)
        2. Check if context is sufficient for prompt composition
        3. If NOT sufficient → supplement with web search
        4. Return combined context
        """
        logger.debug(f"[{self.name}] Building context for user={task.user_id}")

        # Step 1: Get user history from memory
        result = await self.call_tool(
            "memory",
            "get_history",
            user_id=task.user_id,
            limit=10,
        )
        if result.success and result.data:
            state.user_history = result.data.get("history", [])

        # Step 2: Get art references if subject specified
        subject = task.requirements.get("subject", "")
        if subject:
            result = await self.call_tool(
                "memory",
                "get_references",
                query=subject,
                limit=5,
            )
            if result.success and result.data:
                state.art_references = result.data.get("references", [])

        # Step 3: Check if RAG context is sufficient
        if self._is_rag_context_sufficient(task, state):
            logger.debug(f"[{self.name}] RAG context sufficient, skipping web search")
            return state

        # Step 4: RAG context insufficient - supplement with web search
        logger.info(f"[{self.name}] RAG context insufficient, supplementing with web search")
        state = await self._supplement_with_web_search(task, state)

        return state

    def _is_rag_context_sufficient(self, task: PlanningTask, state: PromptState) -> bool:
        """
        Determine if RAG context provides enough info for prompt composition.

        Context is considered sufficient when:
        - We have art references OR user history to draw from
        - OR the request is well-specified with enough detail fields
        """
        # Count available context items
        art_refs_count = len(state.art_references)
        history_count = len(state.user_history)
        total_refs = art_refs_count + history_count

        # If we have enough references, context is sufficient
        if total_refs >= self.MIN_CONTEXT_ITEMS:
            return True

        # If user provided detailed requirements, may not need more context
        requirements = task.requirements
        detail_fields = ["style", "aesthetic", "composition", "background", "mood", "colors"]
        provided_details = sum(1 for f in detail_fields if requirements.get(f))

        # Well-specified requests (3+ detail fields) don't need extra search
        if provided_details >= 3:
            return True

        # SIMPLE mode requests typically don't need web search
        if task.complexity == "simple":
            return True

        # Otherwise, we need more context
        return False

    async def _supplement_with_web_search(
        self,
        task: PlanningTask,
        state: PromptState,
    ) -> PromptState:
        """
        Supplement context with web search results.

        Searches for style references, design trends, and inspiration.
        """
        requirements = task.requirements

        # Build search query from requirements
        subject = requirements.get("subject", "")
        style = requirements.get("style", "")
        product_type = task.product_type or ""

        query_parts = [subject]
        if style:
            query_parts.append(style)
        if product_type and product_type != "general":
            query_parts.append(f"{product_type} design")

        query = " ".join(query_parts)
        if not query.strip():
            return state

        # Add design/art context to query
        search_query = f"{query} design style reference"

        try:
            result = await self.call_tool(
                "search",
                "search",
                query=search_query,
                max_results=3,
                include_answer=True,
            )

            if result.success and result.data:
                # Store web search results
                state.web_results = result.data.get("results", [])
                state.web_answer = result.data.get("answer")

                # Add to RAG sources for tracking
                state.rag_sources.append(f"web_search:{result.data.get('provider', 'unknown')}")

                logger.debug(
                    f"[{self.name}] Web search returned {len(state.web_results)} results"
                )

        except Exception as e:
            logger.warning(f"[{self.name}] Web search failed: {e}")

        return state

    async def _select_dimensions(self, task: PlanningTask, state: PromptState) -> PromptState:
        """Select dimensions using DimensionTool."""
        logger.debug(f"[{self.name}] Selecting dimensions for mode={state.mode}")

        result = await self.call_tool(
            "dimension",
            "select_dimensions",
            mode=state.mode,
            requirements=task.requirements,
            product_type=task.product_type,
            print_method=task.print_method,
        )

        if result.success and result.data:
            state.dimensions = PromptDimensions.from_dict(result.data)
        else:
            # Fallback to basic dimensions from requirements
            state.dimensions = PromptDimensions(
                subject=task.requirements.get("subject", ""),
                aesthetic=task.requirements.get("style"),
                mood=task.requirements.get("mood"),
            )
            if task.requirements.get("colors"):
                state.dimensions.color = ", ".join(task.requirements["colors"])

        return state

    async def _compose_prompt(self, task: PlanningTask, state: PromptState) -> PromptState:
        """Compose prompt using PromptComposerService."""
        logger.debug(f"[{self.name}] Composing prompt")

        try:
            composer = await self._get_prompt_composer()

            # Build context dict for composer
            context_dict = {
                "user_history": state.user_history[:3],  # Limit for token efficiency
                "art_references": state.art_references[:3],
                "requirements": task.requirements,
            }

            # Include web search results if available
            if state.web_results:
                context_dict["web_references"] = [
                    {
                        "title": r.get("title", ""),
                        "snippet": r.get("snippet", ""),
                        "url": r.get("url", ""),
                    }
                    for r in state.web_results[:3]
                ]
            if state.web_answer:
                context_dict["web_summary"] = state.web_answer

            result = await composer.compose_prompt(
                mode=state.mode,
                dimensions=state.dimensions,
                context=context_dict,
            )

            state.prompt = result.positive_prompt
            state.negative_prompt = result.negative_prompt

        except Exception as e:
            logger.warning(f"[{self.name}] Composer failed, using fallback: {e}")
            # Fallback to simple template
            state.prompt = self._build_fallback_prompt(state)
            state.negative_prompt = self._build_fallback_negative()

        # Build/enhance provider_params based on mode and style if not already set
        state.provider_params = self._build_provider_params(task, state)

        return state

    async def _evaluate_prompt(self, task: PlanningTask, state: PromptState) -> PromptState:
        """Evaluate prompt quality using PromptQualityTool."""
        logger.debug(f"[{self.name}] Evaluating prompt quality")

        result = await self.call_tool(
            "prompt_quality",
            "assess_prompt_quality",
            prompt=state.prompt,
            negative_prompt=state.negative_prompt,
            mode=state.mode,
            product_type=task.product_type,
            dimensions=state.dimensions.to_dict() if state.dimensions else {},
        )

        if result.success and result.data:
            state.quality = PromptQualityResult.from_dict(result.data)
        else:
            # Fallback quality assessment
            state.quality = PromptQualityResult(
                overall=0.7,
                dimensions={},
                mode=state.mode,
                threshold=0.7,
                decision="PASS" if len(state.prompt) > 20 else "FIX_REQUIRED",
            )

        return state

    async def _refine_prompt(self, task: PlanningTask, state: PromptState) -> PromptState:
        """Refine prompt based on quality feedback."""
        logger.debug(f"[{self.name}] Refining prompt (revision {state.revision_count + 1})")

        # Store previous prompt for history
        state.revision_history.append(state.prompt)

        result = await self.call_tool(
            "prompt_quality",
            "revise_prompt",
            prompt=state.prompt,
            quality_result=state.quality.to_dict() if state.quality else {},
        )

        if result.success and result.data:
            state.prompt = result.data.get("revised_prompt", state.prompt)
        else:
            # Simple fallback refinement
            if state.quality and state.quality.failed_dimensions:
                # Add more detail for failed dimensions
                additions = []
                for dim in state.quality.failed_dimensions[:2]:
                    if dim == "clarity":
                        additions.append("detailed")
                    elif dim == "coverage":
                        additions.append("comprehensive")
                if additions:
                    state.prompt = ", ".join(additions) + ", " + state.prompt

        state.revision_count += 1
        state.quality = None  # Reset for re-evaluation

        return state

    def _check_goal(self, state: PromptState) -> bool:
        """Check if the goal has been satisfied."""
        if not state.has_prompt:
            return False
        if not state.has_quality:
            return False
        if state.quality.decision == "PASS":
            return True
        if state.revision_count >= self.max_revisions:
            return True  # Stop after max revisions even if not passing
        return False

    def _build_fallback_prompt(self, state: PromptState) -> str:
        """Build a simple fallback prompt from dimensions."""
        parts = []

        if state.dimensions:
            if state.dimensions.subject:
                parts.append(state.dimensions.subject)
            if state.dimensions.aesthetic:
                parts.append(f"{state.dimensions.aesthetic} style")
            if state.dimensions.color:
                parts.append(f"{state.dimensions.color} colors")
            if state.dimensions.mood:
                parts.append(f"{state.dimensions.mood} mood")
            if state.dimensions.background:
                parts.append(f"{state.dimensions.background} background")
            if state.dimensions.lighting:
                parts.append(f"{state.dimensions.lighting} lighting")

        if not parts:
            parts.append("high quality digital art")

        return ", ".join(parts)

    def _build_fallback_negative(self) -> str:
        """Build a standard fallback negative prompt."""
        return "blurry, low quality, distorted, watermark, signature, text"

    def _build_provider_params(
        self,
        task: PlanningTask,
        state: PromptState,
    ) -> Dict[str, Any]:
        """
        Build provider-specific generation parameters.

        Merges user-specified params with mode-based defaults.
        These are passed through to the final image generation.

        Args:
            task: The planning task with requirements
            state: Current prompt state with mode and dimensions

        Returns:
            Dict of provider parameters for generation
        """
        # Start with any user-specified params
        params = dict(state.provider_params)

        # Apply mode-based defaults if not already set
        mode = state.mode.upper()

        # Steps: higher for complex/quality modes
        if "steps" not in params:
            if mode == "ADVANCED" or mode == "COMPLEX":
                params["steps"] = 40
            elif mode == "STANDARD":
                params["steps"] = 30
            else:  # SIMPLE/RELAX
                params["steps"] = 25

        # Guidance scale: higher for more prompt adherence
        if "guidance_scale" not in params:
            if mode == "ADVANCED" or mode == "COMPLEX":
                params["guidance_scale"] = 8.0
            elif mode == "STANDARD":
                params["guidance_scale"] = 7.5
            else:
                params["guidance_scale"] = 7.0

        # Product-specific adjustments
        product_type = task.product_type or ""
        if product_type in ["poster", "wall_art", "canvas"]:
            # Higher quality for wall art
            if "steps" not in state.provider_params:
                params["steps"] = max(params.get("steps", 30), 35)

        # Print method adjustments
        if task.print_method:
            if task.print_method == "screen_print":
                # Screen print needs cleaner edges
                params.setdefault("scheduler", "euler_ancestral")
            elif task.print_method == "dtg":
                # DTG can handle more detail
                if "steps" not in state.provider_params:
                    params["steps"] = max(params.get("steps", 30), 35)

        # Style-based adjustments from dimensions
        if state.dimensions:
            aesthetic = state.dimensions.aesthetic or ""
            if "photorealistic" in aesthetic.lower() or "realistic" in aesthetic.lower():
                params.setdefault("scheduler", "dpm_2m_karras")
            elif "cartoon" in aesthetic.lower() or "anime" in aesthetic.lower():
                params.setdefault("scheduler", "euler")

        # Preserve seed if specified in requirements
        if task.requirements.get("seed") is not None:
            params["seed"] = task.requirements["seed"]

        return params

    async def __aenter__(self) -> "ReactPromptAgent":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()
