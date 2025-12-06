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

from src.utils.logger import get_logger

from palet8_agents.core.agent import BaseAgent, AgentContext, AgentResult
from palet8_agents.tools.base import BaseTool, ToolResult
from palet8_agents.models.planning import PlanningTask, PromptPlan, ContextSummary
from palet8_agents.models.prompt import PromptDimensions, PromptQualityResult
from palet8_agents.services.text_llm_service import TextLLMService, TextLLMServiceError

logger = get_logger(__name__)


class ReactAction(Enum):
    """Actions available in the ReAct loop."""
    EVALUATE_CONTEXT = "evaluate_context"  # Check what context is available/missing
    GENERATE_QUESTIONS = "generate_questions"  # Generate clarification questions if needed
    BUILD_CONTEXT = "build_context"
    SELECT_DIMENSIONS = "select_dimensions"
    COMPOSE_PROMPT = "compose_prompt"
    EVALUATE_PROMPT = "evaluate_prompt"
    REFINE_PROMPT = "refine_prompt"
    DONE = "done"


@dataclass
class ClarificationQuestion:
    """Question to route through Pali to user.

    Question types:
    - "text": Free-form text input
    - "selector": UI selector component (dropdown, grid, etc.)
    - "image_upload": Image file upload

    Frontend components use selector_id to determine which UI to show.
    """
    question_type: str  # "text", "selector", "image_upload"
    field: str
    question_text: str
    selector_id: Optional[str] = None  # Frontend component ID
    component: Optional[str] = None  # Frontend component name (e.g., "StyleSelector")
    options: Optional[List[str]] = None  # Options for selector
    placeholder: Optional[str] = None  # Placeholder for text input
    accept: Optional[str] = None  # File types for image upload (e.g., "image/*")
    required: bool = False
    priority: int = 0  # Lower = higher priority

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "question_type": self.question_type,
            "field": self.field,
            "question_text": self.question_text,
            "selector_id": self.selector_id,
            "options": self.options,
            "required": self.required,
            "priority": self.priority,
        }
        # Include optional fields if present
        if self.component:
            result["component"] = self.component
        if self.placeholder:
            result["placeholder"] = self.placeholder
        if self.accept:
            result["accept"] = self.accept
        return result


# Field to selector mapping for clarification questions
# Format: field -> (selector_id, question_type, component, placeholder/accept)
FIELD_SELECTOR_MAP = {
    # UI Selector components (template HTML selectors from customer app)
    "style": {
        "type": "selector",
        "selector_id": "aesthetic_style",
        "component": "StyleSelector",
    },
    "product_type": {
        "type": "selector",
        "selector_id": "product_category",
        "component": "ProductCategoryGrid",
    },
    "dimensions": {
        "type": "selector",
        "selector_id": "aspect_ratio",
        "component": "DimensionPicker",
    },
    "character": {
        "type": "selector",
        "selector_id": "system_character",
        "component": "CharacterSelector",
    },
    # Text input fields
    "mood": {
        "type": "text",
        "placeholder": "Describe the mood or atmosphere you want...",
    },
    "colors": {
        "type": "text",
        "placeholder": "What colors would you like? (e.g., 'warm tones', 'blue and gold')",
    },
    "subject": {
        "type": "text",
        "placeholder": "What is the main subject of your image?",
    },
    # Image upload
    "reference_image": {
        "type": "image_upload",
        "selector_id": "reference_image",
        "accept": "image/*",
    },
    # Text overlay content
    "text_content": {
        "type": "text",
        "selector_id": "text_in_image",
        "placeholder": "What text should appear in the image?",
    },
}


@dataclass
class PromptState:
    """Internal state for the ReAct loop."""
    # Context evaluation state (NEW)
    context_evaluated: bool = False
    available_fields: List[str] = field(default_factory=list)
    missing_fields: List[str] = field(default_factory=list)
    needs_clarification: bool = False
    priority_field: Optional[str] = None
    questions_generated: bool = False
    clarification_questions: List[ClarificationQuestion] = field(default_factory=list)

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
            context: Shared execution context containing planning_task and generation_plan
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

            # Extract GenerationPlan from context (from GenPlan agent)
            generation_plan_data = context.metadata.get("generation_plan")

            task = PlanningTask.from_dict(planning_task_data)
            logger.info(
                "react_prompt.run.start",
                job_id=task.job_id,
                phase=task.phase,
                has_previous_plan=task.previous_plan is not None,
                has_generation_plan=generation_plan_data is not None,
            )

            # Initialize state with generation plan
            state = self._init_state(task, generation_plan_data)
            steps = 0

            # ReAct loop
            while not state.goal_satisfied and steps < self.max_steps:
                # THINK: Decide next action
                next_action = await self._think(state, task)
                logger.debug(
                    "react_prompt.step.action",
                    step=steps + 1,
                    action=next_action.value,
                )

                # ACT: Execute the action
                if next_action == ReactAction.EVALUATE_CONTEXT:
                    state = await self._evaluate_context(task, state)
                elif next_action == ReactAction.GENERATE_QUESTIONS:
                    state = await self._generate_questions(task, state)
                    # If questions were generated, return early with clarification result
                    if state.needs_clarification and state.clarification_questions:
                        logger.info(
                            "react_prompt.clarification.needed",
                            job_id=task.job_id,
                            missing_fields=state.missing_fields,
                            question_count=len(state.clarification_questions),
                        )
                        # Return clarification result to Planner for routing through Pali
                        return self._create_result(
                            success=True,
                            data={
                                "needs_clarification": True,
                                "missing_fields": state.missing_fields,
                                "priority_field": state.priority_field,
                                "questions": [q.to_dict() for q in state.clarification_questions],
                            },
                            error=None,
                            error_code=None,
                        )
                elif next_action == ReactAction.BUILD_CONTEXT:
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
                "react_prompt.run.complete",
                job_id=task.job_id,
                quality_score=round(prompt_plan.quality_score, 2),
                quality_acceptable=prompt_plan.quality_acceptable,
                steps=steps,
                revision_count=prompt_plan.revision_count,
                mode=prompt_plan.mode,
            )

            return self._create_result(
                success=True,
                data={"prompt_plan": prompt_plan.to_dict()},
                next_agent="planner",
            )

        except Exception as e:
            logger.error(
                "react_prompt.run.error",
                error_detail=str(e),
                error_type=type(e).__name__,
                exc_info=True,
            )
            return self._create_result(
                success=False,
                data=None,
                error_detail=str(e),
                error_code="REACT_ERROR",
            )

    def _init_state(
        self,
        task: PlanningTask,
        generation_plan_data: Optional[Dict[str, Any]] = None,
    ) -> PromptState:
        """
        Initialize state based on task phase and generation plan.

        Args:
            task: The planning task with requirements
            generation_plan_data: GenerationPlan from GenPlan agent (contains
                provider_params, model_input_params, complexity, etc.)

        Returns:
            Initialized PromptState
        """
        state = PromptState()

        # Get mode/complexity from generation_plan (preferred) or task
        if generation_plan_data:
            complexity = generation_plan_data.get("complexity", "standard")
            state.mode = complexity.upper()
            # Get provider_params from GenPlan (model-specific parameters)
            state.provider_params = generation_plan_data.get("provider_params", {})
            # Merge model_input_params (steps, guidance_scale, etc.)
            model_input_params = generation_plan_data.get("model_input_params", {})
            for key in ["steps", "guidance_scale", "width", "height", "seed"]:
                if key in model_input_params and key not in state.provider_params:
                    state.provider_params[key] = model_input_params[key]
            logger.debug(
                "react_prompt.init_state.from_generation_plan",
                mode=state.mode,
                provider_params_keys=list(state.provider_params.keys()),
            )
        else:
            # Fallback to task complexity
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
            # Preserve provider_params from previous plan if not overridden by generation_plan
            if not state.provider_params and task.previous_plan.get("provider_params"):
                state.provider_params = task.previous_plan.get("provider_params", {})

        return state

    async def _think(self, state: PromptState, task: PlanningTask) -> ReactAction:
        """
        Decide next action based on current state and task phase.

        The thinking logic varies by phase:
        - initial: Full pipeline from context evaluation to quality
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
        # Step 1: Evaluate context sufficiency
        if not state.context_evaluated:
            return ReactAction.EVALUATE_CONTEXT
        # Step 2: Generate questions if clarification needed
        if state.needs_clarification and not state.questions_generated:
            return ReactAction.GENERATE_QUESTIONS
        # Step 3+: Continue with normal flow if context is sufficient
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

    # Field priority for clarification questions (lower = higher priority)
    FIELD_PRIORITY = {
        "subject": 1,
        "product_type": 2,
        "style": 3,
        "dimensions": 4,
        "character": 5,
        "mood": 6,
        "colors": 7,
        "reference_image": 8,
        "text_content": 9,
    }

    # Minimum context sufficiency score for different complexity levels
    MIN_CONTEXT_SCORE = {
        "simple": 0.3,      # subject only is sufficient
        "standard": 0.5,    # subject + style preferred
        "complex": 0.7,     # subject + style + mood + detailed direction needed
    }

    async def _evaluate_context(self, task: PlanningTask, state: PromptState) -> PromptState:
        """
        Evaluate if context is sufficient for generation.

        Checks what's provided vs what's missing based on complexity level
        (from GenerationPlan).

        Flow:
        1. Get available fields from requirements
        2. Determine required fields based on complexity
        3. Calculate sufficiency score
        4. Mark if clarification is needed
        """
        logger.info(
            "react_prompt.context.evaluate.start",
            job_id=task.job_id,
            complexity=state.mode.lower(),
        )

        requirements = task.requirements or {}
        complexity = state.mode.lower()

        # Check what fields are available
        available = []
        missing = []

        # Required field: subject
        if requirements.get("subject"):
            available.append("subject")
        else:
            missing.append("subject")

        # Check optional fields
        optional_fields = [
            ("style", requirements.get("style") or requirements.get("aesthetic")),
            ("product_type", requirements.get("product_type")),
            ("mood", requirements.get("mood")),
            ("colors", requirements.get("colors")),
            ("dimensions", requirements.get("dimensions") or requirements.get("aspect_ratio")),
            ("reference_image", requirements.get("reference_image") or requirements.get("has_reference")),
            ("text_content", requirements.get("text_content") or requirements.get("text")),
            ("character", requirements.get("character")),
        ]

        for field_name, field_value in optional_fields:
            if field_value:
                available.append(field_name)
            else:
                missing.append(field_name)

        # Calculate sufficiency score
        total_fields = len(available) + len(missing)
        score = len(available) / total_fields if total_fields > 0 else 0.0

        # Determine if clarification is needed based on complexity
        min_score = self.MIN_CONTEXT_SCORE.get(complexity, 0.5)
        needs_clarification = score < min_score

        # Subject is always required
        if "subject" in missing:
            needs_clarification = True

        # Find highest priority missing field
        priority_field = None
        if missing:
            sorted_missing = sorted(missing, key=lambda f: self.FIELD_PRIORITY.get(f, 99))
            priority_field = sorted_missing[0]

        # Update state
        state.context_evaluated = True
        state.available_fields = available
        state.missing_fields = missing
        state.needs_clarification = needs_clarification
        state.priority_field = priority_field

        logger.info(
            "react_prompt.context.evaluate.complete",
            job_id=task.job_id,
            available_count=len(available),
            missing_count=len(missing),
            score=round(score, 2),
            needs_clarification=needs_clarification,
            priority_field=priority_field,
        )

        return state

    async def _generate_questions(self, task: PlanningTask, state: PromptState) -> PromptState:
        """
        Generate clarification questions for missing fields.

        Maps missing fields to question types:
        - selector: UI selector component (style, product_type, etc.)
        - text: Free-form text input
        - image_upload: Reference image upload

        Each field maps to delivery method from FIELD_SELECTOR_MAP which includes:
        - type: Question type
        - selector_id: Frontend component ID
        - component: Frontend component name
        - placeholder: Text for text inputs
        - accept: File types for image upload
        """
        logger.info(
            "react_prompt.questions.generate.start",
            job_id=task.job_id,
            missing_fields=state.missing_fields,
        )

        questions = []

        # Limit to 3 questions at a time to avoid overwhelming user
        for field in state.missing_fields[:3]:
            # Get delivery method from mapping (new dict format)
            delivery = FIELD_SELECTOR_MAP.get(field, {"type": "text"})
            question_type = delivery.get("type", "text")
            selector_id = delivery.get("selector_id")
            component = delivery.get("component")
            placeholder = delivery.get("placeholder")
            accept = delivery.get("accept")

            # Generate question text based on field
            question_text = self._get_question_text(field)

            # Get options for selector type
            options = self._get_field_options(field) if question_type == "selector" else None

            question = ClarificationQuestion(
                question_type=question_type,
                field=field,
                question_text=question_text,
                selector_id=selector_id,
                component=component,
                options=options,
                placeholder=placeholder,
                accept=accept,
                required=(field == "subject"),  # Only subject is truly required
                priority=self.FIELD_PRIORITY.get(field, 99),
            )
            questions.append(question)

        # Sort by priority
        questions.sort(key=lambda q: q.priority)

        state.questions_generated = True
        state.clarification_questions = questions

        logger.info(
            "react_prompt.questions.generate.complete",
            job_id=task.job_id,
            question_count=len(questions),
            priority_field=state.priority_field,
        )

        return state

    def _get_question_text(self, field: str) -> str:
        """Get natural language question text for a field."""
        question_map = {
            "subject": "What would you like to create? Please describe the main subject of your image.",
            "style": "What visual style would you like for your image?",
            "product_type": "What type of product are you creating? (e.g., poster, social media, album art)",
            "mood": "What mood or feeling should the image convey?",
            "colors": "Do you have any color preferences for your image?",
            "dimensions": "What dimensions or aspect ratio would you like?",
            "reference_image": "Would you like to upload a reference image for inspiration?",
            "text_content": "Do you want any text or typography in your image? If so, what text?",
            "character": "Would you like to use a specific character or persona?",
        }
        return question_map.get(field, f"Please provide information about the {field}.")

    def _get_field_options(self, field: str) -> Optional[List[str]]:
        """Get predefined options for selector fields."""
        options_map = {
            "style": ["photorealistic", "illustration", "vintage", "minimalist", "bold", "artistic"],
            "product_type": ["poster", "social media", "album art", "banner", "icon", "product shot"],
            "dimensions": ["square (1:1)", "landscape (16:9)", "portrait (9:16)", "wide (21:9)"],
        }
        return options_map.get(field)

    async def _build_context(self, task: PlanningTask, state: PromptState) -> PromptState:
        """
        Build context using RAG, memory, and conditional web search.

        Flow:
        1. First get context from memory (user history, art references)
        2. Check if context is sufficient for prompt composition
        3. If NOT sufficient → supplement with web search
        4. Return combined context
        """
        logger.info(
            "react_prompt.context.start",
            job_id=task.job_id,
            user_id=task.user_id,
        )

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
            logger.info(
                "react_prompt.context.complete",
                job_id=task.job_id,
                history_count=len(state.user_history),
                art_refs_count=len(state.art_references),
                web_search_used=False,
            )
            return state

        # Step 4: RAG context insufficient - supplement with web search
        logger.info(
            "react_prompt.context.web_search_triggered",
            job_id=task.job_id,
            history_count=len(state.user_history),
            art_refs_count=len(state.art_references),
        )
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

                logger.info(
                    "react_prompt.web_search.complete",
                    results_count=len(state.web_results),
                    has_answer=state.web_answer is not None,
                    provider=result.data.get("provider", "unknown"),
                )

        except Exception as e:
            logger.warning(
                "react_prompt.web_search.failed",
                error_detail=str(e),
                error_type=type(e).__name__,
            )

        return state

    async def _select_dimensions(self, task: PlanningTask, state: PromptState) -> PromptState:
        """Select dimensions using DimensionTool."""
        logger.info(
            "react_prompt.dimensions.start",
            job_id=task.job_id,
            mode=state.mode,
        )

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
            logger.info(
                "react_prompt.dimensions.complete",
                job_id=task.job_id,
                subject=state.dimensions.subject,
                aesthetic=state.dimensions.aesthetic,
                used_fallback=False,
            )
        else:
            # Fallback to basic dimensions from requirements
            state.dimensions = PromptDimensions(
                subject=task.requirements.get("subject", ""),
                aesthetic=task.requirements.get("style"),
                mood=task.requirements.get("mood"),
            )
            if task.requirements.get("colors"):
                state.dimensions.color = ", ".join(task.requirements["colors"])
            logger.warning(
                "react_prompt.dimensions.fallback",
                job_id=task.job_id,
                subject=state.dimensions.subject,
            )

        return state

    async def _compose_prompt(self, task: PlanningTask, state: PromptState) -> PromptState:
        """Compose prompt using PromptComposerService."""
        logger.info(
            "react_prompt.compose.start",
            job_id=task.job_id,
            mode=state.mode,
        )

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
            logger.info(
                "react_prompt.compose.complete",
                job_id=task.job_id,
                prompt_length=len(state.prompt),
                negative_prompt_length=len(state.negative_prompt),
                used_fallback=False,
            )

        except Exception as e:
            logger.warning(
                "react_prompt.compose.fallback",
                job_id=task.job_id,
                error_detail=str(e),
                error_type=type(e).__name__,
            )
            # Fallback to simple template
            state.prompt = self._build_fallback_prompt(state)
            state.negative_prompt = self._build_fallback_negative()

        # Note: provider_params are now set by GenPlan agent and passed via
        # generation_plan in context. They are initialized in _init_state().

        return state

    async def _evaluate_prompt(self, task: PlanningTask, state: PromptState) -> PromptState:
        """Evaluate prompt quality using PromptQualityTool."""
        logger.info(
            "react_prompt.quality.start",
            job_id=task.job_id,
            prompt_length=len(state.prompt),
        )

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

        logger.info(
            "react_prompt.quality.scored",
            job_id=task.job_id,
            overall=state.quality.overall,
            decision=state.quality.decision,
            threshold=state.quality.threshold,
            failed_dimensions=state.quality.failed_dimensions,
        )

        return state

    async def _refine_prompt(self, task: PlanningTask, state: PromptState) -> PromptState:
        """Refine prompt based on quality feedback."""
        logger.info(
            "react_prompt.refine.start",
            job_id=task.job_id,
            revision=state.revision_count + 1,
            failed_dimensions=state.quality.failed_dimensions if state.quality else [],
        )

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

        logger.info(
            "react_prompt.refine.complete",
            job_id=task.job_id,
            revision=state.revision_count,
            new_prompt_length=len(state.prompt),
        )

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

    # NOTE: _build_provider_params has been removed.
    # Provider parameters (steps, guidance_scale, scheduler, etc.) are now
    # handled by GenPlan agent and passed via generation_plan in context.
    # See GenPlanAgent._extract_parameters() for the implementation.

    async def __aenter__(self) -> "ReactPromptAgent":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()
