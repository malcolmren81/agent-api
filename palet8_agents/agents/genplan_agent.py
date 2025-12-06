"""
GenPlan Agent - Generation Planning using ReAct pattern.

This agent determines the generation strategy through a Think-Act-Observe loop:
1. Analyze complexity (simple/standard/complex)
2. Parse user info from requirements
3. Determine genflow (single vs dual pipeline)
4. Select optimal model
5. Extract generation parameters
6. Validate the complete plan

Uses services:
- ContextAnalysisService: For complexity questioning (triggers user questions via Pali)
- GenflowService: For single/dual pipeline decision
- ModelSelectionService: For model selection
- ModelInfoService: For model specifications and parameters

Documentation Reference: Section 5.2.2 (GenPlan Agent)
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from src.utils.logger import get_logger

from palet8_agents.core.agent import BaseAgent, AgentContext, AgentResult
from palet8_agents.tools.base import BaseTool
from palet8_agents.models.genplan import (
    UserParseResult,
    GenflowConfig,
    GenerationPlan,
    GenPlanState,
)
from palet8_agents.models.generation import PipelineConfig
from palet8_agents.services.context_analysis_service import ContextAnalysisService
from palet8_agents.services.genflow_service import GenflowService
from palet8_agents.services.model_selection_service import ModelSelectionService

logger = get_logger(__name__)

# Path to image models config
MODELS_CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "image_models_config.yaml"

# Cache for loaded config
_models_config_cache: Optional[Dict[str, Any]] = None


def _load_models_config() -> Dict[str, Any]:
    """Load and cache the image models config."""
    global _models_config_cache
    if _models_config_cache is not None:
        return _models_config_cache

    try:
        if MODELS_CONFIG_PATH.exists():
            with open(MODELS_CONFIG_PATH, "r") as f:
                _models_config_cache = yaml.safe_load(f)
        else:
            logger.warning(
                "genplan.config.not_found",
                path=str(MODELS_CONFIG_PATH),
            )
            _models_config_cache = {}
    except Exception as e:
        logger.error(
            "genplan.config.load_error",
            error_detail=str(e),
        )
        _models_config_cache = {}

    return _models_config_cache


def _get_model_specs_from_config(model_id: str) -> Dict[str, Any]:
    """
    Load model specs from image_models_config.yaml.

    Returns the full model config including specs, provider_params, etc.
    Returns empty dict if model not found.
    """
    config = _load_models_config()
    model_registry = config.get("model_registry", {})

    model_config = model_registry.get(model_id, {})
    if not model_config:
        logger.warning(
            "genplan.model_config.not_found",
            model_id=model_id,
        )
        return {}

    return model_config


class GenPlanAction(Enum):
    """Actions available in the GenPlan ReAct loop."""
    ANALYZE_COMPLEXITY = "analyze_complexity"
    PARSE_USER_INFO = "parse_user_info"
    DETERMINE_GENFLOW = "determine_genflow"
    SELECT_MODEL = "select_model"
    EXTRACT_PARAMETERS = "extract_parameters"
    VALIDATE_PLAN = "validate_plan"
    DONE = "done"


# =============================================================================
# Complexity Triggers
# =============================================================================
# Keywords that influence complexity determination

COMPLEXITY_TRIGGERS = {
    "complex": [
        # Text in image triggers
        "text", "typography", "lettering", "font", "words",
        "title", "headline", "quote", "caption", "label",
        # Character refinement triggers
        "character edit", "face fix", "expression change",
        "pose adjust", "facial expression", "character pose",
        # Multi-element triggers
        "multiple subjects", "complex composition", "layered design",
        "multiple characters", "group scene",
        # Production quality triggers
        "print-ready", "production", "high accuracy",
        "4k", "poster", "billboard", "large format",
        "professional quality",
    ],
    "simple": [
        # Simple/quick iteration indicators
        "simple", "quick", "basic", "rough", "draft",
        "sketch", "concept", "test", "try",
    ],
}




# Load system prompt from file
def _load_system_prompt() -> str:
    """Load the system prompt from file."""
    prompt_path = Path(__file__).parent.parent.parent / "prompts" / "genplan_system.txt"
    try:
        return prompt_path.read_text()
    except FileNotFoundError:
        logger.warning(f"System prompt file not found: {prompt_path}")
        return "You are a generation planning agent. Determine optimal generation strategy."


GENPLAN_SYSTEM = _load_system_prompt()


class GenPlanAgent(BaseAgent):
    """
    ReAct agent for generation planning.

    Uses a Think-Act-Observe loop to:
    1. Analyze request complexity
    2. Parse user requirements into structured info
    3. Determine generation flow (single vs dual pipeline)
    4. Select optimal model for the request
    5. Extract model-specific parameters
    6. Validate and return complete GenerationPlan
    """

    def __init__(
        self,
        tools: Optional[List[BaseTool]] = None,
        context_service: Optional[ContextAnalysisService] = None,
        genflow_service: Optional[GenflowService] = None,
        model_selection_service: Optional[ModelSelectionService] = None,
    ):
        """
        Initialize the GenPlanAgent.

        Args:
            tools: List of tools available to this agent
            context_service: Optional ContextAnalysisService for complexity questioning
            genflow_service: Optional GenflowService for pipeline selection
            model_selection_service: Optional ModelSelectionService for model selection
        """
        super().__init__(
            name="genplan",
            description="Determines generation strategy: complexity, genflow, model, parameters",
            tools=tools,
        )

        self._context_service = context_service
        self._genflow_service = genflow_service
        self._model_selection_service = model_selection_service
        self._owns_services = (
            context_service is None or
            genflow_service is None or
            model_selection_service is None
        )

        self.system_prompt = GENPLAN_SYSTEM
        self.model_profile = "genplan"
        self.max_steps = 10

    async def _get_context_service(self) -> ContextAnalysisService:
        """Get or create context analysis service."""
        if self._context_service is None:
            self._context_service = ContextAnalysisService()
        return self._context_service

    async def _get_genflow_service(self) -> GenflowService:
        """Get or create genflow service."""
        if self._genflow_service is None:
            self._genflow_service = GenflowService()
        return self._genflow_service

    async def _get_model_selection_service(self) -> ModelSelectionService:
        """Get or create model selection service."""
        if self._model_selection_service is None:
            self._model_selection_service = ModelSelectionService()
        return self._model_selection_service

    async def close(self) -> None:
        """Close resources."""
        if self._owns_services:
            if self._context_service and hasattr(self._context_service, 'close'):
                await self._context_service.close()
            if self._genflow_service and hasattr(self._genflow_service, 'close'):
                await self._genflow_service.close()
            if self._model_selection_service and hasattr(self._model_selection_service, 'close'):
                await self._model_selection_service.close()

    async def run(
        self,
        context: AgentContext,
        user_input: Optional[str] = None,
    ) -> AgentResult:
        """
        Execute the GenPlan ReAct loop to build a GenerationPlan.

        Args:
            context: Shared execution context containing requirements
            user_input: Optional user input (not typically used)

        Returns:
            AgentResult with GenerationPlan data or clarification request
        """
        self._start_execution()

        try:
            # Extract requirements from context (stored at context.requirements, not metadata)
            requirements = context.requirements or {}
            job_id = context.job_id
            user_id = context.user_id

            logger.info(
                "genplan.run.start",
                job_id=job_id,
                has_requirements=bool(requirements),
            )

            # Initialize state
            state = GenPlanState(job_id=job_id, user_id=user_id)
            steps = 0

            # ReAct loop
            while not state.is_complete and steps < self.max_steps:
                # THINK: Decide next action
                next_action = await self._think(state, requirements)
                logger.debug(
                    "genplan.step.action",
                    step=steps + 1,
                    action=next_action.value,
                )

                # Check if we need user clarification (from context service)
                if next_action == GenPlanAction.ANALYZE_COMPLEXITY:
                    needs_question, question_data = await self._check_complexity_question(
                        requirements, state
                    )
                    if needs_question:
                        return self._create_clarification_result(question_data, state)

                # ACT: Execute the action
                if next_action == GenPlanAction.ANALYZE_COMPLEXITY:
                    state = await self._analyze_complexity(state, requirements)
                elif next_action == GenPlanAction.PARSE_USER_INFO:
                    state = await self._parse_user_info(state, requirements)
                elif next_action == GenPlanAction.DETERMINE_GENFLOW:
                    state = await self._determine_genflow(state, requirements)
                elif next_action == GenPlanAction.SELECT_MODEL:
                    state = await self._select_model(state, requirements)
                elif next_action == GenPlanAction.EXTRACT_PARAMETERS:
                    state = await self._extract_parameters(state, requirements)
                elif next_action == GenPlanAction.VALIDATE_PLAN:
                    state = await self._validate_plan(state, requirements)
                elif next_action == GenPlanAction.DONE:
                    break

                # OBSERVE: Update step tracking
                steps += 1
                state.current_step = steps
                state.steps_completed.append(next_action.value)

            # Build final GenerationPlan
            generation_plan = state.to_generation_plan()

            # Store in context for next agent
            context.metadata["generation_plan"] = generation_plan.to_dict()

            logger.info(
                "genplan.run.complete",
                job_id=job_id,
                complexity=generation_plan.complexity,
                genflow_type=generation_plan.genflow.flow_type if generation_plan.genflow else None,
                model_id=generation_plan.model_id,
                steps=steps,
                is_valid=generation_plan.is_valid,
            )

            return self._create_result(
                success=True,
                data={"generation_plan": generation_plan.to_dict()},
                next_agent="planner",
            )

        except Exception as e:
            logger.error(
                "genplan.run.error",
                error_detail=str(e),
                exception_type=type(e).__name__,
                exc_info=True,
            )
            return self._create_result(
                success=False,
                data=None,
                error_detail=str(e),
                error_code="GENPLAN_ERROR",
            )

    # =========================================================================
    # THINK: Determine next action
    # =========================================================================

    async def _think(self, state: GenPlanState, requirements: Dict[str, Any]) -> GenPlanAction:
        """
        Determine next action based on current state.

        Follows the sequence:
        1. ANALYZE_COMPLEXITY (if not done)
        2. PARSE_USER_INFO (if not done)
        3. DETERMINE_GENFLOW (if not done)
        4. SELECT_MODEL (if not done)
        5. EXTRACT_PARAMETERS (if not done)
        6. VALIDATE_PLAN (if not done)
        7. DONE
        """
        if state.complexity is None:
            return GenPlanAction.ANALYZE_COMPLEXITY
        if state.user_info is None:
            return GenPlanAction.PARSE_USER_INFO
        if state.genflow is None:
            return GenPlanAction.DETERMINE_GENFLOW
        if state.model_id is None:
            return GenPlanAction.SELECT_MODEL
        if not state.parameters_extracted:
            return GenPlanAction.EXTRACT_PARAMETERS
        if not state.validated:
            return GenPlanAction.VALIDATE_PLAN
        return GenPlanAction.DONE

    # =========================================================================
    # ACT: Execute actions
    # =========================================================================

    async def _check_complexity_question(
        self,
        requirements: Dict[str, Any],
        state: GenPlanState,
    ) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Check if we need to ask the user about complexity level.

        Uses ContextAnalysisService to determine if complexity is missing
        and needs user clarification.

        Returns:
            Tuple of (needs_question, question_data)
        """
        # If complexity is already provided in requirements, no question needed
        if requirements.get("complexity"):
            return False, None

        # Use context service to evaluate if we need to ask
        context_service = await self._get_context_service()
        completeness = context_service.evaluate_completeness(requirements)

        # Check if complexity is in missing required fields
        if "complexity" in completeness.metadata.get("required_missing", []):
            # Generate question for complexity
            question = context_service._generate_question("complexity")
            if question:
                return True, {
                    "field": "complexity",
                    "question_type": "selector",
                    "question_text": question,
                    "selector_id": "generation_mode",
                    "options": [
                        {"value": "simple", "label": "Quick/Relax", "description": "Fast generation, less detail"},
                        {"value": "standard", "label": "Standard", "description": "Balanced quality and speed"},
                        {"value": "complex", "label": "Complex/Pro", "description": "Maximum detail and quality"},
                    ],
                    "required": True,
                }

        return False, None

    async def _analyze_complexity(
        self,
        state: GenPlanState,
        requirements: Dict[str, Any],
    ) -> GenPlanState:
        """
        Analyze request complexity based on requirements.

        THINK: Assess complexity based on:
        - Number of elements/subjects
        - Technical requirements (text, character editing)
        - Production quality needs
        - User's explicit complexity hints

        ACT: Analyze triggers and determine complexity level

        OBSERVE: Set state.complexity and state.complexity_rationale
        """
        logger.info("genplan.complexity.start", job_id=state.job_id)

        # Check if complexity is explicitly provided
        explicit_complexity = requirements.get("complexity", "").lower()
        if explicit_complexity in ["simple", "standard", "complex"]:
            state.complexity = explicit_complexity
            state.complexity_rationale = "User-specified complexity level"
            logger.info(
                "genplan.complexity.determined",
                job_id=state.job_id,
                complexity=state.complexity,
                source="explicit",
            )
            return state

        # Build text content for trigger analysis
        text_content = self._build_text_for_analysis(requirements)

        # Check for complex triggers
        complex_score = 0
        simple_score = 0
        triggers_found = []

        for trigger in COMPLEXITY_TRIGGERS["complex"]:
            if trigger in text_content:
                complex_score += 1
                triggers_found.append(trigger)

        for trigger in COMPLEXITY_TRIGGERS["simple"]:
            if trigger in text_content:
                simple_score += 1

        # Determine complexity
        if complex_score >= 2:
            state.complexity = "complex"
            state.complexity_rationale = f"Multiple complex triggers detected: {triggers_found[:3]}"
        elif complex_score == 1:
            state.complexity = "standard"
            state.complexity_rationale = f"One complex trigger detected: {triggers_found[0]}"
        elif simple_score > 0:
            state.complexity = "simple"
            state.complexity_rationale = "Simple/quick iteration indicators found"
        else:
            state.complexity = "standard"
            state.complexity_rationale = "Default complexity for general request"

        logger.info(
            "genplan.complexity.determined",
            job_id=state.job_id,
            complexity=state.complexity,
            rationale=state.complexity_rationale,
            triggers_found=triggers_found[:5],
        )

        return state

    async def _parse_user_info(
        self,
        state: GenPlanState,
        requirements: Dict[str, Any],
    ) -> GenPlanState:
        """
        Parse structured user info from requirements.

        THINK: Extract structured information about what the user wants

        ACT: Parse requirements into UserParseResult fields

        OBSERVE: Set state.user_info
        """
        logger.info("genplan.user_info.start", job_id=state.job_id)

        # Extract core fields
        subject = requirements.get("subject", "")
        style = requirements.get("style") or requirements.get("aesthetic")
        mood = requirements.get("mood")
        colors = requirements.get("colors", [])
        if isinstance(colors, str):
            colors = [c.strip() for c in colors.split(",")]

        product_type = requirements.get("product_type", "general")
        print_method = requirements.get("print_method")

        # Check for reference image
        has_reference = bool(requirements.get("reference_image_url"))
        reference_image_url = requirements.get("reference_image_url")

        # Extract text content (for text overlays)
        text_content = requirements.get("text_content") or requirements.get("text_in_image")

        # Extract intents from the request
        extracted_intents = self._extract_intents(requirements)

        state.user_info = UserParseResult(
            subject=subject,
            style=style,
            mood=mood,
            colors=colors,
            product_type=product_type,
            print_method=print_method,
            has_reference=has_reference,
            reference_image_url=reference_image_url,
            text_content=text_content,
            extracted_intents=extracted_intents,
            metadata={
                "original_prompt": requirements.get("prompt", ""),
                "parsed_from": list(requirements.keys()),
            },
        )

        logger.info(
            "genplan.user_info.parsed",
            job_id=state.job_id,
            subject=subject[:50] if subject else None,
            style=style,
            product_type=product_type,
            has_reference=has_reference,
            has_text_content=bool(text_content),
            intents=extracted_intents,
        )

        return state

    async def _determine_genflow(
        self,
        state: GenPlanState,
        requirements: Dict[str, Any],
    ) -> GenPlanState:
        """
        Determine generation flow (single vs dual pipeline).

        THINK: Based on complexity and user_info, determine if this
        requires single or dual pipeline generation.

        ACT: Call GenflowService.determine_genflow()

        OBSERVE: Set state.genflow
        """
        logger.info("genplan.genflow.start", job_id=state.job_id)

        genflow_service = await self._get_genflow_service()

        # Build prompt text for additional trigger detection
        prompt = requirements.get("prompt", "")

        state.genflow = genflow_service.determine_genflow(
            requirements=requirements,
            complexity=state.complexity,
            prompt=prompt,
            user_info=state.user_info.to_dict() if state.user_info else None,
        )

        logger.info(
            "genplan.genflow.selected",
            job_id=state.job_id,
            flow_type=state.genflow.flow_type,
            flow_name=state.genflow.flow_name,
            rationale=state.genflow.rationale,
            triggered_by=state.genflow.triggered_by,
        )

        return state

    async def _select_model(
        self,
        state: GenPlanState,
        requirements: Dict[str, Any],
    ) -> GenPlanState:
        """
        Select optimal model for generation.

        THINK: Select model based on:
        - Genflow type (single vs dual, which pipeline)
        - Whether reference image exists
        - Style requirements (art vs photo)
        - Cost/quality tradeoffs

        ACT: Call ModelSelectionService or use pipeline predefined models
        For each model selected, load actual specs from image_models_config.yaml

        OBSERVE: Set state.model_id, model_rationale, model_alternatives, model_specs
        """
        logger.info("genplan.model.start", job_id=state.job_id)

        # For dual pipeline, use predefined models from genflow
        if state.genflow and state.genflow.is_dual:
            genflow_service = await self._get_genflow_service()
            pipeline_config = genflow_service.get_dual_pipeline_config(state.genflow.flow_name)

            if pipeline_config and pipeline_config.get("stage_1_model"):
                stage_1_model = pipeline_config.get("stage_1_model")
                stage_2_model = pipeline_config.get("stage_2_model")

                # Load actual model specs from config for BOTH models
                stage_1_config = _get_model_specs_from_config(stage_1_model)
                stage_2_config = _get_model_specs_from_config(stage_2_model) if stage_2_model else {}

                state.model_id = stage_1_model
                state.model_rationale = f"Stage 1 model for {state.genflow.flow_name} pipeline"
                state.model_alternatives = []  # Pipeline models are fixed

                # Store structured specs for all models in the pipeline
                state.model_specs = {
                    "pipeline_type": "dual",
                    "pipeline_name": state.genflow.flow_name,
                    "stage_1": {
                        "model": stage_1_model,
                        "purpose": pipeline_config.get("stage_1_purpose"),
                        "specs": stage_1_config.get("specs", {}),
                        "provider_params": stage_1_config.get("specs", {}).get("provider_params", {}),
                        "air_id": stage_1_config.get("air_id"),
                        "provider": stage_1_config.get("provider"),
                    },
                    "stage_2": {
                        "model": stage_2_model,
                        "purpose": pipeline_config.get("stage_2_purpose"),
                        "specs": stage_2_config.get("specs", {}),
                        "provider_params": stage_2_config.get("specs", {}).get("provider_params", {}),
                        "air_id": stage_2_config.get("air_id"),
                        "provider": stage_2_config.get("provider"),
                    },
                }

                # Build pipeline config
                state.pipeline = PipelineConfig(
                    pipeline_type="dual",
                    pipeline_name=state.genflow.flow_name,
                    stage_1_model=stage_1_model,
                    stage_1_purpose=pipeline_config.get("stage_1_purpose"),
                    stage_2_model=stage_2_model,
                    stage_2_purpose=pipeline_config.get("stage_2_purpose"),
                )

                logger.info(
                    "genplan.model.selected",
                    job_id=state.job_id,
                    model_id=state.model_id,
                    source="dual_pipeline",
                    pipeline=state.genflow.flow_name,
                    stage_1_has_specs=bool(stage_1_config.get("specs")),
                    stage_2_has_specs=bool(stage_2_config.get("specs")),
                )
                return state

        # For single pipeline, use model selection service
        model_service = await self._get_model_selection_service()

        # Determine scenario
        has_reference = state.user_info.has_reference if state.user_info else False
        style = state.user_info.style if state.user_info else ""
        is_photorealistic = style and any(
            w in style.lower() for w in ["photo", "realistic", "product", "lifestyle"]
        )

        # Determine scenario for model selection
        if is_photorealistic:
            scenario = "photo_with_reference" if has_reference else "photo_no_reference"
        else:
            scenario = "art_with_reference" if has_reference else "art_no_reference"

        # Select model - service returns tuple: (model_id, rationale, alternatives, model_specs)
        model_id, rationale, alternatives, service_model_specs = await model_service.select_model(
            mode=state.complexity or "STANDARD",
            requirements={**requirements, "scenario": scenario},
        )

        # Also load specs from config to ensure we have the full picture
        config_model_specs = _get_model_specs_from_config(model_id)

        state.model_id = model_id
        state.model_rationale = rationale
        state.model_alternatives = alternatives

        # Merge service specs with config specs, preferring config for specs section
        state.model_specs = {
            "pipeline_type": "single",
            "stage_1": {
                "model": model_id,
                "purpose": "Generate final image",
                "specs": config_model_specs.get("specs", service_model_specs.get("specs", {})),
                "provider_params": config_model_specs.get("specs", {}).get("provider_params", {}),
                "air_id": config_model_specs.get("air_id") or service_model_specs.get("air_id"),
                "provider": config_model_specs.get("provider") or service_model_specs.get("provider"),
            },
            # Legacy fields for backward compatibility
            "air_id": config_model_specs.get("air_id") or service_model_specs.get("air_id"),
            "provider": config_model_specs.get("provider") or service_model_specs.get("provider"),
            "specs": config_model_specs.get("specs", service_model_specs.get("specs", {})),
        }

        # Build single pipeline config
        state.pipeline = PipelineConfig(
            pipeline_type="single",
            pipeline_name="single_standard",
            stage_1_model=state.model_id,
            stage_1_purpose="Generate final image",
        )

        logger.info(
            "genplan.model.selected",
            job_id=state.job_id,
            model_id=state.model_id,
            scenario=scenario,
            rationale=state.model_rationale,
            alternatives=state.model_alternatives,
            has_specs=bool(state.model_specs.get("specs")),
        )

        return state

    async def _extract_parameters(
        self,
        state: GenPlanState,
        requirements: Dict[str, Any],
    ) -> GenPlanState:
        """
        Extract generation parameters for all models in the pipeline.

        THINK: Extract correct parameters for each model in the pipeline,
        including both standard generation params and provider-specific.
        Only add steps/cfg_scale if the model's config includes specs.steps.

        For dual pipelines, extract params for both stage_1 and stage_2 models.

        ACT: Build model_input_params and provider_params for each stage

        OBSERVE: Set state.model_input_params, provider_params, parameters_extracted
        """
        logger.info(
            "genplan.parameters.start",
            job_id=state.job_id,
            model_id=state.model_id,
            pipeline_type=state.model_specs.get("pipeline_type", "unknown"),
        )

        # Common dimensions based on product type
        product_type = state.user_info.product_type if state.user_info else "general"
        width, height = self._get_dimensions_for_product(product_type, requirements)

        # Check pipeline type
        is_dual = state.model_specs.get("pipeline_type") == "dual"

        if is_dual:
            # Extract parameters for each stage model
            stage_1_params = self._extract_stage_params(
                stage_specs=state.model_specs.get("stage_1", {}),
                complexity=state.complexity,
                width=width,
                height=height,
                requirements=requirements,
                stage_name="stage_1",
                job_id=state.job_id,
            )

            stage_2_params = self._extract_stage_params(
                stage_specs=state.model_specs.get("stage_2", {}),
                complexity=state.complexity,
                width=width,
                height=height,
                requirements=requirements,
                stage_name="stage_2",
                job_id=state.job_id,
            )

            # Store nested structure for AssemblyService (dual pipeline execution)
            state.model_input_params_by_stage = {
                "stage_1": stage_1_params["model_input_params"],
                "stage_2": stage_2_params["model_input_params"],
            }
            state.provider_params_by_stage = {
                "stage_1": stage_1_params["provider_params"],
                "stage_2": stage_2_params["provider_params"],
            }

            # Flatten to stage_1 params for downstream compatibility (ReactPrompt, Planner)
            # These components expect flat structure
            state.model_input_params = stage_1_params["model_input_params"]
            state.provider_params = stage_1_params["provider_params"]

            logger.info(
                "genplan.parameters.extracted",
                job_id=state.job_id,
                pipeline_type="dual",
                stage_1_model=state.model_specs.get("stage_1", {}).get("model"),
                stage_1_has_steps="steps" in stage_1_params["model_input_params"],
                stage_2_model=state.model_specs.get("stage_2", {}).get("model"),
                stage_2_has_steps="steps" in stage_2_params["model_input_params"],
                flattened_to="stage_1",
            )
        else:
            # Single pipeline - extract params for stage_1 only
            stage_specs = state.model_specs.get("stage_1", {})
            if not stage_specs:
                # Legacy fallback - use top-level specs
                stage_specs = {
                    "specs": state.model_specs.get("specs", {}),
                    "provider_params": state.model_specs.get("provider_params", {}),
                    "model": state.model_id,
                }

            stage_params = self._extract_stage_params(
                stage_specs=stage_specs,
                complexity=state.complexity,
                width=width,
                height=height,
                requirements=requirements,
                stage_name="stage_1",
                job_id=state.job_id,
            )

            # For single pipeline, store flat (not nested) for backward compatibility
            state.model_input_params = stage_params["model_input_params"]
            state.provider_params = stage_params["provider_params"]

            logger.info(
                "genplan.parameters.extracted",
                job_id=state.job_id,
                pipeline_type="single",
                model_id=state.model_id,
                has_steps="steps" in state.model_input_params,
                steps=state.model_input_params.get("steps"),
                has_cfg_scale="cfg_scale" in state.model_input_params,
                cfg_scale=state.model_input_params.get("cfg_scale"),
                provider_param_keys=list(state.provider_params.keys()) if state.provider_params else [],
            )

        state.parameters_extracted = True
        return state

    def _extract_stage_params(
        self,
        stage_specs: Dict[str, Any],
        complexity: str,
        width: int,
        height: int,
        requirements: Dict[str, Any],
        stage_name: str,
        job_id: str,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Extract parameters for a single stage model.

        Args:
            stage_specs: The stage config containing model, specs, provider_params
            complexity: Complexity level (simple/standard/complex)
            width: Image width
            height: Image height
            requirements: Original requirements
            stage_name: Name of the stage (stage_1 or stage_2)
            job_id: Job ID for logging

        Returns:
            Dict with model_input_params and provider_params
        """
        model_input_params = {}
        provider_params = {}

        model_id = stage_specs.get("model", "unknown")
        specs = stage_specs.get("specs", {})

        # Set dimensions
        model_input_params["width"] = width
        model_input_params["height"] = height

        # Check if model supports steps parameter from config (specs.steps)
        supports_steps = "steps" in specs
        supports_cfg_scale = "cfg_scale" in specs

        logger.info(
            f"genplan.parameters.{stage_name}_capability",
            job_id=job_id,
            model_id=model_id,
            supports_steps=supports_steps,
            supports_cfg_scale=supports_cfg_scale,
            specs_keys=list(specs.keys()) if specs else [],
        )

        # Only add steps for models that support them (from config)
        if supports_steps:
            steps_config = specs.get("steps", {})
            max_steps = steps_config.get("max", 50) if isinstance(steps_config, dict) else 50

            # Steps based on complexity - clamped to model's max
            if complexity == "complex":
                model_input_params["steps"] = min(40, max_steps)
            elif complexity == "standard":
                model_input_params["steps"] = min(30, max_steps)
            else:  # simple
                model_input_params["steps"] = min(25, max_steps)

            logger.info(
                f"genplan.parameters.{stage_name}_steps_added",
                job_id=job_id,
                model_id=model_id,
                steps=model_input_params["steps"],
                max_steps=max_steps,
            )
        else:
            logger.info(
                f"genplan.parameters.{stage_name}_steps_skipped",
                job_id=job_id,
                model_id=model_id,
                reason="Model config does not include specs.steps",
            )

        # Only add cfg_scale for models that support it
        if supports_cfg_scale:
            cfg_config = specs.get("cfg_scale", {})
            default_cfg = cfg_config.get("default", 7.5) if isinstance(cfg_config, dict) else 7.5

            if complexity == "complex":
                model_input_params["cfg_scale"] = 8.0
            elif complexity == "standard":
                model_input_params["cfg_scale"] = 7.5
            else:  # simple
                model_input_params["cfg_scale"] = default_cfg

            logger.info(
                f"genplan.parameters.{stage_name}_cfg_scale_added",
                job_id=job_id,
                model_id=model_id,
                cfg_scale=model_input_params["cfg_scale"],
            )

        # Number of images
        model_input_params["num_images"] = requirements.get("num_images", 1)

        # Seed if specified
        if requirements.get("seed") is not None:
            model_input_params["seed"] = requirements["seed"]

        # Provider-specific params from stage config
        provider_params_config = stage_specs.get("provider_params", {})
        if isinstance(provider_params_config, dict):
            for param_name, param_config in provider_params_config.items():
                if isinstance(param_config, dict):
                    # Check for complexity-specific default
                    if complexity == "complex" and "complex_default" in param_config:
                        provider_params[param_name] = param_config["complex_default"]
                    elif "default" in param_config:
                        provider_params[param_name] = param_config["default"]

        # Override with user-specified provider params
        if requirements.get("provider_params"):
            provider_params.update(requirements["provider_params"])

        return {
            "model_input_params": model_input_params,
            "provider_params": provider_params,
        }

    async def _validate_plan(
        self,
        state: GenPlanState,
        requirements: Dict[str, Any],
    ) -> GenPlanState:
        """
        Validate the complete generation plan.

        THINK: Validate all required fields are populated and coherent.

        ACT: Check each component of the plan

        OBSERVE: Set state.validated and state.validation_errors
        """
        logger.info("genplan.validate.start", job_id=state.job_id)

        errors = []

        # Validate complexity
        if not state.complexity:
            errors.append("Missing complexity level")

        # Validate user_info
        if not state.user_info:
            errors.append("Missing user info")
        elif not state.user_info.subject:
            errors.append("Missing subject in user info")

        # Validate genflow
        if not state.genflow:
            errors.append("Missing genflow configuration")

        # Validate model
        if not state.model_id:
            errors.append("Missing model selection")

        # Validate parameters
        if not state.parameters_extracted:
            errors.append("Parameters not extracted")
        elif not state.model_input_params.get("width"):
            errors.append("Missing width in parameters")

        # Validate pipeline for dual flow
        if state.genflow and state.genflow.is_dual:
            if not state.pipeline:
                errors.append("Missing pipeline config for dual flow")
            elif not state.pipeline.stage_2_model:
                errors.append("Missing stage 2 model for dual pipeline")

        state.validation_errors = errors
        state.validated = len(errors) == 0

        logger.info(
            "genplan.validate.complete",
            job_id=state.job_id,
            is_valid=state.validated,
            error_count=len(errors),
            errors=errors if errors else None,
        )

        return state

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _build_text_for_analysis(self, requirements: Dict[str, Any]) -> str:
        """Build combined text content for analysis."""
        parts = [
            str(requirements.get("subject", "")),
            str(requirements.get("prompt", "")),
            str(requirements.get("description", "")),
            str(requirements.get("style", "")),
            str(requirements.get("product_type", "")),
        ]
        return " ".join(parts).lower()

    def _extract_intents(self, requirements: Dict[str, Any]) -> List[str]:
        """Extract user intents from requirements."""
        intents = []
        text_content = self._build_text_for_analysis(requirements)

        intent_keywords = {
            "poster": ["poster", "wall art", "print"],
            "portrait": ["portrait", "face", "headshot"],
            "product": ["product", "item", "merchandise"],
            "text_overlay": ["text", "typography", "words", "title"],
            "vintage": ["vintage", "retro", "old school"],
            "modern": ["modern", "contemporary", "minimalist"],
            "character": ["character", "person", "figure"],
        }

        for intent, keywords in intent_keywords.items():
            if any(kw in text_content for kw in keywords):
                intents.append(intent)

        return intents

    def _get_dimensions_for_product(
        self,
        product_type: str,
        requirements: Dict[str, Any],
    ) -> Tuple[int, int]:
        """Get appropriate dimensions for product type."""
        # Check for explicit dimensions
        if requirements.get("width") and requirements.get("height"):
            return requirements["width"], requirements["height"]

        # Check for aspect ratio
        aspect_ratio = requirements.get("aspect_ratio", "")

        # Product-specific defaults
        dimension_map = {
            "poster": (1344, 768),    # 16:9
            "t-shirt": (1024, 1024),  # 1:1
            "phone_case": (768, 1344),  # 9:16
            "canvas": (1152, 896),    # 4:3
            "social": (1024, 1024),   # 1:1
            "banner": (1536, 640),    # 2.4:1
            "general": (1024, 1024),  # 1:1 default
        }

        # Aspect ratio overrides
        if aspect_ratio:
            aspect_map = {
                "1:1": (1024, 1024),
                "16:9": (1344, 768),
                "9:16": (768, 1344),
                "4:3": (1152, 896),
                "3:4": (896, 1152),
            }
            if aspect_ratio in aspect_map:
                return aspect_map[aspect_ratio]

        return dimension_map.get(product_type, dimension_map["general"])

    def _create_clarification_result(
        self,
        question_data: Dict[str, Any],
        state: GenPlanState,
    ) -> AgentResult:
        """Create result requesting user clarification."""
        logger.info(
            "genplan.clarification.requested",
            job_id=state.job_id,
            field=question_data.get("field"),
        )

        return self._create_result(
            success=True,
            data={
                "needs_clarification": True,
                "clarification_request": question_data,
                "partial_state": {
                    "complexity": state.complexity,
                    "steps_completed": state.steps_completed,
                },
            },
            requires_user_input=True,
            next_agent="pali",  # Route through Pali for user communication
        )

    # =========================================================================
    # Resource Management
    # =========================================================================

    async def __aenter__(self) -> "GenPlanAgent":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()
