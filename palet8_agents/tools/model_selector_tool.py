"""
Model Selector Tool - Select optimal model and pipeline.

This tool wraps ModelSelectionService to provide agents with
the ability to select image generation models and pipelines.
"""

from typing import Any, Dict, Optional
import logging

from palet8_agents.tools.base import BaseTool, ToolParameter, ParameterType, ToolResult
from palet8_agents.services.model_selection_service import (
    ModelSelectionService,
    ModelSelectionError,
)

logger = logging.getLogger(__name__)


class ModelSelectorTool(BaseTool):
    """
    Tool for selecting image generation models and pipelines.

    Actions:
    - select_model: Select optimal model based on requirements
    - select_pipeline: Decide single vs dual pipeline
    - estimate_cost: Estimate cost and latency
    - get_pipelines: Get available pipeline configurations
    """

    def __init__(
        self,
        model_service: Optional[ModelSelectionService] = None,
    ):
        """
        Initialize the Model Selector Tool.

        Args:
            model_service: Optional service instance for dependency injection
        """
        super().__init__(
            name="model_selector",
            description="Select optimal model and pipeline for image generation",
            parameters=[
                ToolParameter(
                    name="action",
                    type=ParameterType.STRING,
                    description="Action to perform",
                    required=True,
                    enum=[
                        "select_model",
                        "select_pipeline",
                        "estimate_cost",
                        "get_pipelines",
                    ],
                ),
                ToolParameter(
                    name="mode",
                    type=ParameterType.STRING,
                    description="Generation mode: RELAX, STANDARD, COMPLEX",
                    required=False,
                    enum=["RELAX", "STANDARD", "COMPLEX"],
                    default="STANDARD",
                ),
                ToolParameter(
                    name="requirements",
                    type=ParameterType.OBJECT,
                    description="User requirements dict",
                    required=False,
                ),
                ToolParameter(
                    name="prompt",
                    type=ParameterType.STRING,
                    description="Generated prompt for pipeline decision",
                    required=False,
                ),
                ToolParameter(
                    name="num_images",
                    type=ParameterType.INTEGER,
                    description="Number of images to generate",
                    required=False,
                    default=1,
                ),
                ToolParameter(
                    name="model_info_context",
                    type=ParameterType.OBJECT,
                    description="Context with available models and compatibility info",
                    required=False,
                ),
            ],
        )

        self._model_service = model_service
        self._owns_service = model_service is None

    async def _get_service(self) -> ModelSelectionService:
        """Get or create model service."""
        if self._model_service is None:
            self._model_service = ModelSelectionService()
        return self._model_service

    async def close(self) -> None:
        """Close resources."""
        if self._model_service and self._owns_service:
            await self._model_service.close()
            self._model_service = None

    async def execute(self, **kwargs) -> ToolResult:
        """
        Execute model selection action.

        Args:
            action: The action to perform
            mode: Generation mode
            requirements: User requirements
            prompt: Generated prompt
            num_images: Number of images
            model_info_context: Available models context

        Returns:
            ToolResult with model/pipeline data
        """
        action = kwargs.get("action")
        mode = kwargs.get("mode", "STANDARD")
        requirements = kwargs.get("requirements", {})
        prompt = kwargs.get("prompt", "")
        num_images = kwargs.get("num_images", 1)
        model_info_context = kwargs.get("model_info_context")

        try:
            if action == "select_model":
                return await self._select_model(mode, requirements, model_info_context)
            elif action == "select_pipeline":
                return await self._select_pipeline(requirements, prompt)
            elif action == "estimate_cost":
                return await self._estimate_cost(requirements, prompt, num_images)
            elif action == "get_pipelines":
                return await self._get_pipelines()
            else:
                return ToolResult(
                    success=False,
                    data=None,
                    error=f"Unknown action: {action}",
                    error_code="INVALID_ACTION",
                )

        except ModelSelectionError as e:
            logger.error(f"Model selection error: {e}")
            return ToolResult(
                success=False,
                data=None,
                error=f"Model selection error: {e}",
                error_code="MODEL_ERROR",
            )
        except Exception as e:
            logger.error(f"Unexpected error in ModelSelectorTool: {e}", exc_info=True)
            return ToolResult(
                success=False,
                data=None,
                error=f"Unexpected error: {e}",
                error_code="INTERNAL_ERROR",
            )

    async def _select_model(
        self,
        mode: str,
        requirements: Dict[str, Any],
        model_info_context: Optional[Dict[str, Any]],
    ) -> ToolResult:
        """Select optimal model based on requirements."""
        service = await self._get_service()
        model_id, rationale, alternatives, model_specs = await service.select_model(
            mode=mode,
            requirements=requirements,
            model_info_context=model_info_context,
        )

        return ToolResult(
            success=True,
            data={
                "model_id": model_id,
                "rationale": rationale,
                "alternatives": alternatives,
                "model_specs": model_specs,
            },
        )

    async def _select_pipeline(
        self,
        requirements: Dict[str, Any],
        prompt: str,
    ) -> ToolResult:
        """Decide single vs dual pipeline."""
        service = await self._get_service()
        pipeline = service.select_pipeline(
            requirements=requirements,
            prompt=prompt,
        )

        return ToolResult(
            success=True,
            data={
                "pipeline_type": pipeline.pipeline_type,
                "pipeline_name": pipeline.pipeline_name,
                "stage_1_model": pipeline.stage_1_model,
                "stage_1_purpose": pipeline.stage_1_purpose,
                "stage_2_model": pipeline.stage_2_model,
                "stage_2_purpose": pipeline.stage_2_purpose,
                "decision_rationale": pipeline.decision_rationale,
            },
        )

    async def _estimate_cost(
        self,
        requirements: Dict[str, Any],
        prompt: str,
        num_images: int,
    ) -> ToolResult:
        """Estimate cost and latency."""
        service = await self._get_service()

        # First select pipeline to estimate properly
        pipeline = service.select_pipeline(requirements, prompt)
        cost, latency = service.estimate_cost(pipeline, num_images=num_images)

        return ToolResult(
            success=True,
            data={
                "estimated_cost": cost,
                "estimated_latency_ms": latency,
                "pipeline_type": pipeline.pipeline_type,
                "num_images": num_images,
            },
        )

    async def _get_pipelines(self) -> ToolResult:
        """Get available pipeline configurations."""
        service = await self._get_service()
        pipelines = service.get_available_pipelines()
        triggers = service.get_pipeline_triggers()

        return ToolResult(
            success=True,
            data={
                "pipelines": pipelines,
                "triggers": triggers,
            },
        )

    async def __aenter__(self) -> "ModelSelectorTool":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()
