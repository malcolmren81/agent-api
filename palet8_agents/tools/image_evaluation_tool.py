"""
Image Evaluation Tool - Evaluate generated image quality.

This tool wraps ResultEvaluationService to provide agents with
the ability to evaluate generated images for quality and prompt adherence.
"""

from typing import Any, Dict, List, Optional
import logging

from palet8_agents.tools.base import BaseTool, ToolParameter, ParameterType, ToolResult
from palet8_agents.services.result_evaluation_service import (
    ResultEvaluationService,
    ResultEvaluationError,
)
from palet8_agents.models import EvaluationPlan, ResultQualityResult

logger = logging.getLogger(__name__)


class ImageEvaluationTool(BaseTool):
    """
    Tool for evaluating generated image quality.

    Actions:
    - evaluate_image: Evaluate single image quality
    - evaluate_set: Evaluate multiple images
    - should_retry: Check if generation should be retried
    - get_weights: Get evaluation weights for a mode
    - get_thresholds: Get evaluation thresholds for a mode
    """

    def __init__(
        self,
        evaluation_service: Optional[ResultEvaluationService] = None,
    ):
        """
        Initialize the Image Evaluation Tool.

        Args:
            evaluation_service: Optional service instance for dependency injection
        """
        super().__init__(
            name="image_evaluation",
            description="Evaluate generated images for quality and prompt adherence",
            parameters=[
                ToolParameter(
                    name="action",
                    type=ParameterType.STRING,
                    description="Action to perform",
                    required=True,
                    enum=[
                        "evaluate_image",
                        "evaluate_set",
                        "should_retry",
                        "get_weights",
                        "get_thresholds",
                    ],
                ),
                ToolParameter(
                    name="image_data",
                    type=ParameterType.OBJECT,
                    description="Image metadata and analysis results",
                    required=False,
                ),
                ToolParameter(
                    name="images",
                    type=ParameterType.ARRAY,
                    description="Array of image data for evaluate_set",
                    required=False,
                ),
                ToolParameter(
                    name="plan",
                    type=ParameterType.OBJECT,
                    description="EvaluationPlan with prompt and context",
                    required=False,
                ),
                ToolParameter(
                    name="result",
                    type=ParameterType.OBJECT,
                    description="ResultQualityResult for should_retry check",
                    required=False,
                ),
                ToolParameter(
                    name="attempt_count",
                    type=ParameterType.INTEGER,
                    description="Current attempt count for retry check",
                    required=False,
                    default=1,
                ),
                ToolParameter(
                    name="mode",
                    type=ParameterType.STRING,
                    description="Generation mode for weights/thresholds",
                    required=False,
                    enum=["RELAX", "STANDARD", "COMPLEX"],
                    default="STANDARD",
                ),
            ],
        )

        self._evaluation_service = evaluation_service
        self._owns_service = evaluation_service is None

    async def _get_service(self) -> ResultEvaluationService:
        """Get or create evaluation service."""
        if self._evaluation_service is None:
            self._evaluation_service = ResultEvaluationService()
        return self._evaluation_service

    async def close(self) -> None:
        """Close resources."""
        if self._evaluation_service and self._owns_service:
            await self._evaluation_service.close()
            self._evaluation_service = None

    async def execute(self, **kwargs) -> ToolResult:
        """
        Execute image evaluation action.

        Args:
            action: The action to perform
            image_data: Image metadata
            images: Array of images for set evaluation
            plan: Evaluation plan
            result: Previous result for retry check
            attempt_count: Current attempt count
            mode: Generation mode

        Returns:
            ToolResult with evaluation data
        """
        action = kwargs.get("action")
        image_data = kwargs.get("image_data", {})
        images = kwargs.get("images", [])
        plan = kwargs.get("plan")
        result = kwargs.get("result")
        attempt_count = kwargs.get("attempt_count", 1)
        mode = kwargs.get("mode", "STANDARD")

        try:
            if action == "evaluate_image":
                return await self._evaluate_image(image_data, plan)
            elif action == "evaluate_set":
                return await self._evaluate_set(images, plan)
            elif action == "should_retry":
                return await self._should_retry(result, attempt_count)
            elif action == "get_weights":
                return await self._get_weights(mode)
            elif action == "get_thresholds":
                return await self._get_thresholds(mode)
            else:
                return ToolResult(
                    success=False,
                    data=None,
                    error=f"Unknown action: {action}",
                    error_code="INVALID_ACTION",
                )

        except ResultEvaluationError as e:
            logger.error(f"Result evaluation error: {e}")
            return ToolResult(
                success=False,
                data=None,
                error=f"Result evaluation error: {e}",
                error_code="EVALUATION_ERROR",
            )
        except Exception as e:
            logger.error(f"Unexpected error in ImageEvaluationTool: {e}", exc_info=True)
            return ToolResult(
                success=False,
                data=None,
                error=f"Unexpected error: {e}",
                error_code="INTERNAL_ERROR",
            )

    async def _evaluate_image(
        self,
        image_data: Dict[str, Any],
        plan: Optional[Dict[str, Any]],
    ) -> ToolResult:
        """Evaluate single image quality."""
        if not plan:
            return ToolResult(
                success=False,
                data=None,
                error="plan is required for evaluate_image",
                error_code="MISSING_PARAMETER",
            )

        # Convert dict to EvaluationPlan
        eval_plan = EvaluationPlan.from_dict(plan)

        service = await self._get_service()
        result = await service.evaluate_image(image_data, eval_plan)

        return ToolResult(
            success=True,
            data={
                "overall": result.overall,
                "dimensions": result.dimensions,
                "mode": result.mode,
                "threshold": result.threshold,
                "decision": result.decision,
                "feedback": result.feedback,
                "failed_dimensions": result.failed_dimensions,
                "retry_suggestions": [
                    {"dimension": s.dimension, "suggested_changes": s.suggested_changes}
                    for s in result.retry_suggestions
                ],
                "is_acceptable": result.is_acceptable,
            },
        )

    async def _evaluate_set(
        self,
        images: List[Dict[str, Any]],
        plan: Optional[Dict[str, Any]],
    ) -> ToolResult:
        """Evaluate multiple images."""
        if not images:
            return ToolResult(
                success=False,
                data=None,
                error="images is required for evaluate_set",
                error_code="MISSING_PARAMETER",
            )

        if not plan:
            return ToolResult(
                success=False,
                data=None,
                error="plan is required for evaluate_set",
                error_code="MISSING_PARAMETER",
            )

        eval_plan = EvaluationPlan.from_dict(plan)

        service = await self._get_service()
        results = await service.evaluate_image_set(images, eval_plan)

        return ToolResult(
            success=True,
            data={
                "results": [
                    {
                        "overall": r.overall,
                        "decision": r.decision,
                        "feedback": r.feedback,
                        "is_acceptable": r.is_acceptable,
                    }
                    for r in results
                ],
                "count": len(results),
                "passed_count": sum(1 for r in results if r.is_acceptable),
            },
        )

    async def _should_retry(
        self,
        result: Optional[Dict[str, Any]],
        attempt_count: int,
    ) -> ToolResult:
        """Check if generation should be retried."""
        if not result:
            return ToolResult(
                success=False,
                data=None,
                error="result is required for should_retry",
                error_code="MISSING_PARAMETER",
            )

        # Convert dict to ResultQualityResult
        quality_result = ResultQualityResult.from_dict(result)

        service = await self._get_service()
        should_retry = service.should_retry(quality_result, attempt_count)

        return ToolResult(
            success=True,
            data={
                "should_retry": should_retry,
                "attempt_count": attempt_count,
                "decision": quality_result.decision,
                "mode": quality_result.mode,
            },
        )

    async def _get_weights(self, mode: str) -> ToolResult:
        """Get evaluation weights for a mode."""
        service = await self._get_service()
        weights = service.get_weights(mode)

        return ToolResult(
            success=True,
            data={
                "mode": mode,
                "weights": weights,
            },
        )

    async def _get_thresholds(self, mode: str) -> ToolResult:
        """Get evaluation thresholds for a mode."""
        service = await self._get_service()
        thresholds = service.get_thresholds(mode)

        return ToolResult(
            success=True,
            data={
                "mode": mode,
                "thresholds": thresholds,
            },
        )

    async def __aenter__(self) -> "ImageEvaluationTool":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()
