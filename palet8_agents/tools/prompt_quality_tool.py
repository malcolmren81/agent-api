"""
Prompt Quality Tool - Assess and revise prompt quality.

This tool wraps PromptEvaluationService to provide agents with
the ability to assess prompt quality and propose revisions.
"""

from typing import Any, Dict, Optional
import logging

from palet8_agents.tools.base import BaseTool, ToolParameter, ParameterType, ToolResult
from palet8_agents.services.prompt_evaluation_service import (
    PromptEvaluationService,
    PromptEvaluationError,
)
from palet8_agents.models import PromptQualityResult

logger = logging.getLogger(__name__)


class PromptQualityTool(BaseTool):
    """
    Tool for assessing and revising prompt quality.

    Actions:
    - assess_quality: Assess prompt quality across dimensions
    - revise_prompt: Propose improved prompt based on feedback
    - get_weights: Get weights for a mode
    - get_thresholds: Get thresholds for a mode
    """

    def __init__(
        self,
        prompt_service: Optional[PromptEvaluationService] = None,
    ):
        """
        Initialize the Prompt Quality Tool.

        Args:
            prompt_service: Optional service instance for dependency injection
        """
        super().__init__(
            name="prompt_quality",
            description="Assess prompt quality and propose revisions",
            parameters=[
                ToolParameter(
                    name="action",
                    type=ParameterType.STRING,
                    description="Action to perform",
                    required=True,
                    enum=[
                        "assess_quality",
                        "revise_prompt",
                        "get_weights",
                        "get_thresholds",
                    ],
                ),
                ToolParameter(
                    name="prompt",
                    type=ParameterType.STRING,
                    description="Positive prompt to evaluate",
                    required=False,
                ),
                ToolParameter(
                    name="negative_prompt",
                    type=ParameterType.STRING,
                    description="Negative prompt",
                    required=False,
                    default="",
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
                    name="product_type",
                    type=ParameterType.STRING,
                    description="Target product type",
                    required=False,
                ),
                ToolParameter(
                    name="print_method",
                    type=ParameterType.STRING,
                    description="Print method (screen_print, DTG, etc.)",
                    required=False,
                ),
                ToolParameter(
                    name="dimensions",
                    type=ParameterType.OBJECT,
                    description="Requested dimensions for coverage check",
                    required=False,
                ),
                ToolParameter(
                    name="quality_result",
                    type=ParameterType.OBJECT,
                    description="Previous quality result for revision",
                    required=False,
                ),
            ],
        )

        self._prompt_service = prompt_service
        self._owns_service = prompt_service is None

    async def _get_service(self) -> PromptEvaluationService:
        """Get or create prompt service."""
        if self._prompt_service is None:
            self._prompt_service = PromptEvaluationService()
        return self._prompt_service

    async def close(self) -> None:
        """Close resources."""
        if self._prompt_service and self._owns_service:
            await self._prompt_service.close()
            self._prompt_service = None

    async def execute(self, **kwargs) -> ToolResult:
        """
        Execute prompt quality action.

        Args:
            action: The action to perform
            prompt: Positive prompt
            negative_prompt: Negative prompt
            mode: Generation mode
            product_type: Target product
            print_method: Print method
            dimensions: Requested dimensions
            quality_result: Previous quality result

        Returns:
            ToolResult with quality data
        """
        action = kwargs.get("action")
        prompt = kwargs.get("prompt", "")
        negative_prompt = kwargs.get("negative_prompt", "")
        mode = kwargs.get("mode", "STANDARD")
        product_type = kwargs.get("product_type")
        print_method = kwargs.get("print_method")
        dimensions = kwargs.get("dimensions", {})
        quality_result = kwargs.get("quality_result")

        try:
            if action == "assess_quality":
                return await self._assess_quality(
                    prompt, negative_prompt, mode, product_type, print_method, dimensions
                )
            elif action == "revise_prompt":
                return await self._revise_prompt(prompt, quality_result)
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

        except PromptEvaluationError as e:
            logger.error(f"Prompt evaluation error: {e}")
            return ToolResult(
                success=False,
                data=None,
                error=f"Prompt evaluation error: {e}",
                error_code="PROMPT_ERROR",
            )
        except Exception as e:
            logger.error(f"Unexpected error in PromptQualityTool: {e}", exc_info=True)
            return ToolResult(
                success=False,
                data=None,
                error=f"Unexpected error: {e}",
                error_code="INTERNAL_ERROR",
            )

    async def _assess_quality(
        self,
        prompt: str,
        negative_prompt: str,
        mode: str,
        product_type: Optional[str],
        print_method: Optional[str],
        dimensions: Dict[str, Any],
    ) -> ToolResult:
        """Assess prompt quality across dimensions."""
        if not prompt:
            return ToolResult(
                success=False,
                data=None,
                error="prompt is required for assess_quality",
                error_code="MISSING_PARAMETER",
            )

        service = await self._get_service()
        result = service.assess_quality(
            prompt=prompt,
            negative_prompt=negative_prompt,
            mode=mode,
            product_type=product_type,
            print_method=print_method,
            dimensions=dimensions,
        )

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
                "acceptable": service.is_acceptable(result),
            },
        )

    async def _revise_prompt(
        self,
        prompt: str,
        quality_result: Optional[Dict[str, Any]],
    ) -> ToolResult:
        """Propose improved prompt based on feedback."""
        if not prompt:
            return ToolResult(
                success=False,
                data=None,
                error="prompt is required for revise_prompt",
                error_code="MISSING_PARAMETER",
            )

        if not quality_result:
            return ToolResult(
                success=False,
                data=None,
                error="quality_result is required for revise_prompt",
                error_code="MISSING_PARAMETER",
            )

        # Convert dict to PromptQualityResult
        result = PromptQualityResult.from_dict(quality_result)

        service = await self._get_service()
        revised = await service.propose_revision(prompt, result)

        return ToolResult(
            success=True,
            data={
                "original_prompt": prompt,
                "revised_prompt": revised,
                "feedback_addressed": result.feedback[:3],
            },
        )

    async def _get_weights(self, mode: str) -> ToolResult:
        """Get weights for a mode."""
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
        """Get thresholds for a mode."""
        service = await self._get_service()
        thresholds = service.get_thresholds(mode)

        return ToolResult(
            success=True,
            data={
                "mode": mode,
                "thresholds": thresholds,
            },
        )

    async def __aenter__(self) -> "PromptQualityTool":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()
