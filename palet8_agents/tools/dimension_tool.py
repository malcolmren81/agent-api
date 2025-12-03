"""
Dimension Tool - Select and fill prompt dimensions.

This tool wraps DimensionSelectionService to provide agents with
the ability to select dimensions based on mode, product, and style.
"""

from typing import Any, Dict, Optional
import logging

from palet8_agents.tools.base import BaseTool, ToolParameter, ParameterType, ToolResult
from palet8_agents.services.dimension_selection_service import (
    DimensionSelectionService,
    DimensionSelectionError,
)

logger = logging.getLogger(__name__)


class DimensionTool(BaseTool):
    """
    Tool for selecting prompt dimensions.

    Actions:
    - select_dimensions: Select dimensions for prompt composition
    - get_required: Get required dimensions for a mode
    - get_missing: Get missing dimensions for a mode
    - map_requirements: Map requirements to dimension fields
    """

    def __init__(
        self,
        dimension_service: Optional[DimensionSelectionService] = None,
    ):
        """
        Initialize the Dimension Tool.

        Args:
            dimension_service: Optional service instance for dependency injection
        """
        super().__init__(
            name="dimension",
            description="Select dimensions for prompt composition based on mode and requirements",
            parameters=[
                ToolParameter(
                    name="action",
                    type=ParameterType.STRING,
                    description="Action to perform",
                    required=True,
                    enum=[
                        "select_dimensions",
                        "get_required",
                        "get_missing",
                        "map_requirements",
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
                    name="product_type",
                    type=ParameterType.STRING,
                    description="Target product type (apparel, mug, poster, etc.)",
                    required=False,
                ),
                ToolParameter(
                    name="print_method",
                    type=ParameterType.STRING,
                    description="Print method (screen_print, DTG, embroidery, etc.)",
                    required=False,
                ),
                ToolParameter(
                    name="dimensions",
                    type=ParameterType.OBJECT,
                    description="Current dimensions for get_missing action",
                    required=False,
                ),
            ],
        )

        self._dimension_service = dimension_service
        self._owns_service = dimension_service is None

    async def _get_service(self) -> DimensionSelectionService:
        """Get or create dimension service."""
        if self._dimension_service is None:
            self._dimension_service = DimensionSelectionService()
        return self._dimension_service

    async def close(self) -> None:
        """Close resources."""
        if self._dimension_service and self._owns_service:
            await self._dimension_service.close()
            self._dimension_service = None

    async def execute(self, **kwargs) -> ToolResult:
        """
        Execute dimension selection action.

        Args:
            action: The action to perform
            mode: Generation mode
            requirements: User requirements
            product_type: Target product
            print_method: Print method
            dimensions: Current dimensions

        Returns:
            ToolResult with dimension data
        """
        action = kwargs.get("action")
        mode = kwargs.get("mode", "STANDARD")
        requirements = kwargs.get("requirements", {})
        product_type = kwargs.get("product_type")
        print_method = kwargs.get("print_method")
        dimensions = kwargs.get("dimensions", {})

        try:
            if action == "select_dimensions":
                return await self._select_dimensions(
                    mode, requirements, product_type, print_method
                )
            elif action == "get_required":
                return await self._get_required(mode)
            elif action == "get_missing":
                return await self._get_missing(dimensions, mode)
            elif action == "map_requirements":
                return await self._map_requirements(requirements)
            else:
                return ToolResult(
                    success=False,
                    data=None,
                    error=f"Unknown action: {action}",
                    error_code="INVALID_ACTION",
                )

        except DimensionSelectionError as e:
            logger.error(f"Dimension selection error: {e}")
            return ToolResult(
                success=False,
                data=None,
                error=f"Dimension selection error: {e}",
                error_code="DIMENSION_ERROR",
            )
        except Exception as e:
            logger.error(f"Unexpected error in DimensionTool: {e}", exc_info=True)
            return ToolResult(
                success=False,
                data=None,
                error=f"Unexpected error: {e}",
                error_code="INTERNAL_ERROR",
            )

    async def _select_dimensions(
        self,
        mode: str,
        requirements: Dict[str, Any],
        product_type: Optional[str],
        print_method: Optional[str],
    ) -> ToolResult:
        """Select and fill dimensions for prompt composition."""
        service = await self._get_service()
        dimensions = await service.select_dimensions(
            mode=mode,
            requirements=requirements,
            product_type=product_type,
            print_method=print_method,
        )

        return ToolResult(
            success=True,
            data={
                "dimensions": dimensions.to_dict(),
                "mode": mode,
                "has_technical": bool(dimensions.technical),
            },
        )

    async def _get_required(self, mode: str) -> ToolResult:
        """Get required dimensions for a mode."""
        service = await self._get_service()
        required = service.get_required_dimensions(mode)

        return ToolResult(
            success=True,
            data={
                "required_dimensions": required,
                "mode": mode,
                "count": len(required),
            },
        )

    async def _get_missing(
        self,
        dimensions: Dict[str, Any],
        mode: str,
    ) -> ToolResult:
        """Get missing dimensions for a mode."""
        service = await self._get_service()

        # Convert dict to PromptDimensions if needed
        from palet8_agents.models import PromptDimensions
        if isinstance(dimensions, dict):
            dim_obj = PromptDimensions.from_dict(dimensions)
        else:
            dim_obj = dimensions

        missing = service.get_missing_dimensions(dim_obj, mode)

        return ToolResult(
            success=True,
            data={
                "missing_dimensions": missing,
                "mode": mode,
                "count": len(missing),
            },
        )

    async def _map_requirements(
        self,
        requirements: Dict[str, Any],
    ) -> ToolResult:
        """Map requirements to dimension fields."""
        service = await self._get_service()
        dimensions = service.map_requirements(requirements)

        return ToolResult(
            success=True,
            data={
                "dimensions": dimensions.to_dict(),
            },
        )

    async def __aenter__(self) -> "DimensionTool":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()
