"""
Requirements Tool - Analyze conversation for design requirements.

This tool wraps RequirementsAnalysisService to provide agents with
the ability to extract and analyze design requirements from conversations.
"""

from typing import Any, Dict, List, Optional
import logging

from palet8_agents.tools.base import BaseTool, ToolParameter, ParameterType, ToolResult
from palet8_agents.services.requirements_analysis_service import (
    RequirementsAnalysisService,
    RequirementsAnalysisError,
)

logger = logging.getLogger(__name__)


class RequirementsTool(BaseTool):
    """
    Tool for analyzing design requirements from conversation.

    Actions:
    - analyze_requirements: Extract requirements from conversation
    - get_completeness: Calculate completeness score
    - get_missing_fields: Get list of missing required/recommended fields
    - is_complete: Check if minimum requirements are met
    """

    def __init__(
        self,
        requirements_service: Optional[RequirementsAnalysisService] = None,
    ):
        """
        Initialize the Requirements Tool.

        Args:
            requirements_service: Optional service instance for dependency injection
        """
        super().__init__(
            name="requirements",
            description="Analyze conversation to extract and score design requirements",
            parameters=[
                ToolParameter(
                    name="action",
                    type=ParameterType.STRING,
                    description="Action to perform",
                    required=True,
                    enum=[
                        "analyze_requirements",
                        "get_completeness",
                        "get_missing_fields",
                        "is_complete",
                    ],
                ),
                ToolParameter(
                    name="conversation",
                    type=ParameterType.OBJECT,
                    description="Conversation history to analyze (list of message dicts)",
                    required=False,
                ),
                ToolParameter(
                    name="requirements",
                    type=ParameterType.OBJECT,
                    description="Requirements dict for completeness checks",
                    required=False,
                ),
            ],
        )

        self._requirements_service = requirements_service
        self._owns_service = requirements_service is None

    async def _get_service(self) -> RequirementsAnalysisService:
        """Get or create requirements service."""
        if self._requirements_service is None:
            self._requirements_service = RequirementsAnalysisService()
        return self._requirements_service

    async def close(self) -> None:
        """Close resources."""
        if self._requirements_service and self._owns_service:
            await self._requirements_service.close()
            self._requirements_service = None

    async def execute(self, **kwargs) -> ToolResult:
        """
        Execute requirements analysis action.

        Args:
            action: The action to perform
            conversation: Conversation history for analysis
            requirements: Requirements dict for completeness checks

        Returns:
            ToolResult with analysis results
        """
        action = kwargs.get("action")
        conversation = kwargs.get("conversation")
        requirements = kwargs.get("requirements", {})

        try:
            if action == "analyze_requirements":
                return await self._analyze_requirements(conversation)
            elif action == "get_completeness":
                return await self._get_completeness(requirements)
            elif action == "get_missing_fields":
                return await self._get_missing_fields(requirements)
            elif action == "is_complete":
                return await self._is_complete(requirements)
            else:
                return ToolResult(
                    success=False,
                    data=None,
                    error=f"Unknown action: {action}",
                    error_code="INVALID_ACTION",
                )

        except RequirementsAnalysisError as e:
            logger.error(f"Requirements analysis error: {e}")
            return ToolResult(
                success=False,
                data=None,
                error=f"Requirements analysis error: {e}",
                error_code="REQUIREMENTS_ERROR",
            )
        except Exception as e:
            logger.error(f"Unexpected error in RequirementsTool: {e}", exc_info=True)
            return ToolResult(
                success=False,
                data=None,
                error=f"Unexpected error: {e}",
                error_code="INTERNAL_ERROR",
            )

    async def _analyze_requirements(
        self,
        conversation: Optional[List[Dict[str, Any]]],
    ) -> ToolResult:
        """Analyze conversation to extract requirements."""
        if not conversation:
            return ToolResult(
                success=False,
                data=None,
                error="conversation is required for analyze_requirements",
                error_code="MISSING_PARAMETER",
            )

        service = await self._get_service()
        result = await service.analyze_conversation(conversation)

        return ToolResult(
            success=True,
            data={
                "requirements": result.to_dict(),
                "completeness_score": result.completeness_score,
                "is_complete": result.is_complete,
                "missing_fields": result.missing_fields,
            },
        )

    async def _get_completeness(
        self,
        requirements: Dict[str, Any],
    ) -> ToolResult:
        """Calculate completeness score for requirements."""
        service = await self._get_service()
        score = service.calculate_completeness(requirements)

        return ToolResult(
            success=True,
            data={
                "completeness_score": score,
                "is_sufficient": score >= 0.5,
            },
        )

    async def _get_missing_fields(
        self,
        requirements: Dict[str, Any],
    ) -> ToolResult:
        """Get list of missing fields."""
        service = await self._get_service()
        missing = service.get_missing_fields(requirements)

        return ToolResult(
            success=True,
            data={
                "missing_fields": missing,
                "count": len(missing),
            },
        )

    async def _is_complete(
        self,
        requirements: Dict[str, Any],
    ) -> ToolResult:
        """Check if minimum requirements are met."""
        service = await self._get_service()
        complete = service.is_complete(requirements)

        return ToolResult(
            success=True,
            data={
                "is_complete": complete,
                "has_subject": bool(requirements.get("subject")),
            },
        )

    async def __aenter__(self) -> "RequirementsTool":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()
